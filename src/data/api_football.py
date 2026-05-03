"""api-football.com (api-sports.io v3) client.

Free tier: 100 requests / day, sin tarjeta.
Limitations on free plan:
    - No `last`/`next` parameters
    - No access to seasons 2025+ when filtered by season+league. BUT:
      `params={'date': 'YYYY-MM-DD'}` (without season filter) works for
      any date including current — we filter by league_id in Python.
    - Some markets/odds endpoints may be limited

Functions are best-effort: every call returns None on error rather than raising,
so /analizar can keep working even if the API is temporarily down or quota
is exhausted.

Setup: API_FOOTBALL_KEY in .env (sign up at https://dashboard.api-football.com/).
"""
from __future__ import annotations

import asyncio
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import httpx
from loguru import logger


API_BASE = "https://v3.football.api-sports.io"

# Map our internal slug -> api-football league_id
LEAGUE_ID_BY_SLUG = {
    "premier_league":   39,
    "liga_betplay":     239,
    "champions_league": 2,
    "libertadores":     13,
    "sudamericana":     11,
}


@dataclass
class LineupPlayer:
    name: str
    number: int | None
    pos: str         # G, D, M, F


@dataclass
class TeamLineup:
    team_name: str
    formation: str
    start_xi: list[LineupPlayer] = field(default_factory=list)
    substitutes: list[LineupPlayer] = field(default_factory=list)
    coach_name: str = ""


@dataclass
class Injury:
    player_name: str
    team_name: str
    type: str         # "Missing Fixture" | "Questionable"
    reason: str       # description


@dataclass
class OddsRow:
    bookmaker: str
    market: str       # "1x2" | "ou_2.5" | "btts" | "ah_-1.5" | etc
    selection: str
    odds: float


@dataclass
class FixtureMeta:
    fixture_id: int
    league_id: int
    league_name: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    kickoff_utc: datetime
    status_short: str  # NS, 1H, HT, 2H, FT, etc.


# ---------- HTTP plumbing ----------

def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


async def _get(api_key: str, endpoint: str, params: dict) -> dict | None:
    """Single GET; return parsed JSON or None on error/timeout."""
    if not api_key:
        return None
    url = f"{API_BASE}{endpoint}"
    headers = {"x-apisports-key": api_key}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            errs = data.get("errors")
            if errs:
                # api-football returns errors as either dict or list
                if isinstance(errs, dict) and errs:
                    logger.warning(f"api-football {endpoint} errors: {errs}")
                elif isinstance(errs, list) and errs:
                    logger.warning(f"api-football {endpoint} errors: {errs}")
            remaining = resp.headers.get("x-ratelimit-requests-remaining")
            if remaining is not None and int(remaining) < 10:
                logger.warning(f"api-football quota LOW: {remaining}/100 remaining today")
            return data
    except httpx.HTTPError as exc:
        logger.warning(f"api-football {endpoint} failed: {exc}")
        return None


# ---------- Fixture lookup ----------

# Cache fixtures by date so we don't fetch the same date twice in one process
_FIXTURES_CACHE: dict[str, list[dict]] = {}


async def find_fixture(
    api_key: str, kickoff_date: date,
    league_slug: str | None,
    home_query: str, away_query: str,
) -> FixtureMeta | None:
    """Find the api-football fixture for a match in our DB by (date, teams).

    Costs 1 API call per unique date (cached).
    """
    if not api_key:
        return None
    key = kickoff_date.isoformat()
    if key not in _FIXTURES_CACHE:
        data = await _get(api_key, "/fixtures", {"date": key})
        if data is None:
            return None
        _FIXTURES_CACHE[key] = data.get("response", []) or []
    candidates = _FIXTURES_CACHE[key]
    target_league_id = LEAGUE_ID_BY_SLUG.get(league_slug or "") if league_slug else None
    h_norm = _norm(home_query)
    a_norm = _norm(away_query)
    for f in candidates:
        if target_league_id and f.get("league", {}).get("id") != target_league_id:
            continue
        teams = f.get("teams", {})
        ah = _norm(teams.get("home", {}).get("name", ""))
        aa = _norm(teams.get("away", {}).get("name", ""))
        if ((h_norm in ah or ah in h_norm) and (a_norm in aa or aa in a_norm)):
            try:
                kt = datetime.fromisoformat(f["fixture"]["date"].replace("Z", "+00:00"))
            except Exception:
                kt = datetime.now(tz=timezone.utc)
            return FixtureMeta(
                fixture_id=int(f["fixture"]["id"]),
                league_id=int(f["league"]["id"]),
                league_name=f["league"].get("name", ""),
                home_team=teams["home"]["name"],
                away_team=teams["away"]["name"],
                home_team_id=int(teams["home"]["id"]),
                away_team_id=int(teams["away"]["id"]),
                kickoff_utc=kt,
                status_short=f["fixture"]["status"]["short"],
            )
    return None


# ---------- Lineups ----------

def _player(p: dict) -> LineupPlayer:
    return LineupPlayer(
        name=p.get("name") or "?",
        number=p.get("number"),
        pos=p.get("pos") or "?",
    )


async def fetch_lineups(api_key: str, fixture_id: int) -> list[TeamLineup]:
    """Return [home_lineup, away_lineup]. Empty if not yet announced
    (lineups appear ~1h before kickoff)."""
    if not api_key:
        return []
    data = await _get(api_key, "/fixtures/lineups", {"fixture": fixture_id})
    if data is None:
        return []
    out: list[TeamLineup] = []
    for team in data.get("response", []):
        team_obj = team.get("team", {})
        coach = team.get("coach", {}) or {}
        start_xi_raw = team.get("startXI") or []
        subs_raw = team.get("substitutes") or []
        out.append(TeamLineup(
            team_name=team_obj.get("name", "?"),
            formation=team.get("formation") or "",
            coach_name=coach.get("name") or "",
            start_xi=[_player(p["player"]) for p in start_xi_raw if p.get("player")],
            substitutes=[_player(p["player"]) for p in subs_raw if p.get("player")],
        ))
    return out


# ---------- Injuries ----------

async def fetch_injuries(api_key: str, fixture_id: int) -> list[Injury]:
    """Injury / suspension list for both teams in a fixture. Often empty for
    smaller leagues / older matches."""
    if not api_key:
        return []
    data = await _get(api_key, "/injuries", {"fixture": fixture_id})
    if data is None:
        return []
    out: list[Injury] = []
    for inj in data.get("response", []):
        p = inj.get("player", {}) or {}
        t = inj.get("team", {}) or {}
        out.append(Injury(
            player_name=p.get("name", "?"),
            team_name=t.get("name", "?"),
            type=p.get("type", "") or "",
            reason=p.get("reason", "") or "",
        ))
    return out


# ---------- Odds ----------

# Bookmaker IDs we treat as priority signals (Pinnacle is the sharp book)
PRIORITY_BOOKIES = {
    "Pinnacle", "Bet365", "Bwin", "10Bet", "Marathonbet", "Unibet", "William Hill",
}


@dataclass
class OddsConsensus:
    market: str
    selection: str
    pinnacle_odds: float | None
    best_odds: float
    best_bookie: str
    median_odds: float
    n_books: int


def _api_to_internal_market(bet_name: str, value) -> tuple[str, str] | None:
    """Translate api-football's bet name + value -> (market, selection) we use.

    api-football sometimes returns numeric values (1/2/3) instead of strings;
    coerce both to string for safety.
    """
    bn = (bet_name or "").lower()
    v = str(value if value is not None else "").strip()

    if bn in ("match winner", "fulltime result", "1x2"):
        if v.lower() in ("home", "1"):
            return "1x2", "home"
        if v.lower() in ("draw", "x"):
            return "1x2", "draw"
        if v.lower() in ("away", "2"):
            return "1x2", "away"
    elif bn in ("goals over/under", "goals over/under home", "goals over/under away", "total goals"):
        # value formats: "Over 2.5", "Under 2.5"
        parts = v.split()
        if len(parts) == 2 and parts[1] in ("1.5", "2.5", "3.5"):
            sel = "over" if parts[0].lower() == "over" else "under"
            return f"ou_{parts[1]}", sel
    elif bn in ("both teams to score", "btts"):
        if v.lower() == "yes":
            return "btts", "yes"
        if v.lower() == "no":
            return "btts", "no"
    return None


async def fetch_odds_consensus(api_key: str, fixture_id: int) -> list[OddsConsensus]:
    """Pull all books' odds for this fixture, aggregate per market+selection.
    Returns a list with best/median/Pinnacle for each (market, selection).

    Free tier note: this endpoint sometimes returns one bookmaker at a time
    on free; we'll get whatever they give us."""
    if not api_key:
        return []
    data = await _get(api_key, "/odds", {"fixture": fixture_id})
    if data is None:
        return []
    if not data.get("response"):
        return []

    # Aggregate: (market, sel) -> [(bookie, price)]
    by_sel: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for fixture_block in data.get("response", []):
        for bm in fixture_block.get("bookmakers", []):
            bookie = bm.get("name", "?")
            for bet in bm.get("bets", []):
                bet_name = bet.get("name", "")
                for v in bet.get("values", []):
                    try:
                        price = float(v.get("odd", 0))
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.01:
                        continue
                    sel = _api_to_internal_market(bet_name, v.get("value", ""))
                    if sel is None:
                        continue
                    by_sel.setdefault(sel, []).append((bookie, price))

    out: list[OddsConsensus] = []
    for (market, sel), book_prices in by_sel.items():
        if not book_prices:
            continue
        prices = [p for _, p in book_prices]
        prices_sorted = sorted(prices)
        n = len(prices_sorted)
        median = prices_sorted[n // 2] if n % 2 == 1 else (prices_sorted[n // 2 - 1] + prices_sorted[n // 2]) / 2
        best_bm, best_p = max(book_prices, key=lambda kv: kv[1])
        pinnacle = next((p for b, p in book_prices if b.lower() == "pinnacle"), None)
        out.append(OddsConsensus(
            market=market, selection=sel,
            pinnacle_odds=pinnacle,
            best_odds=best_p, best_bookie=best_bm,
            median_odds=median, n_books=n,
        ))
    return out


# ---------- Quota ----------

async def fetch_quota(api_key: str) -> dict | None:
    """Returns {plan, requests_today, requests_limit_day} or None."""
    if not api_key:
        return None
    data = await _get(api_key, "/status", {})
    if data is None:
        return None
    resp = data.get("response", {}) or {}
    sub = resp.get("subscription", {}) or {}
    req = resp.get("requests", {}) or {}
    return {
        "plan": sub.get("plan", "?"),
        "active": sub.get("active", False),
        "requests_today": req.get("current", 0),
        "requests_limit_day": req.get("limit_day", 100),
    }
