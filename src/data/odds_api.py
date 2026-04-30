"""The Odds API client — aggregates odds from many bookmakers in one call.

Service: https://the-odds-api.com
Free tier: 500 requests/month. Enough for once-daily polling of our two
leagues (~30 fixtures × 2 markets × 1 call/day = 60 req/day → 1800/month).
We cache results and only refresh when needed.

Returns the BEST price across all monitored bookmakers per (match, market,
selection). The detector uses that best price for value calculations,
which gives the user the highest possible CLV automatically.

Setup: add ODDS_API_KEY to .env (sign up free at the-odds-api.com).
Without a key, this module is a no-op (returns empty list) and the rest
of the pipeline keeps working with Wplay-only odds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from loguru import logger


ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Map our internal league slug -> The Odds API sport_key
SPORT_KEY_BY_SLUG = {
    "premier_league": "soccer_epl",
    "liga_betplay": "soccer_colombia_primera",
    # add more as needed: spain_la_liga, italy_serie_a, etc.
}


@dataclass
class MultiBookieOdds:
    league_slug: str
    home_team: str
    away_team: str
    commence_time: datetime
    market: str          # "1x2" | "ou_2.5" | "btts" | etc.
    selection: str       # "home" | "draw" | "away" | "over" | "under" | "yes" | "no"
    best_odds: float
    best_bookmaker: str  # name of the casa with the best price
    casas_seen: int      # how many bookmakers offered this market


def _market_label(api_market: str) -> str:
    """Translate The Odds API market key to our internal market name."""
    return {
        "h2h": "1x2",
        "totals": "ou_2.5",  # default; we filter by point=2.5 below
        "btts": "btts",
    }.get(api_market, api_market)


async def fetch_multi_bookie_odds(
    league_slug: str, api_key: str, regions: str = "uk,eu,us",
    markets: str = "h2h,totals,btts",
) -> list[MultiBookieOdds]:
    """Hit /v4/sports/{sport_key}/odds with the given markets.

    Returns the BEST price per (match, market, selection) across all bookies.
    """
    sport_key = SPORT_KEY_BY_SLUG.get(league_slug)
    if not sport_key:
        return []
    if not api_key:
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning(f"odds-api {league_slug} failed: {exc}")
        return []

    out: list[MultiBookieOdds] = []
    for event in data:
        commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        home = event["home_team"]
        away = event["away_team"]
        # bookmakers list -> markets list -> outcomes list (with name + price)
        per_selection: dict[tuple[str, str], tuple[float, str, int]] = {}
        # (market, selection) -> (best_price, best_bookie, casas_seen)
        for bm in event.get("bookmakers", []):
            bookie_name = bm.get("title", bm.get("key", "?"))
            for mkt in bm.get("markets", []):
                api_market = mkt["key"]
                # totals market repeats per line (point); pull only 2.5
                point = mkt.get("point") if api_market == "totals" else None
                for outcome in mkt.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = float(outcome.get("price", 0))
                    if price <= 1.01:
                        continue

                    # Translate to our (market, selection)
                    if api_market == "h2h":
                        if name == home:
                            sel = "home"
                        elif name == away:
                            sel = "away"
                        elif name.lower() in ("draw", "tie"):
                            sel = "draw"
                        else:
                            continue
                        market_key = "1x2"
                    elif api_market == "totals":
                        if outcome.get("point") not in (2.5, point):
                            # The Odds API repeats point on each outcome
                            pass
                        if (point or outcome.get("point")) != 2.5:
                            continue
                        sel = "over" if name.lower() == "over" else "under"
                        market_key = "ou_2.5"
                    elif api_market == "btts":
                        sel = "yes" if name.lower() in ("yes", "btts yes") else "no"
                        market_key = "btts"
                    else:
                        continue

                    key = (market_key, sel)
                    prev = per_selection.get(key)
                    if prev is None or price > prev[0]:
                        per_selection[key] = (price, bookie_name,
                                              (prev[2] if prev else 0) + 1)
                    else:
                        per_selection[key] = (prev[0], prev[1], prev[2] + 1)

        for (market_key, sel), (best_price, best_bm, casas) in per_selection.items():
            out.append(MultiBookieOdds(
                league_slug=league_slug,
                home_team=home, away_team=away,
                commence_time=commence,
                market=market_key, selection=sel,
                best_odds=best_price, best_bookmaker=best_bm,
                casas_seen=casas,
            ))

    logger.info(f"odds-api {league_slug}: {len(data)} matches, {len(out)} best-price rows")
    return out


# Quota helper

async def fetch_remaining_quota(api_key: str) -> dict | None:
    """X-Requests-Remaining header is returned with every request. We check
    it via a minimal GET to /sports."""
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key})
            resp.raise_for_status()
            return {
                "remaining": resp.headers.get("X-Requests-Remaining", "?"),
                "used": resp.headers.get("X-Requests-Used", "?"),
                "last_cost": resp.headers.get("X-Requests-Last", "?"),
            }
    except httpx.HTTPError:
        return None
