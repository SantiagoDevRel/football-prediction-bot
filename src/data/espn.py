"""ESPN public scoreboard API client.

Endpoint: https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard
No auth, no key, no quota. Used by ESPN's own apps and by la-polla in production.

Provides:
    - Current and recent fixtures (status, score, kickoff, teams)
    - Live status with displayClock
    - Historical scoreboard via ?dates=YYYYMMDD (one day at a time)

Does NOT provide:
    - xG (use Understat for Premier League; no source for Liga BetPlay)
    - Lineups (lightweight; use a separate endpoint or skip for v1)
    - Detailed odds (we have Wplay scraper for that)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
from loguru import logger

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

ESPN_LEAGUE_BY_SLUG: dict[str, str] = {
    "premier_league": "eng.1",
    "liga_betplay": "col.1",
    "sudamericana": "conmebol.sudamericana",
    "libertadores": "conmebol.libertadores",
}


@dataclass
class ESPNMatch:
    espn_id: str
    league_slug: str
    home_team: str
    away_team: str
    home_team_id: str
    away_team_id: str
    kickoff_utc: datetime
    status: str             # "scheduled" | "live" | "finished" | "cancelled"
    home_goals: int | None
    away_goals: int | None
    minute: int | None
    venue: str | None


def _map_status(espn_status_name: str) -> str:
    if espn_status_name in ("STATUS_SCHEDULED", "STATUS_DELAYED"):
        return "scheduled"
    live_set = {
        "STATUS_FIRST_HALF", "STATUS_HALFTIME", "STATUS_SECOND_HALF",
        "STATUS_END_OF_PERIOD", "STATUS_OVERTIME",
        "STATUS_FIRST_HALF_EXTRA_TIME", "STATUS_SECOND_HALF_EXTRA_TIME",
        "STATUS_END_OF_EXTRA_TIME", "STATUS_SHOOTOUT",
    }
    if espn_status_name in live_set:
        return "live"
    if espn_status_name in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_END_OF_REGULATION"):
        return "finished"
    if espn_status_name in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FORFEIT", "STATUS_ABANDONED"):
        return "cancelled"
    return "scheduled"  # safe default


def _parse_minute(display_clock: str | None) -> int | None:
    if not display_clock:
        return None
    s = display_clock.strip()
    if not s:
        return None
    if s.upper() == "HT":
        return 45
    if s.upper() == "FT":
        return None
    import re
    m = re.match(r"^(\d+)['′]?(?:\s*\+\s*(\d+)['′]?)?$", s)
    if not m:
        return None
    base = int(m.group(1))
    extra = int(m.group(2)) if m.group(2) else 0
    return base + extra


def _parse_event(event: dict, league_slug: str) -> ESPNMatch | None:
    """Convert raw ESPN event JSON -> ESPNMatch. Returns None on parse failures."""
    try:
        comp = event["competitions"][0]
        competitors = comp["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")

        status_obj = event["status"]
        status_name = status_obj["type"]["name"]

        kickoff = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))

        def parse_score(raw) -> int | None:
            if raw is None:
                return None
            s = str(raw).strip()
            if not s:
                return None
            try:
                return int(s)
            except ValueError:
                return None

        venue = None
        try:
            venue = comp.get("venue", {}).get("fullName")
        except (AttributeError, TypeError):
            pass

        return ESPNMatch(
            espn_id=str(event["id"]),
            league_slug=league_slug,
            home_team=home["team"]["displayName"],
            away_team=away["team"]["displayName"],
            home_team_id=str(home["team"]["id"]),
            away_team_id=str(away["team"]["id"]),
            kickoff_utc=kickoff,
            status=_map_status(status_name),
            home_goals=parse_score(home.get("score")),
            away_goals=parse_score(away.get("score")),
            minute=_parse_minute(status_obj.get("displayClock")),
            venue=venue,
        )
    except (KeyError, StopIteration, ValueError) as exc:
        logger.warning(f"failed to parse ESPN event {event.get('id', '?')}: {exc}")
        return None


async def fetch_scoreboard(
    league_slug: str, target_date: date | None = None, client: httpx.AsyncClient | None = None
) -> list[ESPNMatch]:
    """Fetch a league scoreboard. Defaults to today's matches.

    For historical pulls, pass target_date. ESPN keeps roughly the current and
    prior season accessible via the dates param.
    """
    league_code = ESPN_LEAGUE_BY_SLUG.get(league_slug)
    if not league_code:
        logger.warning(f"unknown league slug: {league_slug}")
        return []

    url = f"{ESPN_BASE}/{league_code}/scoreboard"
    params = {}
    if target_date is not None:
        params["dates"] = target_date.strftime("%Y%m%d")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=15.0, headers={"accept": "application/json"})

    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    finally:
        if own_client:
            await client.aclose()

    events = data.get("events", [])
    matches = [m for m in (_parse_event(e, league_slug) for e in events) if m is not None]
    return matches


async def fetch_match_goal_events(league_slug: str, event_id: str) -> list[dict] | None:
    """Extract goal events from the /summary keyEvents array.

    Returns a list of {minute, player_name, espn_player_id, team_api_id,
    team_name, is_penalty, is_own_goal}. Returns None on fetch failure,
    empty list when match has no goals.
    """
    league_code = ESPN_LEAGUE_BY_SLUG.get(league_slug)
    if not league_code:
        return None
    url = f"{ESPN_BASE}/{league_code}/summary"
    try:
        async with httpx.AsyncClient(timeout=15.0, headers={"accept": "application/json"}) as client:
            resp = await client.get(url, params={"event": event_id})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning(f"ESPN summary {event_id} failed: {exc}")
        return None

    out: list[dict] = []
    for event in data.get("keyEvents", []) or []:
        et = (event.get("type") or {}).get("text", "")
        if not event.get("scoringPlay"):
            # Not a goal
            continue
        # Minute parsing: clock.displayValue like "87'" or "45'+1'"
        clock = (event.get("clock") or {}).get("displayValue", "")
        minute = None
        if clock:
            import re as _re
            m = _re.match(r"(\d+)", clock)
            if m:
                minute = int(m.group(1))
        team = event.get("team") or {}
        # The scoring player is always participants[0].athlete (assists are [1])
        participants = event.get("participants") or []
        if not participants:
            continue
        scorer = (participants[0].get("athlete") or {})
        out.append({
            "minute": minute,
            "espn_player_id": str(scorer.get("id", "")),
            "player_name": scorer.get("displayName", ""),
            "team_api_id": str(team.get("id", "")),
            "team_name": team.get("displayName", ""),
            "is_penalty": "penalty" in et.lower(),
            "is_own_goal": "own goal" in et.lower(),
        })
    return out


async def fetch_match_summary(league_slug: str, event_id: str) -> dict | None:
    """ESPN's /summary endpoint returns boxscore stats: cards, corners,
    fouls, shots, possession per team. Used to backfill match_stats table.

    Returns dict with home_*/away_* keys for each metric, or None on failure.
    """
    league_code = ESPN_LEAGUE_BY_SLUG.get(league_slug)
    if not league_code:
        return None
    url = f"{ESPN_BASE}/{league_code}/summary"
    try:
        async with httpx.AsyncClient(timeout=15.0, headers={"accept": "application/json"}) as client:
            resp = await client.get(url, params={"event": event_id})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning(f"ESPN summary {event_id} failed: {exc}")
        return None

    teams = (data.get("boxscore") or {}).get("teams") or []
    if len(teams) < 2:
        return None

    def stats_dict(team_block: dict) -> dict[str, str]:
        return {
            s["name"]: s.get("displayValue", "")
            for s in team_block.get("statistics", []) if isinstance(s, dict)
        }

    def home_or_away(t: dict) -> str:
        return t.get("homeAway", "")

    home_block = next((t for t in teams if home_or_away(t) == "home"), teams[0])
    away_block = next((t for t in teams if home_or_away(t) == "away"), teams[1])
    home_stats = stats_dict(home_block)
    away_stats = stats_dict(away_block)

    def safe_int(s: str) -> int | None:
        try:
            return int(float(s)) if s not in ("", None) else None
        except (TypeError, ValueError):
            return None

    def safe_float(s: str) -> float | None:
        try:
            return float(s) if s not in ("", None) else None
        except (TypeError, ValueError):
            return None

    return {
        "home_yellow_cards": safe_int(home_stats.get("yellowCards", "")),
        "away_yellow_cards": safe_int(away_stats.get("yellowCards", "")),
        "home_red_cards":    safe_int(home_stats.get("redCards", "")),
        "away_red_cards":    safe_int(away_stats.get("redCards", "")),
        "home_corners":      safe_int(home_stats.get("wonCorners", "")),
        "away_corners":      safe_int(away_stats.get("wonCorners", "")),
        "home_fouls":        safe_int(home_stats.get("foulsCommitted", "")),
        "away_fouls":        safe_int(away_stats.get("foulsCommitted", "")),
        "home_shots":        safe_int(home_stats.get("totalShots", "")),
        "away_shots":        safe_int(away_stats.get("totalShots", "")),
        "home_shots_on_target": safe_int(home_stats.get("shotsOnTarget", "")),
        "away_shots_on_target": safe_int(away_stats.get("shotsOnTarget", "")),
        "home_possession":   safe_float(home_stats.get("possessionPct", "")),
        "away_possession":   safe_float(away_stats.get("possessionPct", "")),
    }


async def fetch_season_history(
    league_slug: str, season_start: date, season_end: date, request_delay_s: float = 0.15
) -> list[ESPNMatch]:
    """Iterate day-by-day through a date range.

    Used for backtest data when no CSV source exists. For Premier League prefer
    football-data.co.uk (single CSV, instant + closing odds).

    Args:
        request_delay_s: pause between day requests; 0.15s = ~6 req/s, conservative.
    """
    days = (season_end - season_start).days + 1
    if days <= 0:
        return []

    all_matches: list[ESPNMatch] = []
    seen: set[str] = set()
    async with httpx.AsyncClient(timeout=15.0, headers={"accept": "application/json"}) as client:
        for offset in range(days):
            d = season_start + timedelta(days=offset)
            try:
                day_matches = await fetch_scoreboard(league_slug, d, client=client)
            except httpx.HTTPError as exc:
                logger.warning(f"ESPN fetch failed for {league_slug} on {d}: {exc}")
                continue
            for m in day_matches:
                if m.espn_id not in seen:
                    seen.add(m.espn_id)
                    all_matches.append(m)
            await asyncio.sleep(request_delay_s)

    logger.info(
        f"ESPN history {league_slug} {season_start}..{season_end}: {len(all_matches)} unique matches"
    )
    return all_matches


# Convenience for tests / CLI
async def _smoke_test() -> None:
    today = datetime.now(tz=timezone.utc).date()
    for slug in ("premier_league", "liga_betplay"):
        matches = await fetch_scoreboard(slug, today)
        print(f"\n{slug} on {today}: {len(matches)} matches")
        for m in matches[:5]:
            print(f"  [{m.status}] {m.home_team} {m.home_goals}-{m.away_goals} {m.away_team} ({m.kickoff_utc.isoformat()})")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
