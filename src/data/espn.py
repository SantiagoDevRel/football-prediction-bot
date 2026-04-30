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

Phase 1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Map our internal league slug -> ESPN league code.
# Aligned with la-polla's lib/espn/client.ts so we can cross-reference.
ESPN_LEAGUE_BY_SLUG: dict[str, str] = {
    "premier_league": "eng.1",
    "liga_betplay": "col.1",
    # Future leagues: laliga="esp.1", seriea="ita.1", champions="uefa.champions", etc.
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
    minute: int | None      # current minute if live
    venue: str | None


async def fetch_scoreboard(league_slug: str, target_date: date | None = None) -> list[ESPNMatch]:
    """Fetch a league scoreboard. Defaults to today's matches.

    For historical pulls, pass target_date. ESPN keeps about 1 year of history
    accessible via the dates param. For older data, fall back to football-data.co.uk
    or Understat.
    """
    raise NotImplementedError("Phase 1")


async def fetch_season_history(
    league_slug: str, season_start: date, season_end: date
) -> list[ESPNMatch]:
    """Iterate day-by-day through a date range. ~10 req/sec is safe.

    Used for backtest data when we don't have a CSV source. For Premier League
    prefer football-data.co.uk (single CSV, instant). For Liga BetPlay this is
    the primary historical source.
    """
    raise NotImplementedError("Phase 1")
