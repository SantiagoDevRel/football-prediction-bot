"""football-data.co.uk historical CSV downloader.

NOT to be confused with football-data.org (different service, paid above free tier).

football-data.co.uk hosts free CSV files with historical fixtures, results, and
closing odds for major European leagues going back ~25 years. Maintained for
research/betting analysis. Standard URL pattern:

    https://www.football-data.co.uk/mmz4281/{season_yy}{next_yy}/{league_code}.csv

Examples:
    Premier League 2023/24: https://www.football-data.co.uk/mmz4281/2324/E0.csv
    Premier League 2022/23: https://www.football-data.co.uk/mmz4281/2223/E0.csv

Coverage:
    - England (Premier=E0, Championship=E1, etc.)
    - Spain, Germany, Italy, France, Netherlands, Belgium, Portugal, Turkey, Greece, Scotland
    - DOES NOT cover Liga BetPlay / South American leagues. For those use ESPN.

Columns include: HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H/D/A (Bet365 closing odds),
Avg odds across bookmakers, etc. Critical for backtesting because the closing odds are
the benchmark for CLV.

Phase 1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


# League codes used by football-data.co.uk
LEAGUE_CODES: dict[str, str] = {
    "premier_league": "E0",
    # add others when needed: la_liga="SP1", bundesliga="D1", ...
}


@dataclass
class HistoricalMatch:
    league_slug: str
    season: int           # starting year, e.g. 2023 for 2023/24
    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    # Closing odds (Bet365 if available, else fallback to avg)
    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    odds_over_2_5: float | None
    odds_under_2_5: float | None
    odds_btts_yes: float | None
    odds_btts_no: float | None


async def download_season(league_slug: str, season: int) -> list[HistoricalMatch]:
    """Download and parse one season CSV.

    Args:
        league_slug: our internal slug, e.g. "premier_league"
        season: starting year. season=2023 means 2023/24.
    """
    raise NotImplementedError("Phase 1")
