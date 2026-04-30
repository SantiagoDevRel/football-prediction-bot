"""football-data.co.uk historical CSV downloader.

NOT to be confused with football-data.org (paid above free tier).

football-data.co.uk hosts free CSV files with historical fixtures, results and
closing odds for major European leagues going back ~25 years. URL pattern:

    https://www.football-data.co.uk/mmz4281/{season_yy}{next_yy}/{league_code}.csv

Examples:
    Premier 2024/25:  https://www.football-data.co.uk/mmz4281/2425/E0.csv
    Premier 2023/24:  https://www.football-data.co.uk/mmz4281/2324/E0.csv

Coverage:
    - England (Premier=E0, Championship=E1, etc.)
    - Spain, Germany, Italy, France, NL, Belgium, Portugal, Turkey, Greece, Scotland
    - DOES NOT cover Liga BetPlay / South American leagues. For those use ESPN.

Columns we care about:
    Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    B365H/D/A (Bet365 closing 1X2), B365>2.5/B365<2.5 (over/under 2.5)
    BTTS columns: GBH/GBD/GBA or PBHb/Lay (varies). We default to AvgH/AvgD/AvgA when B365 is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO

import httpx
import pandas as pd
from loguru import logger


LEAGUE_CODES: dict[str, str] = {
    "premier_league": "E0",
    # Future: la_liga="SP1", bundesliga="D1", ...
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

    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    odds_over_2_5: float | None
    odds_under_2_5: float | None
    odds_btts_yes: float | None
    odds_btts_no: float | None


def _season_url(league_slug: str, season: int) -> str:
    code = LEAGUE_CODES.get(league_slug)
    if not code:
        raise ValueError(f"unsupported league for football-data.co.uk: {league_slug}")
    yy = str(season % 100).zfill(2)
    next_yy = str((season + 1) % 100).zfill(2)
    return f"https://www.football-data.co.uk/mmz4281/{yy}{next_yy}/{code}.csv"


def _pick_first_present(row: pd.Series, candidates: list[str]) -> float | None:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (ValueError, TypeError):
                continue
    return None


def _parse_date(raw: str) -> date | None:
    """football-data.co.uk uses both DD/MM/YY and DD/MM/YYYY across seasons."""
    s = str(raw).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_row(row: pd.Series, league_slug: str, season: int) -> HistoricalMatch | None:
    try:
        match_date = _parse_date(row.get("Date", ""))
        if match_date is None:
            return None
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        hg = int(row["FTHG"])
        ag = int(row["FTAG"])
    except (KeyError, ValueError, TypeError):
        return None

    # 1X2 closing odds: prefer Bet365, fall back to average
    odds_h = _pick_first_present(row, ["B365H", "AvgH", "BWH", "PSH"])
    odds_d = _pick_first_present(row, ["B365D", "AvgD", "BWD", "PSD"])
    odds_a = _pick_first_present(row, ["B365A", "AvgA", "BWA", "PSA"])

    # Over/Under 2.5 closing
    odds_over = _pick_first_present(row, ["B365>2.5", "Avg>2.5", "P>2.5"])
    odds_under = _pick_first_present(row, ["B365<2.5", "Avg<2.5", "P<2.5"])

    # BTTS: column names vary. Most common: BTTSH/BTTSA OR GBH/GBA. We try several.
    odds_btts_yes = _pick_first_present(row, ["BFEH", "GBH", "PBHb"])
    odds_btts_no = _pick_first_present(row, ["BFEA", "GBA", "PBLa"])

    return HistoricalMatch(
        league_slug=league_slug,
        season=season,
        match_date=match_date,
        home_team=home,
        away_team=away,
        home_goals=hg,
        away_goals=ag,
        odds_home=odds_h,
        odds_draw=odds_d,
        odds_away=odds_a,
        odds_over_2_5=odds_over,
        odds_under_2_5=odds_under,
        odds_btts_yes=odds_btts_yes,
        odds_btts_no=odds_btts_no,
    )


async def download_season(league_slug: str, season: int) -> list[HistoricalMatch]:
    """Download and parse one season CSV.

    Args:
        league_slug: our internal slug, e.g. "premier_league"
        season: starting year. season=2023 means 2023/24.
    """
    url = _season_url(league_slug, season)
    logger.info(f"downloading {league_slug} {season}/{season+1} from {url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        # Some seasons use latin-1 due to historical encoding quirks
        try:
            text = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            text = resp.content.decode("latin-1")

    df = pd.read_csv(StringIO(text))
    # Strip empty rows that some CSVs end with
    df = df.dropna(how="all")
    matches = [
        m for m in (_parse_row(row, league_slug, season) for _, row in df.iterrows())
        if m is not None
    ]
    logger.info(f"  parsed {len(matches)} matches (raw rows: {len(df)})")
    return matches


async def download_seasons(league_slug: str, seasons: list[int]) -> list[HistoricalMatch]:
    all_matches: list[HistoricalMatch] = []
    for s in seasons:
        try:
            ms = await download_season(league_slug, s)
            all_matches.extend(ms)
        except httpx.HTTPError as exc:
            logger.warning(f"failed season {s}: {exc}")
    return all_matches


# Smoke test
async def _smoke() -> None:
    ms = await download_season("premier_league", 2023)
    print(f"Premier 2023/24: {len(ms)} matches")
    if ms:
        m = ms[0]
        print(f"  first: {m.match_date} {m.home_team} {m.home_goals}-{m.away_goals} {m.away_team}")
        print(f"         1X2 odds: H={m.odds_home} D={m.odds_draw} A={m.odds_away}")
        print(f"         O/U 2.5: O={m.odds_over_2_5} U={m.odds_under_2_5}")
        m_last = ms[-1]
        print(f"  last:  {m_last.match_date} {m_last.home_team} {m_last.home_goals}-{m_last.away_goals} {m_last.away_team}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke())
