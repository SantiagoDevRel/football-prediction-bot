"""Understat scraper for xG / xGA historical data.

Understat is the public source of expected goals data. No API, scrape from JSON
embedded in HTML. Premier League: https://understat.com/league/EPL

NOT useful for Liga BetPlay (Understat doesn't cover Colombian football). For
BetPlay we'll have to live without xG initially or compute a simpler proxy from
shot data.

Phase 1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class UnderstatMatch:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_goals: int
    away_goals: int
    date: datetime


def scrape_league_season(league: str, season: int) -> list[UnderstatMatch]:
    """Scrape all matches of a league season.

    Args:
        league: "EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"
        season: starting year (e.g. 2024 for 2024/25 season)
    """
    raise NotImplementedError("Phase 1")
