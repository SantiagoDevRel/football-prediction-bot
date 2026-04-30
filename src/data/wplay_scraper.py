"""Wplay live odds scraper. READ-ONLY.

ARCHITECTURAL CONSTRAINT (do not change without explicit approval):
    This module ONLY reads odds. It NEVER places bets, logs in with user
    credentials, or interacts with the betting form. Wplay's T&Cs prohibit
    automated betting -> account closure + balance forfeiture.

Uses Playwright headless. Wplay is a JS-heavy SPA; httpx alone won't work.

Phase 1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class WplayOdds:
    match_name: str
    league: str
    kickoff_utc: datetime | None
    market: str             # "1x2", "ou_2.5", "btts", etc.
    selection: str          # "home", "draw", "away", "over", "yes", etc.
    odds: float
    is_live: bool
    captured_at: datetime


async def scrape_live_odds(league_filter: list[str] | None = None) -> list[WplayOdds]:
    """Scrape current live odds. Optionally filter by league names.

    Implementation notes:
        - Use Playwright with stealth-like flags to avoid trivial bot detection
        - Always identify ourselves as a desktop browser (no headless: true visible markers)
        - Rate-limit: max 1 request per 30s to avoid spamming
        - Save raw HTML to data/raw/wplay/<timestamp>.html for debugging
    """
    raise NotImplementedError("Phase 1")


async def scrape_prematch_odds(date: str) -> list[WplayOdds]:
    """Scrape pre-match odds for fixtures on a given date (YYYY-MM-DD)."""
    raise NotImplementedError("Phase 1")
