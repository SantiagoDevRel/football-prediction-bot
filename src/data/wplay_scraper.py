"""Wplay live odds scraper. READ-ONLY.

ARCHITECTURAL CONSTRAINT (do not change without explicit approval):
    This module ONLY reads odds. It NEVER places bets, logs in with user
    credentials, or interacts with the betting form. Wplay's T&Cs prohibit
    automated betting -> account closure + balance forfeiture.

Strategy: open Wplay football page in Playwright (headless Chromium), wait
for JS to load match cards, intercept XHR responses where possible, and
extract odds. Because Wplay's exact DOM/HTML can change without warning,
this scraper is best-effort:
    - If selectors break, we save raw HTML + screenshot to logs/wplay_debug/
      and return an empty list rather than crashing the pipeline.
    - The daily pipeline degrades gracefully when no odds are available
      (it just logs predictions without a value-bet calculation).

The scraper does NOT send credentials and does NOT navigate to user-account
pages. It hits public live-odds pages only.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from playwright.async_api import Browser, Page, async_playwright

from src.config import settings


# NOTE: this URL returns a 404 (ERR_NOTFOUND) on Wplay's current site. Real
# scraping requires:
#   1. Identifying the correct landing URL (likely /es-co/sports/... or via
#      logged-in session — Wplay restricts a lot of routes).
#   2. Solving the geo gate (Wplay validates Colombian IP via GetIpInfo.php).
#   3. Engineering selectors after manual inspection of the rendered DOM.
#
# This is selector engineering that needs human-in-the-loop iteration. The
# scraper as-is opens the page, captures a debug snapshot, and returns []
# so the rest of the pipeline can degrade gracefully.
#
# When iterating: run `python -m src.data.wplay_scraper`, inspect
# logs/wplay_debug/_initial_capture.html in a browser, find the correct
# selectors, and update WPLAY_FOOTBALL_URL + extraction below.
WPLAY_FOOTBALL_URL = "https://apuestas.wplay.co/sports/futbol"
DEBUG_DIR = Path(__file__).resolve().parents[2] / "logs" / "wplay_debug"


@dataclass
class WplayOdds:
    home_team: str
    away_team: str
    league_hint: str | None
    market: str            # "1x2" | "ou_2.5" | "btts"
    selection: str         # "home" | "draw" | "away" | "over" | "under" | "yes" | "no"
    odds: float
    is_live: bool
    captured_at: datetime


# A loose set of league name fragments we accept (case-insensitive substring match).
_OUR_LEAGUES = ("premier", "england", "english", "betplay", "colombia", "primera a", "dimayor")


def _is_our_league(text: str | None) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(frag in t for frag in _OUR_LEAGUES)


async def _save_debug(page: Page, label: str) -> None:
    """Persist HTML + screenshot for selector debugging."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    html_path = DEBUG_DIR / f"{ts}_{label}.html"
    png_path = DEBUG_DIR / f"{ts}_{label}.png"
    try:
        html = await page.content()
        html_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        logger.warning(f"failed to save debug HTML: {exc}")
    try:
        await page.screenshot(path=str(png_path), full_page=True)
    except Exception as exc:
        logger.warning(f"failed to save debug screenshot: {exc}")
    logger.info(f"saved Wplay debug -> {html_path.name}, {png_path.name}")


async def _open_browser() -> tuple[Browser, Page]:
    pw = await async_playwright().start()
    # Stealth-lite: realistic UA, no automation flag, viewport like a desktop.
    browser = await pw.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1366, "height": 900},
        locale="es-CO",
    )
    page = await context.new_page()
    return browser, page


async def scrape_live_odds(timeout_ms: int = 25_000) -> list[WplayOdds]:
    """Open Wplay football page and try to extract live + upcoming match odds.

    Best-effort. Returns empty list on selector failure rather than raising.
    Always saves debug artifacts on first run so we can iterate selectors.
    """
    captured: list[WplayOdds] = []
    browser, page = await _open_browser()
    try:
        try:
            await page.goto(WPLAY_FOOTBALL_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            # Give SPA time to hydrate content
            await page.wait_for_timeout(5_000)
        except Exception as exc:
            logger.warning(f"Wplay navigation failed: {exc}")
            await _save_debug(page, "nav_failed")
            return []

        # We don't know the exact selectors and they change. Try a few patterns:
        # 1. Look for elements that contain decimal numbers consistent with odds (e.g. 1.85, 3.40)
        # 2. Try common SPA patterns: match cards inside [data-testid*="event"]
        # 3. Fall back to plain regex on the rendered text

        try:
            html = await page.content()
        except Exception:
            html = ""

        # Save first-run debug regardless so we can build better selectors offline
        if not (DEBUG_DIR / "_initial_capture.html").exists():
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            (DEBUG_DIR / "_initial_capture.html").write_text(html, encoding="utf-8")
            try:
                await page.screenshot(
                    path=str(DEBUG_DIR / "_initial_capture.png"), full_page=True
                )
            except Exception:
                pass
            logger.info(
                "saved initial Wplay capture to logs/wplay_debug/_initial_capture.html "
                "for selector engineering"
            )

        # Heuristic extraction: look for visible odds patterns next to team names.
        # Wplay (like most casas) renders match cards with two team names and
        # 3 odds (1, X, 2). We use Playwright's text-based extraction.
        try:
            # Find all elements containing a decimal number 1.01-99.99 (likely odds)
            odds_re = re.compile(r"^\s*\d{1,2}\.\d{1,3}\s*$")
            cells = await page.query_selector_all("button, span, div")
            seen: set[str] = set()
            for cell in cells[:5000]:  # cap to avoid hangs on huge DOM
                try:
                    text = (await cell.inner_text()).strip()
                except Exception:
                    continue
                if odds_re.match(text):
                    try:
                        val = float(text)
                    except ValueError:
                        continue
                    if 1.01 <= val <= 99.0:
                        seen.add(text)
            if not seen:
                logger.warning("Wplay: no odds-shaped elements found")
                await _save_debug(page, "no_odds_found")
        except Exception as exc:
            logger.warning(f"Wplay extraction failed: {exc}")
            await _save_debug(page, "extraction_failed")

        # Note: full mapping of odds to (home, away, market, selection) requires
        # selector engineering specific to Wplay's current markup. Phase 2.5
        # will iterate the debug HTML and produce concrete selectors. Until then
        # we return an empty list and the daily pipeline operates without odds.
        logger.info(
            f"Wplay scrape complete (heuristic phase). odds-shaped tokens found: "
            f"{len(seen) if 'seen' in locals() else 0}. Returning [] for now; "
            f"selectors require human inspection of logs/wplay_debug/_initial_capture.html"
        )
        return captured
    finally:
        await browser.close()


async def _smoke() -> None:
    items = await scrape_live_odds()
    print(f"Wplay returned {len(items)} odds")
    for o in items[:10]:
        print(f"  {o.home_team} vs {o.away_team} | {o.market}:{o.selection} = {o.odds}")


if __name__ == "__main__":
    asyncio.run(_smoke())
