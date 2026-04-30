"""Understat scraper for xG / xGA historical data.

Understat is the public source of expected goals data for the top 5 European
leagues. Their league page is a JS-rendered SPA showing ONE matchweek at a
time via a calendar widget. To get a full season we click "previous" until
no new matches appear.

Coverage: EPL (Premier League). For other leagues, change the league code.

Output: list of UnderstatMatch with home_xG, away_xG, scores, date.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from playwright.async_api import async_playwright


UNDERSTAT_LEAGUE_BASE = "https://understat.com/league"
DEBUG_DIR = Path(__file__).resolve().parents[2] / "logs" / "understat_debug"


@dataclass
class UnderstatMatch:
    match_id: str
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_goals: int
    away_goals: int
    match_date: datetime
    is_finished: bool


# ---------- Parsing helpers ----------

_MATCH_HREF_RE = re.compile(r"match/(\d+)")
_DECIMAL_RE = re.compile(r"^([0-9.]+)")


async def _parse_xg_text(elem) -> float | None:
    """xG is rendered as e.g. '2.<small>18</small>'. inner_text returns '2.18'."""
    if elem is None:
        return None
    try:
        text = (await elem.inner_text()).strip()
    except Exception:
        return None
    m = _DECIMAL_RE.match(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


_DATE_FMT = "%A, %B %d, %Y"  # "Friday, August 16, 2024"


def _parse_understat_date(text: str) -> datetime:
    text = (text or "").strip()
    if not text:
        return datetime.now(tz=timezone.utc)
    try:
        return datetime.strptime(text, _DATE_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(tz=timezone.utc)


async def _parse_calendar_card(card, match_date: datetime) -> UnderstatMatch | None:
    try:
        # Home / away team names
        home_name_a = await card.query_selector(".block-home .team-title a")
        away_name_a = await card.query_selector(".block-away .team-title a")
        if home_name_a is None or away_name_a is None:
            return None
        home_team = (await home_name_a.inner_text()).strip()
        away_team = (await away_name_a.inner_text()).strip()

        match_link = await card.query_selector("a.match-info")
        if match_link is None:
            return None
        href = (await match_link.get_attribute("href")) or ""
        m = _MATCH_HREF_RE.search(href)
        if not m:
            return None
        match_id = m.group(1)
        is_result = ((await match_link.get_attribute("data-isresult")) or "").lower() == "true"

        home_goals_elem = await card.query_selector(".teams-goals .team-home")
        away_goals_elem = await card.query_selector(".teams-goals .team-away")
        if home_goals_elem and away_goals_elem and is_result:
            try:
                hg = int((await home_goals_elem.inner_text()).strip())
                ag = int((await away_goals_elem.inner_text()).strip())
            except (ValueError, AttributeError):
                hg = ag = 0
        else:
            hg = ag = 0

        home_xg_elem = await card.query_selector(".teams-xG .team-home")
        away_xg_elem = await card.query_selector(".teams-xG .team-away")
        home_xg = await _parse_xg_text(home_xg_elem) or 0.0
        away_xg = await _parse_xg_text(away_xg_elem) or 0.0

        return UnderstatMatch(
            match_id=match_id,
            home_team=home_team, away_team=away_team,
            home_xg=home_xg, away_xg=away_xg,
            home_goals=hg, away_goals=ag,
            match_date=match_date,
            is_finished=is_result,
        )
    except Exception as exc:
        logger.debug(f"understat card parse error: {exc}")
        return None


# ---------- Top-level scrape ----------

async def _open_page():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True, args=["--disable-blink-features=AutomationControlled"]
    )
    ctx = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1366, "height": 900},
    )
    page = await ctx.new_page()
    return browser, page


async def scrape_league_season(
    league: str, season: int, max_iterations: int = 60,
) -> list[UnderstatMatch]:
    """Walk the calendar widget backwards until no new matches load.

    Args:
        league: 'EPL' (case-sensitive Understat path).
        season: starting year. e.g. 2024 -> 2024/25.
        max_iterations: safety cap on prev-clicks (a season has ~38 matchweeks).

    Returns:
        Deduplicated list of UnderstatMatch across the season.
    """
    url = f"{UNDERSTAT_LEAGUE_BASE}/{league}/{season}"
    browser, page = await _open_page()
    out: dict[str, UnderstatMatch] = {}
    try:
        try:
            await page.goto(url, wait_until="networkidle", timeout=30_000)
        except Exception as exc:
            logger.warning(f"understat nav failed: {exc}")
            return []
        await page.wait_for_timeout(2_000)

        for i in range(max_iterations):
            containers = await page.query_selector_all(".calendar-date-container")
            new_count_before = len(out)
            for container in containers:
                date_elem = await container.query_selector(".calendar-date")
                date_str = (await date_elem.inner_text()).strip() if date_elem else ""
                match_date = _parse_understat_date(date_str)
                cards = await container.query_selector_all(".calendar-game")
                for card in cards:
                    parsed = await _parse_calendar_card(card, match_date)
                    if parsed and parsed.match_id not in out:
                        out[parsed.match_id] = parsed
            new_added = len(out) - new_count_before
            logger.debug(
                f"understat {league} {season} iter {i}: containers={len(containers)} "
                f"new={new_added} total={len(out)}"
            )

            # Click "previous"
            prev_btn = await page.query_selector(".calendar-prev")
            if prev_btn is None:
                break
            cls = (await prev_btn.get_attribute("class")) or ""
            if "disabled" in cls.lower():
                break
            try:
                await prev_btn.click(timeout=4_000)
                await page.wait_for_timeout(700)  # let the new matchweek render
            except Exception as exc:
                logger.debug(f"prev-click failed: {exc}")
                break

        logger.info(
            f"understat {league} {season}: scraped {len(out)} matches "
            f"({sum(1 for m in out.values() if m.is_finished)} finished)"
        )

        # Save debug snapshot of final DOM state
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        try:
            (DEBUG_DIR / f"{league}_{season}_{ts}.html").write_text(
                await page.content(), encoding="utf-8"
            )
        except Exception:
            pass

        return list(out.values())
    finally:
        try:
            await browser.close()
        except Exception:
            pass


# ---------- Smoke test ----------

async def _smoke() -> None:
    matches = await scrape_league_season("EPL", 2024, max_iterations=45)
    finished = [m for m in matches if m.is_finished]
    print(f"\nUnderstat EPL 2024: {len(matches)} total, {len(finished)} finished")
    if finished:
        finished.sort(key=lambda m: m.match_date)
        for m in finished[:5]:
            print(f"  {m.match_date.date()}  {m.home_team:18} {m.home_goals}-{m.away_goals} "
                  f"{m.away_team:18}  xG {m.home_xg:.2f}-{m.away_xg:.2f}")
        print("  …")
        for m in finished[-3:]:
            print(f"  {m.match_date.date()}  {m.home_team:18} {m.home_goals}-{m.away_goals} "
                  f"{m.away_team:18}  xG {m.home_xg:.2f}-{m.away_xg:.2f}")


if __name__ == "__main__":
    asyncio.run(_smoke())
