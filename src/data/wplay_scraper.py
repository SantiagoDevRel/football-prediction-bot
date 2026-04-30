"""Wplay live odds scraper. READ-ONLY.

ARCHITECTURAL CONSTRAINT (do not change without explicit approval):
    This module ONLY reads odds. It NEVER places bets, logs in with user
    credentials, or interacts with the betting form. Wplay's T&Cs prohibit
    automated betting -> account closure + balance forfeiture.

Architecture (validated 2026-04-30 from saved debug HTML):
    Wplay runs on OpenBet/SBTech. The league page renders matches as
    <tr class="mkt mkt_content mkt-{mkt_id}" data-mkt_id="{mkt_id}"> rows.
    Each row contains 3 buttons (home / draw / away) with decimal odds in
    <span class="price dec">VALUE</span>. Match URL slug carries team names.

League URLs:
    Premier:  https://apuestas.wplay.co/es/t/19157/Inglaterra-Premier-League
    BetPlay:  https://apuestas.wplay.co/es/t/19311/Colombia-Primera-A
"""
from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from loguru import logger
from playwright.async_api import Page, async_playwright


WPLAY_LEAGUE_URLS: dict[str, str] = {
    "premier_league": "https://apuestas.wplay.co/es/t/19157/Inglaterra-Premier-League",
    "liga_betplay":   "https://apuestas.wplay.co/es/t/19311/Colombia-Primera-A",
}

DEBUG_DIR = Path(__file__).resolve().parents[2] / "logs" / "wplay_debug"


@dataclass
class WplayOdds:
    league_slug: str
    home_team: str
    away_team: str
    event_id: str
    market: str           # "1x2"
    selection: str        # "home" | "draw" | "away"
    odds: float
    captured_at: datetime


# ---------- Normalization helpers ----------

def normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop generic FC/CD/SC suffixes."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(fc|cd|sc|cf|cdf|de|football|club)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_SLUG_TEAMS_RE = re.compile(r"^/es/e/(\d+)/(.+)$")


def parse_event_url(href: str) -> tuple[str, str, str] | None:
    """Parse '/es/e/30923707/Newcastle-v-Brighton' -> (event_id, home, away).

    Some real slugs include URL-encoded whitespace at the start (\t\r\n) and
    accented chars (e.g. %C3%A9). We URL-decode then strip control chars.
    """
    href = href.strip()
    m = _SLUG_TEAMS_RE.match(href)
    if not m:
        return None
    event_id = m.group(1)
    slug = unquote(m.group(2))
    # Strip URL-encoded leading whitespace artifacts (\t \r \n etc.)
    slug = re.sub(r"^\s+", "", slug)
    # Split on '-v-' (both sides padded with hyphens, since slug is hyphen-joined)
    if "-v-" not in slug:
        return None
    home_slug, _, away_slug = slug.partition("-v-")
    home = home_slug.replace("-", " ").strip()
    away = away_slug.replace("-", " ").strip()
    return event_id, home, away


# ---------- Browser plumbing ----------

async def _open_page() -> tuple[object, Page]:
    pw = await async_playwright().start()
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


async def _save_debug(page: Page, label: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    try:
        (DEBUG_DIR / f"{ts}_{label}.html").write_text(await page.content(), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"failed to save debug HTML: {exc}")
    try:
        await page.screenshot(path=str(DEBUG_DIR / f"{ts}_{label}.png"), full_page=True)
    except Exception as exc:
        logger.warning(f"failed to save debug screenshot: {exc}")


# ---------- Extraction ----------

_ODDS_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,3})\s*$")


async def _extract_from_page(page: Page, league_slug: str) -> list[WplayOdds]:
    """Read all match rows currently in the DOM and emit WplayOdds.

    The selector tr[data-mkt_id] matches one match. Inside we expect:
      - <a href="/es/e/{event_id}/{slug}">  somewhere (in the mkt-count cell)
      - 3 <td class="seln"> cells, each with a <button class="price">
        containing <span class="price dec">{decimal_odds}</span>
    """
    out: list[WplayOdds] = []
    captured_at = datetime.now(tz=timezone.utc)

    rows = await page.query_selector_all("tr[data-mkt_id]")
    logger.info(f"[{league_slug}] found {len(rows)} mkt rows")

    for row in rows:
        mkt_id = await row.get_attribute("data-mkt_id")
        if not mkt_id:
            continue

        # Event link
        link = await row.query_selector("a[href*='/es/e/']")
        if not link:
            continue
        href = (await link.get_attribute("href")) or ""
        parsed = parse_event_url(href)
        if not parsed:
            continue
        event_id, home_team, away_team = parsed

        # 3 selection buttons in order: home, draw, away
        buttons = await row.query_selector_all("td.seln button.price")
        if len(buttons) < 3:
            continue

        selections = ("home", "draw", "away")
        odds_for_row: list[tuple[str, float]] = []
        for sel, btn in zip(selections, buttons[:3]):
            # Decimal odds span inside the button
            dec_span = await btn.query_selector("span.price.dec")
            if dec_span is None:
                continue
            try:
                txt = (await dec_span.inner_text()).strip()
            except Exception:
                continue
            m = _ODDS_RE.match(txt)
            if not m:
                continue
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if 1.01 <= val <= 200.0:
                odds_for_row.append((sel, val))

        if len(odds_for_row) < 3:
            continue

        for sel, val in odds_for_row:
            out.append(WplayOdds(
                league_slug=league_slug,
                home_team=home_team,
                away_team=away_team,
                event_id=event_id,
                market="1x2",
                selection=sel,
                odds=val,
                captured_at=captured_at,
            ))

    return out


async def scrape_league(league_slug: str, timeout_ms: int = 30_000) -> list[WplayOdds]:
    """Open a Wplay league page and return all 1X2 odds.

    Best-effort. Saves debug HTML+screenshot once on every run.
    """
    if league_slug not in WPLAY_LEAGUE_URLS:
        raise ValueError(f"unknown league slug: {league_slug}")
    url = WPLAY_LEAGUE_URLS[league_slug]

    browser, page = await _open_page()
    try:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as exc:
            logger.warning(f"Wplay nav {league_slug} failed: {exc}")
            await _save_debug(page, f"{league_slug}_nav_failed")
            return []

        # Wait for the match list to render. The real selector is the mkt
        # row, not the football expander.
        try:
            await page.wait_for_selector("tr[data-mkt_id]", timeout=15_000)
        except Exception:
            logger.warning(f"[{league_slug}] no mkt rows within 15s")
            await _save_debug(page, f"{league_slug}_no_rows")
            return []

        # Settle: let ALL events render (the page may stream more via XHR)
        await page.wait_for_timeout(2_000)

        odds = await _extract_from_page(page, league_slug)
        logger.info(f"[{league_slug}] extracted {len(odds)} price rows")

        await _save_debug(page, f"{league_slug}_extracted")
        return odds
    finally:
        try:
            await browser.close()
        except Exception:
            pass


async def scrape_match_markets(
    event_id: str, league_slug: str, home_team: str, away_team: str,
    timeout_ms: int = 25_000,
) -> list[WplayOdds]:
    """Visit /es/e/{event_id}/... and extract all markets visible inline.

    Markets pulled today (must be present in initial HTML, no expand-on-click):
      - 1X2 (Resultado Tiempo Completo)
      - BTTS (Ambos Equipos Anotan)
      - O/U 1.5, 2.5, 3.5 (from "Total Goles Más/Menos de" group market)

    Lazy-loaded markets (hándicap asiático, marcador correcto) are skipped.
    """
    url = f"https://apuestas.wplay.co/es/e/{event_id}/"
    browser, page = await _open_page()
    out: list[WplayOdds] = []
    captured_at = datetime.now(tz=timezone.utc)
    try:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as exc:
            logger.warning(f"match nav {event_id} failed: {exc}")
            return out
        # Let JS render markets
        try:
            await page.wait_for_selector("button.price[title]", timeout=15_000)
        except Exception:
            logger.warning(f"[match {event_id}] no price buttons rendered in time")
            return out
        await page.wait_for_timeout(2_000)

        # 1X2: prices in mkt with name "Resultado Tiempo Completo"
        try:
            buttons_1x2 = await page.query_selector_all(
                f"button.price[title='{home_team} ']"
            )
            home_btn = buttons_1x2[0] if buttons_1x2 else None
            buttons_draw = await page.query_selector_all(
                "button.price[title='Empate ']"
            )
            draw_btn = buttons_draw[0] if buttons_draw else None
            buttons_away = await page.query_selector_all(
                f"button.price[title='{away_team} ']"
            )
            away_btn = buttons_away[0] if buttons_away else None

            for sel, btn in (("home", home_btn), ("draw", draw_btn), ("away", away_btn)):
                if btn is None:
                    continue
                dec = await btn.query_selector("span.price.dec")
                if dec is None:
                    continue
                val = _safe_dec(await dec.inner_text())
                if val is None:
                    continue
                out.append(WplayOdds(
                    league_slug=league_slug, home_team=home_team, away_team=away_team,
                    event_id=event_id, market="1x2", selection=sel, odds=val,
                    captured_at=captured_at,
                ))
        except Exception as exc:
            logger.warning(f"[match {event_id}] 1X2 parse error: {exc}")

        # BTTS: title="Si " and title="No "
        try:
            for sel, title in (("yes", "Si "), ("no", "No ")):
                btn = await page.query_selector(f"button.price[title='{title}']")
                if btn is None:
                    continue
                dec = await btn.query_selector("span.price.dec")
                if dec is None:
                    continue
                val = _safe_dec(await dec.inner_text())
                if val is None:
                    continue
                out.append(WplayOdds(
                    league_slug=league_slug, home_team=home_team, away_team=away_team,
                    event_id=event_id, market="btts", selection=sel, odds=val,
                    captured_at=captured_at,
                ))
        except Exception as exc:
            logger.warning(f"[match {event_id}] BTTS parse error: {exc}")

        # O/U total goals: "Más de 1.5", "Menos de 1.5", "Más de 2.5", "Menos de 2.5", etc.
        for line in (1.5, 2.5, 3.5):
            for sel, label in (("over", f"Más de {line}"), ("under", f"Menos de {line}")):
                btn = await page.query_selector(f"button.price[title='{label}']")
                if btn is None:
                    continue
                dec = await btn.query_selector("span.price.dec")
                if dec is None:
                    continue
                val = _safe_dec(await dec.inner_text())
                if val is None:
                    continue
                out.append(WplayOdds(
                    league_slug=league_slug, home_team=home_team, away_team=away_team,
                    event_id=event_id, market=f"ou_{line}", selection=sel, odds=val,
                    captured_at=captured_at,
                ))

        logger.info(f"[match {event_id}] extracted {len(out)} odds across markets")
        return out
    finally:
        try:
            await browser.close()
        except Exception:
            pass


def _safe_dec(text: str | None) -> float | None:
    if not text:
        return None
    s = text.strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if 1.01 <= v <= 200.0:
        return v
    return None


async def scrape_all_with_markets() -> list[WplayOdds]:
    """Two-phase scrape:
        1. Visit league pages for both leagues -> list of (event_id, home, away)
        2. For each event, visit the match page -> all markets we can read

    Slower than scrape_all() (~30-60s instead of ~10s) but returns 5-10x more
    odds rows because we get O/U + BTTS in addition to 1X2.
    """
    out: list[WplayOdds] = []
    league_results: list[tuple[str, str, str, str]] = []  # (slug, event_id, home, away)
    for slug in WPLAY_LEAGUE_URLS:
        rows = await scrape_league(slug)
        # rows already give us 1X2; gather unique events
        seen: set[str] = set()
        for r in rows:
            if r.event_id in seen:
                continue
            seen.add(r.event_id)
            league_results.append((slug, r.event_id, r.home_team, r.away_team))
        # keep the 1X2 odds we already have so we don't waste them
        out.extend(rows)

    # Now visit each match for its multi-market data. Skip 1X2 (already have it).
    for slug, event_id, home, away in league_results:
        try:
            extra = await scrape_match_markets(event_id, slug, home, away)
        except Exception as exc:
            logger.warning(f"[match {event_id}] market scrape failed: {exc}")
            continue
        # Drop 1X2 from extra (already in `out` from league pages)
        extra_no_1x2 = [o for o in extra if o.market != "1x2"]
        out.extend(extra_no_1x2)

    return out


async def scrape_all() -> list[WplayOdds]:
    """Scrape both target leagues sequentially. Returns 1X2 only (fast path)."""
    out: list[WplayOdds] = []
    for slug in WPLAY_LEAGUE_URLS:
        out.extend(await scrape_league(slug))
    return out


# Smoke test
async def _smoke() -> None:
    items = await scrape_all()
    print(f"\nWplay returned {len(items)} odds rows")
    seen_matches: set[tuple[str, str]] = set()
    for o in items:
        if o.market == "1x2" and o.selection == "home":
            seen_matches.add((o.home_team, o.away_team))
    for o in items[:30]:
        print(f"  [{o.league_slug:14}] {o.home_team[:25]:25} v {o.away_team[:25]:25}  "
              f"{o.market}:{o.selection:5} = {o.odds:>6.2f}")
    print(f"\nUnique matches with full 1X2: {len(seen_matches)}")


if __name__ == "__main__":
    asyncio.run(_smoke())
