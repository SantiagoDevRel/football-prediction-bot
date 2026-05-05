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
    "premier_league":   "https://apuestas.wplay.co/es/t/19157/Inglaterra-Premier-League",
    "liga_betplay":     "https://apuestas.wplay.co/es/t/19311/Colombia-Primera-A",
    "champions_league": "https://apuestas.wplay.co/es/t/19161/UEFA-Champions-League",
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

_NAME_ALIASES: dict[str, str] = {
    # Long form -> canonical short, so DB names and Wplay short names collapse
    # to the same string. Substring match in cmd_analizar requires equality
    # (or one inside the other) after this normalization.
    "paris saint germain": "psg",
    "paris saint-germain": "psg",
    "manchester united": "man utd",
    "man united": "man utd",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "monchengladbach",
    "bayer leverkusen": "leverkusen",
    "internazionale": "inter",
    "atletico madrid": "atletico madrid",
    "real betis balompie": "real betis",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, flatten dashes/apostrophes, collapse whitespace,
    drop generic FC/CD/SC suffixes, then map known long forms to a canonical
    short. The alias step is what lets DB 'Paris Saint-Germain' match Wplay 'PSG'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    for ch in ("-", "'", "’", ".", ",", "/"):
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(fc|cd|sc|cf|cdf|de|football|club)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _NAME_ALIASES.get(s, s)


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


WPLAY_INPLAY_URL = "https://apuestas.wplay.co/es/live"


async def scrape_inplay() -> list[WplayOdds]:
    """Scrape Wplay's in-play (live) section and return 1X2-shaped odds for
    every match currently being played. Wplay drops live matches off the
    league page, so /envivo can't resolve them via scrape_league(slug).

    Returns rows tagged with league_slug="live" since the page mixes leagues.
    Caller filters by team-name match against the DB row.

    Note: live 1X2 prices fluctuate; scrape close to use, don't cache.
    """
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled"],
    )
    ctx = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1366, "height": 900},
        locale="es-CO",
    )
    page = await ctx.new_page()
    try:
        try:
            await page.goto(WPLAY_INPLAY_URL, wait_until="domcontentloaded", timeout=45_000)
        except Exception as exc:
            logger.warning(f"inplay nav failed: {exc}")
            return []
        await page.wait_for_timeout(8_000)  # in-play SPA needs more time
        try:
            await page.wait_for_selector("tr[data-mkt_id]", timeout=15_000)
        except Exception:
            logger.warning("inplay: no mkt rows after 15s")
            return []
        odds = await _extract_from_page(page, "live")
        logger.info(f"inplay: extracted {len(odds)} live rows")
        return odds
    finally:
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await pw.stop()
        except Exception:
            pass


async def scrape_league(
    league_slug: str, timeout_ms: int = 30_000, max_attempts: int = 2,
) -> list[WplayOdds]:
    """Open a Wplay league page and return all 1X2 odds.

    Best-effort. Saves debug HTML+screenshot once on every run.

    Retries once on nav failure or empty-DOM (Wplay sometimes serves an
    interstitial / CDN warmup that needs a second hit).
    """
    if league_slug not in WPLAY_LEAGUE_URLS:
        raise ValueError(f"unknown league slug: {league_slug}")
    url = WPLAY_LEAGUE_URLS[league_slug]

    last_failure = "unknown"
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            await asyncio.sleep(3)
        browser, page = await _open_page()
        try:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception as exc:
                last_failure = f"nav_failed: {exc}"
                logger.warning(f"Wplay nav {league_slug} attempt {attempt} failed: {exc}")
                await _save_debug(page, f"{league_slug}_nav_failed_a{attempt}")
                continue

            try:
                await page.wait_for_selector("tr[data-mkt_id]", timeout=15_000)
            except Exception:
                last_failure = "no_mkt_rows_within_15s"
                logger.warning(f"[{league_slug}] no mkt rows within 15s (attempt {attempt})")
                await _save_debug(page, f"{league_slug}_no_rows_a{attempt}")
                continue

            # Settle: let ALL events render (the page may stream more via XHR)
            await page.wait_for_timeout(2_000)

            odds = await _extract_from_page(page, league_slug)
            logger.info(f"[{league_slug}] extracted {len(odds)} price rows (attempt {attempt})")
            await _save_debug(page, f"{league_slug}_extracted")
            return odds
        finally:
            try:
                await browser.close()
            except Exception:
                pass

    logger.warning(f"[{league_slug}] all {max_attempts} attempts failed: {last_failure}")
    return []


# Match Wplay's "Tiros de Esquina - 3 Opciones (N)" section name → integer line.
_CORNERS_SECTION_RE = re.compile(r"Tiros de Esquina\s*-\s*3 Opciones\s*\((\d+)\)", re.I)
# Match "Total de Tarjetas (4.5)" → float line.
_CARDS_SECTION_RE = re.compile(r"Total de Tarjetas\s*\((\d+(?:\.\d+)?)\)", re.I)
# "Más de N(.X)" / "Menos de N(.X)" / "Exacto N" — pure label, no suffix.
_MAS_RE = re.compile(r"^M[áa]s de\s+(\d+(?:\.\d+)?)\s*$", re.I)
_MENOS_RE = re.compile(r"^Menos de\s+(\d+(?:\.\d+)?)\s*$", re.I)
_EXACTO_RE = re.compile(r"^Exacto\s+(\d+(?:\.\d+)?)\s*$", re.I)


async def _extract_market_sections(page) -> list[dict]:
    """Walk every `span.mkt-name` and return its tightest `expander mkt` ancestor
    along with the (title, decimal-text) pairs of every `button.price` inside.

    Section-scoped extraction is required because the same button title (e.g.
    'Más de 4.5') appears under both 'Total Goles' and 'Total de Tarjetas';
    a global title query would conflate the two markets."""
    return await page.evaluate(
        """() => {
            const out = [];
            document.querySelectorAll('span.mkt-name').forEach(n => {
                const name = (n.innerText || '').trim();
                if (!name) return;
                let p = n.parentElement; let depth = 0;
                while (p && depth < 10) {
                    const cls = (p.className || '').toString();
                    if (cls.includes('expander') && cls.includes('mkt')) {
                        const items = [];
                        p.querySelectorAll(':scope button.price[title]').forEach(b => {
                            const t = b.getAttribute('title');
                            const dec = b.querySelector('span.price.dec');
                            const v = dec ? (dec.innerText || '').trim() : null;
                            if (t && v) items.push([t, v]);
                        });
                        if (items.length > 0) out.push({name, items});
                        return;
                    }
                    p = p.parentElement; depth++;
                }
            });
            return out;
        }"""
    )


def _parse_sections(
    sections: list[dict], league_slug: str, home_team: str, away_team: str,
    event_id: str, captured_at: datetime,
    *, include_cards: bool = False,
) -> list[WplayOdds]:
    """Map scraped DOM sections into WplayOdds rows. include_cards=True is used
    on the second pass after the 'Tarjetas' tab click."""
    out: list[WplayOdds] = []
    home_t = home_team.strip()
    away_t = away_team.strip()

    def _push(market: str, sel: str, val_text: str) -> None:
        v = _safe_dec(val_text)
        if v is None:
            return
        out.append(WplayOdds(
            league_slug=league_slug, home_team=home_team, away_team=away_team,
            event_id=event_id, market=market, selection=sel, odds=v,
            captured_at=captured_at,
        ))

    for sec in sections:
        name = sec["name"]
        items = sec["items"]

        # 1X2 — section name "Resultado Tiempo Completo"
        if name == "Resultado Tiempo Completo":
            for title, val in items:
                t = title.strip()
                if t == home_t:
                    _push("1x2", "home", val)
                elif t == "Empate":
                    _push("1x2", "draw", val)
                elif t == away_t:
                    _push("1x2", "away", val)
            continue

        # BTTS — "Ambos Equipos Anotan"
        if name == "Ambos Equipos Anotan":
            for title, val in items:
                t = title.strip()
                if t == "Si":
                    _push("btts", "yes", val)
                elif t == "No":
                    _push("btts", "no", val)
            continue

        # Goals O/U — full-match section "Total Goles Más/Menos de".
        # Buttons inside come at half-integer lines we care about (1.5, 2.5, 3.5)
        # plus higher lines (4.5, 5.5, 6.5) that we don't model but still keep.
        if name == "Total Goles Más/Menos de":
            for title, val in items:
                t = title.strip()
                m = _MAS_RE.match(t)
                if m:
                    line = float(m.group(1))
                    if line in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
                        _push(f"ou_{line}", "over", val)
                    continue
                m = _MENOS_RE.match(t)
                if m:
                    line = float(m.group(1))
                    if line in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
                        _push(f"ou_{line}", "under", val)
            continue

        # Corners 3-way — "Tiros de Esquina - 3 Opciones (N)" with buttons
        # 'Más de N', 'Exacto N', 'Menos de N'. N is integer (8, 9, 10, 11, 12).
        m = _CORNERS_SECTION_RE.search(name)
        if m:
            line = int(m.group(1))
            for title, val in items:
                t = title.strip()
                if t == f"Más de {line}":
                    _push(f"corners_{line}", "over", val)
                elif t == f"Menos de {line}":
                    _push(f"corners_{line}", "under", val)
                elif t == f"Exacto {line}":
                    _push(f"corners_{line}", "exact", val)
            continue

        # Cards — only parsed on second pass after Tarjetas tab click
        if include_cards:
            m = _CARDS_SECTION_RE.search(name)
            if m:
                line = float(m.group(1))
                for title, val in items:
                    t = title.strip()
                    if _MAS_RE.match(t) and float(_MAS_RE.match(t).group(1)) == line:
                        _push(f"cards_{line}", "over", val)
                    elif _MENOS_RE.match(t) and float(_MENOS_RE.match(t).group(1)) == line:
                        _push(f"cards_{line}", "under", val)
                continue

    return out


async def scrape_match_markets(
    event_id: str, league_slug: str, home_team: str, away_team: str,
    timeout_ms: int = 25_000,
) -> list[WplayOdds]:
    """Visit /es/e/{event_id}/... and extract all markets we can read.

    Two-pass extraction:
      Pass 1 (default 'Todos' view): 1X2, BTTS, Goals O/U (1.5/2.5/3.5/+),
        Corners 3-way at lines 8–12 (Wplay names them 'Tiros de Esquina -
        3 Opciones (N)' with Más de/Exacto/Menos de buttons).
      Pass 2 (after clicking 'Tarjetas (N)' tab): Cards Total at line 4.5
        from the 'Total de Tarjetas (4.5)' section. Wplay lazy-loads the
        cards section — without the tab click those buttons don't render.

    All matching is section-scoped (DOM ancestor `expander mkt`) because
    'Más de 4.5' as a global selector matches both Total Goles and Total
    de Tarjetas, conflating two different markets.
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
        try:
            await page.wait_for_selector("button.price[title]", timeout=15_000)
        except Exception:
            logger.warning(f"[match {event_id}] no price buttons rendered in time")
            return out
        await page.wait_for_timeout(2_000)

        # Pass 1 — default view
        try:
            sections = await _extract_market_sections(page)
            out.extend(_parse_sections(
                sections, league_slug, home_team, away_team, event_id, captured_at,
                include_cards=False,
            ))
        except Exception as exc:
            logger.warning(f"[match {event_id}] pass-1 parse error: {exc}")

        # Pass 2 — click the Tarjetas tab and re-extract just the cards section.
        # Best effort: if click fails or no cards section appears, log and move on.
        try:
            clicked = await page.evaluate(
                """() => {
                    const links = document.querySelectorAll('a, [role=tab]');
                    for (const l of links) {
                        const t = (l.innerText || '').trim();
                        if (/^Tarjetas\\s*\\(\\d+\\)$/i.test(t)) {
                            l.click();
                            return true;
                        }
                    }
                    return false;
                }"""
            )
            if clicked:
                await page.wait_for_timeout(2_500)
                card_sections = await _extract_market_sections(page)
                out.extend(_parse_sections(
                    card_sections, league_slug, home_team, away_team,
                    event_id, captured_at, include_cards=True,
                ))
            else:
                logger.info(f"[match {event_id}] no Tarjetas tab found — skipping cards")
        except Exception as exc:
            logger.warning(f"[match {event_id}] pass-2 (cards) failed: {exc}")

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
