"""Prospect Colombian bookmakers for scrape feasibility.

For each candidate site:
  - Try to load the BetPlay (Liga Dimayor) page.
  - Save HTML + screenshot for offline inspection.
  - Count how many price-like elements we can find with common selectors.
  - Report a feasibility score (rough): how easy it'd be to scrape.

This is a one-shot diagnostic — does NOT extract odds yet, just measures
the difficulty of doing so.
"""
import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright  # noqa: E402


CANDIDATES = [
    # (label, url, search_terms_in_text)
    ("betplay", "https://betplay.com.co/apuestas/futbol/colombia/primera-a", ["alianza", "millonarios", "santa fe"]),
    ("betplay_alt", "https://www.betplay.com.co/apuestas/futbol", ["liga", "dimayor"]),
    ("codere", "https://www.codere.com.co/deportes/futbol/colombia-liga-betplay-dimayor", ["alianza", "millonarios"]),
    ("rushbet", "https://www.rushbet.co/?page=sports#prematch/event-view/colombia-liga-bet-play", ["liga", "alianza"]),
    ("yajuego", "https://www.yajuego.co/apuestas/futbol/colombia-liga-betplay", ["liga", "alianza"]),
    ("stake_co", "https://stake.com.co/sports/soccer/colombia/colombia-liga-betplay-dimayor", ["liga", "alianza"]),
    ("luckia", "https://www.luckia.com.co/sports/futbol/colombia/liga-bet-play-dimayor", ["alianza", "millonarios"]),
]


async def probe(label: str, url: str, search_terms: list[str], pw) -> dict:
    """Try to load the page, measure scrape difficulty."""
    out = {
        "label": label, "url": url,
        "loaded": False, "html_size": 0, "price_like_count": 0,
        "team_names_found": [], "block_reason": None,
    }
    browser = await pw.chromium.launch(
        headless=True, args=["--disable-blink-features=AutomationControlled"]
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
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        if resp:
            status = resp.status
            if status >= 400:
                out["block_reason"] = f"HTTP {status}"
                await browser.close()
                return out
        await page.wait_for_timeout(6_000)  # let SPA hydrate
        out["loaded"] = True
        html = await page.content()
        out["html_size"] = len(html)

        # Count "price-like" elements: numbers in 1.01..50.00 range
        text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        prices = re.findall(r"\b(\d{1,2}\.\d{1,2})\b", text)
        valid_prices = [float(p) for p in prices if 1.01 <= float(p) <= 50.0]
        out["price_like_count"] = len(valid_prices)

        # Did we see any of the expected team names / keywords?
        text_lower = text.lower()
        out["team_names_found"] = [t for t in search_terms if t in text_lower]

        # Save html + screenshot
        debug = ROOT / "logs" / "co_bookie_probes"
        debug.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        (debug / f"{label}_{ts}.html").write_text(html, encoding="utf-8")
        try:
            await page.screenshot(path=str(debug / f"{label}_{ts}.png"), full_page=False)
        except Exception:
            pass
    except Exception as exc:
        out["block_reason"] = f"exception: {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        await browser.close()
    return out


async def main() -> None:
    async with async_playwright() as pw:
        results = []
        for label, url, terms in CANDIDATES:
            print(f"\n[{label}] probing {url} ...")
            r = await probe(label, url, terms, pw)
            results.append(r)
            print(f"  loaded={r['loaded']}  html_size={r['html_size']}  prices={r['price_like_count']}  teams={r['team_names_found']}  block={r['block_reason']}")

    print("\n\n=== FEASIBILITY SUMMARY ===")
    print(f"{'site':15} {'loaded':>7} {'prices':>7} {'teams':>15} {'block':>25}")
    for r in results:
        teams_str = ",".join(r["team_names_found"][:2]) or "-"
        print(f"{r['label']:15} {str(r['loaded']):>7} {r['price_like_count']:>7} {teams_str:>15} {r['block_reason'] or '-':>25}")

    # Quick scoring
    print("\nFeasibility ranking (higher = easier):")
    for r in sorted(results, key=lambda x: (x["loaded"], x["price_like_count"], len(x["team_names_found"])), reverse=True):
        score = (10 if r["loaded"] else 0) + min(r["price_like_count"] // 5, 30) + len(r["team_names_found"]) * 5
        print(f"  {r['label']:15} score={score}")


if __name__ == "__main__":
    asyncio.run(main())
