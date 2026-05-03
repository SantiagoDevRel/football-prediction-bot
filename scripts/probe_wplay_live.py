"""Probe Wplay's in-play / live section.

Goal: find a URL that lists currently-live matches with their event_ids and
1X2 odds, so cmd_envivo can resolve event_ids without depending on the
pre-match league listing.
"""
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright  # noqa: E402


CANDIDATE_URLS = [
    "https://apuestas.wplay.co/es/inplay",
    "https://apuestas.wplay.co/es/inplay/",
    "https://apuestas.wplay.co/es/live",
    "https://apuestas.wplay.co/es/sports/futbol/inplay",
    "https://apuestas.wplay.co/es/sports/futbol/en-vivo",
    "https://apuestas.wplay.co/es/sports/inplay/futbol",
    "https://apuestas.wplay.co/es/en-vivo",
]


async def main() -> None:
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
        locale="es-CO",
    )
    page = await ctx.new_page()

    for url in CANDIDATE_URLS:
        print(f"\n--- {url} ---")
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            status = resp.status if resp else "?"
            await page.wait_for_timeout(5_000)
            html = await page.content()
            # Look for any /es/e/{id} match links and BetPlay team names
            event_links = re.findall(r"/es/e/(\d+)/[^\"'\s]+", html)
            unique_events = list(set(event_links))
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")
            tl = text.lower()
            betplay_hits = [t for t in ["alianza", "millonarios", "santa fe", "tolima", "fortaleza", "medellin", "águilas", "aguilas"] if t in tl]
            print(f"  status={status}  events={len(unique_events)}  size={len(html)}b  betplay_teams_found={betplay_hits}")
            if unique_events[:5]:
                print(f"  sample event_ids: {unique_events[:5]}")
        except Exception as exc:
            print(f"  EXC: {type(exc).__name__}: {str(exc)[:100]}")

    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
