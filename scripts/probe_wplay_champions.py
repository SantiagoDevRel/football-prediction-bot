"""Discover the Wplay tournament URL/ID for UEFA Champions League.

Strategy:
    1. Open Wplay's "Fútbol" landing page in headless Chromium.
    2. Look for a link whose text contains 'champions league' (case-insensitive).
    3. Print the href so we can hard-code it in WPLAY_LEAGUE_URLS.

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_wplay_champions.py
"""
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright  # noqa: E402


CANDIDATE_PAGES = [
    "https://apuestas.wplay.co/es/sports/futbol/",
    "https://apuestas.wplay.co/es/",
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

    found: dict[str, str] = {}
    for url in CANDIDATE_PAGES:
        print(f"\n--- {url} ---")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as exc:
            print(f"  goto failed: {exc}")
            continue
        await page.wait_for_timeout(5_000)

        # Get all anchor hrefs+text
        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({href: e.href, text: (e.innerText || '').trim()}))"
        )
        print(f"  scanned {len(links)} anchors")

        # Filter for champions / uefa keywords
        kw_re = re.compile(r"champions|uefa|liga.de.campeones", re.IGNORECASE)
        matches = [l for l in links if kw_re.search(l["text"]) or kw_re.search(l["href"])]
        for l in matches:
            print(f"  HIT: {l['text'][:60]:60}  ->  {l['href']}")
            # capture every distinct href
            found[l["href"]] = l["text"]

        # Save full page HTML for offline grep
        out = ROOT / "logs" / "wplay_debug" / f"futbol_landing_{url.split('/')[-2] or 'root'}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            html = await page.content()
            out.write_text(html, encoding="utf-8")
            print(f"  saved html -> {out}")
        except Exception:
            pass

    print("\n\n=== Summary ===")
    if not found:
        print("No Champions League link found in landing pages.")
        print("Try manually visiting the site and inspecting the menu.")
    else:
        print(f"Found {len(found)} candidate URLs:")
        for href, text in found.items():
            print(f"  {text!r:60} -> {href}")

    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
