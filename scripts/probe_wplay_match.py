"""One-shot probe: open a Wplay match page in headless Chromium, save HTML.

Usage:
    python scripts/probe_wplay_match.py 30923707  # event_id
"""
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright  # noqa: E402


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/probe_wplay_match.py <event_id>")
        sys.exit(1)
    event_id = sys.argv[1]
    url = f"https://apuestas.wplay.co/es/e/{event_id}/"

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
    print(f"navigating to {url}")
    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    await page.wait_for_timeout(6_000)  # give SPA time to render markets

    html = await page.content()
    out = ROOT / "logs" / "wplay_debug" / f"match_{event_id}_probe.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    png = out.with_suffix(".png")
    try:
        await page.screenshot(path=str(png), full_page=True)
    except Exception:
        pass

    print(f"saved -> {out.name} ({len(html)} bytes)")
    print(f"saved -> {png.name}")
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
