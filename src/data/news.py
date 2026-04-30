"""News fetcher for a specific match.

Uses Google News RSS feed, which is free and doesn't require auth. We query
for both team names and grab the top N article titles + snippets.

Output is a list of short snippets ready to feed into the LLM extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

import httpx
from loguru import logger


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


@dataclass
class NewsSnippet:
    title: str
    description: str
    pub_date: str


async def fetch_match_news(
    home_team: str, away_team: str, language: str = "es", region: str = "CO",
    max_items: int = 8,
) -> list[NewsSnippet]:
    """Search Google News for articles mentioning both teams.

    The RSS XML uses a simple <item> structure with <title> + <description> +
    <pubDate>. We don't need a full XML parser — a regex pull is enough since
    the format is consistent.
    """
    query = f"{home_team} {away_team}"
    url = (
        f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}"
        f"&hl={language}-{region}&gl={region}&ceid={region}:{language}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"accept": "application/rss+xml"})
            resp.raise_for_status()
            text = resp.text
    except Exception as exc:
        logger.warning(f"news fetch failed for {home_team} vs {away_team}: {exc}")
        return []

    # Pull <item>…</item> blocks. Inside each, regex out title/description/pubDate.
    import re
    items: list[NewsSnippet] = []
    for item_match in re.finditer(r"<item>(.*?)</item>", text, re.DOTALL):
        block = item_match.group(1)
        title = _between(block, "<title>", "</title>")
        desc = _between(block, "<description>", "</description>")
        pub = _between(block, "<pubDate>", "</pubDate>")
        if not title:
            continue
        items.append(NewsSnippet(
            title=_strip_cdata(title)[:300],
            description=_strip_html(desc)[:400] if desc else "",
            pub_date=pub or "",
        ))
        if len(items) >= max_items:
            break
    return items


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        return ""
    return text[i + len(start):j].strip()


def _strip_cdata(text: str) -> str:
    if text.startswith("<![CDATA[") and text.endswith("]]>"):
        return text[9:-3]
    return text


def _strip_html(text: str) -> str:
    import re
    s = _strip_cdata(text)
    return re.sub(r"<[^>]+>", "", s).strip()


# Smoke test
async def _smoke() -> None:
    items = await fetch_match_news("Arsenal", "Fulham")
    print(f"\nFetched {len(items)} news items")
    for it in items[:3]:
        print(f"  • {it.title[:80]}")
        if it.description:
            print(f"    {it.description[:120]}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke())
