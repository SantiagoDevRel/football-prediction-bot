"""Enrich upcoming matches with LLM-extracted qualitative flags.

Workflow:
    1. Find upcoming scheduled matches (next 2 days) for our leagues.
    2. For each: fetch Google News snippets → run Claude Haiku → store
       structured flags in qualitative_features table.
    3. The XGBoost feature builder later reads these flags from DB and
       turns them into one-hot features.

Designed to run once or twice a day. Cost: ~30 matches × $0.005 = $0.15/day.
"""
import asyncio
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.config import settings  # noqa: E402
from src.data.news import fetch_match_news  # noqa: E402
from src.data.persist import get_conn  # noqa: E402
from src.llm.feature_extractor import LLMFeatureExtractor  # noqa: E402


async def _enrich_one(extractor: LLMFeatureExtractor, match: dict) -> dict | None:
    """Fetch news, run LLM, persist flags. Returns the inserted row dict."""
    snippets = await fetch_match_news(match["home_name"], match["away_name"])
    if not snippets:
        logger.info(f"no news for {match['home_name']} vs {match['away_name']}")
        return None

    snippet_texts = [f"{s.title}. {s.description}" for s in snippets if s.title]
    feats = await extractor.extract(
        home_team=match["home_name"],
        away_team=match["away_name"],
        league=match["league_name"],
        news_snippets=snippet_texts,
    )

    with get_conn() as conn:
        # Upsert: delete prior row for this match, insert fresh
        conn.execute(
            "DELETE FROM qualitative_features WHERE match_id = ?",
            (match["id"],),
        )
        cur = conn.execute(
            """
            INSERT INTO qualitative_features
                (match_id, flags, summary, raw_news, model_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                match["id"],
                json.dumps(feats.flags),
                feats.summary,
                "\n".join(snippet_texts)[:2000],
                feats.model_used,
            ),
        )
    logger.info(
        f"enriched #{match['id']} {match['home_name']} v {match['away_name']}: "
        f"{len(feats.flags)} flags, confidence {feats.confidence:.2f}"
    )
    return {"match_id": match["id"], "flags": feats.flags, "summary": feats.summary}


async def main_async() -> None:
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set in .env. Aborting.")
        return

    today = date.today()
    end = today + timedelta(days=2)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
                   h.name AS home_name, a.name AS away_name, l.name AS league_name
              FROM matches m
              JOIN teams h ON m.home_team_id = h.id
              JOIN teams a ON m.away_team_id = a.id
              JOIN leagues l ON m.league_id = l.id
             WHERE m.status = 'scheduled'
               AND date(m.kickoff_utc) >= ?
               AND date(m.kickoff_utc) <= ?
            """,
            (today.isoformat(), end.isoformat()),
        ).fetchall()
    matches = [dict(r) for r in rows]
    print(f"Enriching {len(matches)} upcoming matches…")

    extractor = LLMFeatureExtractor(settings.anthropic_api_key)
    results = []
    for m in matches:
        try:
            r = await _enrich_one(extractor, m)
            if r:
                results.append(r)
        except Exception as exc:
            logger.warning(f"enrich failed for #{m['id']}: {exc}")

    print(f"\nEnriched {len(results)} matches.")
    for r in results[:10]:
        flags_str = ", ".join(r["flags"]) if r["flags"] else "(no flags)"
        print(f"  #{r['match_id']:5}  flags: {flags_str}")
        if r["summary"]:
            print(f"           summary: {r['summary'][:160]}")


if __name__ == "__main__":
    asyncio.run(main_async())
