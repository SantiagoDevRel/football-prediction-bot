"""Backfill match_stats with cards, corners, fouls, shots from ESPN summary.

For each finished match in our DB that doesn't yet have stats, hit ESPN's
/summary?event=ID endpoint and persist the box-score numbers.

Cost: ~1 sec per match, ~1700 finished matches → ~30 min wall time once.
The endpoint is free (no auth, no quota), but we rate-limit to ~5 req/s.

Usage:
    python scripts/backfill_boxscore.py
    python scripts/backfill_boxscore.py --max 200       # only 200 matches
    python scripts/backfill_boxscore.py --league premier_league
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.data.espn import fetch_match_summary  # noqa: E402
from src.data.persist import get_conn  # noqa: E402


async def backfill(max_matches: int | None, league_filter: str | None) -> None:
    league_filter_sql = ""
    league_filter_args: list = []
    if league_filter == "premier_league":
        league_filter_sql = " AND l.name = 'Premier League'"
    elif league_filter == "liga_betplay":
        league_filter_sql = " AND l.name = 'Liga BetPlay Dimayor'"

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT m.id, m.api_id, l.name AS league_name
              FROM matches m
              JOIN leagues l ON m.league_id = l.id
              LEFT JOIN match_stats s ON s.match_id = m.id
             WHERE m.status = 'finished'
               AND m.api_id IS NOT NULL
               AND s.match_id IS NULL
               {league_filter_sql}
             ORDER BY m.kickoff_utc DESC
            """
        ).fetchall()

    if max_matches is not None:
        rows = rows[:max_matches]
    print(f"backfilling {len(rows)} matches…")

    inserted = 0
    failed = 0
    for r in rows:
        league_slug = (
            "premier_league" if "Premier" in r["league_name"] else "liga_betplay"
        )
        stats = await fetch_match_summary(league_slug, str(r["api_id"]))
        if stats is None:
            failed += 1
            continue

        # Skip if all-None (game without stats)
        if all(v is None for v in stats.values()):
            failed += 1
            continue

        with get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO match_stats
                    (match_id, home_yellow_cards, away_yellow_cards,
                     home_red_cards, away_red_cards,
                     home_corners, away_corners,
                     home_fouls, away_fouls,
                     home_shots, away_shots,
                     home_shots_on_target, away_shots_on_target,
                     home_possession, away_possession)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["id"],
                    stats["home_yellow_cards"], stats["away_yellow_cards"],
                    stats["home_red_cards"], stats["away_red_cards"],
                    stats["home_corners"], stats["away_corners"],
                    stats["home_fouls"], stats["away_fouls"],
                    stats["home_shots"], stats["away_shots"],
                    stats["home_shots_on_target"], stats["away_shots_on_target"],
                    stats["home_possession"], stats["away_possession"],
                ),
            )
        inserted += 1
        if inserted % 50 == 0:
            print(f"  …{inserted} done")
        await asyncio.sleep(0.2)  # ~5 req/s

    print(f"\nDONE  inserted={inserted}  failed={failed}")


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--league", default=None,
                        choices=["premier_league", "liga_betplay"])
    args = parser.parse_args()
    await backfill(args.max, args.league)


if __name__ == "__main__":
    asyncio.run(main_async())
