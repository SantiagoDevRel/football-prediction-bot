"""Backfill goal_events from ESPN summary keyEvents.

For each finished match in our DB without goal events yet, hit ESPN's
/summary?event=ID and persist the goals (scorer, minute, penalty/own-goal).

Cost: ~1 sec per match. Free, no auth.

Usage:
    python scripts/backfill_goals.py
    python scripts/backfill_goals.py --max 200
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.data.espn import fetch_match_goal_events  # noqa: E402
from src.data.persist import get_conn  # noqa: E402


async def backfill(max_matches: int | None) -> None:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.api_id, l.name AS league_name, m.home_goals, m.away_goals
              FROM matches m
              JOIN leagues l ON m.league_id = l.id
              LEFT JOIN goal_events g ON g.match_id = m.id
             WHERE m.status = 'finished'
               AND m.api_id IS NOT NULL
               AND (m.home_goals + m.away_goals) > 0
               AND g.id IS NULL
             GROUP BY m.id
             ORDER BY m.kickoff_utc DESC
            """
        ).fetchall()
    if max_matches is not None:
        rows = rows[:max_matches]
    print(f"backfilling goal events for {len(rows)} matches…")

    inserted_goals = 0
    failed = 0
    for r in rows:
        league_slug = (
            "premier_league" if "Premier" in r["league_name"]
            else "liga_betplay" if "BetPlay" in r["league_name"]
            else "sudamericana" if "Sudamericana" in r["league_name"]
            else "libertadores" if "Libertadores" in r["league_name"]
            else None
        )
        if not league_slug:
            continue
        events = await fetch_match_goal_events(league_slug, str(r["api_id"]))
        if events is None:
            failed += 1
            continue

        with get_conn() as conn:
            for e in events:
                conn.execute(
                    """
                    INSERT INTO goal_events
                        (match_id, espn_player_id, player_name, team_api_id,
                         team_name, minute, is_penalty, is_own_goal)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (r["id"], e["espn_player_id"], e["player_name"],
                     e["team_api_id"], e["team_name"], e["minute"],
                     1 if e["is_penalty"] else 0,
                     1 if e["is_own_goal"] else 0),
                )
                inserted_goals += 1
        if len(events) and inserted_goals % 100 == 0:
            print(f"  …{inserted_goals} goals so far")
        await asyncio.sleep(0.2)

    print(f"\nDONE  goals inserted={inserted_goals}  matches failed={failed}")


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    await backfill(args.max)


if __name__ == "__main__":
    asyncio.run(main_async())
