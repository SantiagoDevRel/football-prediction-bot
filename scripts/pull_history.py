"""Pull historical fixtures from ESPN and persist to SQLite.

Used to bootstrap the DB with enough history to train Dixon-Coles, Elo, etc.

Usage:
    python scripts/pull_history.py [--league premier_league] [--from 2024-08-01] [--to 2025-05-31]
    python scripts/pull_history.py --all-recent     # last 18 months, all leagues
"""
import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.espn import fetch_season_history  # noqa: E402
from src.data.persist import bulk_upsert_espn  # noqa: E402


async def pull_one(league_slug: str, start: date, end: date) -> int:
    print(f"\n[{league_slug}] pulling {start} -> {end} from ESPN ...")
    matches = await fetch_season_history(league_slug, start, end)
    print(f"[{league_slug}] fetched {len(matches)} unique matches")
    n = bulk_upsert_espn(matches)
    print(f"[{league_slug}] persisted {n}")
    return n


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier_league",
                        choices=["premier_league", "liga_betplay", "sudamericana", "libertadores",
                                 "champions_league"])
    parser.add_argument("--from", dest="from_date", help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", help="YYYY-MM-DD")
    parser.add_argument("--all-recent", action="store_true",
                        help="pull last 18 months for both leagues")
    args = parser.parse_args()

    if args.all_recent:
        end = date.today()
        start = end - timedelta(days=540)  # ~18 months
        for slug in ("premier_league", "liga_betplay", "sudamericana", "libertadores",
                     "champions_league"):
            await pull_one(slug, start, end)
        return

    start = date.fromisoformat(args.from_date) if args.from_date else date.today() - timedelta(days=365)
    end = date.fromisoformat(args.to_date) if args.to_date else date.today()
    await pull_one(args.league, start, end)


if __name__ == "__main__":
    asyncio.run(main_async())
