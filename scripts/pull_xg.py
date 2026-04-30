"""Pull Understat xG for past Premier League seasons and update DB.

Usage:
    python scripts/pull_xg.py                         # default: 2020..2024
    python scripts/pull_xg.py --seasons 2023,2024    # specific seasons
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.persist import update_xg_from_understat  # noqa: E402
from src.data.understat import scrape_league_season  # noqa: E402


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", default="2024",
        help="Comma-separated season starts (e.g. '2020,2021,2022,2023,2024')",
    )
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]

    grand_total_matched = 0
    for season in seasons:
        print(f"\n=== Season {season}/{(season + 1) % 100:02d} ===")
        try:
            matches = await scrape_league_season("EPL", season)
        except Exception as exc:
            print(f"  scrape failed: {exc}")
            continue
        print(f"  scraped {len(matches)} matches")
        finished = [m for m in matches if m.is_finished]
        print(f"  finished: {len(finished)}")

        result = update_xg_from_understat(finished)
        print(
            f"  DB update: matched={result['matched']}  "
            f"missing={result['missing']}  xG_updated={result['updated_xg']}"
        )
        grand_total_matched += result["matched"]

    print(f"\nTOTAL xG rows persisted across seasons: {grand_total_matched}")


if __name__ == "__main__":
    asyncio.run(main_async())
