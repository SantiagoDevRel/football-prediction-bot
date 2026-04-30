"""Train Dixon-Coles + Elo on persisted historical matches.

Usage:
    python scripts/train_models.py [--league premier_league|liga_betplay|both]
"""
import argparse
import pickle
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.models.elo import Elo  # noqa: E402
from src.models.xgboost_model import XGBoostModel  # noqa: E402

ARTIFACTS_DIR = ROOT / "models_artifacts"


def load_matches_for_league(league_name: str, exclude_after: date | None = None) -> list[dict]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT m.kickoff_utc, m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
               m.league_id
          FROM matches m
          JOIN leagues l ON m.league_id = l.id
         WHERE l.name = ? AND m.status = 'finished'
           AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
    """
    params: list = [league_name]
    if exclude_after is not None:
        sql += " AND date(m.kickoff_utc) <= ?"
        params.append(exclude_after.isoformat())
    sql += " ORDER BY m.kickoff_utc ASC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [
        {
            "home_team_id": r["home_team_id"],
            "away_team_id": r["away_team_id"],
            "home_goals": r["home_goals"],
            "away_goals": r["away_goals"],
            "league_id": r["league_id"],
            "kickoff_date": date.fromisoformat(r["kickoff_utc"][:10]),
        }
        for r in rows
    ]


def train_for_league(league_name: str) -> None:
    print(f"\n=== {league_name} ===")
    matches = load_matches_for_league(league_name)
    print(f"  {len(matches)} finished matches in DB")
    if len(matches) < 100:
        print("  not enough matches to train. skipping.")
        return

    # Train Dixon-Coles
    dc = DixonColes(xi=0.0019)
    dc.fit(matches)

    # Train Elo
    elo = Elo()
    elo.fit(matches)

    # Train XGBoost
    try:
        xgb_model = XGBoostModel(db_path=str(settings.db_path))
        xgb_model.fit(matches)
    except Exception as exc:
        print(f"  XGBoost fit failed: {exc}")
        xgb_model = None

    # Train stacking ensemble (DC + Elo + XGBoost)
    try:
        from src.models.stacking import StackingEnsemble
        stack = StackingEnsemble(base_models=[
            DixonColes(xi=0.0019),
            Elo(),
            XGBoostModel(db_path=str(settings.db_path)),
        ])
        stack.fit(matches)
    except Exception as exc:
        print(f"  Stacking fit failed: {exc}")
        stack = None

    # Print top 5 teams by attack (Dixon-Coles) and Elo
    teams_attack = sorted(
        ((dc.idx_to_team[i], dc.attack[i]) for i in range(len(dc.attack))),
        key=lambda x: -x[1],
    )
    teams_elo = sorted(elo.ratings.items(), key=lambda x: -x[1])

    # Resolve team names
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    name_by_id = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM teams")}
    conn.close()

    print(f"\n  Top 5 attack (DC): ")
    for tid, score in teams_attack[:5]:
        print(f"    {name_by_id.get(tid, '?')[:30]:30} attack={score:+.3f}")
    print(f"\n  Top 5 Elo: ")
    for tid, rating in teams_elo[:5]:
        print(f"    {name_by_id.get(tid, '?')[:30]:30} {rating:.0f}")

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_slug = league_name.lower().replace(" ", "_").replace(".", "")
    with open(ARTIFACTS_DIR / f"dixon_coles_{safe_slug}.pkl", "wb") as f:
        pickle.dump(dc, f)
    with open(ARTIFACTS_DIR / f"elo_{safe_slug}.pkl", "wb") as f:
        pickle.dump(elo, f)
    if xgb_model is not None:
        with open(ARTIFACTS_DIR / f"xgboost_{safe_slug}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
    if stack is not None:
        with open(ARTIFACTS_DIR / f"stacking_{safe_slug}.pkl", "wb") as f:
            pickle.dump(stack, f)
    print(f"\n  saved -> {ARTIFACTS_DIR}/[dixon_coles|elo|xgboost|stacking]_{safe_slug}.pkl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["premier_league", "liga_betplay", "both"], default="both")
    args = parser.parse_args()

    name_map = {"premier_league": "Premier League", "liga_betplay": "Liga BetPlay Dimayor"}
    if args.league == "both":
        for k in name_map.values():
            train_for_league(k)
    else:
        train_for_league(name_map[args.league])


if __name__ == "__main__":
    main()
