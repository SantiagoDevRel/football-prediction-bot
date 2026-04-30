"""Backtest harness: chronological replay with no leakage.

For each match in the test period, train (or update) the model only on
matches BEFORE that match, predict, then compare with the actual result.

Metrics:
    - Accuracy on 1X2 (top-1)
    - Brier score (multiclass)
    - Log-loss
    - Calibration buckets (when we say 60%, do they win 60%?)
    - ROI: simulate flat $1 stake on the most-likely outcome
    - ROI with ¼ Kelly when an edge >5% exists vs implied closing odds (if we have them)

Usage:
    python scripts/backtest.py --league premier_league --train-end 2025-12-31 --test-from 2026-01-01
"""
import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.models.elo import Elo  # noqa: E402
from src.models.xgboost_model import XGBoostModel  # noqa: E402


LEAGUE_NAMES = {
    "premier_league": "Premier League",
    "liga_betplay": "Liga BetPlay Dimayor",
}


@dataclass
class BacktestResult:
    model: str
    league: str
    n: int
    accuracy_1x2: float
    log_loss: float
    brier: float
    roi_flat_top1: float
    calibration: dict


def _load_matches(league_name: str, end: date | None = None) -> list[dict]:
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
    if end is not None:
        sql += " AND date(m.kickoff_utc) <= ?"
        params.append(end.isoformat())
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


def _bucket(p: float) -> str:
    edges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for e in edges:
        if p < e:
            return f"<{int(e*100)}%"
    return ">=90%"


def evaluate(model_name: str, league_slug: str, train_end: date, test_from: date) -> BacktestResult:
    league_name = LEAGUE_NAMES[league_slug]
    print(f"\n--- {model_name} on {league_name} | train<={train_end} test>={test_from} ---")
    train_matches = _load_matches(league_name, end=train_end)
    test_all = _load_matches(league_name)
    test_matches = [m for m in test_all if m["kickoff_date"] >= test_from]
    print(f"  train={len(train_matches)} matches | test={len(test_matches)} matches")

    if model_name == "dixon_coles":
        model = DixonColes(xi=0.0019)
    elif model_name == "elo":
        model = Elo()
    elif model_name == "xgboost":
        model = XGBoostModel(db_path=str(settings.db_path))
    else:
        raise ValueError(f"unknown model {model_name}")

    model.fit(train_matches)

    n = 0
    correct = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    pnl_top1 = 0.0  # flat $1 stake on top-1 1X2 pick at FAIR odds (1/p_top)
    cal_buckets: dict[str, list[int]] = defaultdict(list)

    # Realized 1X2 outcome encoded as 0=H,1=D,2=A
    for m in test_matches:
        try:
            pred = model.predict_match(
                m["home_team_id"], m["away_team_id"],
                kickoff_date=m["kickoff_date"], league_id=m.get("league_id"),
            )
        except KeyError:
            # team not in training set — common for promoted teams. skip.
            continue
        probs = [pred.p_home_win, pred.p_draw, pred.p_away_win]
        # Normalize defensively
        s = sum(probs)
        if s <= 0:
            continue
        probs = [p / s for p in probs]

        if m["home_goals"] > m["away_goals"]:
            actual = 0
        elif m["home_goals"] == m["away_goals"]:
            actual = 1
        else:
            actual = 2

        # Top-1 accuracy
        top = max(range(3), key=lambda i: probs[i])
        correct += int(top == actual)

        # Log loss
        p_actual = max(probs[actual], 1e-12)
        log_loss_sum += -math.log(p_actual)

        # Brier (multiclass)
        target = [0, 0, 0]
        target[actual] = 1
        brier_sum += sum((probs[i] - target[i]) ** 2 for i in range(3))

        # Flat $1 stake on top-1 at fair-odds (1/p_top * (1 - 0.05 vig))
        # This isolates whether the model's top pick wins more than its prob.
        # Real ROI tracking will use Wplay live odds in production.
        bookmaker_vig = 0.05
        fair_odds = 1.0 / max(probs[top], 1e-6)
        bookmaker_odds = fair_odds * (1.0 - bookmaker_vig)  # what a casa would offer
        if top == actual:
            pnl_top1 += bookmaker_odds - 1.0
        else:
            pnl_top1 += -1.0

        # Calibration: bucket the probability assigned to the actual outcome
        cal_buckets[_bucket(probs[actual])].append(1)
        for i in range(3):
            if i != actual:
                cal_buckets[_bucket(probs[i])].append(0)

        n += 1

    if n == 0:
        print("  no test matches with both teams in training — increase data")
        return BacktestResult(model_name, league_name, 0, 0, 0, 0, 0, {})

    accuracy = correct / n
    log_loss = log_loss_sum / n
    brier = brier_sum / n
    roi_flat = pnl_top1 / n

    cal_summary = {
        bucket: (sum(cal_buckets[bucket]) / len(cal_buckets[bucket]), len(cal_buckets[bucket]))
        for bucket in sorted(cal_buckets.keys())
    }

    print(f"  N predicted: {n}")
    print(f"  Accuracy(1X2 top-1): {accuracy:.1%}")
    print(f"  Log-loss: {log_loss:.4f}  (lower is better; uniform=1.099)")
    print(f"  Brier:    {brier:.4f}     (lower is better; uniform=0.667)")
    print(f"  ROI flat top-1 @ casa-odds(vig=5%): {roi_flat:+.2%} per match")
    print(f"  Calibration (bucket -> realized rate, n):")
    for bucket, (rate, count) in cal_summary.items():
        print(f"     {bucket:8} -> {rate:.1%} ({count})")

    return BacktestResult(model_name, league_name, n, accuracy, log_loss, brier, roi_flat, cal_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["premier_league", "liga_betplay"], default="premier_league")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-from", default="2026-01-01")
    parser.add_argument("--models", default="dixon_coles,elo",
                        help="comma-separated subset of: dixon_coles,elo")
    args = parser.parse_args()

    train_end = date.fromisoformat(args.train_end)
    test_from = date.fromisoformat(args.test_from)
    results = []
    for model_name in args.models.split(","):
        results.append(evaluate(model_name.strip(), args.league, train_end, test_from))

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r.model:13} acc={r.accuracy_1x2:.1%}  ll={r.log_loss:.4f}  brier={r.brier:.4f}  roi={r.roi_flat_top1:+.2%}  n={r.n}")


if __name__ == "__main__":
    main()
