"""Manually log a paper pick for a specific match + odds.

Use case: the auto-Wplay scraper is not wired yet. You see odds on Wplay,
type them here, the bot:
  - Looks up the match in the DB
  - Runs the loaded ensemble (Dixon-Coles + Elo) for the match
  - Computes edge vs the odds you provided
  - If edge > threshold, logs a paper pick at full Kelly-fractional stake
  - Otherwise, refuses and tells you why

Usage:
    python scripts/place_paper_pick.py \
        --home "Arsenal" --away "Fulham" \
        --market 1x2 --selection home --odds 1.65

    python scripts/place_paper_pick.py --list   # show today's predictions
"""
import argparse
import pickle
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.betting.value_detector import detect_value, OddsLine  # noqa: E402
from src.config import settings  # noqa: E402
from src.tracking.pick_logger import get_current_bankroll, log_pick  # noqa: E402

ARTIFACTS = ROOT / "models_artifacts"
LEAGUE_NAME = {"premier_league": "Premier League", "liga_betplay": "Liga BetPlay Dimayor"}


def list_today() -> None:
    today = date.today().isoformat()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.id, m.kickoff_utc, h.name AS home, a.name AS away, l.name AS league
          FROM matches m
          JOIN teams h ON m.home_team_id = h.id
          JOIN teams a ON m.away_team_id = a.id
          JOIN leagues l ON m.league_id = l.id
         WHERE date(m.kickoff_utc) >= ?
           AND m.status = 'scheduled'
         ORDER BY m.kickoff_utc ASC
        """, (today,)
    ).fetchall()
    if not rows:
        print("(no upcoming scheduled matches)")
        return
    for r in rows:
        league_short = "EPL" if "Premier" in r["league"] else "BetPlay"
        print(f"  [{league_short}] {r['kickoff_utc'][:16]:16}  {r['home']} vs {r['away']}")


def find_match(home: str, away: str) -> dict | None:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
               h.name AS home_name, a.name AS away_name, l.name AS league_name
          FROM matches m
          JOIN teams h ON m.home_team_id = h.id
          JOIN teams a ON m.away_team_id = a.id
          JOIN leagues l ON m.league_id = l.id
         WHERE m.status = 'scheduled'
           AND lower(h.name) LIKE lower(?) AND lower(a.name) LIKE lower(?)
         ORDER BY m.kickoff_utc ASC LIMIT 1
        """, (f"%{home}%", f"%{away}%")
    ).fetchone()
    return dict(rows) if rows else None


def league_slug_from_name(name: str) -> str:
    return "premier_league" if "Premier" in name else "liga_betplay"


def load_models(slug: str) -> dict:
    nice = LEAGUE_NAME[slug].lower().replace(" ", "_").replace(".", "")
    out = {}
    for m in ("dixon_coles", "elo"):
        p = ARTIFACTS / f"{m}_{nice}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                out[m] = pickle.load(f)
    return out


def ensemble_predict(models: dict, h: int, a: int):
    parts = []
    for m in models.values():
        try:
            parts.append(m.predict_match(h, a))
        except KeyError:
            continue
    if not parts:
        return None
    n = len(parts)
    proto = parts[0]
    return type(proto)(
        p_home_win=sum(p.p_home_win for p in parts) / n,
        p_draw=sum(p.p_draw for p in parts) / n,
        p_away_win=sum(p.p_away_win for p in parts) / n,
        p_over_2_5=sum(p.p_over_2_5 for p in parts) / n,
        p_under_2_5=sum(p.p_under_2_5 for p in parts) / n,
        p_btts_yes=sum(p.p_btts_yes for p in parts) / n,
        p_btts_no=sum(p.p_btts_no for p in parts) / n,
        expected_home_goals=sum(p.expected_home_goals for p in parts) / n,
        expected_away_goals=sum(p.expected_away_goals for p in parts) / n,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list upcoming matches")
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--market", choices=["1x2", "ou_2.5", "btts"])
    parser.add_argument("--selection", choices=["home", "draw", "away", "over", "under", "yes", "no"])
    parser.add_argument("--odds", type=float)
    parser.add_argument("--force", action="store_true",
                        help="log even if edge is below threshold (for record-keeping)")
    args = parser.parse_args()

    if args.list:
        list_today()
        return

    if not all([args.home, args.away, args.market, args.selection, args.odds]):
        parser.error("must supply --home, --away, --market, --selection, --odds (or use --list)")

    match = find_match(args.home, args.away)
    if not match:
        print(f"no upcoming match found matching {args.home} vs {args.away}")
        sys.exit(1)
    print(f"matched: {match['home_name']} vs {match['away_name']} ({match['league_name']}) @ {match['kickoff_utc'][:16]}")

    slug = league_slug_from_name(match["league_name"])
    models = load_models(slug)
    if not models:
        print(f"no trained models for {slug}. Run scripts/train_models.py first.")
        sys.exit(1)

    pred = ensemble_predict(models, match["home_team_id"], match["away_team_id"])
    if pred is None:
        print("could not generate prediction (team missing from training set)")
        sys.exit(1)

    # Show all probs for context
    print(f"\nModel ensemble probabilities:")
    print(f"  1X2: H={pred.p_home_win:.1%}  D={pred.p_draw:.1%}  A={pred.p_away_win:.1%}")
    print(f"  O/U 2.5: O={pred.p_over_2_5:.1%}  U={pred.p_under_2_5:.1%}")
    print(f"  BTTS: Y={pred.p_btts_yes:.1%}  N={pred.p_btts_no:.1%}")

    bankroll = get_current_bankroll("paper")
    odds_line = OddsLine(args.market, args.selection, args.odds, bookmaker="wplay_manual")

    picks = detect_value(
        match_id=match["id"],
        home_team=match["home_name"],
        away_team=match["away_name"],
        league=match["league_name"],
        prediction=pred,
        odds_lines=[odds_line],
        bankroll=bankroll,
    )
    if not picks:
        # Compute what the user passed for transparency
        from src.betting.kelly import edge as edge_fn
        from src.betting.value_detector import _select_prob
        prob = _select_prob(pred, args.market, args.selection)
        e = edge_fn(args.odds, prob) if prob is not None else 0
        print(f"\nNo value at given odds. Model says {prob:.1%}, your odds {args.odds:.2f} → edge {e:+.1%} (threshold: +{settings.min_edge:.0%})")
        if not args.force:
            sys.exit(0)
        # Force-log via direct ValueBet construction
        from src.betting.value_detector import ValueBet
        from src.betting.kelly import kelly_stake
        stake = kelly_stake(bankroll=bankroll, odds=args.odds, probability=prob,
                            fraction=settings.kelly_fraction, max_stake_pct=0.05)
        vb = ValueBet(
            match_id=match["id"], home_team=match["home_name"], away_team=match["away_name"],
            league=match["league_name"], market=args.market, selection=args.selection,
            odds=args.odds, bookmaker="wplay_manual", model_probability=prob,
            fair_odds=1.0/prob, edge=e, confidence=None, recommended_stake=max(stake, 0),
            reasoning="forced (--force) below threshold",
        )
        log_pick(vb, mode="paper")
        return

    p = picks[0]
    print(f"\n✓ Value detected: edge {p.edge:+.1%}, fair odds {p.fair_odds:.2f}")
    print(f"  Stake recommended: ${p.recommended_stake:,.0f} (¼ Kelly)")
    log_pick(p, mode="paper")


if __name__ == "__main__":
    main()
