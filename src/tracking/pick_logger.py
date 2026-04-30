"""Persist picks, resolve them after the match, compute CLV.

CLV (Closing Line Value):
    clv = (odds_taken / closing_odds) - 1
    Sustained positive CLV is the ONLY proof of edge.

Modes:
    - "paper": stake from a virtual bankroll (settings.paper_bankroll_initial)
    - "real":  user's real money. Activated explicitly via env var.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Literal

from loguru import logger

from src.betting.value_detector import ValueBet
from src.config import settings
from src.data.persist import get_conn


PaperOrReal = Literal["paper", "real"]


def get_current_bankroll(mode: PaperOrReal) -> float:
    """Return current bankroll for the given mode.

    Paper: starts at settings.paper_bankroll_initial, then walks via bankroll_history.
    Real:  must be seeded by the user via a manual deposit row.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
    if row is not None:
        return float(row["balance"])
    if mode == "paper":
        # Seed initial paper bankroll
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
                "VALUES (?, NULL, ?, ?, 'initial seed')",
                (mode, settings.paper_bankroll_initial, settings.paper_bankroll_initial),
            )
        logger.info(f"seeded paper bankroll: ${settings.paper_bankroll_initial:,.0f}")
        return float(settings.paper_bankroll_initial)
    # Real mode without a deposit row → 0
    logger.warning("real bankroll has no deposit row; returning 0")
    return 0.0


def log_pick(pick: ValueBet, mode: PaperOrReal = "paper") -> int:
    """Insert a pick row + corresponding bankroll history row (stake out). Returns pick id."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO picks
                (match_id, market, selection, odds_taken, bookmaker,
                 model_probability, edge, confidence, stake, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (pick.match_id, pick.market, pick.selection, pick.odds, pick.bookmaker,
             pick.model_probability, pick.edge, pick.confidence,
             pick.recommended_stake, mode),
        )
        pick_id = cur.lastrowid

        # Subtract stake from bankroll
        prev = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
        prev_balance = float(prev["balance"]) if prev else 0.0
        new_balance = prev_balance - pick.recommended_stake
        conn.execute(
            "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (mode, pick_id, -pick.recommended_stake, new_balance,
             f"stake on {pick.market}:{pick.selection} @ {pick.odds}"),
        )
    logger.info(
        f"[{mode}] logged pick #{pick_id}: {pick.home_team} vs {pick.away_team} "
        f"{pick.market}:{pick.selection} @ {pick.odds} | edge {pick.edge:+.1%} | "
        f"stake ${pick.recommended_stake:,.0f} | new balance ${new_balance:,.0f}"
    )
    return pick_id  # type: ignore[return-value]


def resolve_pick(pick_id: int, won: bool, closing_odds: float | None = None) -> None:
    """Mark a pick as resolved, compute payout/CLV, update bankroll."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT odds_taken, stake, mode FROM picks WHERE id = ?", (pick_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"pick {pick_id} not found")
        odds_taken = float(row["odds_taken"])
        stake = float(row["stake"])
        mode = row["mode"]

        if won:
            payout = stake * odds_taken  # gross
            net = payout - stake
        else:
            payout = 0.0
            net = -stake

        clv = None
        if closing_odds and closing_odds > 0:
            clv = (odds_taken / closing_odds) - 1.0

        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """
            UPDATE picks
               SET won = ?, payout = ?, closing_odds = ?, clv = ?, resolved_at = ?
             WHERE id = ?
            """,
            (1 if won else 0, payout, closing_odds, clv, now, pick_id),
        )

        # Bankroll update: ADD payout (gross). Stake was already subtracted at log time.
        prev = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
        prev_balance = float(prev["balance"]) if prev else 0.0
        new_balance = prev_balance + payout
        conn.execute(
            "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (mode, pick_id, payout, new_balance,
             f"payout on pick #{pick_id} ({'won' if won else 'lost'})"),
        )

    logger.info(
        f"[{mode}] resolved pick #{pick_id}: {'WON' if won else 'LOST'} | "
        f"net ${net:+,.0f} | CLV {clv:+.2%} | balance ${new_balance:,.0f}"
        if clv is not None else
        f"[{mode}] resolved pick #{pick_id}: {'WON' if won else 'LOST'} | "
        f"net ${net:+,.0f} | balance ${new_balance:,.0f}"
    )


def compute_rolling_metrics(mode: PaperOrReal, days: int = 30) -> dict:
    """Aggregate metrics for the last N days."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT odds_taken, stake, won, payout, clv
              FROM picks
             WHERE mode = ? AND won IS NOT NULL
               AND date(placed_at) >= date('now', ?)
            """,
            (mode, f"-{days} days"),
        ).fetchall()
    if not rows:
        return {"n": 0, "win_rate": 0.0, "roi": 0.0, "avg_clv": 0.0, "total_pnl": 0.0}
    n = len(rows)
    wins = sum(1 for r in rows if r["won"] == 1)
    total_stake = sum(float(r["stake"]) for r in rows)
    total_pnl = sum(float(r["payout"]) - float(r["stake"]) for r in rows)
    clvs = [float(r["clv"]) for r in rows if r["clv"] is not None]
    return {
        "n": n,
        "win_rate": wins / n,
        "roi": total_pnl / total_stake if total_stake > 0 else 0.0,
        "avg_clv": sum(clvs) / len(clvs) if clvs else None,
        "total_pnl": total_pnl,
    }
