"""Logs picks to SQLite, computes CLV after match resolution.

CLV (Closing Line Value):
    clv = (odds_taken / closing_odds) - 1
    Positive CLV means we got a better price than the market closed at.
    Sustained positive CLV is the only proof of edge.

Phase 1 implementation.
"""
from __future__ import annotations

from src.betting.value_detector import ValueBet


def log_pick(pick: ValueBet, mode: str = "paper") -> int:
    """Insert a pick row, return its id."""
    raise NotImplementedError("Phase 1")


def resolve_pick(pick_id: int, won: bool, closing_odds: float | None) -> None:
    """Mark pick as resolved, compute payout and CLV, update bankroll history."""
    raise NotImplementedError("Phase 1")


def compute_rolling_metrics(model: str, market: str, days: int = 30) -> dict:
    """Compute log-loss, Brier, win-rate, avg CLV, ROI for the period."""
    raise NotImplementedError("Phase 1")
