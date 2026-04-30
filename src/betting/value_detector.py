"""Compares model predictions against bookmaker odds to find value bets.

A pick is suggested only when ALL gates pass:
    1. edge > MIN_EDGE (default 5%)
    2. confidence > MIN_CONFIDENCE (default 60%, Bayesian)
    3. liquidity / market depth ok (skip exotic markets with thin lines)
    4. model has been calibrated for this league/market in training

Phase 1 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueBet:
    match_id: int
    market: str
    selection: str
    odds: float
    bookmaker: str
    model_probability: float
    fair_odds: float
    edge: float
    confidence: float | None
    recommended_stake: float
    reasoning: str  # short string for the Telegram message


def detect_value(
    predictions, odds_snapshots, bankroll: float
) -> list[ValueBet]:
    """Cross-reference model predictions with current odds and return value bets."""
    raise NotImplementedError("Phase 1")
