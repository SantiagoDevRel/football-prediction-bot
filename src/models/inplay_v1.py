"""In-play v1 — minute-bucketed goal-rate model.

Improvement over v0 (flat Poisson over remaining minutes): we use the
empirical distribution of WHEN goals are scored (from goal_events table)
to weight remaining lambda by what's actually realistic.

Empirical observation: top European leagues score noticeably more in
the second half (especially 60-75 min, 80-90 min) than the first half.
Using a flat 1/90 rate underestimates over-2.5 probability when there's
already 1 goal in the first half.

The model learns the league-level goal distribution by minute bucket,
then conditions on the current minute to compute remaining-minutes lambda.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from math import exp, factorial

import numpy as np
from loguru import logger

from src.models.base import MatchProbabilities


# Minute buckets (0-90). 6 buckets of 15 minutes each.
BUCKET_EDGES = [0, 15, 30, 45, 60, 75, 90]
BUCKET_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]


def _which_bucket(minute: int) -> int:
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= minute < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_EDGES) - 2  # fallback to last bucket


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


class InPlayV1:
    """Wraps in-play prediction with empirical goal-time distribution."""

    name = "inplay_v1"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # Fraction of total goals scored in each bucket (sums to 1.0).
        self.bucket_weights: list[float] = [1.0 / len(BUCKET_LABELS)] * len(BUCKET_LABELS)
        self.fitted_at: datetime | None = None

    def fit(self, training_data=None) -> None:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT minute FROM goal_events WHERE minute IS NOT NULL AND is_own_goal = 0"
        ).fetchall()
        conn.close()
        if not rows:
            logger.warning("inplay_v1: no goal_events to train on, using uniform")
            return

        bucket_counts = [0] * (len(BUCKET_EDGES) - 1)
        for r in rows:
            idx = _which_bucket(int(r[0]))
            bucket_counts[idx] += 1
        total = sum(bucket_counts)
        if total > 0:
            self.bucket_weights = [c / total for c in bucket_counts]
        self.fitted_at = datetime.now()
        logger.info(
            f"inplay_v1: trained on {total} goals; bucket fractions = "
            + ", ".join(f"{b}:{w:.0%}" for b, w in zip(BUCKET_LABELS, self.bucket_weights))
        )

    def remaining_lambda(self, full_lambda: float, current_minute: int,
                        full_match_minutes: int = 90) -> float:
        """Given a team's TOTAL expected goals over a 90-min match, return
        the expected goals over the REMAINING minutes weighted by the
        empirical bucket distribution."""
        if current_minute >= full_match_minutes:
            return 0.0
        remaining_fraction = self._remaining_fraction(current_minute, full_match_minutes)
        return full_lambda * remaining_fraction

    def _remaining_fraction(self, current_minute: int, full_match_minutes: int = 90) -> float:
        """Fraction of total-match goal-rate that remains after current_minute,
        per the empirical bucket distribution."""
        if current_minute >= full_match_minutes:
            return 0.0
        # Sum bucket weights for minutes >= current_minute
        remaining = 0.0
        for i, (lo, hi) in enumerate(zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])):
            if hi <= current_minute:
                continue  # bucket fully past
            if lo >= current_minute:
                remaining += self.bucket_weights[i]
            else:
                # Partial bucket: prorate by minute share within the bucket
                share = (hi - current_minute) / (hi - lo)
                remaining += self.bucket_weights[i] * share
        return remaining

    def condition_on_state(
        self, pre_match: MatchProbabilities, current_home: int, current_away: int,
        minute: int, full_match_minutes: int = 90,
    ) -> MatchProbabilities:
        """Same interface as inplay_v0.condition_on_state but uses the
        learned bucket weights to compute remaining lambda more accurately."""
        if minute >= full_match_minutes:
            from src.models.inplay_v0 import _deterministic_at_final
            return _deterministic_at_final(current_home, current_away, pre_match)

        remaining_lam = self.remaining_lambda(
            pre_match.expected_home_goals, minute, full_match_minutes
        )
        remaining_mu = self.remaining_lambda(
            pre_match.expected_away_goals, minute, full_match_minutes
        )

        cap = 7
        sm_extra = np.zeros((cap + 1, cap + 1))
        for i in range(cap + 1):
            for j in range(cap + 1):
                sm_extra[i, j] = _poisson_pmf(i, remaining_lam) * _poisson_pmf(j, remaining_mu)
        s = sm_extra.sum()
        if s > 0:
            sm_extra = sm_extra / s

        idx_h, idx_a = np.indices(sm_extra.shape)
        final_h = idx_h + current_home
        final_a = idx_a + current_away
        margin = final_h - final_a
        total = final_h + final_a

        p_home_win = float(sm_extra[margin > 0].sum())
        p_draw = float(sm_extra[margin == 0].sum())
        p_away_win = float(sm_extra[margin < 0].sum())
        p_over_1_5 = float(sm_extra[total > 1].sum())
        p_over_2_5 = float(sm_extra[total > 2].sum())
        p_over_3_5 = float(sm_extra[total > 3].sum())

        if current_home > 0 and current_away > 0:
            p_btts_yes = 1.0
        elif current_home > 0:
            p_btts_yes = 1.0 - _poisson_pmf(0, remaining_mu)
        elif current_away > 0:
            p_btts_yes = 1.0 - _poisson_pmf(0, remaining_lam)
        else:
            p_btts_yes = (1.0 - _poisson_pmf(0, remaining_lam)) * (1.0 - _poisson_pmf(0, remaining_mu))

        p_home_minus_1_5 = float(sm_extra[margin >= 2].sum())

        return MatchProbabilities(
            p_home_win=p_home_win,
            p_draw=p_draw,
            p_away_win=p_away_win,
            p_over_2_5=p_over_2_5,
            p_under_2_5=float(1.0 - p_over_2_5),
            p_over_1_5=p_over_1_5,
            p_under_1_5=float(1.0 - p_over_1_5),
            p_over_3_5=p_over_3_5,
            p_under_3_5=float(1.0 - p_over_3_5),
            p_btts_yes=p_btts_yes,
            p_btts_no=float(1.0 - p_btts_yes),
            p_home_minus_1_5=p_home_minus_1_5,
            p_away_plus_1_5=float(1.0 - p_home_minus_1_5),
            expected_home_goals=current_home + remaining_lam,
            expected_away_goals=current_away + remaining_mu,
            features={
                "model": "inplay_v1",
                "minute": minute,
                "current_home": current_home,
                "current_away": current_away,
                "remaining_lambda": remaining_lam,
                "remaining_mu": remaining_mu,
                "remaining_fraction": self._remaining_fraction(minute, full_match_minutes),
            },
        )
