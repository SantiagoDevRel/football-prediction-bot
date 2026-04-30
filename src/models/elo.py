"""Dynamic Elo rating for football, FiveThirtyEight-style.

Adaptations vs. chess Elo:
    - Goal margin matters: K is multiplied by ln(GD+1) * (2.2 / (rating_diff*0.001 + 2.2))
      (the FTE goal-diff weighting that prevents rating swings on 6-0 thrashings)
    - Home advantage as a constant offset (configurable, default ~65)
    - Per-match online update (after every finished match)

For prediction, we use a logistic on rating diff to get win/loss probabilities,
then split a portion off as draw probability (calibrated empirically).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from loguru import logger

from src.models.base import MatchProbabilities, Model


class Elo(Model):
    name = "elo"

    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 65.0,
        initial_rating: float = 1500.0,
        draw_share: float = 0.26,  # avg draw rate in top European leagues
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.draw_share = draw_share
        self.ratings: dict[int, float] = {}
        self.fitted_at: datetime | None = None
        # Empirical avg goals to fold into OU/BTTS predictions (computed during fit)
        self.avg_goals_per_match: float = 2.7

    # ---------- Internals ----------

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Probability that A beats B given ratings (logistic, base 10 / 400)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _goal_diff_multiplier(self, gd: int, rating_diff: float) -> float:
        """FiveThirtyEight goal-diff multiplier."""
        if gd <= 1:
            return 1.0
        adj_rd = abs(rating_diff)
        return float(np.log(gd + 1) * (2.2 / (adj_rd * 0.001 + 2.2)))

    # ---------- Fitting ----------

    def fit(self, training_data: list[dict]) -> None:
        """Replay all matches in chronological order to build current ratings.

        Each match: home_team_id, away_team_id, home_goals, away_goals, kickoff_date.
        """
        if not training_data:
            raise ValueError("no training data")

        # Sort chronologically (no leakage if we feed full history)
        sorted_matches = sorted(training_data, key=lambda m: m["kickoff_date"])

        self.ratings = {}
        total_goals = 0
        for m in sorted_matches:
            self.update_after_match(
                m["home_team_id"], m["away_team_id"],
                int(m["home_goals"]), int(m["away_goals"]),
            )
            total_goals += int(m["home_goals"]) + int(m["away_goals"])

        self.avg_goals_per_match = total_goals / max(len(sorted_matches), 1)
        self.fitted_at = datetime.now(tz=timezone.utc)
        logger.info(
            f"Elo fit on {len(sorted_matches)} matches, {len(self.ratings)} teams. "
            f"avg goals/match={self.avg_goals_per_match:.2f}"
        )

    def update_after_match(
        self, home_team_id: int, away_team_id: int, home_goals: int, away_goals: int
    ) -> None:
        rh = self.ratings.get(home_team_id, self.initial_rating)
        ra = self.ratings.get(away_team_id, self.initial_rating)

        rating_diff_with_home_adv = (rh + self.home_advantage) - ra
        expected_home = 1.0 / (1.0 + 10 ** (-rating_diff_with_home_adv / 400.0))

        if home_goals > away_goals:
            actual_home, gd = 1.0, home_goals - away_goals
        elif home_goals < away_goals:
            actual_home, gd = 0.0, away_goals - home_goals
        else:
            actual_home, gd = 0.5, 0

        mult = self._goal_diff_multiplier(gd, rating_diff_with_home_adv)
        delta = self.k_factor * mult * (actual_home - expected_home)
        self.ratings[home_team_id] = rh + delta
        self.ratings[away_team_id] = ra - delta

    # ---------- Prediction ----------

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        rh = self.ratings.get(home_team_id, self.initial_rating)
        ra = self.ratings.get(away_team_id, self.initial_rating)
        rating_diff = (rh + self.home_advantage) - ra

        # Win probability from rating diff
        p_home_raw = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
        p_away_raw = 1.0 - p_home_raw

        # Reserve a draw share, scaled by how close the teams are
        # When the match is close (rating_diff~0), draw is most likely; when lopsided, draw shrinks
        closeness = float(np.exp(-(rating_diff ** 2) / (2 * 200 ** 2)))
        p_draw = self.draw_share * closeness + 0.10 * (1 - closeness)

        # Re-normalize so totals sum to 1
        remaining = 1.0 - p_draw
        p_home = p_home_raw * remaining
        p_away = p_away_raw * remaining

        # Estimate expected goals: use league avg, skewed by rating diff via a simple ratio
        # log goal ratio ~ rating_diff / 600 (empirical heuristic)
        goal_ratio = float(np.exp(rating_diff / 600.0))
        total = self.avg_goals_per_match
        # split total into home/away by goal_ratio
        away_g = total / (1.0 + goal_ratio)
        home_g = total - away_g

        # Approximate OU/BTTS via Poisson on those rates
        from math import factorial as _f
        from math import exp as _e

        def _pmf(k: int, lam: float) -> float:
            if lam <= 0:
                return 1.0 if k == 0 else 0.0
            return _e(-lam) * (lam ** k) / _f(k)

        # Build a small score matrix
        cap = 7
        sm = np.zeros((cap + 1, cap + 1))
        for i in range(cap + 1):
            for j in range(cap + 1):
                sm[i, j] = _pmf(i, home_g) * _pmf(j, away_g)
        sm = sm / sm.sum()

        idx_i, idx_j = np.indices(sm.shape)
        tot = idx_i + idx_j
        p_over_1_5 = float(sm[tot > 1].sum())
        p_over_2_5 = float(sm[tot > 2].sum())
        p_over_3_5 = float(sm[tot > 3].sum())
        p_btts_yes = float(sm[1:, 1:].sum())
        p_btts_no = float(1.0 - p_btts_yes)

        margin = idx_i - idx_j
        p_home_minus_1_5 = float(sm[margin >= 2].sum())

        return MatchProbabilities(
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            p_over_2_5=p_over_2_5,
            p_under_2_5=float(1.0 - p_over_2_5),
            p_over_1_5=p_over_1_5,
            p_under_1_5=float(1.0 - p_over_1_5),
            p_over_3_5=p_over_3_5,
            p_under_3_5=float(1.0 - p_over_3_5),
            p_btts_yes=p_btts_yes,
            p_btts_no=p_btts_no,
            p_home_minus_1_5=p_home_minus_1_5,
            p_away_plus_1_5=float(1.0 - p_home_minus_1_5),
            expected_home_goals=home_g,
            expected_away_goals=away_g,
            features={"model": "elo", "rh": rh, "ra": ra, "diff": rating_diff},
        )
