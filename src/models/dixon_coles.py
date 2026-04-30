"""Dixon-Coles model (Poisson with low-score correction + temporal decay).

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".

Key ideas:
    - Each team has attack and defense parameters
    - Goals modeled as bivariate Poisson with correction tau() at low scores
    - Older matches weighted less via exp(-xi * days_ago)

Phase 1 implementation. This is our baseline — every other model has to beat it
in CLV to justify being in the ensemble.
"""
from __future__ import annotations

from src.models.base import MatchProbabilities, Model


class DixonColes(Model):
    name = "dixon_coles"

    def __init__(self, xi: float = 0.0019) -> None:
        # xi = decay rate. ~0.0019 means matches from 1 year ago weigh ~50%.
        self.xi = xi
        self.attack: dict[int, float] = {}
        self.defense: dict[int, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0  # low-score correction

    def fit(self, training_data) -> None:
        raise NotImplementedError("Phase 1")

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        raise NotImplementedError("Phase 1")
