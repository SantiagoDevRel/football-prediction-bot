"""Hierarchical Bayesian model (PyMC).

Why Bayesian:
    - Quantified uncertainty (posterior std) → critical for Kelly sizing
    - Hierarchical structure: team strengths drawn from league-level distribution
    - Naturally handles small samples (e.g. promoted teams with few PL games)

Phase 2 implementation. Requires pymc extra: `uv sync --extra bayesian`.
"""
from __future__ import annotations

from src.models.base import MatchProbabilities, Model


class BayesianModel(Model):
    name = "bayesian"

    def __init__(self, n_samples: int = 2000, n_chains: int = 4) -> None:
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.trace = None

    def fit(self, training_data) -> None:
        raise NotImplementedError("Phase 2")

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        """Returns prediction with confidence (posterior uncertainty)."""
        raise NotImplementedError("Phase 2")
