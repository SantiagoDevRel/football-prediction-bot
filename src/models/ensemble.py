"""Stacking ensemble: meta-model learns to weight base model predictions.

Strategy:
    1. Each base model produces MatchProbabilities for the match
    2. Concatenate their outputs as features
    3. Add context features (league, market type, days_to_kickoff)
    4. Meta-model (logistic regression or shallow XGBoost) outputs final calibrated probability

Calibration is the final step: even after stacking, raw outputs may need Platt
scaling or isotonic regression to match observed frequencies.

Phase 2 implementation.
"""
from __future__ import annotations

from src.models.base import MatchProbabilities, Model


class EnsembleModel(Model):
    name = "ensemble"

    def __init__(self, base_models: list[Model]) -> None:
        self.base_models = base_models
        self.meta_model_1x2 = None
        self.meta_model_ou = None
        self.meta_model_btts = None
        self.calibrators: dict[str, object] = {}

    def fit(self, training_data) -> None:
        """Trains base models, then trains meta-model on their out-of-fold predictions."""
        raise NotImplementedError("Phase 2")

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        raise NotImplementedError("Phase 2")
