"""Gradient boosting model on engineered features.

Features (initial set, expand as we learn):
    - Rolling averages (xG, xGA, goals scored/conceded) over 5/10/season
    - Form (W/D/L last 5)
    - Days of rest since last match
    - Travel distance (when available)
    - Home/away splits
    - Head-to-head historical
    - LLM qualitative flags (one-hot or embedding)
    - Lineup strength proxy (sum of player ratings if available)

Phase 2 implementation.
"""
from __future__ import annotations

from src.models.base import MatchProbabilities, Model


class XGBoostModel(Model):
    name = "xgboost"

    def __init__(self) -> None:
        self.model_1x2 = None
        self.model_ou_2_5 = None
        self.model_btts = None

    def fit(self, training_data) -> None:
        raise NotImplementedError("Phase 2")

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        raise NotImplementedError("Phase 2")
