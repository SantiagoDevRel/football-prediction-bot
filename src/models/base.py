"""Common interfaces for all predictive models.

Every model returns the SAME shape: MatchProbabilities. That way the ensemble
layer can stack them without knowing model internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MatchProbabilities:
    """Calibrated probabilities for the standard markets.

    All probabilities are mutually consistent (e.g. p_home_win + p_draw + p_away_win == 1).

    Confidence is optional; only Bayesian-style models populate it. It represents
    the posterior std of the prediction or an analogous uncertainty measure.
    """

    # 1X2
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Over/Under
    p_over_2_5: float
    p_under_2_5: float

    # BTTS
    p_btts_yes: float
    p_btts_no: float

    # Expected goals
    expected_home_goals: float
    expected_away_goals: float

    # Optional uncertainty (Bayesian only)
    confidence: float | None = None

    # Free-form features used (for logging/debug)
    features: dict = field(default_factory=dict)


class Model(ABC):
    """Base class. All concrete models inherit and implement predict_match()."""

    name: str = "base"

    @abstractmethod
    def fit(self, training_data) -> None:
        """Train on historical data."""

    @abstractmethod
    def predict_match(self, home_team_id: int, away_team_id: int, **context) -> MatchProbabilities:
        """Predict a single match. context may include venue, date, lineups, etc."""

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
