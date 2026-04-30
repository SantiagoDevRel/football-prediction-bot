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

    All probabilities are mutually consistent within their market group:
        - 1X2: home + draw + away == 1
        - O/U: over_X + under_X == 1
        - BTTS: yes + no == 1
        - Handicap: home_minus_X + push_X + away_plus_X == 1
                    (push = match settles by exactly the line; bookies refund)

    `confidence` is optional; only Bayesian-style models populate it.
    """

    # 1X2
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Over / Under (multiple lines)
    p_over_2_5: float
    p_under_2_5: float
    p_over_1_5: float = 0.0
    p_under_1_5: float = 0.0
    p_over_3_5: float = 0.0
    p_under_3_5: float = 0.0

    # BTTS
    p_btts_yes: float = 0.0
    p_btts_no: float = 0.0

    # European handicap (no push). "Home wins by 2+ goals" / "Home doesn't lose by 2+".
    # Useful for clear favorites where the 1X2 cuota is too short.
    p_home_minus_1_5: float = 0.0   # home wins by >= 2
    p_away_plus_1_5: float = 0.0    # home wins by 0 or 1, OR draws, OR loses
    # Note: p_home_minus_1_5 + p_away_plus_1_5 == 1 (no push since 1.5 is half)

    # Expected goals
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0

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
