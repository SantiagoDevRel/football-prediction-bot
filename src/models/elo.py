"""Dynamic Elo rating for football.

Reference: FiveThirtyEight's club soccer Elo, adapted.

Key adaptations vs. chess Elo:
    - Goal margin matters (goal difference factor)
    - Home advantage as constant offset (~65 Elo points)
    - Different K-factor for league vs cup vs international

Phase 1 implementation.
"""
from __future__ import annotations

from src.models.base import MatchProbabilities, Model


class Elo(Model):
    name = "elo"

    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 65.0,
        initial_rating: float = 1500.0,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: dict[int, float] = {}

    def fit(self, training_data) -> None:
        """Run Elo through historical matches in chronological order."""
        raise NotImplementedError("Phase 1")

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        raise NotImplementedError("Phase 1")

    def update_after_match(
        self, home_team_id: int, away_team_id: int, home_goals: int, away_goals: int
    ) -> None:
        """Online update. Call this after each finished match."""
        raise NotImplementedError("Phase 1")
