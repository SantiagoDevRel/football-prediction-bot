"""LLM-based qualitative feature extraction.

Claude reads contextual info (news, lineups, injuries, recent form chatter) and
returns a structured dict of flags. This is the ONLY place we use the LLM —
never to predict probabilities directly.

Output schema (the LLM must return JSON matching this):
    {
        "flags": [
            "injuries_home:starters_3",
            "rotation_away:cup_priority",
            "altitude_advantage:home",
            "derby:high_intensity",
            "manager_change_recent:home"
        ],
        "summary": "<= 2 sentence English summary",
        "confidence": 0.0 - 1.0    # how much info was actually available
    }

Flags follow the convention: "<category>:<value>". The set of valid flags is
defined in FLAG_VOCABULARY below — anything outside the vocabulary is dropped
to keep the feature space stable.

Phase 2 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

# Controlled vocabulary. Add new flags here, never let the LLM invent free-form ones.
FLAG_VOCABULARY: set[str] = {
    # Injuries / availability
    "injuries_home:starters_1", "injuries_home:starters_2", "injuries_home:starters_3plus",
    "injuries_away:starters_1", "injuries_away:starters_2", "injuries_away:starters_3plus",
    "key_player_out:home_attacker", "key_player_out:home_defender", "key_player_out:home_keeper",
    "key_player_out:away_attacker", "key_player_out:away_defender", "key_player_out:away_keeper",

    # Rotation / motivation
    "rotation_home:cup_priority", "rotation_away:cup_priority",
    "rotation_home:already_qualified", "rotation_away:already_qualified",
    "must_win_home:relegation", "must_win_away:relegation",
    "must_win_home:title_race", "must_win_away:title_race",

    # Context
    "derby:high_intensity",
    "altitude_advantage:home",
    "long_travel:away",
    "short_rest_home:under_72h", "short_rest_away:under_72h",
    "manager_change_recent:home", "manager_change_recent:away",
    "weather_extreme:cold", "weather_extreme:hot", "weather_extreme:rain",
    "off_field_distraction:home", "off_field_distraction:away",
}


@dataclass
class QualitativeFeatures:
    flags: list[str]
    summary: str
    confidence: float
    raw_news: str  # truncated, for audit


class LLMFeatureExtractor:
    def __init__(self, anthropic_api_key: str, model: str = "claude-opus-4-7") -> None:
        self.api_key = anthropic_api_key
        self.model = model

    async def extract(
        self,
        home_team: str,
        away_team: str,
        news_snippets: list[str],
        lineups: dict | None = None,
        injuries: list[dict] | None = None,
    ) -> QualitativeFeatures:
        """Run Claude over the inputs and return structured flags."""
        raise NotImplementedError("Phase 2")
