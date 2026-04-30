"""LLM-based qualitative feature extraction.

Claude reads contextual info (news, lineups, injuries, recent form chatter) and
returns a structured dict of flags. This is the ONLY place we use the LLM —
never to predict probabilities directly.

Output schema (Claude returns JSON matching this):
    {
        "flags": [
            "injuries_home:starters_3plus",
            "rotation_away:cup_priority",
            "altitude_advantage:home"
        ],
        "summary": "<= 2 sentence English summary",
        "confidence": 0.0 - 1.0
    }

Flags follow "<category>:<value>". Anything outside FLAG_VOCABULARY is dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic
from loguru import logger


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

# Mirror Anthropic SDK's expected model id format.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheap+fast for feature extraction


@dataclass
class QualitativeFeatures:
    flags: list[str]
    summary: str
    confidence: float
    raw_input: str = ""
    model_used: str = ""
    cached_input_tokens: int = 0
    new_input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """You are a football match context analyzer. You receive raw news, lineups, and injury reports about an upcoming match, and you output ONLY structured flags from a controlled vocabulary.

Your output MUST be valid JSON with this exact shape:
{
  "flags": ["flag1:value1", "flag2:value2", ...],
  "summary": "<= 2 sentence English summary of relevant context",
  "confidence": 0.0 to 1.0  (how much real info you had to work with)
}

Rules:
- Use ONLY flags from the controlled vocabulary provided. Never invent new flags.
- If unsure, omit the flag rather than guessing.
- Set confidence low when the input is sparse, high when it clearly documents the context.
- Do NOT predict the match outcome. Do NOT mention probabilities. Do NOT say who will win.
- Output JSON only - no preamble, no markdown fences, no explanation outside the JSON.
"""


def _build_user_prompt(
    home_team: str,
    away_team: str,
    league: str,
    news_snippets: list[str],
    lineups: dict | None,
    injuries: list[dict] | None,
) -> str:
    parts = [
        f"Match: {home_team} (home) vs {away_team} (away)",
        f"League: {league}",
        "",
        "Controlled vocabulary (use ONLY these flag values):",
        *sorted(FLAG_VOCABULARY),
        "",
    ]
    if news_snippets:
        parts.append("News snippets:")
        for s in news_snippets:
            parts.append(f"- {s}")
        parts.append("")
    if lineups:
        parts.append("Lineups:")
        parts.append(json.dumps(lineups, indent=2)[:2000])
        parts.append("")
    if injuries:
        parts.append("Injuries / suspensions:")
        for inj in injuries:
            parts.append(f"- {json.dumps(inj)[:300]}")
        parts.append("")
    parts.append("Output the JSON now.")
    return "\n".join(parts)


class LLMFeatureExtractor:
    def __init__(
        self,
        anthropic_api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 800,
    ) -> None:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required")
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def extract(
        self,
        home_team: str,
        away_team: str,
        league: str,
        news_snippets: list[str] | None = None,
        lineups: dict | None = None,
        injuries: list[dict] | None = None,
    ) -> QualitativeFeatures:
        user_prompt = _build_user_prompt(
            home_team, away_team, league,
            news_snippets or [], lineups, injuries,
        )

        # Use prompt caching: system + vocabulary list cached for 5 min.
        # The user content (per match) is not cached.
        msg = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        # Parse JSON response. Claude sometimes wraps in markdown fences despite the rule.
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            # remove leading "json\n" if present
            if text.startswith("json"):
                text = text[4:].lstrip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"LLM JSON parse failed: {exc}; raw={text[:200]}")
            return QualitativeFeatures(
                flags=[], summary="", confidence=0.0,
                raw_input=user_prompt[:500], model_used=self.model,
            )

        # Filter flags against vocabulary. Drop unknowns silently.
        raw_flags = parsed.get("flags", [])
        clean_flags = [f for f in raw_flags if f in FLAG_VOCABULARY]
        if len(clean_flags) != len(raw_flags):
            logger.debug(f"dropped {len(raw_flags) - len(clean_flags)} unknown flag(s)")

        usage = msg.usage
        return QualitativeFeatures(
            flags=clean_flags,
            summary=str(parsed.get("summary", ""))[:400],
            confidence=float(parsed.get("confidence", 0.5)),
            raw_input=user_prompt[:500],
            model_used=self.model,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            new_input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


# Smoke test
async def _smoke() -> None:
    from src.config import settings
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set - skipping LLM smoke test")
        return
    extractor = LLMFeatureExtractor(settings.anthropic_api_key)
    feats = await extractor.extract(
        home_team="Manchester City",
        away_team="Real Madrid",
        league="UEFA Champions League",
        news_snippets=[
            "Manchester City are missing star midfielder Rodri, ruled out with a 6-week injury",
            "Manchester City already qualified for the next round, manager hints at heavy rotation",
            "Real Madrid travel directly from a 5-3 win at Barcelona last weekend, fully fit squad",
            "Rain forecast for Manchester, possible heavy showers during kickoff",
        ],
    )
    print(f"flags: {feats.flags}")
    print(f"summary: {feats.summary}")
    print(f"confidence: {feats.confidence:.2f}")
    print(f"tokens: cached_in={feats.cached_input_tokens} new_in={feats.new_input_tokens} out={feats.output_tokens}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke())
