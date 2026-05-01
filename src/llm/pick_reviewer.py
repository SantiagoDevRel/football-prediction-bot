"""Claude reviews each value-bet candidate before it's shown to the user.

The numerical model finds picks where edge > 5% — but mathematical edge
isn't the same as "good bet". Claude looks at the pick contextually and
returns one of:

    take    — agrees with the model, no concerns
    reduce  — take the bet but with smaller stake (e.g. 50% of recommended)
    skip    — model is probably wrong; don't bet

Plus 1-2 sentences explaining WHY. This is shown to the user in Telegram.

Cost: ~$0.005 per pick (Haiku 4.5). For 8-15 picks per day = $0.05/day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from anthropic import AsyncAnthropic
from loguru import logger


Verdict = Literal["take", "reduce", "skip"]


@dataclass
class PickReview:
    verdict: Verdict
    reasoning: str
    confidence: float  # how confident Claude is in its verdict


SYSTEM_PROMPT = """You are an experienced football betting analyst reviewing model-generated value bets.

For each pick, the user gives you:
- The match (home vs away, league)
- What the bet is (e.g. "Over 2.5 goals", "Home win", "BTTS No")
- The model's predicted probability
- The bookmaker's implied probability
- The mathematical edge (positive = value)
- Recent context if available

Your job is NOT to predict the outcome. Your job is to spot when the model is probably wrong contextually. Common issues:

1. **Sample size warnings**: small leagues, promoted teams, recent transfers, new managers — the model may be overconfident.
2. **BTTS No on heavy scorers**: if a team scores in 18/20 matches, BTTS No at any "edge" is suspicious.
3. **Underdog covers**: large +EV on a clear underdog often means the model under-weighted form/quality.
4. **Derby/altitude/weather**: the model can miss obvious context.
5. **Half-line ranges**: O/U 3.5 over has very low base rate; small edge isn't worth the variance.

Return JSON exactly like:
{
  "verdict": "take" | "reduce" | "skip",
  "reasoning": "<= 200 chars, plain Spanish, casual tone",
  "confidence": 0.0 to 1.0
}

Don't predict winners. Don't say "Atlético Nacional siempre gana" — that's prediction. DO say "BTTS No is risky here because..." which is META analysis of the pick.

Use "take" when nothing looks wrong with the pick.
Use "reduce" when the math is +EV but the variance is uncomfortable (high cuota, small sample, etc.).
Use "skip" when the pick looks like model error (not real value).

Output JSON only — no preamble."""


def _build_pick_prompt(pick: dict) -> str:
    """Build the user-message part for one pick."""
    casa_implied = 1.0 / pick["odds"]
    market_human = {
        "1x2": "1X2", "ou_1.5": "Más/Menos 1.5 goles",
        "ou_2.5": "Más/Menos 2.5 goles", "ou_3.5": "Más/Menos 3.5 goles",
        "btts": "BTTS (ambos marcan)", "ah_-1.5": "Hándicap -1.5",
    }.get(pick["market"], pick["market"])
    sel_human = {
        "home": "Local gana", "draw": "Empate", "away": "Visitante gana",
        "over": "Más de", "under": "Menos de",
        "yes": "Sí", "no": "No",
    }.get(pick["selection"], pick["selection"])
    return (
        f"Match: {pick['home_team']} vs {pick['away_team']} ({pick['league']})\n"
        f"Mercado: {market_human}\n"
        f"Selección: {sel_human}\n"
        f"Cuota casa: {pick['odds']:.2f} (implícita {casa_implied:.0%})\n"
        f"Modelo: {pick['model_probability']:.0%}\n"
        f"Edge matemático: +{pick['edge']*100:.0f}%\n"
        f"Stake recomendado (¼ Kelly): ${pick['recommended_stake']:,.0f}\n\n"
        f"Tu veredicto en JSON."
    )


class PickReviewer:
    def __init__(self, anthropic_api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required")
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model = model

    async def review(self, pick: dict) -> PickReview:
        user_msg = _build_pick_prompt(pick)
        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}},
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:].lstrip()
            parsed = json.loads(text)
            verdict = parsed.get("verdict", "take")
            if verdict not in ("take", "reduce", "skip"):
                verdict = "take"
            return PickReview(
                verdict=verdict,
                reasoning=str(parsed.get("reasoning", ""))[:300],
                confidence=float(parsed.get("confidence", 0.7)),
            )
        except Exception as exc:
            logger.warning(f"pick review failed: {exc}")
            return PickReview(verdict="take", reasoning="(review failed, defaulting to take)", confidence=0.5)
