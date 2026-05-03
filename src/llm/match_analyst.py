"""Claude-powered match analyst for /analizar.

Receives the model's numerical predictions + Wplay odds + the user's
original Spanish message, and returns a 2–3 sentence verdict in Colombian
Spanish. The user's message often carries qualitative context the model
doesn't have (playoff implications, injuries, manager change, derby), so
we pass it verbatim to Claude.

Cost: ~$0.001-$0.002 per call (Haiku 4.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic
from loguru import logger


@dataclass
class MatchVerdict:
    verdict: str          # "TAKE" | "PASS" | "CAUTION"
    reasoning: str        # 2-3 sentences in Spanish
    suggested_market: str = ""    # e.g. "Santa Fe @ 1.94" if verdict=TAKE
    overrides_model: bool = False  # true if Claude flags the math is misleading


SYSTEM_PROMPT = """Eres un analista de apuestas deportivas. El bot ya hizo el cálculo numérico (modelo + cuotas) y te pasa los resultados. Tu trabajo es dar un veredicto SHORT y RAZONADO en español colombiano (tuteo ESTRICTO).

DIALECTO — uso obligatorio del tuteo, está prohibido el voseo:
✅ Correcto: tú, puedes, apuesta, mira, ten en cuenta, deberías
❌ Prohibido: vos, podés, apostá, mirá, tené en cuenta, deberías
✅ "Apuesta al local" / "Mira la cuota" / "No tomes el over"
❌ "Apostá al local" / "Mirá la cuota" / "No tomés el over"

Información que recibes:
- Partido (equipos, liga, hora)
- Probabilidades del modelo (1X2, O/U 2.5, BTTS)
- Cuotas Wplay
- Mejor pick numérico encontrado (con edge) si lo hay
- El mensaje original del usuario (puede tener contexto adicional como "el que gane clasifica", "X está lesionado", "es derby")

Tu output:
1. UN veredicto: "TAKE" (apostá), "PASS" (no apostés), o "CAUTION" (apostá pero stake reducido)
2. Razonamiento de 2-3 oraciones MAX. Incorpora el contexto del usuario si es relevante. Sé directo, sin rodeos. No repitas las cifras (ya las muestra el bot arriba).

Ejemplos:
- "TAKE — Santa Fe es claro favorito en El Campín. El edge de 21% es real porque Wplay subestima la diferencia con Inter Bogotá, que ha venido jugando con suplentes."
- "PASS — Wplay ya pagó ajustada la cuota del local. Si la motivación playoff es real, NO empuja a más goles: equipos en presión juegan cautos. No hay edge."
- "CAUTION — el edge matemático es alto pero el modelo no sabe que Hugo Rodallega está en duda. Si juega, take; si no, baja stake a la mitad."

Reglas:
- NO inventes contexto que no esté en el mensaje del usuario o en los datos.
- Si el usuario menciona algo (lesión, derby, motivación), evalúalo críticamente — no asumas que cambia el resultado.
- Output en JSON exacto vía el tool, sin texto extra."""


_TOOL = {
    "name": "match_verdict",
    "description": "Veredicto razonado del partido en 2-3 oraciones.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["TAKE", "PASS", "CAUTION"],
                "description": "TAKE si recomendas la apuesta, PASS si no hay value real, CAUTION si tomar con stake reducido.",
            },
            "reasoning": {
                "type": "string",
                "description": "2-3 oraciones max en español colombiano (tuteo). Incorpora contexto del usuario si es relevante.",
            },
            "suggested_market": {
                "type": "string",
                "description": "Si verdict=TAKE o CAUTION: 'mercado @ cuota' (ej. 'Santa Fe @ 1.94'). Vacío si PASS.",
            },
            "overrides_model": {
                "type": "boolean",
                "description": "True si tu razonamiento contradice el cálculo matemático del bot por contexto cualitativo (ej: lesión clave que el modelo no ve).",
            },
        },
        "required": ["verdict", "reasoning"],
    },
}


def _humanize_market(market: str, selection: str) -> str:
    if market == "1x2":
        return {"home": "Local", "draw": "Empate", "away": "Visitante"}.get(selection, selection)
    if market.startswith("ou_"):
        line = market.split("_")[1]
        return f"{'Más' if selection == 'over' else 'Menos'} de {line}"
    if market == "btts":
        return "BTTS Sí" if selection == "yes" else "BTTS No"
    return f"{market}:{selection}"


def build_context_block(
    *,
    home: str, away: str, league: str, kickoff: str,
    p_home: float, p_draw: float, p_away: float,
    p_over_2_5: float, p_btts_yes: float,
    casa_odds: dict[tuple[str, str], float] | None = None,
    best_pick: tuple[str, str, float, float, float] | None = None,
    user_message: str = "",
) -> str:
    """Format all the context into a single user message for Claude."""
    lines = [
        f"Partido: {home} vs {away}",
        f"Liga: {league}",
        f"Kickoff: {kickoff}",
        "",
        "Modelo (ensemble):",
        f"  1X2: {p_home:.0%} / {p_draw:.0%} / {p_away:.0%}",
        f"  O 2.5 goles: {p_over_2_5:.0%}",
        f"  BTTS sí: {p_btts_yes:.0%}",
    ]
    if casa_odds:
        lines.append("")
        lines.append("Cuotas Wplay:")
        if ("1x2", "home") in casa_odds:
            lines.append(
                f"  H {casa_odds.get(('1x2','home'),'?')} / "
                f"X {casa_odds.get(('1x2','draw'),'?')} / "
                f"A {casa_odds.get(('1x2','away'),'?')}"
            )
        if ("ou_2.5", "over") in casa_odds:
            lines.append(
                f"  O 2.5 {casa_odds[('ou_2.5','over')]} / U 2.5 {casa_odds[('ou_2.5','under')]}"
            )
        if ("btts", "yes") in casa_odds:
            lines.append(
                f"  BTTS sí {casa_odds[('btts','yes')]} / no {casa_odds[('btts','no')]}"
            )
    if best_pick:
        market, sel, prob, odds, e = best_pick
        lines.append("")
        lines.append(
            f"Mejor pick numérico: {_humanize_market(market, sel)} @ {odds:.2f} "
            f"(modelo {prob:.0%}, edge +{e*100:.0f}%)"
        )
    else:
        lines.append("")
        lines.append("Mejor pick numérico: ninguno con edge >= 5%.")
    if user_message:
        lines.append("")
        lines.append(f"Mensaje original del usuario: {user_message!r}")
        lines.append("(Si menciona contexto adicional — playoff, lesión, derby, clima — incorpóralo en tu reasoning.)")
    return "\n".join(lines)


class MatchAnalyst:
    def __init__(self, anthropic_api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required")
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model = model

    async def analyze(self, context_block: str) -> MatchVerdict:
        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": "match_verdict"},
                messages=[{"role": "user", "content": context_block}],
            )
        except Exception as exc:
            logger.warning(f"match analyst call failed: {exc}")
            return MatchVerdict(verdict="PASS", reasoning="(error consultando análisis cualitativo)")

        block = next((b for b in msg.content if b.type == "tool_use"), None)
        if block is None:
            return MatchVerdict(verdict="PASS", reasoning="(sin respuesta del análisis cualitativo)")
        a = block.input or {}
        return MatchVerdict(
            verdict=str(a.get("verdict", "PASS")),
            reasoning=str(a.get("reasoning", ""))[:500],
            suggested_market=str(a.get("suggested_market", "")),
            overrides_model=bool(a.get("overrides_model", False)),
        )
