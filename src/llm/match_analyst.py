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
    correlation_note: str = ""    # warning if multiple picks are correlated


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
- HASTA 3 picks numéricos ordenados por edge (los más recomendados)
- El mensaje original del usuario (puede tener contexto adicional como "el que gane clasifica", "X está lesionado", "es derby")

Tu output:
1. UN veredicto general: "TAKE" (las picks valen), "PASS" (saltá todo), o "CAUTION" (mejor sólo la #1).
2. Razonamiento de 2-3 oraciones MAX. Incorpora el contexto del usuario si es relevante. Sé directo, sin rodeos. No repitas las cifras (ya las muestra el bot arriba).
3. Si hay varias picks correlacionadas (ej. "Local gana" + "Más 1.5 goles" — si el local gana suele ser con goles), avisá en correlation_note. Si el usuario las apuesta todas, no diversifica de verdad.

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
            "correlation_note": {
                "type": "string",
                "description": "Si las picks ofrecidas están correlacionadas entre sí (ej. 1X2 home + Más 1.5 + BTTS sí — todas dependen de que el local marque), una frase corta avisando. Vacío si son independientes o si solo hay una pick.",
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
    top_picks: list[tuple[str, str, float, float, float]] | None = None,
    user_message: str = "",
    home_form: object = None,         # RecentForm | None
    away_form: object = None,
    h2h: object = None,                # HeadToHead | None
    market_consensus: list[dict] | None = None,   # multi-bookmaker rows
    lineups: list = None,              # list[TeamLineup] from api-football
    injuries: list = None,             # list[Injury] from api-football
    pinnacle_consensus: list = None,   # list[OddsConsensus] from api-football
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
    if top_picks:
        lines.append("")
        lines.append(f"Top {len(top_picks)} picks numéricos (ordenadas por edge):")
        for i, (market, sel, prob, odds, e) in enumerate(top_picks, 1):
            lines.append(
                f"  {i}. {_humanize_market(market, sel)} @ {odds:.2f} "
                f"(modelo {prob:.0%}, edge +{e*100:.0f}%)"
            )
    elif best_pick:
        market, sel, prob, odds, e = best_pick
        lines.append("")
        lines.append(
            f"Mejor pick numérico: {_humanize_market(market, sel)} @ {odds:.2f} "
            f"(modelo {prob:.0%}, edge +{e*100:.0f}%)"
        )
    else:
        lines.append("")
        lines.append("Mejor pick numérico: ninguno con edge >= 5%.")
    # Recent form for both teams
    if home_form is not None:
        lines.append("")
        lines.append(f"Forma reciente {home} (últimos {home_form.n_matches}):")
        lines.append(
            f"  {home_form.streak}  · {home_form.wins}W {home_form.draws}D {home_form.losses}L "
            f"· GF {home_form.goals_for} GA {home_form.goals_against}"
        )
        for s in home_form.recent_summary[:3]:
            lines.append(f"  · {s}")
    if away_form is not None:
        lines.append("")
        lines.append(f"Forma reciente {away} (últimos {away_form.n_matches}):")
        lines.append(
            f"  {away_form.streak}  · {away_form.wins}W {away_form.draws}D {away_form.losses}L "
            f"· GF {away_form.goals_for} GA {away_form.goals_against}"
        )
        for s in away_form.recent_summary[:3]:
            lines.append(f"  · {s}")

    # Head to head
    if h2h is not None and h2h.n_matches > 0:
        lines.append("")
        lines.append(f"H2H últimos {h2h.n_matches}: {home} {h2h.home_wins}-{h2h.draws}-{h2h.away_wins} {away}")
        for r in h2h.last_results[:3]:
            lines.append(f"  · {r}")

    # Lineups (api-football) — starting XI confirmed
    if lineups:
        lines.append("")
        lines.append("Alineaciones confirmadas:")
        for L in lineups:
            xi_names = [p.name for p in (L.start_xi or [])]
            lines.append(f"  {L.team_name} ({L.formation}) DT {L.coach_name}:")
            if xi_names:
                lines.append(f"    XI: {', '.join(xi_names)}")
        lines.append("(Si en la 'Forma reciente' aparece un goleador clave que NO está en el XI, decílo claramente — afecta predicción.)")

    # Injuries / suspensions (api-football)
    if injuries:
        lines.append("")
        lines.append(f"Lesiones / suspensiones ({len(injuries)} reportadas):")
        for inj in injuries[:8]:
            lines.append(f"  · {inj.player_name} ({inj.team_name}) — {inj.type}: {inj.reason}")

    # Pinnacle consensus (api-football) — gold-standard reference book
    if pinnacle_consensus:
        lines.append("")
        lines.append("Cuotas multi-bookmaker (api-football, ~12 casas, incluye Pinnacle):")
        for o in pinnacle_consensus[:8]:
            mkt_h = _humanize_market(o.market, o.selection)
            pin = f"Pinnacle {o.pinnacle_odds:.2f}" if o.pinnacle_odds else "(sin Pinnacle)"
            line = f"  {mkt_h}: mediana {o.median_odds:.2f}, mejor {o.best_odds:.2f} ({o.best_bookie}), {pin}, {o.n_books} casas"
            # Wplay comparison if we have it
            wplay_v = (casa_odds or {}).get((o.market, o.selection))
            if wplay_v is not None:
                # Compare to Pinnacle (preferred reference) or median
                ref = o.pinnacle_odds or o.median_odds
                gap = (wplay_v - ref) / ref
                if gap < -0.05:
                    line += f" · ⚠️ Wplay {wplay_v:.2f} ({gap*100:+.0f}% vs ref, paga MENOS)"
                elif gap > 0.05:
                    line += f" · 🔥 Wplay {wplay_v:.2f} ({gap*100:+.0f}% vs ref, paga MÁS — outlier)"
                else:
                    line += f" · Wplay {wplay_v:.2f} (en línea con mercado)"
            lines.append(line)

    # Multi-bookmaker consensus from the-odds-api (additional source) — flag where Wplay differs
    if market_consensus:
        lines.append("")
        lines.append("Consenso de mercado (otros bookmakers):")
        for row in market_consensus[:6]:  # cap to keep prompt small
            mkt = row.get("market", "?")
            sel = row.get("selection", "?")
            best = row.get("best", 0)
            median = row.get("median", 0)
            worst = row.get("worst", 0)
            n = row.get("n_books", 0)
            wplay = row.get("wplay", None)
            line = (f"  {_humanize_market(mkt, sel)}: "
                    f"mediana {median:.2f} (rango {worst:.2f}-{best:.2f}, {n} casas)")
            if wplay is not None:
                # Highlight when Wplay is way off the market
                if wplay < median * 0.92:
                    line += f" · ⚠️ Wplay {wplay:.2f} (paga MENOS que el mercado)"
                elif wplay > median * 1.08:
                    line += f" · 🔥 Wplay {wplay:.2f} (paga MÁS que el mercado — outlier)"
                else:
                    line += f" · Wplay {wplay:.2f}"
            lines.append(line)

    if user_message:
        lines.append("")
        lines.append(f"Mensaje original del usuario: {user_message!r}")
        lines.append("(Si menciona contexto adicional — playoff, lesión, derby, clima — incorpóralo en tu reasoning.)")

    lines.append("")
    lines.append(
        "Tu trabajo: dar el veredicto incorporando TODO el contexto (forma, H2H, "
        "consenso de mercado, y mensaje del usuario), no solo las cifras del modelo."
    )
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
            correlation_note=str(a.get("correlation_note", ""))[:200],
        )
