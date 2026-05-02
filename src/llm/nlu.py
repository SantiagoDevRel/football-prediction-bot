"""Natural-language intent parser for the Telegram bot.

User types something like:
    "dame picks de betplay del finde"
    "el top pick de hoy en premier"
    "qué hay en vivo"
    "cómo va mi balance"
    "analiza nacional vs millonarios"
    "registra el 3 con 5000"

We let Claude Haiku 4.5 pick ONE tool from the bot's command set + extract
structured args. That tool name + args are returned as an Intent.

Why tool-use vs free-form JSON: tool-use is enforced by the API (the model is
constrained to emit a valid tool_use block), so we don't have to parse markdown
fences or repair JSON. ~1 second latency, ~$0.001 per query (Haiku 4.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from anthropic import AsyncAnthropic
from loguru import logger


Action = Literal[
    "picks", "live", "analyze", "balance", "history", "place_bet",
    "resolve_auto", "help", "smalltalk",
]

# Leagues recognized by the model. Keep in sync with daily_pipeline.LEAGUES.
SUPPORTED_LEAGUES = [
    "premier_league",
    "liga_betplay",
    "sudamericana",
    "libertadores",
    "champions_league",
]

TimeWindow = Literal["today", "tomorrow", "weekend", "week", "any"]


@dataclass
class Intent:
    action: Action
    leagues: list[str] = field(default_factory=list)   # filter, empty = all
    time_window: TimeWindow = "any"                    # filter
    top_only: bool = False                             # return just the best pick
    home_query: str = ""                               # for /analyze
    away_query: str = ""
    pick_number: int | None = None                     # for /place_bet
    stake: int | None = None                           # for /place_bet
    reasoning: str = ""                                # the model's short explanation


# ---------- Tool definitions ----------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_picks",
        "description": (
            "Lista picks (apuestas con value) para próximos partidos. "
            "Usá esto cuando el usuario pida 'picks', 'apuestas', 'qué jugar', "
            "'recomendaciones', 'top pick', 'mejor apuesta', etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "leagues": {
                    "type": "array",
                    "items": {"type": "string", "enum": SUPPORTED_LEAGUES},
                    "description": (
                        "Filtro por liga(s). Vacío o omitido = todas las ligas. "
                        "Mapeos: 'premier'/'inglaterra' → premier_league; "
                        "'betplay'/'colombia'/'liga colombiana'/'dimayor' → liga_betplay; "
                        "'sudamericana' → sudamericana; "
                        "'libertadores' → libertadores; "
                        "'champions'/'champions league'/'uefa' → champions_league."
                    ),
                },
                "time_window": {
                    "type": "string",
                    "enum": ["today", "tomorrow", "weekend", "week", "any"],
                    "description": (
                        "Filtro temporal. 'hoy' → today; 'mañana' → tomorrow; "
                        "'finde'/'fin de semana'/'sábado y domingo' → weekend; "
                        "'esta semana' → week; sin mención → any."
                    ),
                },
                "top_only": {
                    "type": "boolean",
                    "description": (
                        "True si el usuario pide UN SOLO pick ('el top', 'el mejor', "
                        "'una apuesta', 'dame solo uno', 'el #1'). False si pide varios."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_live",
        "description": (
            "Muestra partidos en vivo con predicciones in-play. Usá cuando "
            "pidan 'en vivo', 'partidos jugándose', 'qué hay ahora', 'live'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_match",
        "description": (
            "Análisis detallado de UN partido específico (todos los mercados). "
            "Usá cuando mencionen dos equipos: 'analiza X vs Y', 'cómo viene X-Y', "
            "'qué dice el modelo de Real Madrid Barcelona'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Equipo local (o el primero mencionado)."},
                "away_team": {"type": "string", "description": "Equipo visitante (o el segundo mencionado)."},
            },
            "required": ["home_team", "away_team"],
        },
    },
    {
        "name": "get_balance",
        "description": (
            "Bankroll, picks abiertas, ROI rolling 30d. Usá cuando pidan "
            "'mi balance', 'cuánto tengo', 'bankroll', 'ganancias', 'cómo voy', 'estado'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_history",
        "description": (
            "Últimas 10 picks resueltas con P&L. Usá cuando pidan "
            "'historial', 'últimas apuestas', 'qué gané', 'qué perdí', 'mis picks pasadas'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "place_bet",
        "description": (
            "Registra una apuesta del pick #N. Usá cuando digan 'apuesto el 3', "
            "'aposté el 5 con 10000', 'va el 2 con la sugerida'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pick_number": {"type": "integer", "description": "Número de pick (1, 2, 3, ...)."},
                "stake": {
                    "type": "integer",
                    "description": "Plata apostada en COP. Omitir si el usuario no especifica (usa la sugerida).",
                },
            },
            "required": ["pick_number"],
        },
    },
    {
        "name": "resolve_pending",
        "description": (
            "Resuelve picks de partidos ya terminados. Usá cuando pidan "
            "'resolver', 'cerrar picks', 'actualizar resultados'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "show_help",
        "description": (
            "Muestra el menú de ayuda con comandos disponibles. Usá cuando "
            "pidan 'ayuda', '/help', 'qué podés hacer', 'menú'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "smalltalk",
        "description": (
            "Charla casual o pregunta sin acción ejecutable: 'hola', 'gracias', "
            "preguntas filosóficas sobre apuestas. Devolvé reasoning con la respuesta corta."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "description": "Respuesta corta y amable en español."},
            },
            "required": ["reply"],
        },
    },
]


SYSTEM_PROMPT = """Eres el parser de intenciones de un bot de Telegram de apuestas deportivas.

Tu único trabajo es elegir UNA herramienta (tool) que ejecute lo que el usuario pidió, extrayendo argumentos estructurados. NO predigas resultados de partidos. NO recomiendes apuestas. NO inventes ligas que no están en el enum.

Contexto del bot (lo que SÍ existe):
- Ligas soportadas: Premier League, Liga BetPlay (Colombia), Copa Sudamericana, Copa Libertadores, Champions League.
- Mercados: 1X2, Over/Under 1.5/2.5/3.5, BTTS, hándicap. (Tarjetas/córners/scorers vienen en la próxima fase.)
- Modo: paper trading.
- El usuario es colombiano. Usa español de Colombia con TUTEO ESTRICTO en tus respuestas: "tú", "puedes", "necesitas", "quieres", "dime", "registra". NUNCA uses voseo argentino: nada de "vos", "podés", "necesitás", "querés", "decime", "registrá", "hacé". Esto aplica a smalltalk y a cualquier texto que devuelvas. Ejemplo correcto: "¿Qué necesitas? Puedes pedirme picks." Ejemplo INCORRECTO: "¿Qué necesitás? Podés pedirme picks."

Reglas:
1. Si el usuario nombra una liga, mapeala al enum exacto. Si dice algo que no existe ("La Liga", "Bundesliga"), igual elegí get_picks con leagues vacío y avisá en reasoning.
2. Si pide "el top pick" / "el mejor" / "uno solo" → top_only=true.
3. "finde" en Colombia = sábado y domingo (time_window="weekend").
4. Si el mensaje no mapea a ninguna acción ejecutable, usá smalltalk con una respuesta corta.
5. NUNCA mezcles dos acciones. Una sola tool por turno."""


class IntentParser:
    def __init__(self, anthropic_api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required for IntentParser")
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model = model

    async def parse(self, text: str) -> Intent:
        """Parse free text → Intent. Falls back to action='help' on failure."""
        text = text.strip()
        if not text:
            return Intent(action="help", reasoning="(mensaje vacío)")

        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}},
                ],
                tools=_TOOLS,
                tool_choice={"type": "any"},  # force a tool selection
                messages=[{"role": "user", "content": text}],
            )
        except Exception as exc:
            logger.warning(f"NLU call failed: {exc}")
            return Intent(action="help", reasoning=f"(error: {exc})")

        # Find the tool_use block
        tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
        if tool_block is None:
            logger.warning(f"NLU returned no tool block: {msg.content}")
            return Intent(action="help", reasoning="(no entendí, mostrá ayuda)")

        return _tool_block_to_intent(tool_block.name, tool_block.input or {})


def _tool_block_to_intent(name: str, args: dict[str, Any]) -> Intent:
    if name == "get_picks":
        leagues = args.get("leagues") or []
        # Defensive: drop any non-recognized values that snuck through
        leagues = [l for l in leagues if l in SUPPORTED_LEAGUES]
        return Intent(
            action="picks",
            leagues=leagues,
            time_window=args.get("time_window") or "any",
            top_only=bool(args.get("top_only", False)),
        )
    if name == "get_live":
        return Intent(action="live")
    if name == "analyze_match":
        return Intent(
            action="analyze",
            home_query=str(args.get("home_team", "")).strip(),
            away_query=str(args.get("away_team", "")).strip(),
        )
    if name == "get_balance":
        return Intent(action="balance")
    if name == "get_history":
        return Intent(action="history")
    if name == "place_bet":
        return Intent(
            action="place_bet",
            pick_number=int(args.get("pick_number")) if args.get("pick_number") is not None else None,
            stake=int(args.get("stake")) if args.get("stake") is not None else None,
        )
    if name == "resolve_pending":
        return Intent(action="resolve_auto")
    if name == "show_help":
        return Intent(action="help")
    if name == "smalltalk":
        return Intent(action="smalltalk", reasoning=str(args.get("reply", "")))
    # Unknown tool name
    return Intent(action="help", reasoning=f"(tool desconocida: {name})")
