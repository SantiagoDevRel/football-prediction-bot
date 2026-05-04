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
    "register_externals", "resolve_auto", "set_bankroll", "open_positions",
    "delete_pick", "help", "smalltalk",
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
    raw_text: str = ""                                 # for /register_externals (full message)
    mode_hint: str = ""                                # 'paper'|'real'|'' inferred
    bankroll_amount: int | None = None                 # for set_bankroll
    reasoning: str = ""                                # the model's short explanation


# ---------- Tool definitions ----------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_picks",
        "description": (
            "Lista picks (apuestas con value) para próximos partidos. "
            "Usá esto cuando el usuario pida 'picks', 'apuestas', 'qué jugar', "
            "'recomendaciones', 'top pick', 'mejor apuesta', SIEMPRE QUE NO "
            "haya nombrado dos equipos específicos. Si el mensaje incluye "
            "dos clubes en estructura 'X vs Y', usá analyze_match aunque "
            "el usuario diga la palabra 'picks'."
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
            "Análisis detallado de UN partido específico (todos los mercados + "
            "cuotas + value bets para ESE partido). Usá cuando el mensaje "
            "mencione dos equipos específicos Y NO contenga señales de paste "
            "de boleta (ver register_external_bets). Disparadores típicos: "
            "'analiza X vs Y', 'cómo viene X-Y', 'info del Bayern PSG', "
            "'dame picks/tips/info/recomendaciones del partido X vs Y', "
            "'qué dice el modelo de Real Madrid Barcelona', 'el partido X y Y'. "
            "Esta tool ES la respuesta correcta cuando el usuario quiere "
            "información focalizada en un solo encuentro, aunque diga 'picks'. "
            "PERO si el mensaje incluye monto + cuota + verbo de registro "
            "('aposté', 'guarda', 'registra'), elegí register_external_bets en su lugar."
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
        "name": "register_external_bets",
        "description": (
            "Registra apuestas que el usuario YA hizo en Wplay (u otra casa) y "
            "está reportando para que el bot las trackee. Esta herramienta GANA "
            "sobre analyze_match cuando hay señales claras de paste de boleta. "
            "Usá esta herramienta cuando el mensaje incluya cualquiera de: "
            "(a) verbo de registro en pasado/imperativo: 'aposté', 'jugué', "
            "'guarda', 'registra', 'anota', 'tomá nota', 'metele', "
            "'guarda en mis apuestas'; (b) un monto con separador o símbolo: "
            "'$60,000', '60.000', '60000 COP', '60k'; (c) una cuota decimal: "
            "'1.85', '4.10'; (d) estructura de boleta Wplay: 'Tiros de Esquina', "
            "'Total Goles', 'Resultado Tiempo Completo', 'Más de (X)', "
            "'Menos de (X)', 'Ganancia: $...', 'Monto de Apuesta'. "
            "BASTAN DOS de estas señales — aunque el mensaje también nombre dos "
            "equipos, NO uses analyze_match. Es DISTINTO de place_bet: place_bet "
            "registra UN pick previamente sugerido por el bot (numerado #1..N); "
            "register_external_bets registra apuestas externas que el usuario "
            "hizo por su cuenta, pueden ser muchas a la vez."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["real", "paper"],
                    "description": (
                        "'real' si el usuario apostó plata real (default — palabras como "
                        "'aposté', 'jugué', confirmaciones de Wplay con números de boleta). "
                        "'paper' solo si dice explícitamente 'modo paper' o 'simulación'."
                    ),
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "set_bankroll",
        "description": (
            "Setea el bankroll (saldo) actual del usuario. Disparadores típicos: "
            "'tengo 1500000 en wplay', 'mi saldo es X', 'tengo X de bankroll', "
            "'mi bankroll real es X', 'cargá X como saldo', 'deposité X'. "
            "El monto representa la PLATA EN EFECTIVO en la cuenta del usuario "
            "AHORA (lo que ve en Wplay), NO el total histórico ni el total con "
            "apuestas abiertas incluidas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": (
                        "Monto en COP, sin símbolos ni separadores. "
                        "'1.500.000' / '1,500,000' / '1500k' / '1.5M' → 1500000."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["real", "paper"],
                    "description": (
                        "'real' (default) si el usuario habla de plata real. "
                        "'paper' solo si dice explícitamente 'modo paper' o 'simulación'."
                    ),
                },
            },
            "required": ["amount"],
        },
    },
    {
        "name": "delete_pick",
        "description": (
            "Borra una pick de la base de datos. Usá cuando el usuario diga "
            "'elimina la #N', 'borra la pick #N', 'sacá la #N', 'remove pick "
            "X', 'esa apuesta no es mía bórrala', 'cancela la #N'. SOLO permite "
            "borrar picks que NO están resueltas todavía. Si está resuelta, "
            "responde con smalltalk explicando que no se puede tocar el "
            "histórico — para eso usar resolver con override manual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pick_id": {
                    "type": "integer",
                    "description": "ID de la pick a borrar. Sale del listado de picks abiertas (#15, #19, #34, etc.).",
                },
            },
            "required": ["pick_id"],
        },
    },
    {
        "name": "get_open_positions",
        "description": (
            "Lista todas las picks abiertas del usuario (no resueltas) con su "
            "payout potencial y exposición total. Disparadores: 'qué tengo "
            "abierto', 'mis apuestas pendientes', 'cuánto tengo en juego', "
            "'cuáles me faltan resolver', 'qué cosas tengo pegadas'."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
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
            "Charla casual o pregunta que NO mapea a ninguna otra herramienta. "
            "Antes de elegir esto, asegurate de que NINGUNA de las herramientas "
            "anteriores aplica. Esta es el último recurso. La 'reply' debe "
            "responder concretamente lo que el usuario pidió, no una respuesta "
            "genérica. Si el usuario hace una pregunta sobre datos del bot "
            "(picks, bankroll, partidos), preferí mencionar el comando exacto "
            "que necesita en vez de inventar la respuesta."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": (
                        "Respuesta corta y útil en español colombiano (tuteo). "
                        "Si el usuario pregunta algo del bot que no podés "
                        "responder con seguridad, sugerí el comando o frase "
                        "exacta para obtener la respuesta. Ejemplos: "
                        "'Para ver tus picks abiertas, escribe: qué tengo abierto'."
                    ),
                },
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

Reglas (en orden de prioridad — la regla #1 GANA sobre las siguientes):

1. **PASTE DE BOLETA / "GUARDA ESTA APUESTA" = register_external_bets.**
   Si el mensaje contiene señales claras de una apuesta YA hecha que el
   usuario quiere TRACKEAR, elegí register_external_bets aunque mencione
   dos equipos. Señales fuertes (basta con DOS de éstas para disparar):
   - Verbos en pasado o imperativo de registro: "aposté", "jugué",
     "guarda esta apuesta", "registra", "tomá nota", "guarda en mis
     apuestas", "anota", "metele", "metí".
   - Monto con símbolo o separador de miles: "$60,000", "60.000",
     "60000 COP", "stake 60k".
   - Línea de cuota decimal pegada: "1.85", "4.10", "@ 1.85".
   - Estructura de boleta Wplay: "Tiros de Esquina", "Total Goles",
     "Resultado Tiempo Completo", "Ganancia: $...", "Más de (X)",
     "Menos de (X)" seguido de cuota.
   - Listas con varios partidos + cuotas + montos.
   Esta regla GANA sobre la #2. El usuario está reportando una apuesta
   ejecutada, no pidiendo análisis. Ejemplos:
   - "guarda esta apuesta de everton vs city, menos de 14 corners 1.85,
     aposté 60000" → register_external_bets
   - "aposté 50000 al chelsea forest empate 4.10" → register_external_bets
   - paste con "Monto de Apuesta $50,000" → register_external_bets

2. **DOS EQUIPOS ESPECÍFICOS = analyze_match.** Si el mensaje
   menciona dos equipos específicos en estructura "X vs Y", "X v Y", "X y Y",
   "X contra Y", o cualquier construcción que claramente refiera a UN
   partido entre dos clubes nombrados, elegí analyze_match aunque el
   mensaje también incluya las palabras "picks", "apuestas", "tips",
   "info", "dame", etc. — SIEMPRE QUE NO disparó la regla #1.
   - "dame picks de medellin vs aguilas" → analyze_match (NO get_picks)
   - "info del bayern psg" → analyze_match
   - "qué dice el modelo de chelsea forest" → analyze_match
   - "tips para arsenal atlético" → analyze_match
   El usuario quiere foco en ESE partido. get_picks tirar lista completa
   de la liga es respuesta INCORRECTA.

3. Si el usuario nombra una liga sin nombrar dos equipos específicos,
   mapeala al enum exacto. Si dice algo que no existe ("La Liga",
   "Bundesliga"), igual elegí get_picks con leagues vacío y avisá en reasoning.
4. Si pide "el top pick" / "el mejor" / "uno solo" → top_only=true.
5. "finde" en Colombia = sábado y domingo (time_window="weekend").
6. Si el mensaje no mapea a ninguna acción ejecutable, usá smalltalk con una respuesta corta.
7. NUNCA mezcles dos acciones. Una sola tool por turno.
8. **REFERENCIAS AL ÚLTIMO PARTIDO ANALIZADO.** Si después del system prompt hay un bloque "Contexto del chat: ..." con un partido, y el usuario hace una pregunta que claramente es seguimiento de ESE partido sin nombrar equipos nuevos, usá analyze_match con esos equipos. Disparadores típicos:
   - "y las cuotas?", "y wplay?", "porqué no aparece?", "verifica de nuevo"
   - "cómo va ese partido?", "dame más info", "y los corners?"
   - cualquier pregunta sobre "el partido" / "ese partido" / "eso" sin nombrar otros equipos
   Si el usuario nombra DOS equipos nuevos, ignorá el contexto y usá los equipos nuevos."""


class IntentParser:
    def __init__(self, anthropic_api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required for IntentParser")
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model = model

    async def parse(self, text: str, context_hint: str = "") -> Intent:
        """Parse free text → Intent. Falls back to action='help' on failure.

        context_hint: short string injected as an extra system message describing
        recent chat state (e.g. "Último partido analizado: Bayern Munich vs PSG").
        Used so follow-ups like "y las cuotas?" or "porqué no aparece?" route to
        analyze_match for the same teams instead of falling into smalltalk.
        """
        text = text.strip()
        if not text:
            return Intent(action="help", reasoning="(mensaje vacío)")

        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}},
        ]
        if context_hint:
            system_blocks.append({"type": "text", "text": context_hint})

        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=system_blocks,
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
    if name == "register_external_bets":
        return Intent(
            action="register_externals",
            mode_hint=str(args.get("mode", "real")),
        )
    if name == "set_bankroll":
        return Intent(
            action="set_bankroll",
            bankroll_amount=int(args.get("amount")) if args.get("amount") is not None else None,
            mode_hint=str(args.get("mode", "real")),
        )
    if name == "delete_pick":
        return Intent(
            action="delete_pick",
            pick_number=int(args.get("pick_id")) if args.get("pick_id") is not None else None,
        )
    if name == "get_open_positions":
        return Intent(action="open_positions")
    if name == "resolve_pending":
        return Intent(action="resolve_auto")
    if name == "show_help":
        return Intent(action="help")
    if name == "smalltalk":
        return Intent(action="smalltalk", reasoning=str(args.get("reply", "")))
    # Unknown tool name
    return Intent(action="help", reasoning=f"(tool desconocida: {name})")
