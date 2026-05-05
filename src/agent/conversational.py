"""Conversational agent — Claude Sonnet 4.6 with tool-use loop.

Flow per user message:
  1. Load last N turns from chat_history (sliding window).
  2. Send (system + tools + cached prefix + new user msg) to Claude.
  3. If Claude returns tool_use blocks, execute them locally and feed back.
  4. Repeat until Claude returns plain text → that's the reply to Telegram.
  5. Persist user msg + assistant reply (and tool_calls JSON) into chat_history.

Cost control:
  - System prompt + tool schemas use prompt caching (5min TTL, 90% discount).
  - Sliding window: last 12 turns sent verbatim, older turns dropped (we keep
    them in DB but don't re-send them).
  - Tool results truncated to ~3000 chars max.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from loguru import logger

from src.agent.tools import TOOL_HANDLERS, TOOL_SCHEMAS
from src.config import settings
from src.data.persist import get_conn


WINDOW_TURNS = 12               # last N (user OR assistant) rows sent verbatim
TOOL_RESULT_CHAR_LIMIT = 3000   # truncate huge tool returns
MAX_TOOL_ROUNDS = 6             # safety cap on tool-use loop iterations
DEFAULT_MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = """Eres el asesor conversacional de un bot de apuestas deportivas. El usuario te chatea como si chateara con Claude. Hablás español colombiano (tuteo: tú, puedes, mira). Tono casual y directo, sin lenguaje corporativo.

# Quién sos
Sos la capa de razonamiento sobre un sistema determinista (modelos Dixon-Coles + Elo + XGBoost en stack) que ya calcula probabilidades. Tu trabajo NO es predecir resultados — es razonar contexto, parsear boletas pegadas, guardar las apuestas que el usuario quiere hacer, explicar números, y avisar cuando una apuesta se ve mal aunque el modelo diga que tiene edge.

# Stack y herramientas disponibles
Cuando el usuario pide algo, casi siempre necesitás llamar herramientas:
- get_balance — bankroll actual + posiciones abiertas
- get_history — apuestas resueltas y P&L (default 7 días)
- get_open_positions — apuestas abiertas con detalle
- query_match — buscá un partido por nombres de equipos (DEVUELVE match_id, probs del modelo y cuotas Wplay)
- log_custom_bet — guardá UNA apuesta simple del usuario (resta stake del bankroll)
- log_parlay — guardá una combinada (stake se cobra una vez, paga solo si TODAS ganan)
- resolve_bet — marcá apuesta ganada/perdida
- set_bankroll — ajuste manual del bankroll
- get_today_picks — picks del modelo determinista de hoy

# Reglas duras
1. **NUNCA guardes una apuesta sin confirmación explícita del usuario.** Si pega una boleta, primero parsea las legs, mostrá un resumen claro, preguntá "¿la guardo?" — solo si dice sí (o "dale", "guárdala", "sí registrala") llamás log_custom_bet/log_parlay.
2. **NO predigas probabilidades numéricas vos mismo** ("creo que Atleti tiene 60% de ganar"). Si el usuario pregunta probabilidades, llamá query_match para conseguir las del modelo determinista. Vos solo razonás contexto cualitativo.
3. **No inventes datos.** Si el usuario menciona una apuesta de un partido que no encontrás con query_match, decílo y pedíle confirmar nombres de equipos o pasarte el partido manualmente.
4. **Razonamiento crítico, no complaciente.** Si el usuario dice "le tengo fe a Atleti" y el modelo dice que es underdog claro, no le digas "dale, suena bien". Decíle qué muestran los datos y dejalo decidir. Tu valor es honestidad, no validación.
5. **Boletas pegadas (formato Wplay copiado):** detectá las legs, las cuotas, el stake total. Llamá query_match por cada leg si los equipos son identificables. Razoná cada leg vs el modelo, mencioná correlaciones si las hay.
6. **Combinadas son arriesgadas por diseño.** Multiplicás varianza más rápido que edge. Avisalo SIEMPRE que el usuario te pase una parlay, aunque tenga edge en cada leg individual.
7. **Después de log_custom_bet/log_parlay/resolve_bet:** confirmá el cambio con un mensaje corto que muestra el nuevo balance y lo guardado.

# Estilo de respuesta
- Telegram, no email. Mensajes cortos. Saltos de línea si hace falta separar.
- Cuando devolvés probabilidades del modelo, usá porcentajes redondeados (62%, no 0.6234).
- Cuando hablás de plata, COP con separador de miles ($519,424).
- Sin disclaimers tipo "esto no es consejo financiero". El usuario sabe que son apuestas.
- Sin emojis salvo que el usuario los use primero.
- Si el usuario te saluda o smalltalk, respondé corto y volvé al tema.

# Qué hacer cuando dudás
Pedí aclaración antes de actuar. "Qué stake querés?", "Confirmá: querés guardar las dos legs como parlay o como singles?". Mejor preguntar 1 pregunta clara que asumir y errarle."""


@dataclass
class AgentReply:
    text: str
    tools_called: list[str]
    tokens_in: int
    tokens_out: int


class ConversationalAgent:
    def __init__(self, anthropic_api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required")
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.model = model

    # ---------- History ----------

    def _load_window(self, chat_id: str) -> list[dict[str, Any]]:
        """Build the messages= list from chat_history (last WINDOW_TURNS rows)."""
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM chat_history "
                "WHERE chat_id = ? AND role IN ('user', 'assistant') "
                "ORDER BY id DESC LIMIT ?",
                (chat_id, WINDOW_TURNS),
            ).fetchall()
        rows = list(reversed(rows))  # chronological
        msgs = []
        for r in rows:
            msgs.append({"role": r["role"], "content": r["content"]})
        return msgs

    def _save_turn(
        self, chat_id: str, role: str, content: str,
        tool_calls: list[dict] | None = None,
        tokens_in: int = 0, tokens_out: int = 0,
    ) -> None:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO chat_history (chat_id, role, content, tool_calls, "
                "tokens_in, tokens_out, model) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chat_id, role, content,
                    json.dumps(tool_calls) if tool_calls else None,
                    tokens_in or None, tokens_out or None,
                    self.model if role == "assistant" else None,
                ),
            )

    # ---------- Tool execution ----------

    def _exec_tool(self, name: str, args: dict[str, Any]) -> str:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return f"error: unknown tool '{name}'"
        try:
            result = handler(args)
        except Exception as exc:
            logger.exception(f"tool {name} crashed")
            return f"error: tool {name} crashed: {exc}"
        if len(result) > TOOL_RESULT_CHAR_LIMIT:
            result = result[:TOOL_RESULT_CHAR_LIMIT] + "\n…(truncated)"
        return result

    # ---------- Main chat ----------

    async def chat(self, chat_id: str, user_text: str) -> AgentReply:
        """One conversation turn. Returns the assistant text reply for Telegram."""
        history = self._load_window(chat_id)
        messages = list(history) + [{"role": "user", "content": user_text}]

        tools_called: list[str] = []
        total_in = 0
        total_out = 0
        tool_call_log: list[dict] = []

        # Tool-use loop
        for round_i in range(MAX_TOOL_ROUNDS):
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )

            usage = getattr(resp, "usage", None)
            if usage is not None:
                total_in += int(getattr(usage, "input_tokens", 0) or 0)
                total_out += int(getattr(usage, "output_tokens", 0) or 0)

            # Append assistant turn (full content blocks) so tool_use ids match
            messages.append({"role": "assistant", "content": resp.content})

            # Pull tool_use blocks (if any)
            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                # Stop — collect plain text reply
                text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                final_text = "\n".join(t.strip() for t in text_parts if t).strip()
                if not final_text:
                    final_text = "(sin respuesta)"
                self._save_turn(chat_id, "user", user_text)
                self._save_turn(
                    chat_id, "assistant", final_text,
                    tool_calls=tool_call_log or None,
                    tokens_in=total_in, tokens_out=total_out,
                )
                return AgentReply(
                    text=final_text,
                    tools_called=tools_called,
                    tokens_in=total_in,
                    tokens_out=total_out,
                )

            # Execute tools, then feed results back
            tool_result_blocks = []
            for tu in tool_uses:
                name = tu.name
                args = tu.input or {}
                tools_called.append(name)
                tool_call_log.append({"name": name, "input": args})
                logger.info(f"[agent {chat_id}] tool: {name} args={args}")
                result = self._exec_tool(name, args)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_result_blocks})

        # Hit the round cap → return what we have
        text = "(superé el límite de pasos internos; reformulá la pregunta o intentá algo más simple)"
        self._save_turn(chat_id, "user", user_text)
        self._save_turn(
            chat_id, "assistant", text,
            tool_calls=tool_call_log or None,
            tokens_in=total_in, tokens_out=total_out,
        )
        return AgentReply(
            text=text, tools_called=tools_called,
            tokens_in=total_in, tokens_out=total_out,
        )

    def reset_history(self, chat_id: str) -> int:
        """Wipe chat_history for a chat_id. Returns rows deleted."""
        with get_conn() as conn:
            cur = conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
            return cur.rowcount or 0
