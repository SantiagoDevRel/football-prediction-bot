"""Smoke test: feed the user's exact parlay-paste example into the agent
and print the conversation. Verifies tool-use loop, message persistence,
and parlay-leg parsing without going through Telegram.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_agent.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ConversationalAgent  # noqa: E402
from src.config import settings  # noqa: E402


SAMPLE_PARLAY_MSG = """quiero hacer esta apuesta, le tengo fe al atletico de madrid, te parece bien o hago algo diferente?

Atlético Madrid/Empate
2.30
Doble Oportunidad
Arsenal v Atlético Madrid
$
Ganancia: $0

Eliminar
Bayern Munich
1.75
Se clasificará
Bayern Munich v PSG
$
Ganancia: $0
 Parlay
Doble (1 Apuesta)
$
519424
cuotas: 4.025Ganancia: $2,090,682
Noº total de apuestas: 1
Total importe apostado:  $519,424
Aceptar cualquier cambio de cuotas
Aceptar cuota más alta
Ganancia posible:
$2,090,682"""


async def main() -> None:
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    agent = ConversationalAgent(settings.anthropic_api_key)
    chat_id = "smoke_test_chat"

    # Wipe any prior smoke-test history for a clean run
    n = agent.reset_history(chat_id)
    print(f"[setup] cleared {n} prior turns\n")

    print(f"=== USER ===\n{SAMPLE_PARLAY_MSG}\n")
    reply = await agent.chat(chat_id, SAMPLE_PARLAY_MSG)
    print(f"=== ASSISTANT ===\n{reply.text}\n")
    print(f"--- meta ---")
    print(f"tools_called: {reply.tools_called}")
    print(f"tokens_in={reply.tokens_in} tokens_out={reply.tokens_out}")


if __name__ == "__main__":
    asyncio.run(main())
