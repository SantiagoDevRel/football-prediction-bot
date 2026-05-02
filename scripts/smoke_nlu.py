"""Smoke test for the NLU intent parser. Hits Claude API with a small set of
realistic Spanish inputs and prints the parsed Intent. Run once after wiring
the parser to confirm everything flows.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_nlu.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.llm.nlu import IntentParser  # noqa: E402


CASES = [
    "dame picks solo de la liga betplay este findesemana",
    "el top pick de hoy",
    "qué hay en vivo",
    "analiza nacional vs millonarios",
    "cómo voy de balance",
    "aposté el 3 con 5000",
    "dame algo de champions league mañana",
    "una sola apuesta segura para hoy en premier",
    "hola, qué tal",
    "resolvé los picks vencidos",
    "ayuda",
]


async def main() -> None:
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    parser = IntentParser(settings.anthropic_api_key)
    for text in CASES:
        intent = await parser.parse(text)
        leagues = ",".join(intent.leagues) or "-"
        extras = []
        if intent.top_only:
            extras.append("TOP")
        if intent.home_query or intent.away_query:
            extras.append(f"{intent.home_query} v {intent.away_query}")
        if intent.pick_number is not None:
            extras.append(f"#{intent.pick_number} stake={intent.stake}")
        if intent.reasoning:
            extras.append(f"reply={intent.reasoning[:50]!r}")
        extras_str = " | ".join(extras)
        print(f"[{intent.action:10}] leagues={leagues:30} time={intent.time_window:8} {extras_str}")
        print(f"  > {text!r}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
