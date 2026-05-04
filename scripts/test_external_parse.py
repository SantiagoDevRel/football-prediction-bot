"""Smoke test for external bet parser with the user's actual Wplay paste."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.tracking.external_bets import parse_pasted_text, resolve_bets


USER_TEXT = """Fecha de la apuesta    Número de apuesta    Tipo de apuesta    Apuesta    Monto de Apuesta    cuotas    Bono    Ingreso    Estado
 03 Mayo 00:34
2025351350    Derecha
Ganador    Chelsea v Nottingham Forest
$50,000
4.10    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
04 Mayo 09:00    Chelsea v Nottingham Forest    Resultado Tiempo Completo
Empate
$205,000

RECUPERAR APUESTA?:
$48,000
Imprimir
 03 Mayo 00:34
2025350903    Derecha
Ganador    América de Cali v Deportivo Pereira
$100,000
2.00    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
03 Mayo 17:45    América de Cali v Deportivo Pereira    Total Goles Más/Menos de
Menos de (2.5)
$200,000

RECUPERAR APUESTA?:
$96,000
Imprimir
 03 Mayo 00:33
2025350575    Derecha
Ganador    Alianza FC v Millonarios
$50,000
3.45    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
03 Mayo 15:30    Alianza FC v Millonarios    Resultado Tiempo Completo
Alianza FC
$172,500

RECUPERAR APUESTA?:
$48,000
Imprimir
 03 Mayo 00:33
2025349941    Derecha
Ganador    Manchester United v Liverpool
$100,000
2.60    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
03 Mayo 09:30    Manchester United v Liverpool    Total Goles Más/Menos de
Menos de (2.5)
$260,000

RECUPERAR APUESTA?:
$96,000
Imprimir
 03 Mayo 00:32
2025348985    Derecha
Ganador    Everton v Manchester City
$200,000
1.50    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
04 Mayo 14:00    Everton v Manchester City    Total Goles Más/Menos de
Menos de (3.5)
$300,000

RECUPERAR APUESTA?:
$192,000
Imprimir
 03 Mayo 00:31
2025348488    Derecha
Ganador    Independiente Santa Fe v Internacional de Bogota
$200,000
1.38    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
03 Mayo 15:30    Independiente Santa Fe v Internacional de Bogota    Total Goles Más/Menos de
Más de (1.5)
$276,000

RECUPERAR APUESTA?:
$192,000
Imprimir
 03 Mayo 00:30
2025347809    Derecha
Ganador    Alianza FC v Millonarios
$200,000
1.42    -    -    Abierta
Fecha y Hora del evento    Descripción del evento    Apuesta    Selección    Ganancia posible    Resultado
03 Mayo 15:30    Alianza FC v Millonarios    Total Goles Más/Menos de
Más de (1.5)
$284,000

RECUPERAR APUESTA?:
$192,000"""


async def main() -> None:
    print("=== PARSING ===")
    parsed = await parse_pasted_text(USER_TEXT, settings.anthropic_api_key)
    print(f"\n{len(parsed)} bets parsed:")
    for i, p in enumerate(parsed, 1):
        print(f"  {i}. {p.home_team} v {p.away_team}  "
              f"market={p.market}  sel_hint={p.selection_hint}  "
              f"odds={p.odds}  stake=${p.stake:,.0f}")

    print("\n=== RESOLVING ===")
    resolved, errors = resolve_bets(parsed)
    print(f"\n{len(resolved)} resolved, {len(errors)} errors:")
    for r in resolved:
        print(f"  OK  {r.home_team} v {r.away_team}  ({r.league})")
        print(f"      {r.market}:{r.selection} @ {r.odds}  stake=${r.stake:,.0f}  conf={r.confidence:.2f}")
        if r.note:
            print(f"      note: {r.note}")
    for pb, reason in errors:
        print(f"  ERR  {pb.home_team} v {pb.away_team}  -> {reason}")


if __name__ == "__main__":
    asyncio.run(main())
