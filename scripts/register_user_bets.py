"""One-shot insert of the user's 7 real Wplay bets from 2026-05-03.

Run once. Then auto-resolver picks them up when matches finish.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.tracking.external_bets import (
    parse_pasted_text, resolve_bets, register_resolved_bets,
)


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
    parsed = await parse_pasted_text(USER_TEXT, settings.anthropic_api_key)
    print(f"\n{len(parsed)} apuestas extraídas")
    resolved, errors = resolve_bets(parsed)
    print(f"{len(resolved)} resueltas, {len(errors)} errores")
    if errors:
        for pb, reason in errors:
            print(f"  ERR  {pb.home_team} v {pb.away_team}: {reason}")
        if len(resolved) == 0:
            print("Aborting — nothing to insert.")
            return

    result = register_resolved_bets(resolved, mode="real")
    print(f"\n=== INSERTED {len(result.inserted)} of {len(resolved)} ===")
    for pid, b in result.inserted:
        print(f"  pick #{pid}: {b.home_team} v {b.away_team} {b.market}:{b.selection} @ {b.odds} ${b.stake:,.0f}")
    if result.skipped:
        for pb, reason in result.skipped:
            print(f"  SKIP: {pb.home_team} v {pb.away_team} ({reason})")


if __name__ == "__main__":
    asyncio.run(main())
