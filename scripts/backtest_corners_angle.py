"""Backtest: does the "favorite that fails to win generates corners" angle hold?

The live angle (from the conversation): bet OVER corners on a strong team that
is drawing/losing, because a chasing favorite attacks the wings -> crosses ->
corners. Corners are high-frequency / low-variance, so the causal chain is more
reliable than "domination -> goal".

What this script CAN test (with free-tier data = final boxscore corners + result):
    Do matches where the pre-match FAVORITE fails to win produce more corners
    (total, and the favorite's own) than matches where the favorite wins?
    If yes, the *tendency* the angle relies on is real.

What this script CANNOT test (be honest — CLAUDE.md "ser claro sobre limitaciones"):
    1. Final corners != live timeline. We bet AFTER the favorite trails; by then
       some corners already happened AND the live line has moved up. This measures
       the underlying rate, not the live-price-adjusted edge.
    2. No historical corner ODDS on free tier -> we report HIT-RATE shift, not
       realized ROI. Edge exists only if the live line lags the true rate.
    3. World Cup (national teams) is NOT in this DB. Result is DIRECTIONAL for the
       WC; club-league corner dynamics are the proxy.

Favorite is determined leakage-free via a chronological inline Elo (ratings
updated only AFTER each match is scored).

Usage:
    python scripts/backtest_corners_angle.py
    python scripts/backtest_corners_angle.py --min-fav-prob 0.60 --min-history 5
"""
from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from src.config import settings  # noqa: E402

# -110 American odds break-even; typical Wplay corner-over margin is worse,
# so we also flag the 55% threshold as a more realistic bar.
BREAK_EVEN_110 = 0.524
REALISTIC_BAR = 0.55

HA = 60.0   # home advantage in Elo points
K = 20.0    # Elo update factor


def _load() -> list[dict]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals, l.name AS league,
               s.home_corners, s.away_corners
          FROM matches m
          JOIN match_stats s ON s.match_id = m.id
          JOIN leagues l ON l.id = m.league_id
         WHERE m.status = 'finished'
           AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
           AND s.home_corners IS NOT NULL AND s.away_corners IS NOT NULL
         ORDER BY m.kickoff_utc ASC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _expected_home(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((r_home + HA) - r_away) / 400.0))


def build_records(matches: list[dict], min_history: int) -> list[dict]:
    """Chronological Elo replay -> per-match record with favorite + corners.
    Only matches where BOTH teams have >= min_history prior games are kept
    (so the favorite signal is meaningful)."""
    ratings: dict[int, float] = defaultdict(lambda: 1500.0)
    games: dict[int, int] = defaultdict(int)
    records: list[dict] = []

    for m in matches:
        h, a = m["home_team_id"], m["away_team_id"]
        rh, ra = ratings[h], ratings[a]
        exp_h = _expected_home(rh, ra)

        if games[h] >= min_history and games[a] >= min_history:
            if exp_h >= 0.5:
                fav, fav_prob = "home", exp_h
            else:
                fav, fav_prob = "away", 1.0 - exp_h
            hg, ag = m["home_goals"], m["away_goals"]
            if hg > ag:
                winner = "home"
            elif hg < ag:
                winner = "away"
            else:
                winner = "draw"
            fav_won = (winner == fav)
            hc, ac = m["home_corners"], m["away_corners"]
            fav_corners = hc if fav == "home" else ac
            opp_corners = ac if fav == "home" else hc
            records.append({
                "league": m["league"],
                "fav_prob": fav_prob,
                "fav_won": fav_won,
                "fav_drew": winner == "draw",
                "fav_lost": (not fav_won) and winner != "draw",
                "total_corners": hc + ac,
                "fav_corners": fav_corners,
                "opp_corners": opp_corners,
            })

        # update Elo AFTER recording (no leakage)
        hg, ag = m["home_goals"], m["away_goals"]
        s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        ratings[h] = rh + K * (s_home - exp_h)
        ratings[a] = ra + K * ((1.0 - s_home) - (1.0 - exp_h))
        games[h] += 1
        games[a] += 1

    return records


def _over_rate(values: list[float], line: float) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v > line) / len(values)


def _mark(rate: float) -> str:
    if rate >= REALISTIC_BAR:
        return "✅"
    if rate >= BREAK_EVEN_110:
        return "~"
    return " "


def report(records: list[dict], min_fav_prob: float) -> None:
    did_not_win = [r for r in records if not r["fav_won"]]
    won = [r for r in records if r["fav_won"]]
    clear = [r for r in records if r["fav_prob"] >= min_fav_prob]
    clear_dnw = [r for r in clear if not r["fav_won"]]

    def avg(rs, key):
        return (sum(r[key] for r in rs) / len(rs)) if rs else 0.0

    print("=" * 70)
    print("  BACKTEST — ÁNGULO DE CÓRNERS DEL FAVORITO QUE PERSIGUE")
    print("=" * 70)
    print(f"  Partidos analizados (con historia suficiente): {len(records)}")
    print(f"  Favorito GANÓ:        {len(won):>5}  ({len(won)/max(len(records),1):.0%})")
    print(f"  Favorito NO ganó:     {len(did_not_win):>5}  "
          f"({len(did_not_win)/max(len(records),1):.0%})  ← el escenario del ángulo")
    print()
    print("  PROMEDIOS DE CÓRNERS")
    print("  " + "-" * 66)
    print(f"  {'grupo':<26}{'total':>10}{'favorito':>12}{'rival':>10}")
    print(f"  {'Favorito ganó':<26}{avg(won,'total_corners'):>10.2f}"
          f"{avg(won,'fav_corners'):>12.2f}{avg(won,'opp_corners'):>10.2f}")
    print(f"  {'Favorito NO ganó':<26}{avg(did_not_win,'total_corners'):>10.2f}"
          f"{avg(did_not_win,'fav_corners'):>12.2f}{avg(did_not_win,'opp_corners'):>10.2f}")
    lift = avg(did_not_win, "fav_corners") - avg(won, "fav_corners")
    print(f"  → lift de córners del FAVORITO cuando NO gana: {lift:+.2f}")
    print()

    print("  TASA DE ACIERTO 'OVER' (✅ ≥55% | ~ ≥52.4% break-even @ -110)")
    print("  " + "-" * 66)
    print("  CÓRNERS TOTALES del partido — favorito NO ganó:")
    tot = [r["total_corners"] for r in did_not_win]
    for line in (8.5, 9.5, 10.5, 11.5, 12.5):
        rate = _over_rate(tot, line)
        print(f"     over {line:>4}  {rate:6.1%}  {_mark(rate)}")
    print("  CÓRNERS DEL FAVORITO — favorito NO ganó:")
    favc = [r["fav_corners"] for r in did_not_win]
    for line in (3.5, 4.5, 5.5, 6.5):
        rate = _over_rate(favc, line)
        print(f"     over {line:>4}  {rate:6.1%}  {_mark(rate)}")
    print()

    print(f"  FAVORITO CLARO (prob ≥ {min_fav_prob:.0%}) que NO ganó: "
          f"{len(clear_dnw)} partidos")
    if clear_dnw:
        favc2 = [r["fav_corners"] for r in clear_dnw]
        tot2 = [r["total_corners"] for r in clear_dnw]
        print(f"     córners promedio del favorito: {avg(clear_dnw,'fav_corners'):.2f}")
        for line in (4.5, 5.5, 6.5):
            print(f"     fav over {line}: {_over_rate(favc2, line):.1%}  {_mark(_over_rate(favc2,line))}")
        for line in (9.5, 10.5, 11.5):
            print(f"     total over {line}: {_over_rate(tot2, line):.1%}  {_mark(_over_rate(tot2,line))}")
    print()

    # Per-league breakdown (corner rates vary a lot by league)
    print("  POR LIGA (favorito NO ganó — córners totales promedio):")
    print("  " + "-" * 66)
    by_league: dict[str, list[dict]] = defaultdict(list)
    for r in did_not_win:
        by_league[r["league"]].append(r)
    for lg, rs in sorted(by_league.items(), key=lambda x: -len(x[1])):
        if len(rs) < 20:
            continue
        tot_l = [r["total_corners"] for r in rs]
        print(f"  {lg[:30]:<30} n={len(rs):>4}  "
              f"avg={sum(tot_l)/len(tot_l):5.2f}  "
              f"over9.5={_over_rate(tot_l,9.5):.0%}  over10.5={_over_rate(tot_l,10.5):.0%}")
    print()

    print("  " + "=" * 66)
    print("  LECTURA HONESTA")
    print("  " + "-" * 66)
    print("  • Esto mide la TENDENCIA con córners FINALES, no la línea live.")
    print("    En vivo apostás DESPUÉS de que el favorito va perdiendo: parte de")
    print("    los córners YA pasaron y la línea de over YA subió. El edge real")
    print("    existe sólo si la línea live de Wplay va por detrás de esta tasa.")
    print("  • Sin odds históricas de córners (free tier) medimos acierto, no ROI.")
    print("  • Mundial = selecciones, NO está en esta data. Esto es DIRECCIONAL.")
    print("  • Próximo paso real: el monitor (momentum_panel --watch) puede loguear")
    print("    la línea live de Wplay vs córners reales para medir CLV de verdad.")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-fav-prob", type=float, default=0.60,
                    help="umbral de 'favorito claro' (default 0.60)")
    ap.add_argument("--min-history", type=int, default=5,
                    help="partidos previos mínimos por equipo para contar (default 5)")
    args = ap.parse_args()

    matches = _load()
    if not matches:
        print("No hay match_stats con córners en la DB.")
        return
    print(f"Cargados {len(matches)} partidos finalizados con córners.\n")
    records = build_records(matches, args.min_history)
    if not records:
        print("Ningún partido pasó el filtro de historia mínima.")
        return
    report(records, args.min_fav_prob)


if __name__ == "__main__":
    main()
