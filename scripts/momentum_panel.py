"""Live momentum panel + corner-pressure monitor (free-tier, ESPN boxscore).

Reads ESPN's /summary boxscore (no key, no quota) and renders EVERY team
statistic available, plus derived signals. Two modes:

    snapshot (default): print the panel once and exit.
    --watch N         : poll every N seconds, track corner/cross/shot deltas,
                        and alert when a non-winning side's corner pressure rises
                        (the only live angle with a defensible edge — see CLAUDE.md
                        decision in the conversation: volume != goal quality, but
                        corners are high-frequency / low-variance / causally driven
                        by a chasing team's wing attacks).

IMPORTANT (honest framing baked in): this panel is NOT a green light to bet
GOAL markets. Possession + shot VOLUME are weak predictors of goals — the panel
prints a trap warning for goal markets and only surfaces CORNER markets as
actionable. ESPN gives shot count, not shot quality (xG); we never pretend
otherwise.

Usage:
    python scripts/momentum_panel.py --home Turkey --away Australia
    python scripts/momentum_panel.py --event-id 760421 --league world_cup
    python scripts/momentum_panel.py --home Germany --away Curacao --watch 60
    python scripts/momentum_panel.py --home X --away Y --favorite home --watch 90

Leagues searched by default (national-team + club, live coverage):
    world_cup, euros, copa_america, champions_league, libertadores,
    sudamericana, premier_league, liga_betplay
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252 and choke on emojis/accents in the panel.
with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from src.data.espn import (  # noqa: E402
    fetch_match_full_stats,
    fetch_scoreboard,
)

# Leagues we sweep to resolve team names -> (league_slug, event_id).
SEARCH_LEAGUES = [
    "world_cup", "euros", "copa_america", "champions_league",
    "libertadores", "sudamericana", "premier_league", "liga_betplay",
]


def _norm(s: str) -> str:
    """Accent-insensitive, lowercase, alnum-only (matches 'Turkiye'~'Turquia'
    loosely via substring after stripping)."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


async def resolve_event(home: str, away: str) -> tuple[str, str, str, str] | None:
    """Search live/recent scoreboards for a match. Returns
    (league_slug, event_id, home_team, away_team) or None."""
    hn, an = _norm(home), _norm(away)
    for slug in SEARCH_LEAGUES:
        try:
            matches = await fetch_scoreboard(slug)
        except Exception:
            continue
        for m in matches:
            mh, ma = _norm(m.home_team), _norm(m.away_team)
            pair_ok = (
                (hn in mh or mh in hn) and (an in ma or ma in an)
            ) or (
                # team order can be flipped vs the user's input
                (hn in ma or ma in hn) and (an in mh or mh in an)
            )
            if pair_ok:
                return slug, m.espn_id, m.home_team, m.away_team
    return None


# ---------- derived signals ----------

def _g(stats: dict, name: str) -> float:
    v = stats.get(name, {}).get("value")
    return float(v) if v is not None else 0.0


def derive(side: dict) -> dict:
    """Compute interpretable signals from one team's stat block."""
    shots = _g(side, "totalShots")
    on_t = _g(side, "shotsOnTarget")
    blocked = _g(side, "blockedShots")
    off_t = max(0.0, shots - on_t - blocked)
    corners = _g(side, "wonCorners")
    crosses = _g(side, "totalCrosses")
    acc_crosses = _g(side, "accurateCrosses")
    poss = _g(side, "possessionPct")
    on_t_rate = (on_t / shots) if shots > 0 else 0.0
    # Corner pressure: corners already won + attacking width proxy (crosses).
    # Weights are intentionally simple/transparent, not fitted.
    corner_pressure = corners + 0.25 * crosses + 0.10 * shots
    return {
        "shots": shots, "on_target": on_t, "blocked": blocked,
        "off_target": off_t, "on_target_rate": on_t_rate,
        "corners": corners, "crosses": crosses, "acc_crosses": acc_crosses,
        "possession": poss, "corner_pressure": corner_pressure,
        # volume-without-quality: lots of shots, few on target
        "volume_no_quality": shots >= 12 and on_t_rate < 0.35,
    }


def _result_state(gh: int | None, ga: int | None) -> tuple[str, str]:
    """Return (home_state, away_state) in {winning, drawing, losing}."""
    if gh is None or ga is None:
        return ("?", "?")
    if gh > ga:
        return ("winning", "losing")
    if gh < ga:
        return ("losing", "winning")
    return ("drawing", "drawing")


# ---------- rendering ----------

def render_panel(data: dict, favorite: str | None = None) -> str:
    home, away = data["home_team"], data["away_team"]
    gh, ga = data["home_goals"], data["away_goals"]
    minute = data["minute"]
    status = data["status"]
    hs, as_ = data["home"], data["away"]
    order = data["stat_order"]

    score_str = f"{gh if gh is not None else '-'}-{ga if ga is not None else '-'}"
    min_str = f"{minute}'" if minute is not None else ("FT" if status == "finished" else status)
    lines = []
    lines.append("=" * 64)
    lines.append(f"  {home}  {score_str}  {away}   [{status} {min_str}]")
    lines.append("=" * 64)

    # Side-by-side raw stats (every stat ESPN gives)
    lines.append(f"  {'STAT':<24}{home[:16]:>16}{away[:16]:>16}")
    lines.append("  " + "-" * 56)
    for name in order:
        label = (hs.get(name) or as_.get(name) or {}).get("label", name)
        hv = (hs.get(name) or {}).get("display", "-")
        av = (as_.get(name) or {}).get("display", "-")
        lines.append(f"  {label[:24]:<24}{hv!s:>16}{av!s:>16}")

    # Derived signals
    dh, da = derive(hs), derive(as_)
    h_state, a_state = _result_state(gh, ga)
    lines.append("")
    lines.append("  " + "-" * 56)
    lines.append("  DERIVED SIGNALS")
    lines.append("  " + "-" * 56)
    for nm, d, st in ((home, dh, h_state), (away, da, a_state)):
        lines.append(
            f"  {nm[:20]:<20} [{st}]  "
            f"tiros {int(d['shots'])} (al arco {int(d['on_target'])}, "
            f"bloq {int(d['blocked'])}, fuera {int(d['off_target'])}) "
            f"| on-target {d['on_target_rate']:.0%}"
        )
        lines.append(
            f"  {'':<20} córners {int(d['corners'])} | centros {int(d['crosses'])} "
            f"(precisos {int(d['acc_crosses'])}) | presión-córner {d['corner_pressure']:.1f}"
        )
        if d["volume_no_quality"]:
            lines.append(
                f"  {'':<20} ⚠ VOLUMEN SIN CALIDAD: muchos tiros, pocos al arco "
                f"— NO leer como 'merece gol'."
            )

    total_corners = int(dh["corners"] + da["corners"])
    proj = None
    if minute and minute > 0 and status == "live":
        proj = total_corners * (90.0 / min(minute, 90))

    lines.append("")
    lines.append("  " + "-" * 56)
    lines.append("  LECTURA DE MERCADOS (honesta)")
    lines.append("  " + "-" * 56)
    lines.append(
        "  ⛔ GOL (gana/empata, BTTS): posesión y volumen de tiro son "
        "predictores DÉBILES de gol."
    )
    lines.append(
        "     ESPN da cantidad de tiros, NO calidad (xG). No bancar gol por "
        "'dominio'."
    )

    # Corner angle — the usable one. Highlight a non-winning side that's pressing.
    pressing = []
    for nm, d, st in ((home, dh, h_state), (away, da, a_state)):
        if st in ("drawing", "losing") and d["corner_pressure"] >= 4:
            pressing.append((nm, d, st))
    lines.append(
        f"  ✅ CÓRNERS: total actual = {total_corners}"
        + (f" | proyección 90' ≈ {proj:.0f}" if proj else "")
    )
    if pressing:
        for nm, d, st in pressing:
            lines.append(
                f"     → {nm} va {st.upper()} y genera presión de córner "
                f"({int(d['corners'])} córners, {int(d['crosses'])} centros). "
                f"Ángulo válido si la línea live de over paga."
            )
    else:
        lines.append(
            "     → ningún favorito persiguiendo con presión clara de córner ahora."
        )
    if favorite in ("home", "away"):
        fav_name = home if favorite == "home" else away
        fav_state = h_state if favorite == "home" else a_state
        fav_d = dh if favorite == "home" else da
        lines.append(
            f"  ★ Favorito marcado: {fav_name} [{fav_state}] "
            f"presión-córner={fav_d['corner_pressure']:.1f}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------- monitor (watch) mode ----------

def _now() -> str:
    return datetime.now(tz=UTC).astimezone().strftime("%H:%M:%S")


async def watch(league_slug: str, event_id: str, interval: int,
                favorite: str | None) -> None:
    print(f"[{_now()}] MONITOR cada {interval}s — Ctrl+C para cortar\n")
    prev: dict | None = None
    while True:
        data = await fetch_match_full_stats(league_slug, event_id)
        if data is None:
            print(f"[{_now()}] sin data (reintentando)…")
            await asyncio.sleep(interval)
            continue

        print(render_panel(data, favorite))

        # Deltas vs previous poll → corner-pressure ALERTS
        if prev is not None:
            dh_prev, da_prev = derive(prev["home"]), derive(prev["away"])
            dh, da = derive(data["home"]), derive(data["away"])
            gh, ga = data["home_goals"], data["away_goals"]
            h_state, a_state = _result_state(gh, ga)
            for nm, d_now, d_old, st in (
                (data["home_team"], dh, dh_prev, h_state),
                (data["away_team"], da, da_prev, a_state),
            ):
                d_corners = d_now["corners"] - d_old["corners"]
                d_crosses = d_now["crosses"] - d_old["crosses"]
                # Alert: a non-winning side gaining corners fast = live over angle.
                if st in ("drawing", "losing") and (d_corners >= 2 or d_crosses >= 4):
                    print(
                        f"  🔔 ALERTA [{_now()}] {nm} ({st}) +{int(d_corners)} córners "
                        f"/ +{int(d_crosses)} centros en {interval}s. "
                        f"Mirá la línea live de over córners."
                    )
        prev = data
        if data["status"] == "finished":
            print(f"\n[{_now()}] partido terminado. Fin del monitor.")
            return
        await asyncio.sleep(interval)


# ---------- main ----------

async def _main() -> int:
    ap = argparse.ArgumentParser(description="Panel de momentum + monitor de córners (ESPN)")
    ap.add_argument("--home", help="equipo local (búsqueda por nombre)")
    ap.add_argument("--away", help="equipo visitante")
    ap.add_argument("--event-id", help="ESPN event id (salta la búsqueda)")
    ap.add_argument("--league", default="world_cup",
                    help="slug de liga si usás --event-id (default world_cup)")
    ap.add_argument("--watch", type=int, metavar="SEG",
                    help="modo monitor: poll cada SEG segundos")
    ap.add_argument("--favorite", choices=["home", "away"],
                    help="marca cuál equipo es el favorito (para la alerta de córner)")
    args = ap.parse_args()

    if args.event_id:
        league_slug, event_id = args.league, args.event_id
    elif args.home and args.away:
        resolved = await resolve_event(args.home, args.away)
        if not resolved:
            print(f"No encontré '{args.home}' vs '{args.away}' en los scoreboards de hoy.")
            print("Probá con --event-id + --league, o revisá los nombres.")
            return 1
        league_slug, event_id, h, a = resolved
        print(f"Match: {h} vs {a}  [{league_slug} event={event_id}]\n")
    else:
        print("Pasá --home y --away, o --event-id.")
        return 1

    if args.watch:
        try:
            await watch(league_slug, event_id, args.watch, args.favorite)
        except KeyboardInterrupt:
            print("\nmonitor cortado.")
        return 0

    data = await fetch_match_full_stats(league_slug, event_id)
    if data is None:
        print("No pude traer el boxscore (¿partido sin stats todavía?).")
        return 1
    print(render_panel(data, args.favorite))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
