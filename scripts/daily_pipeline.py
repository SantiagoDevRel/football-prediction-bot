"""Daily prediction pipeline.

End-to-end run:
    1. Pull today's + tomorrow's fixtures from ESPN for both leagues, persist
    2. Load trained Dixon-Coles + Elo models per league
    3. For each upcoming match, run both models and take a simple average
       (later: replace with proper ensemble model from Phase 2)
    4. Persist predictions to DB
    5. Try Wplay scraping for live odds (best effort; degrades gracefully)
    6. If odds available, run value detection and log paper picks
    7. Send Telegram summary (or console fallback)

Designed to be run via cron daily at, say, 09:00 local time.
"""
import asyncio
import pickle
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.betting.value_detector import detect_value, OddsLine  # noqa: E402
from src.config import settings  # noqa: E402
from src.data.espn import fetch_scoreboard  # noqa: E402
from src.data.persist import bulk_upsert_espn, get_conn  # noqa: E402
from src.data.wplay_scraper import (  # noqa: E402
    WplayOdds,
    normalize_name,
    scrape_all as scrape_wplay_all,
)
from src.notifications.telegram_bot import send_message  # noqa: E402
from src.tracking.pick_logger import get_current_bankroll, log_pick  # noqa: E402

LEAGUES = ["premier_league", "liga_betplay"]
LEAGUE_NAME = {"premier_league": "Premier League", "liga_betplay": "Liga BetPlay Dimayor"}
ARTIFACTS = ROOT / "models_artifacts"


def _model_path(model_name: str, league_slug: str) -> Path:
    safe = LEAGUE_NAME[league_slug].lower().replace(" ", "_").replace(".", "")
    return ARTIFACTS / f"{model_name}_{safe}.pkl"


def _load_models(league_slug: str) -> dict:
    out = {}
    for name in ("dixon_coles", "elo"):
        p = _model_path(name, league_slug)
        if p.exists():
            with open(p, "rb") as f:
                out[name] = pickle.load(f)
        else:
            logger.warning(f"model not found: {p}")
    return out


def _persist_prediction(
    match_id: int, market: str, selection: str, model: str,
    probability: float, confidence: float | None = None,
    features: dict | None = None,
) -> None:
    import json
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (match_id, market, selection, model, probability, confidence, features_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (match_id, market, selection, model, probability, confidence,
             json.dumps(features or {})),
        )


async def _pull_fixtures() -> int:
    today = date.today()
    end = today + timedelta(days=2)
    total = 0
    for slug in LEAGUES:
        for offset in range((end - today).days + 1):
            d = today + timedelta(days=offset)
            try:
                ms = await fetch_scoreboard(slug, d)
                bulk_upsert_espn(ms)
                total += len(ms)
            except Exception as exc:
                logger.warning(f"fixture pull {slug} {d} failed: {exc}")
    return total


def _load_upcoming_matches() -> list[dict]:
    """Return matches kicking off today or in the next 2 days."""
    today = date.today()
    end = today + timedelta(days=2)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
                   h.name AS home_name, a.name AS away_name,
                   l.name AS league_name
              FROM matches m
              JOIN teams h ON m.home_team_id = h.id
              JOIN teams a ON m.away_team_id = a.id
              JOIN leagues l ON m.league_id = l.id
             WHERE m.status = 'scheduled'
               AND date(m.kickoff_utc) >= ?
               AND date(m.kickoff_utc) <= ?
             ORDER BY m.kickoff_utc ASC
            """,
            (today.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def _league_slug_from_name(name: str) -> str:
    if "Premier" in name:
        return "premier_league"
    return "liga_betplay"


def _ensemble_predict(models: dict, home_id: int, away_id: int):
    """Simple average of available models. Replace with stacking ensemble in v2."""
    probs = []
    for name, m in models.items():
        try:
            p = m.predict_match(home_id, away_id)
            probs.append(p)
        except KeyError:
            continue
    if not probs:
        return None
    n = len(probs)
    avg = type(probs[0])(
        p_home_win=sum(p.p_home_win for p in probs) / n,
        p_draw=sum(p.p_draw for p in probs) / n,
        p_away_win=sum(p.p_away_win for p in probs) / n,
        p_over_2_5=sum(p.p_over_2_5 for p in probs) / n,
        p_under_2_5=sum(p.p_under_2_5 for p in probs) / n,
        p_btts_yes=sum(p.p_btts_yes for p in probs) / n,
        p_btts_no=sum(p.p_btts_no for p in probs) / n,
        expected_home_goals=sum(p.expected_home_goals for p in probs) / n,
        expected_away_goals=sum(p.expected_away_goals for p in probs) / n,
        features={"models_used": list(models.keys())},
    )
    return avg, probs


# ---------- Telegram message formatting ----------

_DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _humanize_kickoff(iso: str) -> str:
    """'2026-05-02T16:30' -> 'sábado 16:30 (02 may)'."""
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso[:16].replace("T", " ")
    today = date.today()
    diff_days = (dt.date() - today).days
    if diff_days == 0:
        when = "HOY"
    elif diff_days == 1:
        when = "MAÑANA"
    else:
        when = _DAYS_ES[dt.weekday()]
    fecha = f"{dt.day:02d} {_MONTHS_ES[dt.month - 1]}"
    return f"{when} {dt.strftime('%H:%M')} <i>({fecha})</i>"


def _humanize_pick(vb: dict) -> tuple[str, str]:
    """Return (action_line, why_line) for a value bet. Spanish and friendly."""
    market, selection = vb["market"], vb["selection"]
    home, away = vb["home_team"], vb["away_team"]

    if market == "1x2":
        if selection == "home":
            action = f"<b>Gana {home}</b>"
        elif selection == "away":
            action = f"<b>Gana {away}</b> (visitante)"
        else:
            action = "<b>Empate</b> (X)"
    elif market == "ou_2.5":
        action = "<b>Más de 2.5 goles</b>" if selection == "over" else "<b>Menos de 2.5 goles</b>"
    elif market == "btts":
        action = "<b>Ambos equipos marcan</b>" if selection == "yes" else "<b>NO marcan los dos</b>"
    else:
        action = f"<b>{market}:{selection}</b>"

    model_prob = vb["model_probability"]
    fair_odds = vb["fair_odds"]
    casa_odds = vb["odds"]
    edge = vb["edge"]
    casa_implied = 1.0 / casa_odds  # implied prob from the bookie odds

    # Why: tell the math in plain terms
    why = (
        f"Modelo: {model_prob:.0%} probabilidad. "
        f"Wplay paga {casa_odds:.2f} (lo cual sería justo si fuera {casa_implied:.0%}). "
        f"Edge: <b>+{edge*100:.0f}%</b> sobre el mercado."
    )
    return action, why


def _format_telegram_message(
    predictions: list,
    picks: list,
    bankroll: float,
    wplay_odds: list,
) -> str:
    """Compose the Telegram daily message: actionable picks first, summary at bottom."""
    today_str = datetime.now().strftime("%a %d %b").lower()
    header = (
        f"<b>🎯 Picks del día</b> · {today_str}\n"
        f"<i>Bankroll paper: ${bankroll:,.0f} COP</i>"
    )

    if not predictions:
        return f"{header}\n\nNo hay partidos próximos en los próximos 2 días."

    # If no value bets: short, encouraging
    if not picks:
        if not wplay_odds:
            tail = (
                "\n\n<i>⚠️ No pude leer cuotas de Wplay esta vez. "
                "Las predicciones quedaron logueadas, pero no hay análisis de value.</i>"
            )
        else:
            tail = (
                "\n\nMiré <b>{npred} partidos</b> en Premier y Liga BetPlay y "
                "<b>ninguna cuota de Wplay tiene value real</b> sobre el modelo. "
                "Las casas están bien calibradas hoy. Mejor no apostar."
                "\n\n<i>Esto es lo correcto cuando no hay edge — no apostar es ganar.</i>"
            ).format(npred=len(predictions))
        return header + tail

    # Otherwise: actionable picks
    parts = [header, ""]
    parts.append(f"<b>📍 {len(picks)} apuesta(s) recomendada(s):</b>")
    parts.append("")

    for i, vb in enumerate(picks, start=1):
        league_short = "Premier" if "Premier" in vb["league"] else "BetPlay"
        action, why = _humanize_pick(vb)
        # Find kickoff from predictions list
        kickoff_iso = next((p["kickoff"] for p in predictions if p["match_id"] == vb["match_id"]), "")
        when = _humanize_kickoff(kickoff_iso) if kickoff_iso else ""

        parts.append("━━━━━━━━━━━━━━━━━━━━")
        parts.append(f"<b>{i}. {vb['home_team']} vs {vb['away_team']}</b>")
        parts.append(f"<i>{league_short} · {when}</i>")
        parts.append("")
        parts.append(f"➤ Apostá: {action}")
        parts.append(f"💰 Cuota Wplay: <b>{vb['odds']:.2f}</b>")
        parts.append(f"💵 Stake: <b>${vb['recommended_stake']:,.0f} COP</b> <i>(¼ Kelly)</i>")
        parts.append("")
        parts.append(f"<i>🧠 {why}</i>")
        parts.append("")

    parts.append("━━━━━━━━━━━━━━━━━━━━")
    parts.append("")
    parts.append(
        "<i>⚠️ Modo paper trading. No apuestes plata real "
        "hasta tener 100+ picks con CLV positivo.</i>"
    )
    parts.append("")
    parts.append(
        f"<i>Analizados: {len(predictions)} partidos. "
        f"Filtros: edge entre 5% y 30%.</i>"
    )

    return "\n".join(parts)


async def main() -> None:
    logger.info("=== Daily pipeline starting ===")
    logger.info(f"Mode: {settings.betting_mode} | bankroll target: paper")

    # 1. Pull fixtures
    n_fixtures = await _pull_fixtures()
    logger.info(f"fixtures upserted: {n_fixtures}")

    # 2. Load models per league
    models_by_league = {slug: _load_models(slug) for slug in LEAGUES}

    # 3. Get upcoming matches
    upcoming = _load_upcoming_matches()
    logger.info(f"upcoming matches: {len(upcoming)}")

    # 4. Predict + persist
    predictions_summary: list[dict] = []
    for m in upcoming:
        slug = _league_slug_from_name(m["league_name"])
        models = models_by_league.get(slug, {})
        if not models:
            continue
        result = _ensemble_predict(models, m["home_team_id"], m["away_team_id"])
        if result is None:
            continue
        avg, _individual = result

        # Persist
        for market, sel, prob in [
            ("1x2", "home", avg.p_home_win),
            ("1x2", "draw", avg.p_draw),
            ("1x2", "away", avg.p_away_win),
            ("ou_2.5", "over", avg.p_over_2_5),
            ("ou_2.5", "under", avg.p_under_2_5),
            ("btts", "yes", avg.p_btts_yes),
            ("btts", "no", avg.p_btts_no),
        ]:
            _persist_prediction(
                m["id"], market, sel, "ensemble_avg", prob,
                features={"home_xG": avg.expected_home_goals, "away_xG": avg.expected_away_goals},
            )
        predictions_summary.append({
            "match_id": m["id"],
            "kickoff": m["kickoff_utc"],
            "league": m["league_name"],
            "home": m["home_name"],
            "away": m["away_name"],
            "p_home": avg.p_home_win,
            "p_draw": avg.p_draw,
            "p_away": avg.p_away_win,
            "p_over_2_5": avg.p_over_2_5,
            "p_btts": avg.p_btts_yes,
            "xG_h": avg.expected_home_goals,
            "xG_a": avg.expected_away_goals,
            "ensemble": avg,
        })

    logger.info(f"predictions persisted for {len(predictions_summary)} matches")

    # 5. Try Wplay odds (best effort)
    wplay_odds: list = []
    try:
        wplay_odds = await scrape_wplay_all()
        logger.info(f"Wplay odds captured: {len(wplay_odds)}")
    except Exception as exc:
        logger.warning(f"Wplay scrape failed: {exc}")

    # 6. Value detection - only if we have odds
    bankroll = get_current_bankroll("paper")
    picks_made: list = []
    if wplay_odds:
        # Build a lookup: (norm_home, norm_away) -> dict of selection -> odds
        odds_by_match: dict[tuple[str, str], dict[str, float]] = {}
        for o in wplay_odds:
            key = (normalize_name(o.home_team), normalize_name(o.away_team))
            odds_by_match.setdefault(key, {})[o.selection] = o.odds

        # For each prediction, look up the matching Wplay row
        for p in predictions_summary:
            key = (normalize_name(p["home"]), normalize_name(p["away"]))
            casa_odds = odds_by_match.get(key)
            if not casa_odds:
                # try swapping H/A in case ESPN/Wplay flipped them
                key_swap = (normalize_name(p["away"]), normalize_name(p["home"]))
                casa_odds = odds_by_match.get(key_swap)
                if casa_odds:
                    logger.warning(f"team order swapped between sources for {p['home']} vs {p['away']}; skipping")
                continue

            lines: list[OddsLine] = []
            for sel in ("home", "draw", "away"):
                if sel in casa_odds:
                    lines.append(OddsLine("1x2", sel, casa_odds[sel], bookmaker="wplay"))

            value_bets = detect_value(
                match_id=p["match_id"],
                home_team=p["home"],
                away_team=p["away"],
                league=p["league"],
                prediction=p["ensemble"],
                odds_lines=lines,
                bankroll=bankroll,
            )
            for vb in value_bets:
                pick_id = log_pick(vb, mode="paper")
                picks_made.append({"pick_id": pick_id, **vb.__dict__})
        logger.info(f"value bets logged: {len(picks_made)}")
    else:
        logger.info("no Wplay odds — skipping value detection. predictions logged only.")

    # 7. Compose Telegram summary
    msg = _format_telegram_message(predictions_summary, picks_made, bankroll, wplay_odds)

    await send_message(msg)
    logger.info("=== Daily pipeline done ===")


if __name__ == "__main__":
    asyncio.run(main())
