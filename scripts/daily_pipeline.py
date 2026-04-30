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
    """Simple average of available models across all output markets."""
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
    fields = [
        "p_home_win", "p_draw", "p_away_win",
        "p_over_1_5", "p_under_1_5",
        "p_over_2_5", "p_under_2_5",
        "p_over_3_5", "p_under_3_5",
        "p_btts_yes", "p_btts_no",
        "p_home_minus_1_5", "p_away_plus_1_5",
        "expected_home_goals", "expected_away_goals",
    ]
    avg_kwargs = {f: sum(getattr(p, f) for p in probs) / n for f in fields}
    avg = type(probs[0])(features={"models_used": list(models.keys())}, **avg_kwargs)
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


def _format_pick_block(vb: dict, predictions: list, idx: int) -> list[str]:
    """Render one pick as a block of lines."""
    league_short = "Premier" if "Premier" in vb["league"] else "BetPlay"
    action, why = _humanize_pick(vb)
    kickoff_iso = next(
        (p["kickoff"] for p in predictions if p["match_id"] == vb["match_id"]), ""
    )
    when = _humanize_kickoff(kickoff_iso) if kickoff_iso else ""

    return [
        f"<b>#{idx} · {vb['home_team']} vs {vb['away_team']}</b>",
        f"<i>{league_short} · {when}</i>",
        "",
        f"➤ Apostá: {action}",
        f"💰 Cuota Wplay: <b>{vb['odds']:.2f}</b>",
        f"💵 Stake: <b>${vb['recommended_stake']:,.0f} COP</b> <i>(¼ Kelly)</i>",
        "",
        f"<i>🧠 {why}</i>",
        "",
    ]


def _format_telegram_message(
    predictions: list,
    picks: list,
    bankroll: float,
    wplay_odds: list,
) -> str:
    """Compose the Telegram message: split picks into safe (low odds) vs risky (high odds)."""
    today_str = datetime.now().strftime("%a %d %b").lower()
    header_lines = [
        f"<b>🎯 Picks del día</b> · {today_str}",
        f"<i>Bankroll paper: ${bankroll:,.0f} COP</i>",
    ]

    if not predictions:
        return "\n".join(header_lines + ["", "No hay partidos próximos en los siguientes 2 días."])

    # Bucket picks by odds level
    safe_threshold = 2.00
    safe_picks = sorted([p for p in picks if p["odds"] < safe_threshold],
                        key=lambda p: -p["edge"])[:5]
    risky_picks = sorted([p for p in picks if p["odds"] >= safe_threshold],
                         key=lambda p: -p["edge"])[:5]

    parts = list(header_lines) + [""]

    # Header with totals
    if not picks:
        if not wplay_odds:
            parts.extend([
                "<i>⚠️ No pude leer cuotas de Wplay esta vez. Las predicciones quedaron en DB pero sin análisis de value.</i>"
            ])
        else:
            parts.extend([
                f"Miré <b>{len(predictions)} partidos</b> en Premier y BetPlay.",
                "<b>Ninguna cuota de Wplay tiene value sobre el modelo hoy.</b>",
                "Las casas están bien calibradas — mejor no apostar.",
                "",
                "<i>No apostar cuando no hay edge también es ganar.</i>",
            ])
        return "\n".join(parts)

    # Safe section
    parts.append("━━━━━━━━━━━━━━━━━━━━")
    parts.append("<b>💪 CUOTAS SEGURAS</b> <i>(cuota &lt; 2.00)</i>")
    parts.append("━━━━━━━━━━━━━━━━━━━━")
    parts.append("")
    if safe_picks:
        for i, vb in enumerate(safe_picks, start=1):
            parts.extend(_format_pick_block(vb, predictions, i))
    else:
        parts.append("<i>Hoy no hay value en cuotas bajas. El mercado está bien calibrado en favoritos — la casa y el modelo coinciden.</i>")
        parts.append("")

    # Risky section
    parts.append("━━━━━━━━━━━━━━━━━━━━")
    parts.append("<b>🎲 CUOTAS CON VALOR PERO RIESGO</b> <i>(cuota ≥ 2.00)</i>")
    parts.append("━━━━━━━━━━━━━━━━━━━━")
    parts.append("")
    if risky_picks:
        start_idx = len(safe_picks) + 1
        for offset, vb in enumerate(risky_picks):
            parts.extend(_format_pick_block(vb, predictions, start_idx + offset))
    else:
        parts.append("<i>Hoy no hay value en cuotas altas con edge razonable.</i>")
        parts.append("")

    # Footer
    parts.append("━━━━━━━━━━━━━━━━━━━━")
    parts.append("")
    parts.append(
        "<i>⚠️ Modo paper. NO apuestes plata real "
        "hasta tener 100+ picks con CLV positivo.</i>"
    )
    parts.append("")
    parts.append(
        f"<i>Analizados: {len(predictions)} partidos · "
        f"Filtros: edge 5%–30% · Stake: ¼ Kelly · Cap 5%/bankroll</i>"
    )

    return "\n".join(parts)


async def run_pipeline_core(*, persist_predictions_flag: bool = True) -> dict:
    """Shared pipeline used by both the daily cron and the Telegram bot.

    Does NOT auto-log picks. Returns the candidates so the caller decides:
        - cron mode → log all candidates as paper picks
        - bot mode  → stage candidates per chat_id, user confirms with /aposte

    Returns dict with:
        predictions       (list of dicts, one per match)
        value_bets        (list of dicts; ValueBet __dict__ flattened)
        bankroll_paper    (float)
        wplay_odds_count  (int)
        n_fixtures        (int)
    """
    n_fixtures = await _pull_fixtures()
    models_by_league = {slug: _load_models(slug) for slug in LEAGUES}
    upcoming = _load_upcoming_matches()

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

        if persist_predictions_flag:
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
                    features={"home_xG": avg.expected_home_goals,
                              "away_xG": avg.expected_away_goals},
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

    wplay_odds: list = []
    try:
        wplay_odds = await scrape_wplay_all()
    except Exception as exc:
        logger.warning(f"Wplay scrape failed: {exc}")

    bankroll = get_current_bankroll("paper")
    candidates: list[dict] = []
    if wplay_odds:
        odds_by_match: dict[tuple[str, str], dict[str, float]] = {}
        for o in wplay_odds:
            key = (normalize_name(o.home_team), normalize_name(o.away_team))
            odds_by_match.setdefault(key, {})[o.selection] = o.odds

        for p in predictions_summary:
            key = (normalize_name(p["home"]), normalize_name(p["away"]))
            casa = odds_by_match.get(key)
            if not casa:
                continue
            lines = [
                OddsLine("1x2", sel, casa[sel], bookmaker="wplay")
                for sel in ("home", "draw", "away") if sel in casa
            ]
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
                vb_dict = vb.__dict__.copy()
                vb_dict["kickoff"] = p["kickoff"]
                candidates.append(vb_dict)

    return {
        "predictions": predictions_summary,
        "value_bets": candidates,
        "bankroll_paper": bankroll,
        "wplay_odds_count": len(wplay_odds),
        "n_fixtures": n_fixtures,
    }


async def main() -> None:
    """Cron entry point: runs core pipeline, auto-logs picks, sends Telegram summary."""
    logger.info("=== Daily pipeline starting ===")
    logger.info(f"Mode: {settings.betting_mode} | bankroll target: paper")

    result = await run_pipeline_core(persist_predictions_flag=True)
    logger.info(
        f"fixtures={result['n_fixtures']} predictions={len(result['predictions'])} "
        f"wplay_odds={result['wplay_odds_count']} value_candidates={len(result['value_bets'])}"
    )

    # Cron auto-logs all candidates as paper picks
    picks_made: list[dict] = []
    for vb_dict in result["value_bets"]:
        from src.betting.value_detector import ValueBet
        # Reconstruct ValueBet from flattened dict (drop kickoff which we added)
        vb_kwargs = {k: v for k, v in vb_dict.items() if k != "kickoff"}
        vb = ValueBet(**vb_kwargs)
        pick_id = log_pick(vb, mode="paper")
        picks_made.append({"pick_id": pick_id, **vb_dict})

    logger.info(f"value bets logged: {len(picks_made)}")

    # 7. Compose Telegram summary
    msg = _format_telegram_message(
        result["predictions"], picks_made, result["bankroll_paper"], [None] * result["wplay_odds_count"]
    )
    await send_message(msg)
    logger.info("=== Daily pipeline done ===")


if __name__ == "__main__":
    asyncio.run(main())
