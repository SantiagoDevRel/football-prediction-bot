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
from src.data.odds_api import fetch_multi_bookie_odds  # noqa: E402
from src.data.wplay_scraper import (  # noqa: E402
    WplayOdds,
    normalize_name,
    scrape_all_with_markets,
)
from src.notifications.telegram_bot import send_message  # noqa: E402
from src.llm.pick_reviewer import PickReviewer  # noqa: E402
from src.tracking.auto_resolver import auto_resolve_paper_picks  # noqa: E402
from src.tracking.pick_logger import get_current_bankroll, log_pick  # noqa: E402

LEAGUES = ["premier_league", "liga_betplay", "sudamericana", "libertadores", "champions_league"]
# Live-only leagues: surfaced in /envivo so the user sees Colombian matches
# outside Primera A (Primera B, Copa Colombia). No models trained → no picks.
LIVE_ONLY_LEAGUES = ["primera_b", "copa_colombia"]
LEAGUE_NAME = {
    "premier_league": "Premier League",
    "liga_betplay": "Liga BetPlay Dimayor",
    "primera_b": "Primera B Colombia",
    "copa_colombia": "Copa Colombia",
    "sudamericana": "Copa Sudamericana",
    "libertadores": "Copa Libertadores",
    "champions_league": "UEFA Champions League",
}
ARTIFACTS = ROOT / "models_artifacts"


def _model_path(model_name: str, league_slug: str) -> Path:
    safe = LEAGUE_NAME[league_slug].lower().replace(" ", "_").replace(".", "")
    return ARTIFACTS / f"{model_name}_{safe}.pkl"


def _load_models(league_slug: str) -> dict:
    """Load all available models for a league.

    Preference order at predict time:
        1. Stacking (DC + Elo + XGBoost calibrated) — used by _ensemble_predict
        2. If stacking missing, fall back to simple avg of DC + Elo
    """
    out = {}
    for name in ("dixon_coles", "elo", "xgboost", "stacking"):
        p = _model_path(name, league_slug)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    out[name] = pickle.load(f)
            except Exception as exc:
                logger.warning(f"failed to load {name} for {league_slug}: {exc}")
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


def _pull_window_days(slug: str) -> int:
    """How far ahead to fetch fixtures per league. CONMEBOL competitions
    (Libertadores / Sudamericana) and UEFA Champions are weekly midweek
    games — we need 5+ days ahead so they show up when the user asks Sat/Sun
    about Tue/Wed matches. Domestic leagues (Premier, BetPlay) play every
    weekend so 2 days is enough."""
    if slug in ("champions_league", "libertadores", "sudamericana"):
        return 7
    return 2


async def _pull_fixtures() -> int:
    today = date.today()
    total = 0
    for slug in LEAGUES:
        end = today + timedelta(days=_pull_window_days(slug))
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
    """Return matches kicking off today or in the next 7 days (max window
    across all leagues — Champions / Libertadores / Sudamericana have weekly
    midweek matches that need to be visible from the weekend)."""
    today = date.today()
    end = today + timedelta(days=7)
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
    if "Primera B" in name:
        return "primera_b"
    if "Copa Colombia" in name:
        return "copa_colombia"
    if "Sudamericana" in name:
        return "sudamericana"
    if "Libertadores" in name:
        return "libertadores"
    if "Champions" in name:
        return "champions_league"
    return "liga_betplay"


_CARDS_CORNERS_MODELS: dict[str, dict] = {}


def _get_cards_corners_models(db_path: str) -> dict:
    """Cache the cards + corners models in-process so we don't re-fit per match."""
    if _CARDS_CORNERS_MODELS:
        return _CARDS_CORNERS_MODELS
    try:
        from src.models.cards_corners import CardsOrCornersModel
        cm = CardsOrCornersModel(db_path, kind="cards")
        cm.fit(None)
        co = CardsOrCornersModel(db_path, kind="corners")
        co.fit(None)
        _CARDS_CORNERS_MODELS["cards"] = cm
        _CARDS_CORNERS_MODELS["corners"] = co
    except Exception as exc:
        logger.warning(f"cards/corners models init failed: {exc}")
    return _CARDS_CORNERS_MODELS


def _attach_cards_corners(prediction, home_id: int, away_id: int) -> None:
    """Mutate prediction.features to include over/under prob for cards & corners
    half-lines, so value_detector picks them up via market='cards_X.5' / 'corners_X.5'."""
    cc = _get_cards_corners_models(str(settings.db_path))
    for kind, model in cc.items():
        try:
            lam = model.expected_total(home_id, away_id)
            lines = model.predict_over_lines(home_id, away_id)
        except Exception:
            continue
        prediction.features[f"{kind}_lambda"] = lam
        for line, p_over in lines.items():
            prediction.features[f"{kind}_{line}_over"] = p_over


def _ensemble_predict(models: dict, home_id: int, away_id: int, match_id: int | None = None):
    """If a stacking model is available, use it. Otherwise fall back to a
    simple average of Dixon-Coles + Elo. Always attaches cards/corners
    predictions to features dict."""
    p = None
    if "stacking" in models:
        try:
            p = models["stacking"].predict_match(home_id, away_id, match_id=match_id)
        except KeyError:
            return None
        except Exception as exc:
            logger.warning(f"stacking predict failed, falling back: {exc}")

    if p is not None:
        _attach_cards_corners(p, home_id, away_id)
        return p, [p]

    # Fallback: average of available non-stacking models
    fallback_models = {k: v for k, v in models.items() if k in ("dixon_coles", "elo")}
    probs = []
    for name, m in fallback_models.items():
        try:
            p_one = m.predict_match(home_id, away_id)
            probs.append(p_one)
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
    avg = type(probs[0])(features={"models_used": list(fallback_models.keys())}, **avg_kwargs)
    _attach_cards_corners(avg, home_id, away_id)
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
    """Render one pick as a block of lines, including Claude's verdict if any."""
    if "Premier" in vb["league"]:
        league_short = "Premier"
    elif "BetPlay" in vb["league"]:
        league_short = "BetPlay"
    elif "Libertadores" in vb["league"]:
        league_short = "Libertadores"
    elif "Sudamericana" in vb["league"]:
        league_short = "Sudamericana"
    elif "Champions" in vb["league"]:
        league_short = "Champions"
    else:
        league_short = vb["league"][:12]

    action, why = _humanize_pick(vb)
    kickoff_iso = next(
        (p["kickoff"] for p in predictions if p["match_id"] == vb["match_id"]), ""
    )
    when = _humanize_kickoff(kickoff_iso) if kickoff_iso else ""

    block = [
        f"<b>#{idx} · {vb['home_team']} vs {vb['away_team']}</b>",
        f"<i>{league_short} · {when}</i>",
        "",
        f"➤ Apostá: {action}",
        f"💰 Cuota: <b>{vb['odds']:.2f}</b>",
        f"💵 Stake: <b>${vb['recommended_stake']:,.0f} COP</b> <i>(¼ Kelly)</i>",
        "",
        f"<i>🧠 {why}</i>",
    ]
    # Claude's verdict (if reviewer ran)
    verdict = vb.get("claude_verdict")
    reasoning = vb.get("claude_reasoning")
    if verdict:
        icon = {"take": "✅", "reduce": "⚠️", "skip": "🛑"}.get(verdict, "❓")
        verdict_es = {"take": "ok", "reduce": "stake reducido", "skip": "skip"}.get(verdict, verdict)
        if reasoning:
            block.append(f"<i>{icon} Claude ({verdict_es}): {reasoning}</i>")
        else:
            block.append(f"<i>{icon} Claude: {verdict_es}</i>")
    block.append("")
    return block


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
                "<i>⚠️ No pude leer cuotas de Wplay esta vez. Las predicciones "
                "quedaron en DB pero sin análisis de value. Revisá "
                "<code>logs/wplay_debug/</code> si sigue fallando.</i>"
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
        result = _ensemble_predict(models, m["home_team_id"], m["away_team_id"], match_id=m["id"])
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
        wplay_odds = await scrape_all_with_markets()
    except Exception as exc:
        logger.warning(f"Wplay multi-market scrape failed: {exc}")

    # Multi-bookmaker odds (optional, requires ODDS_API_KEY).
    # We merge into the same odds_by_match index, keeping the BEST price
    # available per (market, selection) across all sources.
    odds_api_data: list = []
    if settings.odds_api_key:
        for slug in LEAGUES:
            try:
                rows = await fetch_multi_bookie_odds(slug, settings.odds_api_key)
                odds_api_data.extend(rows)
            except Exception as exc:
                logger.warning(f"odds-api {slug} failed: {exc}")
        logger.info(f"odds-api: {len(odds_api_data)} best-price rows")

    bankroll = get_current_bankroll("paper")
    candidates: list[dict] = []
    if wplay_odds or odds_api_data:
        # Index by (norm_home, norm_away) -> { (market, selection): (odds, source) }
        # We keep the BEST price across all sources.
        odds_by_match: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        odds_source: dict[tuple[str, str], dict[tuple[str, str], str]] = {}

        def _add(key, market_sel, price, source):
            existing = odds_by_match.setdefault(key, {})
            cur = existing.get(market_sel)
            if cur is None or price > cur:
                existing[market_sel] = price
                odds_source.setdefault(key, {})[market_sel] = source

        for o in wplay_odds:
            key = (normalize_name(o.home_team), normalize_name(o.away_team))
            _add(key, (o.market, o.selection), o.odds, "wplay")
        for o in odds_api_data:
            key = (normalize_name(o.home_team), normalize_name(o.away_team))
            _add(key, (o.market, o.selection), o.best_odds, o.best_bookmaker)

        for p in predictions_summary:
            key = (normalize_name(p["home"]), normalize_name(p["away"]))
            casa = odds_by_match.get(key)
            if not casa:
                continue

            sources_for_match = odds_source.get(key, {})
            lines: list[OddsLine] = [
                OddsLine(market, selection, odds,
                         bookmaker=sources_for_match.get((market, selection), "wplay"))
                for (market, selection), odds in casa.items()
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

    # Claude review pass: each candidate gets a verdict (take/reduce/skip) +
    # short reasoning. Reduces stake to 50% on 'reduce', drops 'skip' picks.
    if candidates and settings.anthropic_api_key:
        reviewer = PickReviewer(settings.anthropic_api_key)
        kept: list[dict] = []
        for c in candidates:
            review = await reviewer.review(c)
            c["claude_verdict"] = review.verdict
            c["claude_reasoning"] = review.reasoning
            c["claude_confidence"] = review.confidence
            if review.verdict == "skip":
                logger.info(f"  Claude SKIP: {c['home_team']} v {c['away_team']} "
                            f"{c['market']}:{c['selection']} — {review.reasoning[:120]}")
                continue
            if review.verdict == "reduce":
                c["recommended_stake"] = c["recommended_stake"] * 0.5
            kept.append(c)
        logger.info(f"Claude review: kept {len(kept)} of {len(candidates)} candidates")
        candidates = kept

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

    # Step 0: auto-resolve any picks whose match has finished since last run.
    # Free side-effect — ensures bankroll is up-to-date before logging new picks.
    try:
        resolved = await auto_resolve_paper_picks()
        if resolved:
            logger.info(f"auto-resolved {len(resolved)} picks before today's pipeline")
    except Exception as exc:
        logger.warning(f"auto-resolver failed: {exc}")

    result = await run_pipeline_core(persist_predictions_flag=True)
    logger.info(
        f"fixtures={result['n_fixtures']} predictions={len(result['predictions'])} "
        f"wplay_odds={result['wplay_odds_count']} value_candidates={len(result['value_bets'])}"
    )

    # Cron auto-logs all candidates as paper picks (skip ones rejected by risk mgmt)
    picks_made: list[dict] = []
    rejected: list[str] = []
    for vb_dict in result["value_bets"]:
        from src.betting.value_detector import ValueBet
        # Strip enrichment fields that aren't ValueBet constructor args
        excluded = {"kickoff", "claude_verdict", "claude_reasoning", "claude_confidence"}
        vb_kwargs = {k: v for k, v in vb_dict.items() if k not in excluded}
        vb = ValueBet(**vb_kwargs)
        try:
            pick_id = log_pick(vb, mode="paper")
            picks_made.append({"pick_id": pick_id, **vb_dict})
        except ValueError as exc:
            rejected.append(f"{vb.home_team} v {vb.away_team} {vb.market}:{vb.selection} ({exc})")

    logger.info(f"value bets logged: {len(picks_made)} (rejected by risk: {len(rejected)})")
    for r in rejected:
        logger.info(f"  rejected: {r}")

    # 7. Compose Telegram summary
    msg = _format_telegram_message(
        result["predictions"], picks_made, result["bankroll_paper"], [None] * result["wplay_odds_count"]
    )
    await send_message(msg)
    logger.info("=== Daily pipeline done ===")


if __name__ == "__main__":
    asyncio.run(main())
