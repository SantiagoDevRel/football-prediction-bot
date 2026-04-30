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
    if not predictions_summary:
        msg = (
            f"<b>📊 Daily pipeline report</b>\n"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"No upcoming matches in the next 2 days for our leagues."
        )
    else:
        lines = [
            f"<b>📊 Daily picks ({datetime.now().strftime('%Y-%m-%d')})</b>",
            f"<i>Mode: {settings.betting_mode} | Bankroll: ${bankroll:,.0f}</i>",
            f"<i>{len(predictions_summary)} predictions | {len(picks_made)} value bets detected</i>",
            "",
        ]
        if picks_made:
            lines.append("<b>⭐ VALUE BETS</b>")
            for vb in picks_made:
                short_league = "EPL" if "Premier" in vb["league"] else "BetPlay"
                lines.append(
                    f"<b>[{short_league}] {vb['home_team']} vs {vb['away_team']}</b>\n"
                    f"  {vb['market']}:{vb['selection']} @ Wplay <b>{vb['odds']:.2f}</b>  "
                    f"(model: {vb['model_probability']:.0%}, fair: {vb['fair_odds']:.2f})\n"
                    f"  <b>edge +{vb['edge']*100:.1f}%</b> · stake ¼K: ${vb['recommended_stake']:,.0f}"
                )
            lines.append("")
        lines.append("<b>📋 All predictions</b>")
        for p in predictions_summary[:15]:
            kickoff = p["kickoff"][:16].replace("T", " ")
            league_short = "EPL" if "Premier" in p["league"] else "BetPlay"
            lines.append(
                f"<b>[{league_short} · {kickoff}]</b>\n"
                f"  {p['home']} vs {p['away']}\n"
                f"  <code>1X2: {p['p_home']:.0%} / {p['p_draw']:.0%} / {p['p_away']:.0%}</code>\n"
                f"  <code>O2.5: {p['p_over_2_5']:.0%}  BTTS: {p['p_btts']:.0%}  xG: {p['xG_h']:.1f}-{p['xG_a']:.1f}</code>"
            )
        if len(predictions_summary) > 15:
            lines.append(f"\n... +{len(predictions_summary) - 15} more")
        msg = "\n".join(lines)

    await send_message(msg)
    logger.info("=== Daily pipeline done ===")


if __name__ == "__main__":
    asyncio.run(main())
