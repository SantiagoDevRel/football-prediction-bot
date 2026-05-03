"""Command handlers for the Telegram bot."""
from __future__ import annotations

import sqlite3
import unicodedata
from datetime import datetime, date, timedelta

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.betting.kelly import edge as edge_fn, kelly_stake
from src.betting.value_detector import ValueBet
from src.config import settings
from src.data.persist import get_conn
from src.telegram_app.staging import StagedPick, get_staged, stage_picks
from src.tracking.pick_logger import (
    compute_rolling_metrics,
    get_current_bankroll,
    log_pick,
    resolve_pick,
)


# Map slug -> human league name (mirror of daily_pipeline.LEAGUE_NAME, kept
# here to avoid circular imports). Used by NLU filtering.
SLUG_TO_LEAGUE_NAME = {
    "premier_league": "Premier League",
    "liga_betplay": "Liga BetPlay Dimayor",
    "sudamericana": "Copa Sudamericana",
    "libertadores": "Copa Libertadores",
    "champions_league": "UEFA Champions League",
}


def _filter_value_bets(
    value_bets: list[dict],
    leagues: list[str] | None,
    time_window: str,
) -> list[dict]:
    """Filter value-bet candidates by league slug(s) + time window.

    leagues: empty/None = no filter. time_window: 'today'|'tomorrow'|'weekend'|'week'|'any'.
    Filters defensively: invalid kickoff strings are kept (don't drop unknown).
    """
    out = list(value_bets)

    if leagues:
        target_names = {SLUG_TO_LEAGUE_NAME.get(s, s) for s in leagues}
        out = [v for v in out if v.get("league") in target_names]

    if time_window in ("today", "tomorrow", "weekend", "week"):
        today = date.today()
        if time_window == "today":
            ok = lambda d: d == today
        elif time_window == "tomorrow":
            tom = today + timedelta(days=1)
            ok = lambda d: d == tom
        elif time_window == "weekend":
            # Closest upcoming Sat/Sun pair (if today is in the weekend, current one)
            wd = today.weekday()  # Mon=0..Sun=6
            sat = today + timedelta(days=(5 - wd) % 7)
            if wd >= 5:  # already weekend
                sat = today - timedelta(days=wd - 5)
            sun = sat + timedelta(days=1)
            ok = lambda d: d in (sat, sun)
        else:  # week
            end = today + timedelta(days=7)
            ok = lambda d: today <= d <= end

        def _kickoff_date(v: dict) -> date | None:
            ko = v.get("kickoff") or ""
            try:
                return datetime.fromisoformat(ko).date()
            except (ValueError, TypeError):
                return None

        out = [v for v in out if (d := _kickoff_date(v)) is not None and ok(d)]

    return out


# ---------- Helpers ----------

_DAYS_ES = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
_MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _humanize_kickoff(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso[:16].replace("T", " ")
    today = date.today()
    diff = (dt.date() - today).days
    if diff == 0:
        when = "HOY"
    elif diff == 1:
        when = "MAÑANA"
    else:
        when = _DAYS_ES[dt.weekday()]
    fecha = f"{dt.day:02d} {_MONTHS_ES[dt.month - 1]}"
    return f"{when} {dt.strftime('%H:%M')} ({fecha})"


def _strip_accents(s: str) -> str:
    """Lowercase + strip diacritics. SQLite's LOWER doesn't do diacritics, so
    we have to do this in Python before comparing user query to team names."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


# Common nicknames / city names → substring of team name in the DB.
# Ad-hoc, expand as we find more user phrasings.
_TEAM_ALIASES: dict[str, str] = {
    "rionegro": "aguilas doradas",
    "verdolaga": "atletico nacional",
    "millos": "millonarios",
    "albirrojo": "junior",
    "tiburones": "junior",
    "leones": "santa fe",
    "dim": "independiente medellin",
    "poderoso": "independiente medellin",
}


def _find_team_candidates(query_raw: str) -> list[int]:
    """Return team ids whose normalized name contains the (normalized) query
    OR an aliased form. Done in Python because SQLite doesn't strip accents."""
    q = _strip_accents(query_raw)
    if not q:
        return []
    # Apply aliases (substring match): if any alias key appears in the query,
    # also search the alias's mapped form.
    expanded = [q]
    for alias_key, alias_target in _TEAM_ALIASES.items():
        if alias_key in q:
            expanded.append(alias_target)

    with get_conn() as conn:
        rows = conn.execute("SELECT id, name FROM teams").fetchall()
    out: list[int] = []
    for r in rows:
        n = _strip_accents(r["name"])
        for term in expanded:
            if term in n or n in term:
                out.append(r["id"])
                break
    return out


def _short_league_name(full_name: str) -> str:
    """Map full league name to a short label for compact UI rendering."""
    n = (full_name or "")
    if "Premier" in n:
        return "Premier"
    if "Primera B" in n:
        return "Primera B"
    if "Copa Colombia" in n:
        return "Copa Colombia"
    if "BetPlay" in n or "Dimayor" in n:
        return "BetPlay"
    if "Sudamericana" in n:
        return "Sudamericana"
    if "Libertadores" in n:
        return "Libertadores"
    if "Champions" in n:
        return "Champions"
    return n[:14] or "?"


def _humanize_action(market: str, selection: str, home: str, away: str) -> str:
    if market == "1x2":
        if selection == "home":
            return f"Gana {home}"
        if selection == "away":
            return f"Gana {away} (visitante)"
        if selection == "draw":
            return "Empate (X)"
    if market == "ou_1.5":
        return "Más de 1.5 goles" if selection == "over" else "Menos de 1.5 goles"
    if market == "ou_2.5":
        return "Más de 2.5 goles" if selection == "over" else "Menos de 2.5 goles"
    if market == "ou_3.5":
        return "Más de 3.5 goles" if selection == "over" else "Menos de 3.5 goles"
    if market == "btts":
        return "Ambos equipos marcan" if selection == "yes" else "NO marcan los dos"
    if market == "ah_-1.5":
        return f"{home} gana por 2+ goles" if selection == "home" else f"{away} no pierde por 2+"
    return f"{market}:{selection}"


# ---------- /start ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    bankroll = get_current_bankroll("paper")
    msg = (
        "<b>🎯 Football Prediction Bot</b>\n"
        f"<i>Paper trading · Bankroll: ${bankroll:,.0f} COP</i>\n\n"
        "<b>Comandos:</b>\n"
        "/picks — analizar partidos próximos\n"
        "/envivo — partidos en vivo (con disclaimer)\n"
        "/analizar &lt;eq1&gt; &lt;eq2&gt; — analizar UN partido específico\n"
        "/aposte &lt;n&gt; [stake] — registrar apuesta\n"
        "/resolver_auto — resolver picks ya finalizadas\n"
        "/resolver &lt;pick_id&gt; ganada|perdida — manual\n"
        "/balance — bankroll y picks abiertas\n"
        "/historial — últimas resueltas\n"
        "/help — ver esta ayuda\n\n"
        "<b>💬 También entiendo lenguaje natural:</b>\n"
        "<i>“dame el top pick de hoy en betplay”\n"
        "“qué hay en vivo”\n"
        "“analiza nacional vs millonarios”\n"
        "“cómo va mi balance”</i>\n\n"
        "<i>Apretá un botón, escribí un comando, o dime qué querés.</i>"
    )
    keyboard = [
        [InlineKeyboardButton("📊 Picks de hoy", callback_data="cmd:picks")],
        [InlineKeyboardButton("💰 Mi balance", callback_data="cmd:balance"),
         InlineKeyboardButton("📜 Historial", callback_data="cmd:historial")],
    ]
    await update.message.reply_text(
        msg, parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


# ---------- /picks ----------

async def cmd_picks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Default /picks: all leagues, all upcoming, full list."""
    await _run_and_send_picks(update, context, leagues=[], time_window="any", top_only=False)


async def _run_and_send_picks(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    leagues: list[str],
    time_window: str,
    top_only: bool,
    prefix_note: str | None = None,
) -> None:
    """Shared core for /picks and NLU-dispatched picks. Filters + stages + sends."""
    chat_id = str(update.effective_chat.id)
    filter_desc = ""
    if leagues:
        nice = [SLUG_TO_LEAGUE_NAME.get(s, s) for s in leagues]
        filter_desc += f" · {', '.join(nice)}"
    if time_window != "any":
        es = {"today": "hoy", "tomorrow": "mañana", "weekend": "finde", "week": "esta semana"}.get(time_window, time_window)
        filter_desc += f" · {es}"
    if top_only:
        filter_desc += " · TOP 1"

    wait_msg = (
        f"⏳ Analizando partidos y cuotas (1X2 + BTTS + O/U) en Wplay{filter_desc}…\n"
        "<i>Esto tarda ~1 minuto porque visito cada match page para sacar todos los mercados.</i>"
    )
    if update.message:
        await update.message.reply_text(wait_msg, parse_mode=ParseMode.HTML)
    else:
        await update.callback_query.message.reply_text(wait_msg, parse_mode=ParseMode.HTML)

    # Import here to avoid heavy import at module load
    from scripts.daily_pipeline import run_pipeline_core, LEAGUES as ACTIVE_LEAGUES

    # Short-circuit: if user filtered ONLY by leagues not yet in the pipeline,
    # don't waste 60s running the full scrape.
    if leagues and all(s not in ACTIVE_LEAGUES for s in leagues):
        nice = ", ".join(SLUG_TO_LEAGUE_NAME.get(s, s) for s in leagues)
        await context.bot.send_message(
            chat_id=chat_id,
            text=(f"<b>⚠️ {nice} todavía no está activa en el pipeline.</b>\n"
                  f"<i>Estoy trabajando en ella — vendrá pronto. "
                  f"Mientras tanto preguntame por Premier, BetPlay, Sudamericana o Libertadores.</i>"),
            parse_mode=ParseMode.HTML,
        )
        return

    # Warn (but still run) if user filtered by a mix that includes inactive ones
    if leagues:
        missing = [s for s in leagues if s not in ACTIVE_LEAGUES]
        if missing:
            nice_missing = ", ".join(SLUG_TO_LEAGUE_NAME.get(s, s) for s in missing)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"<i>⚠️ {nice_missing}: todavía no está activa en el pipeline. Filtro las demás.</i>",
                parse_mode=ParseMode.HTML,
            )

    try:
        result = await run_pipeline_core(persist_predictions_flag=True)
    except Exception as exc:
        logger.exception("pipeline failed")
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error corriendo pipeline: {exc}")
        return

    candidates = _filter_value_bets(result["value_bets"], leagues, time_window)
    if top_only and candidates:
        candidates = sorted(candidates, key=lambda v: -v["edge"])[:1]
    bankroll = result["bankroll_paper"]
    n_pred = len(result["predictions"])

    # Stage candidates per chat (numbered 1..N)
    staged = stage_picks(chat_id, candidates)

    if not staged:
        if result["wplay_odds_count"] == 0:
            text = (
                "⚠️ <b>No pude leer cuotas de Wplay esta vez.</b>\n"
                "Las predicciones quedaron en DB pero no hay análisis de value.\n\n"
                "<i>Probá de nuevo en unos minutos. Si sigue fallando, revisá "
                "<code>logs/wplay_debug/</code> — el scraper guarda HTML "
                "y screenshot de cada intento para diagnóstico.</i>"
            )
        else:
            text = (
                f"📊 Analicé <b>{n_pred} partidos</b> próximos.\n\n"
                "<b>Ninguna cuota tiene value sobre el modelo hoy.</b>\n"
                "Las casas están bien calibradas — mejor no apostar.\n\n"
                "<i>No apostar cuando no hay edge también es ganar.</i>"
            )
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
        return

    # Group by safe (odds<2) vs risky (>=2) and produce numbered list
    safe = [s for s in staged if s.odds < 2.0]
    risky = [s for s in staged if s.odds >= 2.0]

    parts: list[str] = [
        f"<b>🎯 Picks del día</b>",
        f"<i>Bankroll: ${bankroll:,.0f} · {n_pred} partidos analizados</i>",
        "",
    ]

    def render_pick(s: StagedPick) -> list[str]:
        league_short = _short_league_name(s.league)
        action = _humanize_action(s.market, s.selection, s.home_team, s.away_team)
        casa_implied = 1.0 / s.odds
        when = _humanize_kickoff(s.kickoff_utc)
        return [
            f"<b>#{s.session_number}</b> · {s.home_team} vs {s.away_team}",
            f"<i>{league_short} · {when}</i>",
            f"➤ <b>{action}</b>",
            f"💰 Cuota: <b>{s.odds:.2f}</b>  ·  Stake sugerido: <b>${s.recommended_stake:,.0f}</b>",
            f"<i>🧠 Modelo: {s.model_probability:.0%} · Wplay: {casa_implied:.0%} · Edge: +{s.edge*100:.0f}%</i>",
            "",
        ]

    if safe:
        parts.append("━━━━━━━━━━━━━━━━━━━")
        parts.append("<b>💪 CUOTAS SEGURAS</b> <i>(&lt; 2.00)</i>")
        parts.append("━━━━━━━━━━━━━━━━━━━")
        parts.append("")
        for s in safe:
            parts.extend(render_pick(s))
    if risky:
        parts.append("━━━━━━━━━━━━━━━━━━━")
        parts.append("<b>🎲 CUOTAS CON VALOR</b> <i>(≥ 2.00)</i>")
        parts.append("━━━━━━━━━━━━━━━━━━━")
        parts.append("")
        for s in risky:
            parts.extend(render_pick(s))

    parts.append("━━━━━━━━━━━━━━━━━━━")
    parts.append(
        "<i>Para apostar:</i> <code>/aposte 3 5000</code> "
        "<i>(reemplazá 3 con el número y 5000 con la plata).</i>\n"
        "<i>Para usar el stake sugerido:</i> <code>/aposte 3</code>"
    )

    msg = "\n".join(parts)
    # Telegram has a 4096 char limit — slice if needed
    if len(msg) > 4000:
        msg = msg[:3990] + "\n…<i>(truncado)</i>"
    await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.HTML)


# ---------- /aposte ----------

async def cmd_aposte(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    args = context.args
    if not args:
        await update.message.reply_text(
            "Uso: <code>/aposte &lt;n&gt; [stake]</code>\n"
            "Ejemplo: <code>/aposte 3 5000</code>\n"
            "<i>Si no pasás stake, uso el sugerido (¼ Kelly).</i>",
            parse_mode=ParseMode.HTML,
        )
        return
    try:
        n = int(args[0])
    except ValueError:
        await update.message.reply_text(f"'{args[0]}' no es un número válido.")
        return

    s = get_staged(chat_id, n)
    if s is None:
        await update.message.reply_text(
            f"No encontré la pick #{n}. Corré /picks primero para ver las opciones."
        )
        return

    # Determine stake
    stake = s.recommended_stake
    if len(args) >= 2:
        try:
            stake = float(args[1])
        except ValueError:
            await update.message.reply_text(f"'{args[1]}' no es un número válido para stake.")
            return
        if stake <= 0:
            await update.message.reply_text("El stake tiene que ser positivo.")
            return

    # Build a ValueBet with the user's chosen stake and log it
    vb = ValueBet(
        match_id=s.match_id, home_team=s.home_team, away_team=s.away_team,
        league=s.league, market=s.market, selection=s.selection, odds=s.odds,
        bookmaker=s.bookmaker, model_probability=s.model_probability,
        fair_odds=s.fair_odds, edge=s.edge, confidence=s.confidence,
        recommended_stake=stake,
        reasoning=s.reasoning,
    )
    try:
        pick_id = log_pick(vb, mode="paper")
    except ValueError as exc:
        await update.message.reply_text(
            f"⚠️ <b>Apuesta rechazada por gestión de riesgo</b>\n\n"
            f"<i>{exc}</i>\n\n"
            f"Para forzar (no recomendado), ajustá los límites en "
            f"<code>src/betting/risk_manager.py</code>.",
            parse_mode=ParseMode.HTML,
        )
        return
    bankroll = get_current_bankroll("paper")
    action = _humanize_action(s.market, s.selection, s.home_team, s.away_team)
    msg = (
        f"✅ <b>Apuesta registrada</b>\n\n"
        f"<b>Pick #{pick_id}</b> · {s.home_team} vs {s.away_team}\n"
        f"➤ {action}\n"
        f"💰 Cuota: <b>{s.odds:.2f}</b>  ·  Stake: <b>${stake:,.0f}</b>\n"
        f"💵 Bankroll restante: <b>${bankroll:,.0f}</b>\n\n"
        f"<i>Cuando termine el partido:</i>\n"
        f"<code>/resolver {pick_id} ganada</code>\n"
        f"<code>/resolver {pick_id} perdida</code>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ---------- /resolver ----------

async def cmd_resolver(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Uso: <code>/resolver &lt;pick_id&gt; ganada|perdida</code>\n"
            "Ejemplo: <code>/resolver 42 ganada</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    try:
        pick_id = int(args[0])
    except ValueError:
        await update.message.reply_text(f"'{args[0]}' no es un pick_id válido.")
        return

    outcome = args[1].lower().strip()
    if outcome in ("ganada", "ganado", "win", "won", "g"):
        won = True
    elif outcome in ("perdida", "perdido", "loss", "lost", "l"):
        won = False
    else:
        await update.message.reply_text(
            f"No entendí '{outcome}'. Usá: ganada o perdida."
        )
        return

    try:
        resolve_pick(pick_id, won=won)
    except ValueError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    bankroll = get_current_bankroll("paper")
    metrics = compute_rolling_metrics("paper", days=30)
    icon = "🟢" if won else "🔴"
    status = "GANADA" if won else "PERDIDA"
    msg = (
        f"{icon} <b>Pick #{pick_id} {status}</b>\n\n"
        f"💵 Bankroll: <b>${bankroll:,.0f}</b>\n"
        f"<i>Últimos 30 días:</i>\n"
        f"  • {metrics['n']} picks resueltas\n"
        f"  • Win rate: {metrics['win_rate']:.0%}\n"
        f"  • ROI: {metrics['roi']:+.1%}\n"
        f"  • P&L total: ${metrics['total_pnl']:+,.0f}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ---------- /balance ----------

async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from src.betting.risk_manager import risk_summary
    chat_id = str(update.effective_chat.id)
    bankroll = get_current_bankroll("paper")
    metrics = compute_rolling_metrics("paper", days=30)
    risk = risk_summary("paper")

    # Open picks (logged but not resolved)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, market, selection, odds_taken, stake, placed_at,
                   (SELECT name FROM teams WHERE id = (SELECT home_team_id FROM matches WHERE id = picks.match_id)) AS home,
                   (SELECT name FROM teams WHERE id = (SELECT away_team_id FROM matches WHERE id = picks.match_id)) AS away
            FROM picks WHERE mode = 'paper' AND won IS NULL
            ORDER BY placed_at DESC
            """
        ).fetchall()
    open_picks = [dict(r) for r in rows]

    parts = [
        "<b>💰 Tu balance</b>",
        "",
        f"<b>Bankroll:</b> ${bankroll:,.0f} COP",
        f"<b>Pico histórico:</b> ${risk['peak']:,.0f} (drawdown {risk['drawdown_pct']*100:.1f}%)",
        f"<b>Picks abiertas:</b> {len(open_picks)}",
        "",
        "<b>Últimos 30 días:</b>",
        f"• Apuestas resueltas: {metrics['n']}",
        f"• Win rate: {metrics['win_rate']:.1%}",
        f"• ROI sobre stakes: {metrics['roi']:+.1%}",
        f"• P&L: ${metrics['total_pnl']:+,.0f}",
        "",
        "<b>Gestión de riesgo:</b>",
        f"• Stakes hoy: ${risk['stakes_today']:,.0f} de ${risk['daily_remaining']:,.0f} restantes",
        f"• Stop-loss: {'🔴 ACTIVO' if risk['stop_loss_active'] else '🟢 ok'} "
        f"(drawdown threshold {risk['stop_loss_threshold']*100:.0f}%)",
        f"• Pérdidas seguidas: {risk['consecutive_losses']} (cooldown a partir de {risk['cooldown_threshold']})",
    ]
    if open_picks:
        parts.append("")
        parts.append("<b>Picks abiertas:</b>")
        for p in open_picks[:10]:
            parts.append(
                f"• #{p['id']} {p['home']} vs {p['away']} · "
                f"{p['market']}:{p['selection']} @ {p['odds_taken']:.2f} · "
                f"${p['stake']:,.0f}"
            )
    await update.message.reply_text("\n".join(parts), parse_mode=ParseMode.HTML)


# ---------- /envivo ----------

async def cmd_envivo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show currently in-play matches with conditional predictions.

    Uses in-play v0 (Poisson-on-remaining-time) — explicit disclaimer:
    this is NOT a model trained on minute-by-minute data.
    """
    from datetime import date as _date
    from src.data.espn import fetch_scoreboard
    from src.data.persist import bulk_upsert_espn
    from src.data.wplay_scraper import scrape_match_markets
    from src.models.inplay_v0 import condition_on_state as v0_condition
    from src.models.inplay_v1 import InPlayV1
    from scripts.daily_pipeline import (
        _ensemble_predict, _load_models,
        LEAGUE_NAME, LEAGUES, LIVE_ONLY_LEAGUES,
    )

    # Lazy-fit v1 once per process
    if not hasattr(cmd_envivo, "_v1"):
        v1 = InPlayV1(str(settings.db_path))
        v1.fit()
        cmd_envivo._v1 = v1
    v1 = cmd_envivo._v1
    use_v1 = v1.fitted_at is not None
    inplay_label = "in-play v1 (minute-bucketed)" if use_v1 else "in-play v0 (flat Poisson)"

    await update.message.reply_text(
        "⏳ Buscando partidos en vivo en ESPN…",
        parse_mode=ParseMode.HTML,
    )

    today = _date.today()
    live_matches: list[dict] = []
    for slug in LEAGUES + LIVE_ONLY_LEAGUES:
        try:
            ms = await fetch_scoreboard(slug, today)
            bulk_upsert_espn(ms)
            for m in ms:
                if m.status == "live":
                    live_matches.append({
                        "event_id": m.espn_id, "league_slug": slug,
                        "home_team": m.home_team, "away_team": m.away_team,
                        "home_goals": m.home_goals or 0, "away_goals": m.away_goals or 0,
                        "minute": m.minute or 0,
                    })
        except Exception as exc:
            logger.warning(f"ESPN live fetch {slug} failed: {exc}")

    if not live_matches:
        await update.message.reply_text(
            "<b>📺 En vivo</b>\n\n"
            "No hay partidos en vivo en este momento.\n\n"
            "<i>Probá de nuevo más tarde.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Match each live event to our DB so we can pull team_ids -> models
    parts = [
        "<b>📺 Partidos en vivo</b>",
        f"<i>{len(live_matches)} partido(s) en juego</i>",
        "",
        f"<b>⚠️ DISCLAIMER:</b> usando <b>{inplay_label}</b>. "
        "Es una aproximación, no un modelo neural in-play de clase mundial. "
        "<b>Apostá CHICO en vivo.</b>",
        "",
    ]

    for lm in live_matches:
        # Find match in DB to pull team ids (may be missing for live-only leagues
        # where the row was just inserted on this same call — we re-query each
        # time so primera_b / copa_colombia matches are also resolved).
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT m.id, m.home_team_id, m.away_team_id,
                       l.name as league_name
                  FROM matches m
                  JOIN leagues l ON m.league_id = l.id
                 WHERE m.api_id = ?
                """, (int(lm["event_id"]),)
            ).fetchone()

        models = _load_models(lm["league_slug"])
        league_name = (
            row["league_name"] if row else LEAGUE_NAME.get(lm["league_slug"], lm["league_slug"])
        )
        league_short = _short_league_name(league_name)
        score = f"{lm['home_goals']}-{lm['away_goals']}"
        parts.append(f"<b>⚽ {lm['home_team']} {score} {lm['away_team']}</b>")
        parts.append(f"<i>{league_short} · min {lm['minute']}'</i>")

        # No model trained for this league (Primera B, Copa Colombia, etc.) →
        # show match info only. Per CLAUDE.md, we do NOT predict markets without
        # quality data.
        if not row or not models:
            parts.append("")
            parts.append(
                "<i>ℹ️ Liga sin modelo entrenado — solo te muestro el live, "
                "no hay predicción ni análisis de value.</i>")
            parts.append("")
            parts.append("━━━━━━━━━━━━━━━━━━━")
            parts.append("")
            continue

        ensemble = _ensemble_predict(models, row["home_team_id"], row["away_team_id"])
        if ensemble is None:
            parts.append("")
            parts.append(
                "<i>ℹ️ Uno de los equipos no estaba en el set de entrenamiento — "
                "no puedo predecir este partido.</i>")
            parts.append("")
            parts.append("━━━━━━━━━━━━━━━━━━━")
            parts.append("")
            continue
        avg, _ = ensemble

        if use_v1:
            live_pred = v1.condition_on_state(
                avg, lm["home_goals"], lm["away_goals"], minute=lm["minute"]
            )
        else:
            live_pred = v0_condition(
                avg, lm["home_goals"], lm["away_goals"], minute=lm["minute"]
            )

        # Try to fetch live Wplay odds for this match
        casa: dict[tuple[str, str], float] = {}
        try:
            full = await scrape_match_markets(
                lm["event_id"], lm["league_slug"],
                lm["home_team"], lm["away_team"],
            )
            for o in full:
                casa[(o.market, o.selection)] = o.odds
        except Exception as exc:
            logger.warning(f"live Wplay scrape failed for {lm['event_id']}: {exc}")

        parts.append("")
        parts.append("<b>Predicciones del partido (final esperado)</b>")
        parts.append(
            f"  1X2: H={live_pred.p_home_win:.0%} · D={live_pred.p_draw:.0%} · A={live_pred.p_away_win:.0%}"
        )
        parts.append(
            f"  O/U 2.5: {live_pred.p_over_2_5:.0%} / {live_pred.p_under_2_5:.0%}  ·  "
            f"BTTS: {live_pred.p_btts_yes:.0%} / {live_pred.p_btts_no:.0%}"
        )

        # Edge analysis if we have casa odds
        if casa:
            market_to_prob = {
                ("1x2", "home"): live_pred.p_home_win,
                ("1x2", "draw"): live_pred.p_draw,
                ("1x2", "away"): live_pred.p_away_win,
                ("btts", "yes"): live_pred.p_btts_yes,
                ("btts", "no"): live_pred.p_btts_no,
                ("ou_1.5", "over"): live_pred.p_over_1_5,
                ("ou_1.5", "under"): live_pred.p_under_1_5,
                ("ou_2.5", "over"): live_pred.p_over_2_5,
                ("ou_2.5", "under"): live_pred.p_under_2_5,
                ("ou_3.5", "over"): live_pred.p_over_3_5,
                ("ou_3.5", "under"): live_pred.p_under_3_5,
            }
            ranked = []
            for (market, sel), odds_val in casa.items():
                prob = market_to_prob.get((market, sel))
                if prob is None or not (0.02 < prob < 0.98):
                    continue
                e = edge_fn(odds_val, prob)
                ranked.append((market, sel, prob, odds_val, e))
            ranked.sort(key=lambda x: -x[4])
            good = [r for r in ranked if settings.min_edge <= r[4] <= settings.max_edge]
            if good:
                parts.append("")
                parts.append("<b>💡 Posible value (con disclaimer)</b>")
                for market, sel, prob, odds_val, e in good[:3]:
                    action = _humanize_action(market, sel, lm["home_team"], lm["away_team"])
                    parts.append(
                        f"  • {action} @ <b>{odds_val:.2f}</b> "
                        f"<i>(modelo {prob:.0%}, edge +{e*100:.0f}%)</i>"
                    )
        parts.append("")
        parts.append("━━━━━━━━━━━━━━━━━━━")
        parts.append("")

    msg = "\n".join(parts)
    if len(msg) > 4000:
        msg = msg[:3990] + "\n…<i>(truncado)</i>"
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ---------- /resolver_auto ----------

async def cmd_resolver_auto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger auto-resolution. Useful right after a match ends so you don't
    have to wait for the cron job."""
    await update.message.reply_text(
        "⏳ Refrescando ESPN y resolviendo picks finalizadas…",
        parse_mode=ParseMode.HTML,
    )
    from src.tracking.auto_resolver import auto_resolve_paper_picks

    resolved = await auto_resolve_paper_picks()
    bankroll = get_current_bankroll("paper")
    if not resolved:
        await update.message.reply_text(
            "<i>No hay picks listas para resolver (ningún partido terminó desde la última vez).</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    parts = [f"<b>✅ {len(resolved)} pick(s) resueltas</b>", ""]
    for p in resolved:
        market, sel = p["market"], p["selection"]
        home, away = p["home_name"], p["away_name"]
        score = f"{p['home_goals']}-{p['away_goals']}"
        action = _humanize_action(market, sel, home, away)
        icon = "🟢" if p["won"] else "🔴"
        status = "GANADA" if p["won"] else "PERDIDA"
        if p["won"]:
            net = float(p["stake"]) * (float(p["odds_taken"]) - 1)
        else:
            net = -float(p["stake"])
        parts.append(
            f"{icon} <b>Pick #{p['id']} {status}</b>\n"
            f"  {home} {score} {away}\n"
            f"  <i>{action} @ {p['odds_taken']:.2f} · stake ${p['stake']:,.0f} · "
            f"P&amp;L ${net:+,.0f}</i>"
        )
    parts.append("")
    parts.append(f"<b>Bankroll:</b> ${bankroll:,.0f}")
    await update.message.reply_text("\n".join(parts), parse_mode=ParseMode.HTML)


# ---------- /historial ----------

async def cmd_historial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, market, selection, odds_taken, stake, won, payout, clv, resolved_at,
                   (SELECT name FROM teams WHERE id = (SELECT home_team_id FROM matches WHERE id = picks.match_id)) AS home,
                   (SELECT name FROM teams WHERE id = (SELECT away_team_id FROM matches WHERE id = picks.match_id)) AS away
            FROM picks
            WHERE mode = 'paper' AND won IS NOT NULL
            ORDER BY resolved_at DESC
            LIMIT 10
            """
        ).fetchall()
    if not rows:
        await update.message.reply_text(
            "<b>📜 Historial</b>\n\nNo hay picks resueltas aún.",
            parse_mode=ParseMode.HTML,
        )
        return
    parts = ["<b>📜 Últimas 10 picks resueltas</b>", ""]
    for r in rows:
        icon = "🟢" if r["won"] == 1 else "🔴"
        net = float(r["payout"]) - float(r["stake"])
        clv_str = f"CLV {r['clv']:+.1%}" if r["clv"] is not None else ""
        parts.append(
            f"{icon} <b>#{r['id']}</b> {r['home']} vs {r['away']}\n"
            f"   {r['market']}:{r['selection']} @ {r['odds_taken']:.2f} · stake ${r['stake']:,.0f} · "
            f"<b>P&L ${net:+,.0f}</b> {clv_str}"
        )
    await update.message.reply_text("\n".join(parts), parse_mode=ParseMode.HTML)


# ---------- /analizar ----------

async def cmd_analizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Find a specific match by team-name fragments and produce a full analysis."""
    from src.data.wplay_scraper import (
        normalize_name, scrape_league as scrape_wplay_league,
        scrape_match_markets,
    )
    from scripts.daily_pipeline import _load_models, _ensemble_predict, _league_slug_from_name

    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Uso: <code>/analizar &lt;equipo1&gt; &lt;equipo2&gt;</code>\n"
            "Ejemplo: <code>/analizar Arsenal Fulham</code>\n"
            "<i>Podés escribir solo parte del nombre.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Strategy: try every possible split point and pick the one that finds a match.
    # If user wrote "once caldas vs nacional", split on " vs " first.
    # Otherwise try every i in 1..len-1 as the split between home and away.
    full = " ".join(args).lower()

    candidates: list[tuple[str, str]] = []
    if " vs " in full:
        h, a = [s.strip() for s in full.split(" vs ", 1)]
        candidates.append((h, a))
    elif " v " in full:
        h, a = [s.strip() for s in full.split(" v ", 1)]
        candidates.append((h, a))
    else:
        # Try all reasonable splits, prefer ones with more even token counts
        for i in range(1, len(args)):
            candidates.append((" ".join(args[:i]).lower(), " ".join(args[i:]).lower()))

    # Diacritic-safe match: build candidate team_id lists in Python
    # (SQLite LOWER doesn't strip accents, so substring LIKE on raw text fails
    # for "medellin" vs DB "Medellín", "aguilas" vs "Águilas Doradas", etc.)
    def _query_by_team_ids(home_ids: list[int], away_ids: list[int]) -> list:
        if not home_ids or not away_ids:
            return []
        ph_h = ",".join(["?"] * len(home_ids))
        ph_a = ",".join(["?"] * len(away_ids))
        with get_conn() as conn:
            return conn.execute(
                f"""
                SELECT m.id, m.kickoff_utc, m.status, m.home_team_id, m.away_team_id,
                       h.name AS home_name, a.name AS away_name, l.name AS league
                  FROM matches m
                  JOIN teams h ON m.home_team_id = h.id
                  JOIN teams a ON m.away_team_id = a.id
                  JOIN leagues l ON m.league_id = l.id
                 WHERE m.home_team_id IN ({ph_h}) AND m.away_team_id IN ({ph_a})
                 ORDER BY m.kickoff_utc DESC LIMIT 5
                """,
                [*home_ids, *away_ids],
            ).fetchall()

    rows: list = []
    for home_q, away_q in candidates:
        h_ids = _find_team_candidates(home_q)
        a_ids = _find_team_candidates(away_q)
        r = _query_by_team_ids(h_ids, a_ids)
        if r:
            rows = r
            break

    if not rows:
        # Try every split with home/away swapped (user may have given them backwards)
        for hq, aq in candidates:
            h_ids = _find_team_candidates(aq)  # swapped
            a_ids = _find_team_candidates(hq)
            r = _query_by_team_ids(h_ids, a_ids)
            if r:
                rows = r
                await update.message.reply_text(
                    "<i>(Invertí el orden — el partido está como visitante/local opuesto a lo que escribiste.)</i>",
                    parse_mode=ParseMode.HTML,
                )
                break

        if not rows:
            await update.message.reply_text(
                f"No encontré ningún partido con esos nombres. "
                f"Probá nombres más completos, o usá explícitamente <code>vs</code>: "
                f"<code>/analizar Once Caldas vs Atlético Nacional</code>\n\n"
                f"También podés correr /picks para ver qué partidos tenemos.",
                parse_mode=ParseMode.HTML,
            )
            return

    # If multiple matches: pick the closest to today (most recent or earliest upcoming)
    target = rows[0]  # already ordered
    if len(rows) > 1:
        scheduled = [r for r in rows if r["status"] == "scheduled"]
        if scheduled:
            target = scheduled[0]

    match_id = target["id"]
    home, away = target["home_name"], target["away_name"]
    league = target["league"]
    status = target["status"]

    # Run ensemble
    league_slug = _league_slug_from_name(league)
    models = _load_models(league_slug)
    if not models:
        await update.message.reply_text(f"No hay modelos entrenados para {league}.")
        return
    pred_result = _ensemble_predict(models, target["home_team_id"], target["away_team_id"])
    if pred_result is None:
        await update.message.reply_text(
            f"Uno de los equipos no estaba en el set de entrenamiento. No puedo predecir."
        )
        return
    avg, _ = pred_result

    # If the match is in the future and in our two leagues, try to grab Wplay odds
    # across all available markets (1X2 + BTTS + O/U lines).
    casa_odds: dict[tuple[str, str], float] = {}  # (market, selection) -> odds
    if status == "scheduled":
        try:
            league_rows = await scrape_wplay_league(league_slug)
            db_home_norm = normalize_name(home)
            db_away_norm = normalize_name(away)

            def _team_match(db_norm: str, wplay_norm: str) -> bool:
                """Substring-tolerant match. Wplay may add city suffixes
                ('Águilas Doradas Rionegro' vs DB 'Águilas Doradas') or
                drop them; either side containing the other is a match."""
                if not db_norm or not wplay_norm:
                    return False
                return db_norm in wplay_norm or wplay_norm in db_norm

            event_id: str | None = None
            wplay_home, wplay_away = home, away
            for r in league_rows:
                if (_team_match(db_home_norm, normalize_name(r.home_team))
                        and _team_match(db_away_norm, normalize_name(r.away_team))):
                    event_id = r.event_id
                    wplay_home, wplay_away = r.home_team, r.away_team
                    break
            if event_id:
                full = await scrape_match_markets(event_id, league_slug, wplay_home, wplay_away)
                for o in full:
                    casa_odds[(o.market, o.selection)] = o.odds
            else:
                # Fall back: at least we have 1X2 from league_rows
                for r in league_rows:
                    if (_team_match(db_home_norm, normalize_name(r.home_team))
                            and _team_match(db_away_norm, normalize_name(r.away_team))):
                        casa_odds[(r.market, r.selection)] = r.odds
        except Exception as exc:
            logger.warning(f"Wplay scrape during /analizar failed: {exc}")

    # Build response
    when = _humanize_kickoff(target["kickoff_utc"])
    parts = [
        f"<b>📊 {home} vs {away}</b>",
        f"<i>{league} · {when} · estado: {status}</i>",
        "",
        "<b>Predicción del modelo</b> <i>(ensemble Dixon-Coles + Elo)</i>",
        f"  1X2:   <b>{avg.p_home_win:.0%}</b> / <b>{avg.p_draw:.0%}</b> / <b>{avg.p_away_win:.0%}</b>",
        f"  xG:    <b>{avg.expected_home_goals:.2f} - {avg.expected_away_goals:.2f}</b>",
        "",
        "<b>Mercados de goles</b>",
        f"  Más 1.5: {avg.p_over_1_5:.0%}  ·  Menos 1.5: {avg.p_under_1_5:.0%}",
        f"  Más 2.5: {avg.p_over_2_5:.0%}  ·  Menos 2.5: {avg.p_under_2_5:.0%}",
        f"  Más 3.5: {avg.p_over_3_5:.0%}  ·  Menos 3.5: {avg.p_under_3_5:.0%}",
        f"  Ambos marcan: {avg.p_btts_yes:.0%}  ·  No: {avg.p_btts_no:.0%}",
        "",
        "<b>Hándicap</b>",
        f"  {home} -1.5 (gana por 2+): {avg.p_home_minus_1_5:.0%}",
        f"  {away} +1.5: {avg.p_away_plus_1_5:.0%}",
    ]

    if casa_odds:
        # Compute edge for every market+selection combo we have casa odds for
        market_to_prob: dict[tuple[str, str], float] = {
            ("1x2", "home"): avg.p_home_win,
            ("1x2", "draw"): avg.p_draw,
            ("1x2", "away"): avg.p_away_win,
            ("btts", "yes"): avg.p_btts_yes,
            ("btts", "no"): avg.p_btts_no,
            ("ou_1.5", "over"): avg.p_over_1_5,
            ("ou_1.5", "under"): avg.p_under_1_5,
            ("ou_2.5", "over"): avg.p_over_2_5,
            ("ou_2.5", "under"): avg.p_under_2_5,
            ("ou_3.5", "over"): avg.p_over_3_5,
            ("ou_3.5", "under"): avg.p_under_3_5,
        }
        ranked: list[tuple[str, str, float, float, float]] = []  # market, sel, prob, odds, edge
        for (market, sel), odds_val in casa_odds.items():
            prob = market_to_prob.get((market, sel))
            if prob is None or not (0.02 < prob < 0.98):
                continue
            e = edge_fn(odds_val, prob)
            ranked.append((market, sel, prob, odds_val, e))
        ranked.sort(key=lambda x: -x[4])

        # Show 1X2 cuotas summary line
        sel_to_odds_1x2 = {sel: casa_odds[("1x2", sel)] for sel in ("home", "draw", "away")
                            if ("1x2", sel) in casa_odds}
        if sel_to_odds_1x2:
            parts.append("")
            parts.append("<b>Cuotas Wplay</b>")
            parts.append(
                f"  1X2: H {sel_to_odds_1x2.get('home', '—')} · "
                f"X {sel_to_odds_1x2.get('draw', '—')} · "
                f"A {sel_to_odds_1x2.get('away', '—')}"
            )
            for line in (1.5, 2.5, 3.5):
                key_o = (f"ou_{line}", "over")
                key_u = (f"ou_{line}", "under")
                if key_o in casa_odds and key_u in casa_odds:
                    parts.append(
                        f"  Más {line}: {casa_odds[key_o]:.2f} · Menos {line}: {casa_odds[key_u]:.2f}"
                    )
            if ("btts", "yes") in casa_odds and ("btts", "no") in casa_odds:
                parts.append(
                    f"  BTTS sí: {casa_odds[('btts', 'yes')]:.2f} · "
                    f"BTTS no: {casa_odds[('btts', 'no')]:.2f}"
                )

        # Top picks with edge between MIN_EDGE and MAX_EDGE
        good = [r for r in ranked if settings.min_edge <= r[4] <= settings.max_edge]
        parts.append("")
        parts.append("━━━━━━━━━━━━━━━━━━━")
        if good:
            best_pick = good[0]
            market, sel, prob, odds_val, e = best_pick
            action = _humanize_action(market, sel, home, away)
            from src.betting.kelly import kelly_stake
            bankroll = get_current_bankroll("paper")
            stake = kelly_stake(prob, odds_val, bankroll, fraction=settings.kelly_fraction)
            parts.append(f"<b>✅ APUESTA RECOMENDADA</b>")
            parts.append(f"➤ <b>{action}</b>  @ <b>{odds_val:.2f}</b>")
            parts.append(
                f"<i>Modelo {prob:.0%} · cuota implica {1/odds_val:.0%} · "
                f"edge <b>+{e*100:.0f}%</b> · stake sugerido ${stake:,.0f} (¼ Kelly)</i>"
            )
            if len(good) > 1:
                parts.append("")
                parts.append("<i>Otras opciones con value:</i>")
                for market, sel, prob, odds_val, e in good[1:4]:
                    action = _humanize_action(market, sel, home, away)
                    parts.append(
                        f"  • {action} @ {odds_val:.2f} <i>(edge +{e*100:.0f}%)</i>"
                    )
        elif ranked:
            best = ranked[0]
            best_action = _humanize_action(best[0], best[1], home, away)
            if best[4] > settings.max_edge:
                parts.append(f"<b>⚠️ NO APUESTES</b>")
                parts.append(
                    f"<i>El mejor edge es +{best[4]*100:.0f}% sobre {best_action} — "
                    f"sospechoso, filtrado por límite de 30% (probable error del modelo).</i>"
                )
            else:
                # Find what the model thinks is the most likely outcome
                top_prob_market = max(
                    [(m, s, p) for (m, s, p) in [
                        ("1x2", "home", avg.p_home_win),
                        ("1x2", "draw", avg.p_draw),
                        ("1x2", "away", avg.p_away_win),
                    ]],
                    key=lambda x: x[2],
                )
                top_action = _humanize_action(top_prob_market[0], top_prob_market[1], home, away)
                parts.append(f"<b>🛑 PASÁ — no hay value</b>")
                parts.append(
                    f"<i>El modelo cree que lo más probable es <b>{top_action}</b> "
                    f"({top_prob_market[2]:.0%}), pero Wplay ya lo paga ajustado. "
                    f"El mejor edge en cualquier mercado es {best[4]*100:+.0f}% — "
                    f"abajo del mínimo de 5% para apostar.</i>"
                )
    else:
        parts.append("")
        parts.append(
            "<i>(No tengo cuotas de Wplay para este partido en este momento. "
            "Probá /picks o /analizar en pre-match cuando estén abiertas las cuotas.)</i>"
        )

    # Top scorers section (anytime scorer model)
    try:
        from src.models.player_scorers import AnytimeScorerModel
        if not hasattr(cmd_analizar, "_scorer_model"):
            sm = AnytimeScorerModel(str(settings.db_path))
            sm.fit()
            cmd_analizar._scorer_model = sm
        sm = cmd_analizar._scorer_model
        home_scorers = sm.top_scorers(target["home_team_id"], n=4,
                                       match_lambda_for_team=avg.expected_home_goals)
        away_scorers = sm.top_scorers(target["away_team_id"], n=4,
                                       match_lambda_for_team=avg.expected_away_goals)
        if home_scorers or away_scorers:
            parts.append("")
            parts.append("<b>👤 Anytime scorer (top candidatos)</b>")
            for s in home_scorers:
                parts.append(
                    f"  <i>{home}:</i> <b>{s.player_name}</b> "
                    f"({s.goals_scored}g) → {s.p_anytime_score:.0%}"
                )
            for s in away_scorers:
                parts.append(
                    f"  <i>{away}:</i> <b>{s.player_name}</b> "
                    f"({s.goals_scored}g) → {s.p_anytime_score:.0%}"
                )
            parts.append(
                "<i>Si Wplay paga &gt; 1/(prob mostrada), hay value. "
                "Limitación: no sabemos si arrancan en el 11 inicial.</i>"
            )
    except Exception as exc:
        logger.warning(f"scorer model failed: {exc}")

    msg = "\n".join(parts)
    if len(msg) > 4000:
        msg = msg[:3990] + "\n…<i>(truncado)</i>"
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ---------- Inline button callback ----------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data == "cmd:picks":
        # Run /picks logic
        update_proxy = Update(update.update_id, message=query.message)
        await cmd_picks(update_proxy, context)
    elif data == "cmd:balance":
        update_proxy = Update(update.update_id, message=query.message)
        await cmd_balance(update_proxy, context)
    elif data == "cmd:historial":
        update_proxy = Update(update.update_id, message=query.message)
        await cmd_historial(update_proxy, context)


# ---------- Natural-language handler (free-text messages) ----------

# Lazy-init the parser once per process; reused across messages.
_NLU_PARSER = None


def _get_nlu_parser():
    global _NLU_PARSER
    if _NLU_PARSER is not None:
        return _NLU_PARSER
    if not settings.anthropic_api_key:
        return None
    try:
        from src.llm.nlu import IntentParser
        _NLU_PARSER = IntentParser(settings.anthropic_api_key)
        return _NLU_PARSER
    except Exception as exc:
        logger.warning(f"NLU parser init failed: {exc}")
        return None


async def _handle_register_externals(
    update: Update, context: ContextTypes.DEFAULT_TYPE,
    *, raw_text: str, mode_hint: str,
) -> None:
    """Parse a pasted Wplay block (or NL bet description), register the bets.

    Always asks for confirmation before persisting if there's any low-confidence
    selection. Otherwise persists immediately.
    """
    from src.tracking.external_bets import (
        parse_pasted_text, resolve_bets, register_resolved_bets,
    )

    chat_id = update.effective_chat.id
    mode = mode_hint if mode_hint in ("paper", "real") else "real"

    await context.bot.send_message(
        chat_id=chat_id,
        text="<i>Procesando tus apuestas… (extraigo + cruzo con la base, ~10s)</i>",
        parse_mode=ParseMode.HTML,
    )

    if not settings.anthropic_api_key:
        await context.bot.send_message(
            chat_id=chat_id,
            text="<i>Necesito ANTHROPIC_API_KEY en .env para parsear apuestas pegadas.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        parsed = await parse_pasted_text(raw_text, settings.anthropic_api_key)
    except Exception as exc:
        logger.exception("external bet parse failed")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"<i>Error parseando: {exc}</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not parsed:
        await context.bot.send_message(
            chat_id=chat_id,
            text="<i>No encontré apuestas en el texto. Pegame la confirmación de Wplay (con el detalle de cada partido) para que las pueda registrar.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    resolved, errors = resolve_bets(parsed)
    result = register_resolved_bets(resolved, mode=mode) if resolved else None

    parts: list[str] = [
        f"<b>📝 Registré {len(result.inserted) if result else 0} apuesta(s) (modo {mode})</b>",
        "",
    ]
    if result and result.inserted:
        for pid, b in result.inserted:
            sel_human = _humanize_action(b.market, b.selection, b.home_team, b.away_team)
            league_short = _short_league_name(b.league)
            parts.append(
                f"<b>#{pid}</b> · {b.home_team} vs {b.away_team}  "
                f"<i>({league_short})</i>"
            )
            parts.append(
                f"  ➤ {sel_human} @ <b>{b.odds:.2f}</b>  ·  stake <b>${b.stake:,.0f}</b>"
            )
            if b.confidence < 0.95:
                parts.append(f"  <i>⚠️ confianza {b.confidence:.0%} — {b.note}</i>")
        parts.append("")

    if errors:
        parts.append("<b>⚠️ No pude registrar:</b>")
        for pb, reason in errors:
            parts.append(f"  • {pb.home_team} v {pb.away_team}: <i>{reason}</i>")
        parts.append("")
        parts.append("<i>Si los nombres están mal escritos o el partido no está en mi base, decime y lo agrego manual.</i>")

    parts.append(
        "<i>Cuando los partidos terminen, el auto-resolver actualiza ganada/perdida automáticamente. "
        "También puedes hacer <code>/resolver_auto</code> para forzar.</i>"
    )

    await context.bot.send_message(
        chat_id=chat_id, text="\n".join(parts), parse_mode=ParseMode.HTML,
    )


async def cmd_natural_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all for plain-text messages (no '/'). Uses Claude to parse intent
    and dispatches to the appropriate command handler.
    """
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text:
        return

    parser = _get_nlu_parser()
    if parser is None:
        await update.message.reply_text(
            "<i>Para entender mensajes en lenguaje natural necesito ANTHROPIC_API_KEY en .env. "
            "Mientras tanto, usá los comandos slash: /picks, /envivo, /balance, /historial, "
            "/analizar &lt;eq1&gt; &lt;eq2&gt;, /aposte &lt;n&gt;.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Show a tiny "thinking" hint so the user knows we received it
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    except Exception:
        pass

    intent = await parser.parse(text)
    logger.info(f"NLU: '{text[:80]}' -> {intent.action} leagues={intent.leagues} time={intent.time_window} top={intent.top_only}")

    # Dispatch
    if intent.action == "picks":
        await _run_and_send_picks(
            update, context,
            leagues=intent.leagues,
            time_window=intent.time_window,
            top_only=intent.top_only,
        )
        return

    if intent.action == "live":
        await cmd_envivo(update, context)
        return

    if intent.action == "balance":
        await cmd_balance(update, context)
        return

    if intent.action == "history":
        await cmd_historial(update, context)
        return

    if intent.action == "resolve_auto":
        await cmd_resolver_auto(update, context)
        return

    if intent.action == "register_externals":
        await _handle_register_externals(update, context, raw_text=text, mode_hint=intent.mode_hint)
        return

    if intent.action == "help":
        await cmd_help(update, context)
        return

    if intent.action == "analyze":
        # cmd_analizar reads team names from context.args; inject them.
        home, away = intent.home_query, intent.away_query
        if not home or not away:
            await update.message.reply_text(
                "<i>No identifiqué bien los dos equipos. Probá: <code>analiza Real Madrid vs Barcelona</code></i>",
                parse_mode=ParseMode.HTML,
            )
            return
        # context.args expects a list of word tokens (cmd_analizar joins them later)
        original_args = context.args
        context.args = home.split() + ["vs"] + away.split()
        try:
            await cmd_analizar(update, context)
        finally:
            context.args = original_args
        return

    if intent.action == "place_bet":
        if intent.pick_number is None:
            await update.message.reply_text(
                "<i>No entendí qué número de pick querés apostar. Probá: <code>aposté el 3</code> o <code>aposté el 5 con 10000</code></i>",
                parse_mode=ParseMode.HTML,
            )
            return
        original_args = context.args
        args_list = [str(intent.pick_number)]
        if intent.stake is not None:
            args_list.append(str(intent.stake))
        context.args = args_list
        try:
            await cmd_aposte(update, context)
        finally:
            context.args = original_args
        return

    if intent.action == "smalltalk":
        reply = intent.reasoning or "🙂"
        await update.message.reply_text(reply, parse_mode=ParseMode.HTML)
        return

    # Fallback
    await cmd_help(update, context)
