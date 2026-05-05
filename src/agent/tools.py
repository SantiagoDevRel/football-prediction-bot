"""Tool implementations for the conversational agent.

Each function takes a single dict (the tool_use input) and returns a string
(what Claude sees as the tool_result). Strings are short and structured —
Claude reads them, the user never does.

Tools never raise to the loop: errors come back as `error: <msg>` strings so
the model can recover or apologize to the user.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from src.config import settings
from src.data.persist import get_conn
from src.tracking.pick_logger import get_current_bankroll


# ---------- Helpers ----------

def _today_iso() -> str:
    return date.today().isoformat()


def _humanize_market(market: str, selection: str) -> str:
    if market == "1x2":
        return {"home": "Local gana", "draw": "Empate", "away": "Visitante gana"}.get(selection, f"{market}:{selection}")
    if market.startswith("ou_"):
        line = market.split("_", 1)[1]
        return f"{'Más' if selection == 'over' else 'Menos'} de {line} goles"
    if market == "btts":
        return "Ambos marcan SÍ" if selection == "yes" else "Ambos marcan NO"
    if market == "double_chance":
        return {"1x": "Local o Empate", "x2": "Empate o Visitante", "12": "Local o Visitante"}.get(selection, f"DC:{selection}")
    return f"{market}:{selection}"


# ---------- Tool: get_balance ----------

def tool_get_balance(_args: dict[str, Any]) -> str:
    mode = settings.betting_mode
    bal = get_current_bankroll(mode)  # type: ignore[arg-type]
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(stake), 0) s "
            "FROM picks WHERE mode = ? AND won IS NULL",
            (mode,),
        ).fetchone()
    open_count = rows["c"]
    open_stake = rows["s"]
    return (
        f"mode={mode}\n"
        f"current_bankroll_cop={bal:,.0f}\n"
        f"open_positions={open_count}\n"
        f"open_stake_cop={open_stake:,.0f}\n"
        f"available_for_new_bets_cop={max(0, bal - 0):,.0f}"
    )


# ---------- Tool: get_history ----------

def tool_get_history(args: dict[str, Any]) -> str:
    days = int(args.get("days", 7))
    mode = settings.betting_mode
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.market, p.selection, p.odds_taken, p.stake, p.won, p.payout,
                   p.placed_at, p.resolved_at, p.source, p.note,
                   ht.name AS home, at_.name AS away
              FROM picks p
              LEFT JOIN matches m ON m.id = p.match_id
              LEFT JOIN teams ht ON ht.id = m.home_team_id
              LEFT JOIN teams at_ ON at_.id = m.away_team_id
             WHERE p.mode = ? AND p.placed_at >= ?
             ORDER BY p.placed_at DESC
             LIMIT 30
            """,
            (mode, since),
        ).fetchall()

    if not rows:
        return f"No bets in last {days} days (mode={mode})"

    won = sum(1 for r in rows if r["won"] == 1)
    lost = sum(1 for r in rows if r["won"] == 0)
    pending = sum(1 for r in rows if r["won"] is None)
    total_stake = sum(float(r["stake"]) for r in rows if r["won"] is not None)
    total_pnl = sum(float(r["payout"] or 0) - float(r["stake"]) for r in rows if r["won"] is not None)
    roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0.0

    out = [
        f"period_days={days} mode={mode}",
        f"summary: {won}W-{lost}L (+{pending} pending), pnl={total_pnl:+,.0f} COP, roi={roi:+.1f}%",
        "",
        "recent_bets:"
    ]
    for r in rows[:15]:
        match = f"{r['home'] or '?'} vs {r['away'] or '?'}" if r["home"] else (r["note"] or "custom")
        m = _humanize_market(r["market"], r["selection"])
        status = "WON" if r["won"] == 1 else ("LOST" if r["won"] == 0 else "OPEN")
        net = f" net={float(r['payout'] or 0) - float(r['stake']):+,.0f}" if r["won"] is not None else ""
        src = f" src={r['source']}" if r["source"] != "model" else ""
        out.append(f"  #{r['id']} [{status}{src}] {match} | {m} @ {r['odds_taken']} | stake={r['stake']:,.0f}{net}")
    return "\n".join(out)


# ---------- Tool: get_open_positions ----------

def tool_get_open_positions(_args: dict[str, Any]) -> str:
    mode = settings.betting_mode
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.market, p.selection, p.odds_taken, p.stake, p.placed_at,
                   p.source, p.parlay_group_id, p.note,
                   ht.name AS home, at_.name AS away,
                   m.kickoff_utc
              FROM picks p
              LEFT JOIN matches m ON m.id = p.match_id
              LEFT JOIN teams ht ON ht.id = m.home_team_id
              LEFT JOIN teams at_ ON at_.id = m.away_team_id
             WHERE p.mode = ? AND p.won IS NULL
             ORDER BY COALESCE(m.kickoff_utc, p.placed_at) ASC
            """,
            (mode,),
        ).fetchall()

    if not rows:
        return "No open positions"

    parlays: dict[int, list[dict]] = {}
    singles: list[dict] = []
    for r in rows:
        d = dict(r)
        if d["parlay_group_id"]:
            parlays.setdefault(d["parlay_group_id"], []).append(d)
        else:
            singles.append(d)

    out = [f"open_positions={len(rows)} mode={mode}"]
    for r in singles:
        match = f"{r['home']} vs {r['away']}" if r["home"] else (r["note"] or "custom")
        m = _humanize_market(r["market"], r["selection"])
        out.append(f"  #{r['id']} [{r['source']}] {match} | {m} @ {r['odds_taken']} | stake={r['stake']:,.0f}")
    for gid, legs in parlays.items():
        total_odds = 1.0
        for leg in legs:
            total_odds *= float(leg["odds_taken"])
        stake = float(legs[0]["stake"])
        ids = ",".join(str(x["id"]) for x in legs)
        out.append(f"  parlay#{gid} [ids={ids}] {len(legs)} legs @ combined={total_odds:.2f} stake={stake:,.0f}")
        for leg in legs:
            match = f"{leg['home']} vs {leg['away']}" if leg["home"] else (leg["note"] or "?")
            m = _humanize_market(leg["market"], leg["selection"])
            out.append(f"     - {match} | {m} @ {leg['odds_taken']}")
    return "\n".join(out)


# ---------- Tool: query_match ----------

def tool_query_match(args: dict[str, Any]) -> str:
    """Find a match by team names. Returns match_id + model probs + Wplay odds if any."""
    home = (args.get("home_team") or "").strip().lower()
    away = (args.get("away_team") or "").strip().lower()
    if not home and not away:
        return "error: provide at least home_team or away_team"

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.kickoff_utc, m.status, m.home_goals, m.away_goals,
                   ht.name AS home, at_.name AS away,
                   l.name AS league
              FROM matches m
              JOIN teams ht ON ht.id = m.home_team_id
              JOIN teams at_ ON at_.id = m.away_team_id
              JOIN leagues l ON l.id = m.league_id
             WHERE m.kickoff_utc >= datetime('now', '-1 day')
             ORDER BY m.kickoff_utc ASC
             LIMIT 500
            """
        ).fetchall()

    candidates = []
    for r in rows:
        h = r["home"].lower()
        a = r["away"].lower()
        if (not home or home in h or h in home) and (not away or away in a or a in away):
            candidates.append(dict(r))

    if not candidates:
        # fallback: just match either team
        for r in rows:
            h = r["home"].lower()
            a = r["away"].lower()
            if (home and (home in h or home in a)) or (away and (away in h or away in a)):
                candidates.append(dict(r))

    if not candidates:
        return f"no match found for home='{home}' away='{away}'"

    out = [f"matches_found={len(candidates)}"]
    for c in candidates[:5]:
        out.append(
            f"  match_id={c['id']} {c['home']} vs {c['away']} "
            f"({c['league']}) ko={c['kickoff_utc']} status={c['status']}"
        )
        # Latest odds snapshot per market
        with get_conn() as conn:
            odds = conn.execute(
                """
                SELECT market, selection, odds, bookmaker
                  FROM odds_snapshots
                 WHERE match_id = ?
                   AND captured_at = (
                       SELECT MAX(captured_at) FROM odds_snapshots o2
                        WHERE o2.match_id = odds_snapshots.match_id
                          AND o2.market = odds_snapshots.market
                          AND o2.selection = odds_snapshots.selection
                          AND o2.bookmaker = odds_snapshots.bookmaker
                   )
                 ORDER BY market, selection
                """,
                (c["id"],),
            ).fetchall()
        if odds:
            line_strs = [f"{o['market']}:{o['selection']}={o['odds']}({o['bookmaker']})" for o in odds[:12]]
            out.append("    odds: " + ", ".join(line_strs))
        # Latest ensemble prediction
        with get_conn() as conn:
            preds = conn.execute(
                "SELECT market, selection, probability FROM predictions "
                "WHERE match_id = ? AND model = 'ensemble' "
                "ORDER BY market, selection",
                (c["id"],),
            ).fetchall()
        if preds:
            pstrs = [f"{p['market']}:{p['selection']}={float(p['probability']):.0%}" for p in preds[:12]]
            out.append("    model: " + ", ".join(pstrs))
    return "\n".join(out)


# ---------- Tool: log_custom_bet ----------

def tool_log_custom_bet(args: dict[str, Any]) -> str:
    """Save a single user-decided bet. Subtracts stake from bankroll."""
    try:
        match_id = args.get("match_id")  # may be None
        market = str(args["market"])
        selection = str(args["selection"])
        odds = float(args["odds"])
        stake = float(args["stake"])
        bookmaker = str(args.get("bookmaker", "wplay"))
        note = args.get("note") or ""
    except (KeyError, TypeError, ValueError) as e:
        return f"error: bad args ({e}); required: market, selection, odds, stake"

    if odds <= 1.0 or stake <= 0:
        return "error: odds must be > 1.0 and stake > 0"

    mode = settings.betting_mode
    with get_conn() as conn:
        prev = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
        prev_balance = float(prev["balance"]) if prev else 0.0
        new_balance = prev_balance - stake

        cur = conn.execute(
            """
            INSERT INTO picks
                (match_id, market, selection, odds_taken, bookmaker,
                 model_probability, edge, confidence, stake, mode, source, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'user_manual', ?)
            """,
            (match_id, market, selection, odds, bookmaker,
             1.0 / odds, 0.0, None, stake, mode, note),
        )
        pick_id = cur.lastrowid

        conn.execute(
            "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (mode, pick_id, -stake, new_balance,
             f"user_manual: {market}:{selection} @ {odds} ({note[:60]})"),
        )

    logger.info(f"[{mode}] user_manual pick #{pick_id}: {market}:{selection} @ {odds} stake=${stake:,.0f}")
    return (
        f"saved pick_id={pick_id} stake_subtracted={stake:,.0f} new_balance={new_balance:,.0f}\n"
        f"market={market} sel={selection} odds={odds} bookmaker={bookmaker}"
    )


# ---------- Tool: log_parlay ----------

def tool_log_parlay(args: dict[str, Any]) -> str:
    """Save a multi-leg user parlay. Each leg is a row sharing a parlay_group_id.
    Stake is taken once (on the parlay, not per-leg). Combined odds are stored
    in the note. Resolution: parlay won iff ALL legs won.
    """
    try:
        legs = args["legs"]
        stake = float(args["stake"])
        bookmaker = str(args.get("bookmaker", "wplay"))
        if not isinstance(legs, list) or len(legs) < 2:
            return "error: legs must be a list of >=2 items. For singles use log_custom_bet."
    except (KeyError, TypeError, ValueError) as e:
        return f"error: {e}; required: legs (list), stake"

    if stake <= 0:
        return "error: stake must be > 0"

    combined = 1.0
    for i, leg in enumerate(legs):
        try:
            o = float(leg["odds"])
            if o <= 1.0:
                return f"error: leg {i} odds must be > 1.0"
            combined *= o
        except (KeyError, TypeError, ValueError) as e:
            return f"error: leg {i} bad: {e}"

    mode = settings.betting_mode
    with get_conn() as conn:
        # Allocate parlay_group_id = next id of first inserted row
        cur = conn.execute("SELECT COALESCE(MAX(parlay_group_id), 0) + 1 AS gid FROM picks")
        gid = int(cur.fetchone()["gid"])

        prev = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
        prev_balance = float(prev["balance"]) if prev else 0.0
        new_balance = prev_balance - stake

        leg_ids = []
        for i, leg in enumerate(legs):
            match_id = leg.get("match_id")
            note_leg = leg.get("note") or ""
            # All legs share the stake (only first row's stake is "real"; others are 0
            # to avoid double-counting in stake aggregations). The first row is the
            # "anchor" of the parlay.
            leg_stake = stake if i == 0 else 0.0
            cur = conn.execute(
                """
                INSERT INTO picks
                    (match_id, market, selection, odds_taken, bookmaker,
                     model_probability, edge, confidence, stake, mode,
                     source, parlay_group_id, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'user_parlay', ?, ?)
                """,
                (
                    match_id, str(leg["market"]), str(leg["selection"]),
                    float(leg["odds"]), bookmaker,
                    1.0 / float(leg["odds"]), 0.0, None,
                    leg_stake, mode, gid, note_leg,
                ),
            )
            leg_ids.append(cur.lastrowid)

        conn.execute(
            "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (mode, leg_ids[0], -stake, new_balance,
             f"user_parlay#{gid}: {len(legs)} legs @ combined={combined:.2f}"),
        )

    logger.info(f"[{mode}] user_parlay #{gid}: {len(legs)} legs combined={combined:.2f} stake=${stake:,.0f}")
    return (
        f"saved parlay_group_id={gid} legs={len(legs)} combined_odds={combined:.2f}\n"
        f"leg_ids={leg_ids} stake={stake:,.0f} potential_payout={stake * combined:,.0f}\n"
        f"new_balance={new_balance:,.0f}"
    )


# ---------- Tool: resolve_bet ----------

def tool_resolve_bet(args: dict[str, Any]) -> str:
    try:
        pick_id = int(args["pick_id"])
        won = bool(args["won"])
    except (KeyError, TypeError, ValueError) as e:
        return f"error: {e}; required: pick_id (int), won (bool)"

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, parlay_group_id, source, won FROM picks WHERE id = ?",
            (pick_id,),
        ).fetchone()
        if row is None:
            return f"error: pick {pick_id} not found"
        if row["won"] is not None:
            return f"error: pick {pick_id} already resolved (won={row['won']})"

    # Parlay legs use a different resolution path (all-or-nothing).
    if row["parlay_group_id"]:
        return _resolve_parlay_leg(pick_id, won)

    # Single bet → reuse existing pick_logger.resolve_pick (handles bankroll)
    from src.tracking.pick_logger import resolve_pick
    try:
        resolve_pick(pick_id, won, closing_odds=None)
    except Exception as e:
        return f"error during resolve: {e}"
    return f"resolved pick_id={pick_id} won={won}"


def _resolve_parlay_leg(pick_id: int, won: bool) -> str:
    """Mark one leg result. Only when the LAST leg is set we update bankroll
    (parlay pays only if ALL legs won)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT parlay_group_id FROM picks WHERE id = ?", (pick_id,)
        ).fetchone()
        gid = row["parlay_group_id"]
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "UPDATE picks SET won = ?, resolved_at = ? WHERE id = ?",
            (1 if won else 0, now, pick_id),
        )

        # Fetch all legs to see if parlay is now fully resolved
        legs = conn.execute(
            "SELECT id, won, odds_taken, stake, mode FROM picks WHERE parlay_group_id = ?",
            (gid,),
        ).fetchall()
        unresolved = [l for l in legs if l["won"] is None]
        if unresolved:
            return (
                f"resolved leg pick_id={pick_id} won={won}; "
                f"parlay#{gid} still has {len(unresolved)} unresolved leg(s)"
            )

        # All legs in: compute payout
        all_won = all(l["won"] == 1 for l in legs)
        anchor = next(l for l in legs if float(l["stake"]) > 0)
        stake = float(anchor["stake"])
        mode = anchor["mode"]
        combined = 1.0
        for l in legs:
            combined *= float(l["odds_taken"])
        payout = stake * combined if all_won else 0.0

        # Mark payout on the anchor leg (so /historial shows correct net)
        conn.execute(
            "UPDATE picks SET payout = ? WHERE id = ?",
            (payout, anchor["id"]),
        )
        # Other legs payout=0 (they don't pay individually)
        for l in legs:
            if l["id"] != anchor["id"]:
                conn.execute("UPDATE picks SET payout = 0 WHERE id = ?", (l["id"],))

        prev = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
        prev_balance = float(prev["balance"]) if prev else 0.0
        new_balance = prev_balance + payout
        conn.execute(
            "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (mode, anchor["id"], payout, new_balance,
             f"parlay#{gid} settled ({'WON' if all_won else 'LOST'}, {len(legs)} legs)"),
        )

    return (
        f"parlay#{gid} fully resolved: {'WON' if all_won else 'LOST'} "
        f"payout={payout:,.0f} new_balance={new_balance:,.0f}"
    )


# ---------- Tool: set_bankroll ----------

def tool_set_bankroll(args: dict[str, Any]) -> str:
    try:
        amount = float(args["amount"])
        note = str(args.get("note") or "manual set")
    except (KeyError, TypeError, ValueError) as e:
        return f"error: {e}; required: amount"
    if amount < 0:
        return "error: amount must be >= 0"

    mode = settings.betting_mode
    with get_conn() as conn:
        prev = conn.execute(
            "SELECT balance FROM bankroll_history WHERE mode = ? ORDER BY id DESC LIMIT 1",
            (mode,),
        ).fetchone()
        prev_balance = float(prev["balance"]) if prev else 0.0
        delta = amount - prev_balance
        conn.execute(
            "INSERT INTO bankroll_history (mode, pick_id, delta, balance, note) "
            "VALUES (?, NULL, ?, ?, ?)",
            (mode, delta, amount, f"set_bankroll: {note}"),
        )
    return f"bankroll set to {amount:,.0f} (delta={delta:+,.0f}) mode={mode}"


# ---------- Tool: get_today_picks ----------

def tool_get_today_picks(_args: dict[str, Any]) -> str:
    """Read pre-staged value-bet candidates from the latest pipeline run."""
    today = _today_iso()
    mode = settings.betting_mode

    with get_conn() as conn:
        # Latest staged session for any chat (assumes daily_pipeline ran today)
        row = conn.execute(
            "SELECT MAX(session_number) AS sn FROM staged_picks "
            "WHERE date(created_at) = ?",
            (today,),
        ).fetchone()
        if row is None or row["sn"] is None:
            # Fallback: open model-sourced picks from today
            picks = conn.execute(
                """
                SELECT p.id, p.market, p.selection, p.odds_taken, p.edge, p.stake,
                       p.confidence, ht.name AS home, at_.name AS away, l.name AS league,
                       m.kickoff_utc
                  FROM picks p
                  LEFT JOIN matches m ON m.id = p.match_id
                  LEFT JOIN teams ht ON ht.id = m.home_team_id
                  LEFT JOIN teams at_ ON at_.id = m.away_team_id
                  LEFT JOIN leagues l ON l.id = m.league_id
                 WHERE p.mode = ? AND p.source = 'model' AND p.won IS NULL
                   AND date(p.placed_at) = ?
                 ORDER BY p.edge DESC
                 LIMIT 15
                """,
                (mode, today),
            ).fetchall()
            if not picks:
                return "no_model_picks_today"
            out = [f"model_picks_today={len(picks)}"]
            for p in picks:
                m = _humanize_market(p["market"], p["selection"])
                out.append(
                    f"  pick_id={p['id']} {p['home']} vs {p['away']} ({p['league']}) "
                    f"ko={p['kickoff_utc']} | {m} @ {p['odds_taken']} edge={float(p['edge']):+.1%} "
                    f"stake={p['stake']:,.0f}"
                )
            return "\n".join(out)

        sn = row["sn"]
        rows = conn.execute(
            "SELECT * FROM staged_picks WHERE session_number = ? "
            "ORDER BY edge DESC LIMIT 20",
            (sn,),
        ).fetchall()

    if not rows:
        return "no_staged_picks_today"

    out = [f"staged_session={sn} picks={len(rows)}"]
    for i, r in enumerate(rows, start=1):
        d = dict(r)
        m = _humanize_market(d["market"], d["selection"])
        out.append(
            f"  #{i} {d['home_team']} vs {d['away_team']} ({d.get('league','')}) "
            f"ko={d.get('kickoff_utc','?')} | {m} @ {d['odds']} "
            f"model={float(d['model_probability']):.0%} edge={float(d.get('edge') or 0):+.1%} "
            f"stake={float(d.get('recommended_stake') or 0):,.0f}"
        )
    return "\n".join(out)


# ---------- Registry ----------

TOOL_HANDLERS = {
    "get_balance": tool_get_balance,
    "get_history": tool_get_history,
    "get_open_positions": tool_get_open_positions,
    "query_match": tool_query_match,
    "log_custom_bet": tool_log_custom_bet,
    "log_parlay": tool_log_parlay,
    "resolve_bet": tool_resolve_bet,
    "set_bankroll": tool_set_bankroll,
    "get_today_picks": tool_get_today_picks,
}


# ---------- Tool schemas (sent to Claude) ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_balance",
        "description": "Devuelve el bankroll actual del usuario, posiciones abiertas, y stake comprometido. Llamala cuando el usuario pregunte 'cuánto tengo', 'mi balance', 'cuánto me queda'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_history",
        "description": "Apuestas recientes con resultado (ganadas/perdidas/abiertas) y P&L. Para 'cómo voy', 'mi historial', 'qué he apostado esta semana'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Cuántos días atrás (default 7).", "minimum": 1, "maximum": 90},
            },
        },
    },
    {
        "name": "get_open_positions",
        "description": "Apuestas abiertas (sin resolver). Singles + parlays con sus legs. Para 'qué tengo abierto', 'apuestas vivas'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_match",
        "description": "Buscá un partido por nombres de equipos. Devuelve match_id, kickoff, y si hay: probabilidades del modelo + cuotas Wplay. Usalo ANTES de log_custom_bet/log_parlay para conseguir el match_id, o cuando el usuario pregunta '¿qué dice el modelo de X vs Y?'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Nombre del equipo local (parcial OK, ej. 'Atletico Madrid' o 'atleti')."},
                "away_team": {"type": "string"},
            },
        },
    },
    {
        "name": "log_custom_bet",
        "description": "Registra UNA apuesta simple del usuario (no del modelo). Resta stake del bankroll. Si el usuario pegó una boleta de Wplay y CONFIRMÓ que quiere guardarla, llamá esto. NO la llames sin confirmación explícita.",
        "input_schema": {
            "type": "object",
            "properties": {
                "match_id": {"type": ["integer", "null"], "description": "Si existe el partido en DB (de query_match). null si no se encontró."},
                "market": {"type": "string", "description": "Ej: '1x2', 'ou_2.5', 'btts', 'double_chance', 'qualify', 'custom'."},
                "selection": {"type": "string", "description": "Ej: 'home', 'draw', 'away', 'over', 'under', 'yes', 'no', '1x', 'x2', '12'."},
                "odds": {"type": "number", "minimum": 1.01},
                "stake": {"type": "number", "minimum": 1, "description": "Stake en COP."},
                "bookmaker": {"type": "string", "default": "wplay"},
                "note": {"type": "string", "description": "Texto libre describiendo la apuesta (ej: 'Bayern se clasifica vs PSG')."},
            },
            "required": ["market", "selection", "odds", "stake"],
        },
    },
    {
        "name": "log_parlay",
        "description": "Registra una combinada (parlay) del usuario, 2+ legs. Stake se cobra UNA vez. Resolución: gana solo si TODAS las legs ganan. Solo llamá esto si el usuario CONFIRMÓ que quiere guardar la boleta combinada.",
        "input_schema": {
            "type": "object",
            "properties": {
                "stake": {"type": "number", "minimum": 1},
                "bookmaker": {"type": "string", "default": "wplay"},
                "legs": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "match_id": {"type": ["integer", "null"]},
                            "market": {"type": "string"},
                            "selection": {"type": "string"},
                            "odds": {"type": "number", "minimum": 1.01},
                            "note": {"type": "string"},
                        },
                        "required": ["market", "selection", "odds"],
                    },
                },
            },
            "required": ["legs", "stake"],
        },
    },
    {
        "name": "resolve_bet",
        "description": "Marcá una apuesta como ganada/perdida. Para parlays, llamala una vez por cada leg; el agente paga solo cuando todas estén resueltas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pick_id": {"type": "integer"},
                "won": {"type": "boolean"},
            },
            "required": ["pick_id", "won"],
        },
    },
    {
        "name": "set_bankroll",
        "description": "Reinicia o ajusta el bankroll a un valor exacto (ej. depósito, retiro, ajuste manual). Solo si el usuario lo pide explícitamente.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "note": {"type": "string"},
            },
            "required": ["amount"],
        },
    },
    {
        "name": "get_today_picks",
        "description": "Devuelve los picks que el modelo determinista detectó hoy con value (edge >5%). Para 'qué picks hay', 'top picks de hoy'.",
        "input_schema": {"type": "object", "properties": {}},
    },
]
