"""Auto-resolver: for each open pick, fetch the final score from ESPN and call
resolve_pick(). Saves the user from typing /resolver after every match.

Trigger options (all use the same core function):
    - Daily pipeline calls it after the predict step (free).
    - /resolver_auto command in the bot (manual trigger).
    - A separate cron job every hour (autonomy).

Today this resolves ONLY win/loss. CLV measurement stays at 0% because we
don't yet snapshot the closing odds at kickoff. That's a Phase 3 add: a
job that captures Wplay odds 1-2 minutes before kickoff for each match.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Iterable

from loguru import logger

from src.data.espn import fetch_scoreboard
from src.data.persist import bulk_upsert_espn, get_conn
from src.tracking.pick_logger import resolve_pick


# ---------- Outcome rules ----------

def compute_outcome(market: str, selection: str, home_goals: int, away_goals: int) -> bool | None:
    """Return True if the bet won, False if it lost, None if we can't decide."""
    if market == "1x2":
        if selection == "home":
            return home_goals > away_goals
        if selection == "draw":
            return home_goals == away_goals
        if selection == "away":
            return away_goals > home_goals
    if market == "btts":
        both = home_goals > 0 and away_goals > 0
        if selection == "yes":
            return both
        if selection == "no":
            return not both
    if market.startswith("ou_"):
        try:
            line = float(market.removeprefix("ou_"))
        except ValueError:
            return None
        total = home_goals + away_goals
        if selection == "over":
            return total > line
        if selection == "under":
            return total < line
    if market == "ah_-1.5":
        margin = home_goals - away_goals
        if selection == "home":
            return margin >= 2
        if selection == "away":
            return margin < 2
    return None


# ---------- Auto-resolution ----------

async def _refresh_espn_for_matches(match_ids: Iterable[int]) -> None:
    """For each unique (league_slug, kickoff_date) implied by the match list,
    fetch scoreboard from ESPN and upsert. Cheap defensive refresh so we have
    the latest finals before resolving."""
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT m.id, m.kickoff_utc, l.name as league_name
              FROM matches m
              JOIN leagues l ON m.league_id = l.id
             WHERE m.id IN ({','.join('?' * len(list(match_ids) or [-1]))})
            """,
            tuple(match_ids),
        ).fetchall()
    if not rows:
        return
    pairs: set[tuple[str, date]] = set()
    for r in rows:
        slug = "premier_league" if "Premier" in r["league_name"] else "liga_betplay"
        try:
            d = datetime.fromisoformat(r["kickoff_utc"]).date()
        except ValueError:
            continue
        pairs.add((slug, d))
    for slug, d in pairs:
        try:
            ms = await fetch_scoreboard(slug, d)
            bulk_upsert_espn(ms)
        except Exception as exc:
            logger.warning(f"refresh ESPN {slug} {d} failed: {exc}")


async def auto_resolve_paper_picks(notify_callback=None) -> list[dict]:
    """Resolve all open paper picks whose match has finished.

    Args:
        notify_callback: optional async function(pick_id, won, payout, balance, ...)
                         called for each resolution. Used by the bot to push
                         Telegram messages.

    Returns list of resolved-pick dicts (for logging/inspection).
    """
    # Find pending picks (won IS NULL) — both paper AND real modes.
    # External Wplay bets registered via /aposte/paste arrive as mode='real'.
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.match_id, p.market, p.selection, p.odds_taken, p.stake, p.mode,
                   m.status, m.home_goals, m.away_goals, m.kickoff_utc,
                   h.name AS home_name, a.name AS away_name
              FROM picks p
              JOIN matches m ON p.match_id = m.id
              LEFT JOIN teams h ON m.home_team_id = h.id
              LEFT JOIN teams a ON m.away_team_id = a.id
             WHERE p.won IS NULL
             ORDER BY m.kickoff_utc ASC
            """
        ).fetchall()

    if not rows:
        return []

    pending = [dict(r) for r in rows]

    # Refresh ESPN for the matches in the candidate set (cheap insurance)
    match_ids = list({r["match_id"] for r in pending})
    await _refresh_espn_for_matches(match_ids)

    # Re-read after refresh so we have the latest scores/status
    with get_conn() as conn:
        placeholders = ",".join("?" * len(match_ids))
        rows = conn.execute(
            f"""
            SELECT id, status, home_goals, away_goals
              FROM matches WHERE id IN ({placeholders})
            """,
            tuple(match_ids),
        ).fetchall()
    fresh = {r["id"]: dict(r) for r in rows}

    resolved: list[dict] = []
    for p in pending:
        m = fresh.get(p["match_id"])
        if not m:
            continue
        if m["status"] != "finished":
            continue
        if m["home_goals"] is None or m["away_goals"] is None:
            continue
        won = compute_outcome(p["market"], p["selection"],
                              int(m["home_goals"]), int(m["away_goals"]))
        if won is None:
            logger.warning(
                f"unsupported market for auto-resolve: {p['market']}:{p['selection']} "
                f"on pick #{p['id']}"
            )
            continue

        try:
            resolve_pick(p["id"], won=won)
        except Exception as exc:
            logger.warning(f"resolve_pick failed for #{p['id']}: {exc}")
            continue

        info = {
            **p,
            "won": won,
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
        }
        resolved.append(info)
        if notify_callback is not None:
            try:
                await notify_callback(info)
            except Exception as exc:
                logger.warning(f"notify_callback failed for #{p['id']}: {exc}")

    logger.info(f"auto-resolver: resolved {len(resolved)} picks (out of {len(pending)} pending)")
    return resolved


def auto_resolve_sync() -> list[dict]:
    """Sync wrapper for cron / CLI usage."""
    return asyncio.run(auto_resolve_paper_picks())
