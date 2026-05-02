"""Persistence helpers: upsert leagues, teams, matches into SQLite.

Idempotent: re-running a sync upserts by api_id. Goal scores get updated
when a previously-scheduled match becomes finished.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import settings
from src.data.espn import ESPNMatch, ESPN_LEAGUE_BY_SLUG
from src.data.football_data_uk import HistoricalMatch


@contextmanager
def get_conn():
    db_path: Path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_league(conn: sqlite3.Connection, slug: str, season: int) -> int:
    """Insert league if missing, return its id."""
    code = ESPN_LEAGUE_BY_SLUG.get(slug, slug)
    name_map = {
        "premier_league": "Premier League",
        "liga_betplay": "Liga BetPlay Dimayor",
        "sudamericana": "Copa Sudamericana",
        "libertadores": "Copa Libertadores",
        "champions_league": "UEFA Champions League",
    }
    country_map = {
        "premier_league": "England",
        "liga_betplay": "Colombia",
        "sudamericana": "South America",
        "libertadores": "South America",
        "champions_league": "Europe",
    }
    cur = conn.execute(
        "SELECT id FROM leagues WHERE name = ? AND season = ?",
        (name_map.get(slug, slug), season),
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO leagues (api_id, name, country, season) VALUES (?, ?, ?, ?)",
        (None, name_map.get(slug, slug), country_map.get(slug, "Unknown"), season),
    )
    return cur.lastrowid  # type: ignore[return-value]


def ensure_team(conn: sqlite3.Connection, name: str, league_id: int, espn_id: str | None = None) -> int:
    """Insert team if missing, return its id."""
    cur = conn.execute("SELECT id FROM teams WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO teams (api_id, name, league_id) VALUES (?, ?, ?)",
        (int(espn_id) if espn_id and espn_id.isdigit() else None, name, league_id),
    )
    return cur.lastrowid  # type: ignore[return-value]


def _infer_season(kickoff: datetime, slug: str) -> int:
    """Infer the football season starting year from kickoff datetime."""
    if slug in ("liga_betplay", "sudamericana", "libertadores"):
        # South-American competitions run within a calendar year
        return kickoff.year
    # Default European convention (also applies to Champions League)
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1


def upsert_espn_match(conn: sqlite3.Connection, match: ESPNMatch) -> int:
    """Insert or update one ESPN match. Returns the matches.id."""
    season = _infer_season(match.kickoff_utc, match.league_slug)
    league_id = ensure_league(conn, match.league_slug, season)
    home_id = ensure_team(conn, match.home_team, league_id, match.home_team_id)
    away_id = ensure_team(conn, match.away_team, league_id, match.away_team_id)

    cur = conn.execute("SELECT id FROM matches WHERE api_id = ?", (int(match.espn_id),))
    row = cur.fetchone()
    now = datetime.now().isoformat(timespec="seconds")
    if row:
        conn.execute(
            """
            UPDATE matches
               SET status = ?, home_goals = ?, away_goals = ?,
                   kickoff_utc = ?, venue = ?, updated_at = ?
             WHERE id = ?
            """,
            (match.status, match.home_goals, match.away_goals,
             match.kickoff_utc.isoformat(), match.venue, now, row["id"]),
        )
        return row["id"]
    cur = conn.execute(
        """
        INSERT INTO matches
            (api_id, league_id, season, home_team_id, away_team_id, kickoff_utc,
             status, home_goals, away_goals, venue, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (int(match.espn_id), league_id, season, home_id, away_id,
         match.kickoff_utc.isoformat(), match.status, match.home_goals, match.away_goals,
         match.venue, now, now),
    )
    return cur.lastrowid  # type: ignore[return-value]


def upsert_historical_match(conn: sqlite3.Connection, match: HistoricalMatch) -> int:
    """Insert or update a historical match from football-data.co.uk.

    These matches don't have a stable external id (the CSV doesn't include one),
    so we synthesize a deterministic key from (league, date, home, away).
    """
    league_id = ensure_league(conn, match.league_slug, match.season)
    home_id = ensure_team(conn, match.home_team, league_id)
    away_id = ensure_team(conn, match.away_team, league_id)

    # Synthetic api_id: large negative numbers to avoid collisions with ESPN ids.
    # hash() is process-stable enough for our use; we fall back to lookup by
    # composite key if no row matches.
    cur = conn.execute(
        """
        SELECT id FROM matches
         WHERE league_id = ? AND home_team_id = ? AND away_team_id = ?
           AND date(kickoff_utc) = date(?)
        """,
        (league_id, home_id, away_id, match.match_date.isoformat()),
    )
    row = cur.fetchone()
    now = datetime.now().isoformat(timespec="seconds")
    kickoff_iso = datetime.combine(match.match_date, datetime.min.time()).isoformat()

    if row:
        conn.execute(
            """
            UPDATE matches
               SET status = 'finished', home_goals = ?, away_goals = ?, updated_at = ?
             WHERE id = ?
            """,
            (match.home_goals, match.away_goals, now, row["id"]),
        )
        match_id = row["id"]
    else:
        cur = conn.execute(
            """
            INSERT INTO matches
                (api_id, league_id, season, home_team_id, away_team_id, kickoff_utc,
                 status, home_goals, away_goals, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'finished', ?, ?, ?, ?)
            """,
            (None, league_id, match.season, home_id, away_id, kickoff_iso,
             match.home_goals, match.away_goals, now, now),
        )
        match_id = cur.lastrowid

    # Save closing odds as snapshots
    odds_to_save = [
        ("1x2", "home", match.odds_home),
        ("1x2", "draw", match.odds_draw),
        ("1x2", "away", match.odds_away),
        ("ou_2.5", "over", match.odds_over_2_5),
        ("ou_2.5", "under", match.odds_under_2_5),
        ("btts", "yes", match.odds_btts_yes),
        ("btts", "no", match.odds_btts_no),
    ]
    for market, selection, price in odds_to_save:
        if price is None:
            continue
        # Avoid duplicate snapshot rows on re-import
        existing = conn.execute(
            """
            SELECT 1 FROM odds_snapshots
             WHERE match_id = ? AND bookmaker = 'football_data_uk'
               AND market = ? AND selection = ? AND is_closing = 1
            """,
            (match_id, market, selection),
        ).fetchone()
        if existing:
            continue
        conn.execute(
            """
            INSERT INTO odds_snapshots
                (match_id, bookmaker, market, selection, odds, is_closing, captured_at)
            VALUES (?, 'football_data_uk', ?, ?, ?, 1, ?)
            """,
            (match_id, market, selection, price, now),
        )
    return match_id  # type: ignore[return-value]


def bulk_upsert_espn(matches: list[ESPNMatch]) -> int:
    n = 0
    with get_conn() as conn:
        for m in matches:
            upsert_espn_match(conn, m)
            n += 1
    logger.info(f"upserted {n} ESPN matches into DB")
    return n


def bulk_upsert_historical(matches: list[HistoricalMatch]) -> int:
    n = 0
    with get_conn() as conn:
        for m in matches:
            upsert_historical_match(conn, m)
            n += 1
    logger.info(f"upserted {n} historical matches into DB")
    return n


def _normalize_team_name(name: str) -> str:
    """Aggressive normalization for cross-source matching (ESPN ↔ Understat ↔ Wplay).
    Strips diacritics, common suffixes, generic words, and trailing 'united'/
    'city'/'hotspur' style decorations that vary across providers.
    """
    import re as _re
    import unicodedata
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    # Drop generic prefixes/suffixes that providers add inconsistently:
    s = _re.sub(r"\b(fc|cd|sc|cf|cdf|afc|de|football|club)\b", "", s)
    # Drop nickname suffixes that one provider uses and another doesn't:
    #   ESPN says "Tottenham Hotspur"; Understat says "Tottenham"
    #   ESPN says "Brighton & Hove Albion"; Understat says "Brighton"
    #   ESPN says "Newcastle United"; Understat says "Newcastle"
    #   ESPN says "Ipswich Town"; Understat says "Ipswich"
    s = _re.sub(r"\b(hotspur|& hove albion|& hove|albion|wanderers|town|united)\b", "", s)
    # Final whitespace cleanup
    return _re.sub(r"\s+", " ", s).strip()


def update_xg_from_understat(understat_matches: list) -> dict:
    """For each UnderstatMatch, find the matching DB row by (date, home, away)
    using fuzzy team-name matching. Update home_xg / away_xg columns.

    Returns counts: {matched, missing, updated_xg}
    """
    matched = missing = updated_xg = 0
    with get_conn() as conn:
        # Pre-build a lookup of all teams in the DB by normalized name
        team_rows = conn.execute("SELECT id, name FROM teams").fetchall()
        norm_to_team_id = {_normalize_team_name(r["name"]): r["id"] for r in team_rows}

        for u in understat_matches:
            h_norm = _normalize_team_name(u.home_team)
            a_norm = _normalize_team_name(u.away_team)
            home_id = norm_to_team_id.get(h_norm)
            away_id = norm_to_team_id.get(a_norm)
            if home_id is None or away_id is None:
                missing += 1
                continue

            row = conn.execute(
                """
                SELECT id FROM matches
                 WHERE home_team_id = ? AND away_team_id = ?
                   AND date(kickoff_utc) = date(?)
                """,
                (home_id, away_id, u.match_date.isoformat()),
            ).fetchone()
            if row is None:
                missing += 1
                continue
            matched += 1
            conn.execute(
                "UPDATE matches SET home_xg = ?, away_xg = ? WHERE id = ?",
                (float(u.home_xg), float(u.away_xg), row["id"]),
            )
            updated_xg += 1

    logger.info(f"understat → DB: matched={matched} missing={missing} xG_updated={updated_xg}")
    return {"matched": matched, "missing": missing, "updated_xg": updated_xg}
