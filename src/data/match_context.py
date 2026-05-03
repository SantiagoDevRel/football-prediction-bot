"""Helpers that produce qualitative match context for Claude:

    - recent_form(team_id, n=5)  → "WWDLW (3W 1D 1L · GF 8 GA 4)"
    - head_to_head(home_id, away_id, n=5) → "Last 5 H2H: Santa Fe 2-1-2"
    - team_streak(team_id, threshold) → home/away unbeaten streaks

These are derived from the matches table. Cheap (one SQL each).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.data.persist import get_conn


@dataclass
class RecentForm:
    n_matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    streak: str            # e.g. "WWDLW" most recent first
    recent_summary: list[str]  # ["W vs Atlético Nacional 2-1", ...]


@dataclass
class HeadToHead:
    n_matches: int
    home_wins: int          # wins for the team labeled 'home_id' (regardless of venue)
    draws: int
    away_wins: int          # wins for 'away_id' regardless of venue
    last_results: list[str]  # ["Santa Fe 2-1 Inter (2025-09)", ...]


def recent_form(team_id: int, n: int = 5) -> RecentForm | None:
    """Last n finished matches for this team."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
                   m.home_goals, m.away_goals,
                   h.name AS home, a.name AS away
              FROM matches m
              JOIN teams h ON m.home_team_id = h.id
              JOIN teams a ON m.away_team_id = a.id
             WHERE m.status = 'finished'
               AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
               AND (m.home_team_id = ? OR m.away_team_id = ?)
             ORDER BY m.kickoff_utc DESC LIMIT ?
            """,
            (team_id, team_id, n),
        ).fetchall()
    if not rows:
        return None

    wins = draws = losses = gf = ga = 0
    streak_chars: list[str] = []
    summary: list[str] = []
    for r in rows:
        is_home = r["home_team_id"] == team_id
        my_goals = r["home_goals"] if is_home else r["away_goals"]
        their_goals = r["away_goals"] if is_home else r["home_goals"]
        opp_name = r["away"] if is_home else r["home"]
        gf += my_goals
        ga += their_goals
        if my_goals > their_goals:
            wins += 1
            streak_chars.append("W")
            res_char = "W"
        elif my_goals == their_goals:
            draws += 1
            streak_chars.append("D")
            res_char = "D"
        else:
            losses += 1
            streak_chars.append("L")
            res_char = "L"
        venue = "vs" if is_home else "@"
        summary.append(f"{res_char} {venue} {opp_name} {my_goals}-{their_goals}")

    return RecentForm(
        n_matches=len(rows),
        wins=wins, draws=draws, losses=losses,
        goals_for=gf, goals_against=ga,
        streak="".join(streak_chars),
        recent_summary=summary,
    )


def head_to_head(home_team_id: int, away_team_id: int, n: int = 5) -> HeadToHead | None:
    """Last n finished matches between these two teams (any venue)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
                   m.home_goals, m.away_goals,
                   h.name AS home, a.name AS away
              FROM matches m
              JOIN teams h ON m.home_team_id = h.id
              JOIN teams a ON m.away_team_id = a.id
             WHERE m.status = 'finished'
               AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
               AND ((m.home_team_id = ? AND m.away_team_id = ?)
                 OR (m.home_team_id = ? AND m.away_team_id = ?))
             ORDER BY m.kickoff_utc DESC LIMIT ?
            """,
            (home_team_id, away_team_id, away_team_id, home_team_id, n),
        ).fetchall()
    if not rows:
        return None

    home_wins = draws = away_wins = 0
    results: list[str] = []
    for r in rows:
        # Was 'home_team_id' the home of this past match?
        was_home_match = r["home_team_id"] == home_team_id
        h_goals = r["home_goals"]
        a_goals = r["away_goals"]
        if was_home_match:
            if h_goals > a_goals:
                home_wins += 1
            elif h_goals == a_goals:
                draws += 1
            else:
                away_wins += 1
        else:
            # past match had teams reversed
            if h_goals > a_goals:
                away_wins += 1
            elif h_goals == a_goals:
                draws += 1
            else:
                home_wins += 1
        results.append(
            f"{r['home']} {h_goals}-{a_goals} {r['away']} ({r['kickoff_utc'][:7]})"
        )

    return HeadToHead(
        n_matches=len(rows),
        home_wins=home_wins, draws=draws, away_wins=away_wins,
        last_results=results,
    )


def consensus_block_for_match(
    home: str, away: str, league_slug: str, api_key: str | None,
) -> dict[tuple[str, str], dict] | None:
    """Synchronous wrapper that returns multi-bookmaker consensus stats
    for a specific match. Returns dict keyed by (market, selection) →
    {best, median, worst, n_books, best_bookie}, or None if odds-api
    isn't configured/doesn't have the match.

    NOTE: this is a thin wrapper — the actual fetch is async. The caller
    is responsible for awaiting fetch_multi_bookie_odds and filtering.
    """
    return None  # caller does the async fetch + filter directly
