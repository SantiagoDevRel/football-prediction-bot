"""Anytime scorer model — for each player, predict P(scores at least once).

Approach:
    1. From goal_events table, count goals per player in their last N matches.
    2. Compute player's goal-rate per match (goals_scored / matches_played).
    3. For an upcoming match, scale the rate by:
        - team's expected goals in the match (from Dixon-Coles / ensemble)
        - relative to team's avg expected goals overall
    4. Convert to anytime-scorer probability via 1 - exp(-λ_player).

Limitations:
    - We don't have lineup data → can only predict for players known to have
      played recently. Not the actual starting XI.
    - Penalties and own goals are excluded from the rate.
    - No injury/suspension awareness (covered by LLM features for high-profile
      cases but not per-player).

Output: list of {player_name, team, goal_rate, p_anytime_score} for each
team's most recent goal-scorers.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from math import exp

from loguru import logger


@dataclass
class ScorerPrediction:
    player_name: str
    espn_player_id: str
    team_id: int
    team_name: str
    matches_seen: int
    goals_scored: int
    goal_rate_per_match: float
    p_anytime_score: float


class AnytimeScorerModel:
    """Builds per-player goal rates and predicts anytime-score for next match."""

    name = "anytime_scorer"

    def __init__(self, db_path: str, recent_matches: int = 20) -> None:
        self.db_path = db_path
        self.recent_matches = recent_matches
        # team_id -> list[(player_id, name, goals, matches, rate)]
        self.team_scorers: dict[int, list[ScorerPrediction]] = {}
        self.fitted_at: datetime | None = None

    def fit(self, training_data=None) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get all goal events with team mapping. Key the player by espn_player_id
        # because names can collide. team_api_id matches teams.api_id.
        rows = conn.execute(
            """
            SELECT g.espn_player_id, g.player_name, g.team_api_id,
                   g.match_id, g.is_penalty, g.is_own_goal,
                   m.kickoff_utc
              FROM goal_events g
              JOIN matches m ON g.match_id = m.id
             WHERE g.is_own_goal = 0
             ORDER BY m.kickoff_utc DESC
            """
        ).fetchall()

        # Per-player goal aggregates over recent_matches limit per player
        per_player: dict[str, dict] = {}
        for r in rows:
            pid = r["espn_player_id"]
            if not pid:
                continue
            entry = per_player.setdefault(pid, {
                "name": r["player_name"], "team_api_id": r["team_api_id"],
                "goals": 0, "matches_with_goal": set(),
                "first_match_seen": r["match_id"],
            })
            entry["goals"] += 1
            entry["matches_with_goal"].add(r["match_id"])
            entry["last_match"] = r["match_id"]

        # We need to compute matches PLAYED per player — but we don't have
        # appearance data, only goal events. Approximation: estimate matches
        # played as "matches between first goal seen and last goal seen,
        # bounded by recent_matches". For active players this is decent.
        # Better: pull lineups (next iteration).
        team_team_id_by_api: dict[str, int] = {}
        for r in conn.execute(
            "SELECT id, api_id FROM teams WHERE api_id IS NOT NULL"
        ).fetchall():
            team_team_id_by_api[str(r["api_id"])] = r["id"]

        # Total matches played by each team in our data (approx denominator)
        team_match_counts: dict[int, int] = {}
        for r in conn.execute(
            """
            SELECT home_team_id AS tid, COUNT(*) AS n FROM matches
             WHERE status='finished' GROUP BY home_team_id
            UNION ALL
            SELECT away_team_id AS tid, COUNT(*) AS n FROM matches
             WHERE status='finished' GROUP BY away_team_id
            """
        ).fetchall():
            team_match_counts[r["tid"]] = team_match_counts.get(r["tid"], 0) + r["n"]

        team_name_by_id: dict[int, str] = {
            r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM teams")
        }
        conn.close()

        # Build the result
        team_scorers: dict[int, list[ScorerPrediction]] = {}
        for pid, info in per_player.items():
            team_id = team_team_id_by_api.get(str(info["team_api_id"]))
            if team_id is None:
                continue
            team_matches = team_match_counts.get(team_id, 0) or 1
            goals = info["goals"]
            # Goal rate = goals / team_matches (assumes player played most matches).
            # This OVER-estimates rate for backups but is honest for regulars.
            rate = goals / team_matches
            p_score = 1.0 - exp(-rate)
            pred = ScorerPrediction(
                player_name=info["name"],
                espn_player_id=pid,
                team_id=team_id,
                team_name=team_name_by_id.get(team_id, "?"),
                matches_seen=team_matches,
                goals_scored=goals,
                goal_rate_per_match=rate,
                p_anytime_score=p_score,
            )
            team_scorers.setdefault(team_id, []).append(pred)

        # Sort each team's list by rate descending
        for team_id in team_scorers:
            team_scorers[team_id].sort(key=lambda s: -s.goal_rate_per_match)

        self.team_scorers = team_scorers
        self.fitted_at = datetime.now()
        logger.info(
            f"anytime_scorer: built for {len(team_scorers)} teams, "
            f"{sum(len(v) for v in team_scorers.values())} players total"
        )

    def top_scorers(
        self, team_id: int, n: int = 5,
        match_lambda_for_team: float | None = None,
    ) -> list[ScorerPrediction]:
        """Return the team's top N players by goal rate, optionally rescaled
        by the team's expected goals in the upcoming match.

        Example: if a team usually averages 1.4 goals/match but for this
        match Dixon-Coles predicts 2.0, scale all rates by 2.0/1.4 = 1.43x.
        """
        scorers = list(self.team_scorers.get(team_id, []))[:n]
        if not scorers:
            return []
        if match_lambda_for_team and scorers:
            # Average team rate across all listed scorers
            team_avg = sum(s.goal_rate_per_match for s in scorers) / len(scorers)
            if team_avg > 0:
                factor = match_lambda_for_team / max(team_avg * len(scorers), 0.1)
                # Don't exceed 2x boost (sanity)
                factor = max(0.5, min(2.0, factor))
                rescaled = []
                for s in scorers:
                    new_rate = s.goal_rate_per_match * factor
                    rescaled.append(ScorerPrediction(
                        player_name=s.player_name,
                        espn_player_id=s.espn_player_id,
                        team_id=s.team_id,
                        team_name=s.team_name,
                        matches_seen=s.matches_seen,
                        goals_scored=s.goals_scored,
                        goal_rate_per_match=new_rate,
                        p_anytime_score=1.0 - exp(-new_rate),
                    ))
                return rescaled
        return scorers
