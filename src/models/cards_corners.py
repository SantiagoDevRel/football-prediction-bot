"""Cards and Corners models — simple but effective Poisson-rate models.

Approach:
    For each team, compute its rolling avg of cards (or corners) "for" and
    "against" — that is, how many YELLOW cards a team typically gets and
    how many its opponents typically commit against it. Same for corners.

    Predicted total in match X vs Y:
        λ_total = (X_for + Y_against) / 2 + (Y_for + X_against) / 2
                ~ X_for + Y_for  (averaged over many windows)

    Then a Poisson over λ gives P(total > 4.5), P(total > 5.5), etc.

This is naive (no referee adjustment, no derby-intensity bonus) but already
useful as a baseline. The same module fits 'cards' and 'corners' — they
share the same Poisson logic, only the underlying counts differ.

Phase 4.5 enhancements (later): per-referee bonus for cards, derby bonus,
home/away rates separately.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from math import exp, factorial
from typing import Iterable, Literal

import numpy as np
from loguru import logger


StatKind = Literal["cards", "corners"]


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def _poisson_over(line: float, lam: float, max_n: int = 25) -> float:
    """P(X > line) where X ~ Poisson(λ). For half-line markets only (no push)."""
    if lam <= 0:
        return 0.0 if line >= 0 else 1.0
    cum = 0.0
    cap = int(line)  # P(X <= cap)
    for k in range(cap + 1):
        cum += _poisson_pmf(k, lam)
    return max(0.0, 1.0 - cum)


class CardsOrCornersModel:
    """One model handles either cards (yellow only) or corners.

    Calling fit() builds team-level rolling averages from the match_stats
    table. Calling predict_total() returns expected total for a match plus
    P(over X.5) for the standard market lines.
    """

    name: str = "cards_or_corners"

    def __init__(self, db_path: str, kind: StatKind = "cards") -> None:
        self.db_path = db_path
        self.kind = kind
        # Per-team rolling avg "for" (cards/corners *committed by* this team
        # — i.e. their attacking style draws corners, their style commits fouls
        # leading to cards) and "against" (drawn against this team).
        self.team_for: dict[int, float] = {}
        self.team_against: dict[int, float] = {}
        self.league_avg: float = 4.5  # league-wide fallback
        self.fitted_at: datetime | None = None
        self.window_size: int = 10  # last N matches per team

    @staticmethod
    def _stat_columns(kind: StatKind) -> tuple[str, str]:
        if kind == "cards":
            return ("home_yellow_cards", "away_yellow_cards")
        if kind == "corners":
            return ("home_corners", "away_corners")
        raise ValueError(kind)

    def fit(self, training_data: list[dict] | None = None) -> None:
        """Compute team-level rolling averages from match_stats.

        We ignore training_data and just read the DB directly because
        match_stats is keyed by match_id and we want all available data.
        """
        h_col, a_col = self._stat_columns(self.kind)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT m.id, m.kickoff_utc, m.home_team_id, m.away_team_id,
                   s.{h_col} AS home_stat, s.{a_col} AS away_stat
              FROM matches m
              JOIN match_stats s ON s.match_id = m.id
             WHERE s.{h_col} IS NOT NULL AND s.{a_col} IS NOT NULL
             ORDER BY m.kickoff_utc DESC
            """
        ).fetchall()
        conn.close()

        if not rows:
            logger.warning(f"{self.name} ({self.kind}): no match_stats rows yet")
            return

        # Aggregate per team — most recent N matches each
        per_team: dict[int, list[tuple[int, int]]] = {}
        # tuples are (this_team_stat, opp_stat)
        all_totals: list[int] = []
        for r in rows:
            home_stat = int(r["home_stat"])
            away_stat = int(r["away_stat"])
            all_totals.append(home_stat + away_stat)
            per_team.setdefault(r["home_team_id"], []).append((home_stat, away_stat))
            per_team.setdefault(r["away_team_id"], []).append((away_stat, home_stat))

        for team_id, samples in per_team.items():
            recent = samples[: self.window_size]
            if not recent:
                continue
            self.team_for[team_id] = float(np.mean([s[0] for s in recent]))
            self.team_against[team_id] = float(np.mean([s[1] for s in recent]))

        self.league_avg = float(np.mean(all_totals))
        self.fitted_at = datetime.now()
        logger.info(
            f"{self.kind}: fit on {len(rows)} matches, {len(per_team)} teams, "
            f"league avg total = {self.league_avg:.2f}"
        )

    def expected_total(self, home_team_id: int, away_team_id: int) -> float:
        """Predicted total cards (or corners) in this match."""
        h_for = self.team_for.get(home_team_id, self.league_avg / 2)
        a_for = self.team_against.get(home_team_id, self.league_avg / 2)
        a2_for = self.team_for.get(away_team_id, self.league_avg / 2)
        h_against = self.team_against.get(away_team_id, self.league_avg / 2)
        # Average each team's typical contribution — for + opponent's against
        home_contrib = (h_for + h_against) / 2
        away_contrib = (a2_for + a_for) / 2
        return home_contrib + away_contrib

    def predict_over_lines(
        self, home_team_id: int, away_team_id: int,
        lines: Iterable[float] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
                                   7.5, 8.5, 9.5, 10.5, 11.5, 12.5),
    ) -> dict[float, float]:
        """Returns dict {line: P(total > line)} via Poisson PMF on the
        expected total. Only half-lines (no pushes)."""
        lam = self.expected_total(home_team_id, away_team_id)
        return {line: _poisson_over(line, lam) for line in lines}

    def summary(self, home_team_id: int, away_team_id: int) -> dict:
        lam = self.expected_total(home_team_id, away_team_id)
        return {
            "kind": self.kind,
            "expected_total": lam,
            "league_avg": self.league_avg,
            "lines": self.predict_over_lines(home_team_id, away_team_id),
        }
