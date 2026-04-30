"""Feature engineering for XGBoost-style models.

Given a match (home_team_id, away_team_id, kickoff_date) and the full DB
of past finished matches, build a numeric feature vector summarizing
recent form, scoring rate, defensive rate, rest, head-to-head and xG.

Strict no-leakage rule: features are computed using ONLY matches that
finished STRICTLY BEFORE the target match's kickoff. Training and
inference share the same code path so this rule is enforced everywhere.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np


# Window sizes for rolling stats. Smaller = more responsive but noisier;
# larger = more stable but lags. We compute several windows so XGBoost can
# pick whichever it likes.
WINDOWS_LAST_N = (3, 5, 10)


@dataclass
class FeatureVector:
    """One row of features for a single (home, away, date) sample."""
    # 1. Form / scoring
    home_goals_for_avg: dict[int, float]    # {n: avg goals scored last n matches}
    home_goals_against_avg: dict[int, float]
    away_goals_for_avg: dict[int, float]
    away_goals_against_avg: dict[int, float]

    # 2. xG (Premier only; zero for BetPlay where xG is unavailable)
    home_xg_avg: dict[int, float]
    home_xga_avg: dict[int, float]
    away_xg_avg: dict[int, float]
    away_xga_avg: dict[int, float]

    # 3. Form (W=3 D=1 L=0 over last n)
    home_form_pts: dict[int, float]
    away_form_pts: dict[int, float]

    # 4. Rest (days since last match)
    home_days_rest: float
    away_days_rest: float

    # 5. Head-to-head (last 5 between these two teams)
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int
    h2h_avg_total_goals: float

    # 6. Has training data flag (low-data warning)
    home_n_matches: int
    away_n_matches: int

    def to_array(self) -> np.ndarray:
        """Flatten to a fixed-order numpy array for XGBoost."""
        parts: list[float] = []
        for n in WINDOWS_LAST_N:
            parts.extend([
                self.home_goals_for_avg.get(n, 0.0),
                self.home_goals_against_avg.get(n, 0.0),
                self.away_goals_for_avg.get(n, 0.0),
                self.away_goals_against_avg.get(n, 0.0),
                self.home_xg_avg.get(n, 0.0),
                self.home_xga_avg.get(n, 0.0),
                self.away_xg_avg.get(n, 0.0),
                self.away_xga_avg.get(n, 0.0),
                self.home_form_pts.get(n, 0.0),
                self.away_form_pts.get(n, 0.0),
            ])
        parts.extend([
            self.home_days_rest, self.away_days_rest,
            float(self.h2h_home_wins), float(self.h2h_draws),
            float(self.h2h_away_wins), self.h2h_avg_total_goals,
            float(self.home_n_matches), float(self.away_n_matches),
        ])
        return np.array(parts, dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        names: list[str] = []
        for n in WINDOWS_LAST_N:
            for prefix in (
                "home_GF", "home_GA", "away_GF", "away_GA",
                "home_xG", "home_xGA", "away_xG", "away_xGA",
                "home_form_pts", "away_form_pts",
            ):
                names.append(f"{prefix}_last{n}")
        names.extend([
            "home_days_rest", "away_days_rest",
            "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals",
            "home_n_matches", "away_n_matches",
        ])
        return names


# ---------- DB-backed feature builder ----------

class FeatureBuilder:
    """Computes feature vectors from the SQLite DB.

    Holds an in-memory snapshot of the matches table to avoid hammering the DB
    with one query per feature. Refresh via .reload().
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._matches: list[dict] = []
        self.reload()

    def reload(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, league_id, home_team_id, away_team_id,
                   home_goals, away_goals, home_xg, away_xg,
                   kickoff_utc, status
              FROM matches
             WHERE status = 'finished'
               AND home_goals IS NOT NULL AND away_goals IS NOT NULL
             ORDER BY kickoff_utc ASC
            """
        ).fetchall()
        conn.close()
        self._matches = [dict(r) for r in rows]

    def _team_history(
        self, team_id: int, before: date, league_id: int | None = None,
    ) -> list[dict]:
        """All matches involving this team that finished before the given date.

        If league_id is provided, restrict to that league (avoids mixing rates
        across very different leagues, e.g. Premier vs BetPlay)."""
        out = []
        for m in self._matches:
            if league_id is not None and m["league_id"] != league_id:
                continue
            try:
                kd = datetime.fromisoformat(m["kickoff_utc"]).date()
            except ValueError:
                continue
            if kd >= before:
                continue
            if m["home_team_id"] == team_id or m["away_team_id"] == team_id:
                out.append(m)
        return out

    def _h2h(
        self, home_id: int, away_id: int, before: date, max_n: int = 5,
    ) -> tuple[int, int, int, float]:
        """Head-to-head: last N matches between these two teams (in any home/away
        configuration), strictly before `before`. Returns (home_wins, draws,
        away_wins, avg_total_goals)."""
        rows = []
        for m in self._matches:
            try:
                kd = datetime.fromisoformat(m["kickoff_utc"]).date()
            except ValueError:
                continue
            if kd >= before:
                continue
            ids = {m["home_team_id"], m["away_team_id"]}
            if home_id in ids and away_id in ids:
                rows.append(m)
        rows = rows[-max_n:]  # most recent N
        if not rows:
            return 0, 0, 0, 0.0
        wins_for_home = draws = wins_for_away = 0
        totals = []
        for m in rows:
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            totals.append(hg + ag)
            margin_for_home = (
                hg - ag if m["home_team_id"] == home_id else ag - hg
            )
            if margin_for_home > 0:
                wins_for_home += 1
            elif margin_for_home == 0:
                draws += 1
            else:
                wins_for_away += 1
        return wins_for_home, draws, wins_for_away, float(np.mean(totals))

    @staticmethod
    def _team_stats_window(
        history: list[dict], team_id: int, n: int,
    ) -> dict:
        """For a team's history, slice to the last n matches and compute stats."""
        last = history[-n:] if len(history) >= 1 else []
        if not last:
            return {
                "gf": 0.0, "ga": 0.0, "xg": 0.0, "xga": 0.0,
                "form_pts": 0.0, "n": 0,
            }
        gf, ga, xg, xga, pts = [], [], [], [], []
        for m in last:
            is_home = m["home_team_id"] == team_id
            tg = int(m["home_goals"] if is_home else m["away_goals"])
            cg = int(m["away_goals"] if is_home else m["home_goals"])
            gf.append(tg); ga.append(cg)
            txg = float(m["home_xg"] if is_home else m["away_xg"]) if m["home_xg"] is not None else 0.0
            cxga = float(m["away_xg"] if is_home else m["home_xg"]) if m["away_xg"] is not None else 0.0
            xg.append(txg); xga.append(cxga)
            if tg > cg:
                pts.append(3)
            elif tg == cg:
                pts.append(1)
            else:
                pts.append(0)
        return {
            "gf": float(np.mean(gf)), "ga": float(np.mean(ga)),
            "xg": float(np.mean(xg)), "xga": float(np.mean(xga)),
            "form_pts": float(np.mean(pts)),
            "n": len(last),
        }

    def build(
        self, home_team_id: int, away_team_id: int, kickoff_date: date,
        league_id: int | None = None,
    ) -> FeatureVector:
        home_hist = self._team_history(home_team_id, kickoff_date, league_id)
        away_hist = self._team_history(away_team_id, kickoff_date, league_id)

        h_for: dict[int, float] = {}
        h_against: dict[int, float] = {}
        a_for: dict[int, float] = {}
        a_against: dict[int, float] = {}
        h_xg: dict[int, float] = {}
        h_xga: dict[int, float] = {}
        a_xg: dict[int, float] = {}
        a_xga: dict[int, float] = {}
        h_form: dict[int, float] = {}
        a_form: dict[int, float] = {}

        for n in WINDOWS_LAST_N:
            hs = self._team_stats_window(home_hist, home_team_id, n)
            as_ = self._team_stats_window(away_hist, away_team_id, n)
            h_for[n], h_against[n] = hs["gf"], hs["ga"]
            a_for[n], a_against[n] = as_["gf"], as_["ga"]
            h_xg[n], h_xga[n] = hs["xg"], hs["xga"]
            a_xg[n], a_xga[n] = as_["xg"], as_["xga"]
            h_form[n], a_form[n] = hs["form_pts"], as_["form_pts"]

        # Days of rest = kickoff_date - last match date (capped at 14 to avoid extremes)
        def days_rest(history: list[dict]) -> float:
            if not history:
                return 14.0
            last = history[-1]
            try:
                last_date = datetime.fromisoformat(last["kickoff_utc"]).date()
            except ValueError:
                return 14.0
            d = (kickoff_date - last_date).days
            return float(min(max(d, 0), 14))

        h_rest = days_rest(home_hist)
        a_rest = days_rest(away_hist)

        h2h = self._h2h(home_team_id, away_team_id, kickoff_date, max_n=5)

        return FeatureVector(
            home_goals_for_avg=h_for, home_goals_against_avg=h_against,
            away_goals_for_avg=a_for, away_goals_against_avg=a_against,
            home_xg_avg=h_xg, home_xga_avg=h_xga,
            away_xg_avg=a_xg, away_xga_avg=a_xga,
            home_form_pts=h_form, away_form_pts=a_form,
            home_days_rest=h_rest, away_days_rest=a_rest,
            h2h_home_wins=h2h[0], h2h_draws=h2h[1], h2h_away_wins=h2h[2],
            h2h_avg_total_goals=h2h[3],
            home_n_matches=len(home_hist), away_n_matches=len(away_hist),
        )
