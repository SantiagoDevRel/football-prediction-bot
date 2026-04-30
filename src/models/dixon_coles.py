"""Dixon-Coles model (Poisson with low-score correction + temporal decay).

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".

Each team has attack and defense parameters. Goals are modeled bivariate Poisson
with a tau() correction at low scores (0-0, 1-0, 0-1, 1-1). Older matches are
weighted less via exp(-xi * days_ago).

Implementation: MLE via scipy minimize. Constraint: mean attack = mean defense = 1
(implemented as a soft penalty in the objective).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from math import exp, factorial, log

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from src.models.base import MatchProbabilities, Model


# We cap goal totals when computing market probabilities. 6 covers ~99.9%
# of real Premier scores.
_MAX_GOALS = 7


def _tau(home_goals: int, away_goals: int, lambda_: float, mu: float, rho: float) -> float:
    """Low-score correction. Equals 1 outside the corrected cells."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_ * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_ * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


@dataclass
class _Match:
    home_idx: int
    away_idx: int
    home_goals: int
    away_goals: int
    weight: float = 1.0  # decays for older matches


class DixonColes(Model):
    name = "dixon_coles"

    def __init__(self, xi: float = 0.0019, max_iters: int = 200) -> None:
        # xi = decay rate per day. ~0.0019 makes 1-year-old matches weigh ~50%.
        self.xi = xi
        self.max_iters = max_iters
        self.team_to_idx: dict[int, int] = {}
        self.idx_to_team: dict[int, int] = {}
        self.attack: np.ndarray = np.array([])
        self.defense: np.ndarray = np.array([])
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.fitted_at: datetime | None = None

    # ---------- Fitting ----------

    def fit(self, training_data: list[dict]) -> None:
        """Fit on a list of finished matches.

        Each match dict must have:
            home_team_id, away_team_id, home_goals, away_goals, kickoff_date (date)
        """
        if not training_data:
            raise ValueError("no training data")

        # Build team index
        teams = sorted(
            set(m["home_team_id"] for m in training_data) |
            set(m["away_team_id"] for m in training_data)
        )
        self.team_to_idx = {t: i for i, t in enumerate(teams)}
        self.idx_to_team = {i: t for t, i in self.team_to_idx.items()}
        n_teams = len(teams)

        # Compute time-decay weights against the most recent match
        ref_date = max(m["kickoff_date"] for m in training_data)
        if isinstance(ref_date, datetime):
            ref_date = ref_date.date()
        matches: list[_Match] = []
        for m in training_data:
            kd = m["kickoff_date"]
            if isinstance(kd, datetime):
                kd = kd.date()
            days_ago = (ref_date - kd).days
            w = exp(-self.xi * days_ago)
            matches.append(_Match(
                home_idx=self.team_to_idx[m["home_team_id"]],
                away_idx=self.team_to_idx[m["away_team_id"]],
                home_goals=int(m["home_goals"]),
                away_goals=int(m["away_goals"]),
                weight=w,
            ))

        # Initial parameters: 2*n_teams (attack+defense) + home_advantage + rho
        # Layout: [att_0..att_{n-1}, def_0..def_{n-1}, ha, rho]
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = 0.25  # home advantage in log space
        x0[2 * n_teams + 1] = -0.1  # rho

        def neg_log_likelihood(params: np.ndarray) -> float:
            att = params[:n_teams]
            defe = params[n_teams:2 * n_teams]
            ha = params[2 * n_teams]
            rho = params[2 * n_teams + 1]

            # Soft constraint: mean(attack) = 0 (in log space).
            # Without it the model is identifiable up to a constant shift.
            penalty = (att.mean() ** 2 + defe.mean() ** 2) * 100.0

            ll = 0.0
            for m in matches:
                lam = exp(att[m.home_idx] + defe[m.away_idx] + ha)  # home goals rate
                mu = exp(att[m.away_idx] + defe[m.home_idx])         # away goals rate
                p_h = _poisson_pmf(m.home_goals, lam)
                p_a = _poisson_pmf(m.away_goals, mu)
                p = max(_tau(m.home_goals, m.away_goals, lam, mu, rho) * p_h * p_a, 1e-15)
                ll += m.weight * log(p)
            return -ll + penalty

        logger.info(
            f"Dixon-Coles fitting on {len(matches)} matches, {n_teams} teams, xi={self.xi}"
        )
        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iters, "disp": False},
            bounds=[(-3, 3)] * n_teams + [(-3, 3)] * n_teams + [(0, 1)] + [(-0.3, 0.3)],
        )
        if not result.success:
            logger.warning(f"  optimizer warning: {result.message}")

        self.attack = result.x[:n_teams]
        self.defense = result.x[n_teams:2 * n_teams]
        self.home_advantage = float(result.x[2 * n_teams])
        self.rho = float(result.x[2 * n_teams + 1])
        self.fitted_at = datetime.now(tz=timezone.utc)
        logger.info(
            f"  done. home_adv={self.home_advantage:.3f} rho={self.rho:.3f} "
            f"final_nll={result.fun:.1f}"
        )

    # ---------- Prediction ----------

    def _expected_goals(self, home_team_id: int, away_team_id: int) -> tuple[float, float]:
        if home_team_id not in self.team_to_idx or away_team_id not in self.team_to_idx:
            raise KeyError(
                f"team not in training set: home={home_team_id} away={away_team_id}"
            )
        h = self.team_to_idx[home_team_id]
        a = self.team_to_idx[away_team_id]
        lam = exp(self.attack[h] + self.defense[a] + self.home_advantage)
        mu = exp(self.attack[a] + self.defense[h])
        return float(lam), float(mu)

    def _score_matrix(self, lam: float, mu: float) -> np.ndarray:
        """Joint pmf of (home_goals, away_goals) up to MAX_GOALS, with tau correction."""
        m = np.zeros((_MAX_GOALS + 1, _MAX_GOALS + 1))
        for i in range(_MAX_GOALS + 1):
            for j in range(_MAX_GOALS + 1):
                m[i, j] = _tau(i, j, lam, mu, self.rho) * _poisson_pmf(i, lam) * _poisson_pmf(j, mu)
        # Renormalize so probabilities sum to 1 (truncation + rho introduces small drift)
        total = m.sum()
        if total > 0:
            m = m / total
        return m

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        lam, mu = self._expected_goals(home_team_id, away_team_id)
        sm = self._score_matrix(lam, mu)

        p_home = float(np.tril(sm, k=-1).sum())   # i > j
        p_draw = float(np.trace(sm))
        p_away = float(np.triu(sm, k=1).sum())    # j > i

        # Over/Under various lines
        idx_i, idx_j = np.indices(sm.shape)
        total_goals = idx_i + idx_j
        p_over_1_5 = float(sm[total_goals > 1].sum())
        p_over_2_5 = float(sm[total_goals > 2].sum())
        p_over_3_5 = float(sm[total_goals > 3].sum())

        # BTTS: both > 0 vs at least one == 0
        p_btts_yes = float(sm[1:, 1:].sum())
        p_btts_no = float(1.0 - p_btts_yes)

        # European handicap -1.5 / +1.5 (no push)
        # Home -1.5: home wins by 2 or more (i - j >= 2)
        margin = idx_i - idx_j
        p_home_minus_1_5 = float(sm[margin >= 2].sum())
        p_away_plus_1_5 = float(1.0 - p_home_minus_1_5)

        return MatchProbabilities(
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            p_over_2_5=p_over_2_5,
            p_under_2_5=float(1.0 - p_over_2_5),
            p_over_1_5=p_over_1_5,
            p_under_1_5=float(1.0 - p_over_1_5),
            p_over_3_5=p_over_3_5,
            p_under_3_5=float(1.0 - p_over_3_5),
            p_btts_yes=p_btts_yes,
            p_btts_no=p_btts_no,
            p_home_minus_1_5=p_home_minus_1_5,
            p_away_plus_1_5=p_away_plus_1_5,
            expected_home_goals=lam,
            expected_away_goals=mu,
            features={"model": "dixon_coles", "rho": self.rho, "home_adv": self.home_advantage},
        )
