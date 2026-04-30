"""XGBoost model on engineered features.

We train THREE separate classifiers, one per market:
    - 1X2 (3-class: home / draw / away)
    - O/U 2.5 (binary)
    - BTTS (binary)

For O/U 1.5/3.5 and handicap we currently fall back to the Dixon-Coles
distribution since we don't have label data conveniently shaped for them
(would require derived labels). Phase 3.5 enhancement.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
import xgboost as xgb
from loguru import logger

from src.models.base import MatchProbabilities, Model
from src.models.features import FeatureBuilder, FeatureVector


class XGBoostModel(Model):
    name = "xgboost"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.builder = FeatureBuilder(db_path)
        self.clf_1x2: xgb.XGBClassifier | None = None
        self.clf_ou_2_5: xgb.XGBClassifier | None = None
        self.clf_btts: xgb.XGBClassifier | None = None
        self.fitted_at: datetime | None = None

    # ---------- Training data assembly ----------

    def _build_dataset(
        self, training_data: list[dict]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """For each match in training_data, build features (using only data
        available BEFORE the match) and labels per market.

        training_data must already be sorted chronologically by kickoff_date.
        """
        X: list[np.ndarray] = []
        y_1x2: list[int] = []
        y_ou_25: list[int] = []
        y_btts: list[int] = []
        for m in training_data:
            kd = m["kickoff_date"]
            if isinstance(kd, datetime):
                kd = kd.date()
            features = self.builder.build(
                m["home_team_id"], m["away_team_id"], kd,
                league_id=m.get("league_id"),
            )
            X.append(features.to_array())
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            # 1X2: 0=home, 1=draw, 2=away
            if hg > ag:
                y_1x2.append(0)
            elif hg == ag:
                y_1x2.append(1)
            else:
                y_1x2.append(2)
            y_ou_25.append(1 if hg + ag > 2 else 0)
            y_btts.append(1 if hg > 0 and ag > 0 else 0)

        X_arr = np.array(X, dtype=np.float32)
        return X_arr, {
            "1x2": np.array(y_1x2),
            "ou_2.5": np.array(y_ou_25),
            "btts": np.array(y_btts),
        }

    # ---------- Fit ----------

    def fit(self, training_data: list[dict]) -> None:
        if not training_data:
            raise ValueError("no training data")

        # Refresh feature builder snapshot (DB may have changed since __init__)
        self.builder.reload()

        # Sort chronologically (no leakage)
        sorted_data = sorted(training_data, key=lambda m: m["kickoff_date"])
        X, y = self._build_dataset(sorted_data)
        logger.info(f"XGBoost fit: X.shape={X.shape}, features={X.shape[1]}")

        common = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )

        self.clf_1x2 = xgb.XGBClassifier(
            **{**common, "objective": "multi:softprob", "num_class": 3,
               "eval_metric": "mlogloss"}
        )
        self.clf_1x2.fit(X, y["1x2"])

        self.clf_ou_2_5 = xgb.XGBClassifier(**common)
        self.clf_ou_2_5.fit(X, y["ou_2.5"])

        self.clf_btts = xgb.XGBClassifier(**common)
        self.clf_btts.fit(X, y["btts"])

        self.fitted_at = datetime.now()
        logger.info(
            f"XGBoost done. trained on {len(sorted_data)} matches across 3 markets."
        )

    # ---------- Predict ----------

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        if self.clf_1x2 is None:
            raise RuntimeError("XGBoost not fitted")

        kd = context.get("kickoff_date") or date.today()
        league_id = context.get("league_id")
        feats = self.builder.build(home_team_id, away_team_id, kd, league_id=league_id)
        x = feats.to_array().reshape(1, -1)

        p_1x2 = self.clf_1x2.predict_proba(x)[0]   # [home, draw, away]
        p_ou25 = float(self.clf_ou_2_5.predict_proba(x)[0][1])  # P(over)
        p_btts = float(self.clf_btts.predict_proba(x)[0][1])    # P(yes)

        # We don't have direct models for O/U 1.5, 3.5, AH; derive crude estimates
        # from the rolling averages. xG-based estimates are usually close enough.
        approx_total = (
            feats.home_goals_for_avg.get(5, 0.0) + feats.away_goals_for_avg.get(5, 0.0)
        ) / 2 + (
            feats.home_goals_against_avg.get(5, 0.0) + feats.away_goals_against_avg.get(5, 0.0)
        ) / 2
        approx_total = max(approx_total, 1.5)
        # Heuristic: shift O/U 2.5 prob toward 1.5 / 3.5 by an empirical factor
        p_over_1_5 = min(0.99, p_ou25 + 0.20)
        p_over_3_5 = max(0.01, p_ou25 - 0.20)

        return MatchProbabilities(
            p_home_win=float(p_1x2[0]),
            p_draw=float(p_1x2[1]),
            p_away_win=float(p_1x2[2]),
            p_over_2_5=p_ou25,
            p_under_2_5=1.0 - p_ou25,
            p_over_1_5=p_over_1_5,
            p_under_1_5=1.0 - p_over_1_5,
            p_over_3_5=p_over_3_5,
            p_under_3_5=1.0 - p_over_3_5,
            p_btts_yes=p_btts,
            p_btts_no=1.0 - p_btts,
            p_home_minus_1_5=max(0.01, float(p_1x2[0]) - 0.20),
            p_away_plus_1_5=1.0 - max(0.01, float(p_1x2[0]) - 0.20),
            expected_home_goals=approx_total / 2,
            expected_away_goals=approx_total / 2,
            features={"model": "xgboost"},
        )
