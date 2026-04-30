"""Stacking ensemble + Platt scaling calibration.

Strategy:
    1. Train base models (Dixon-Coles, Elo, XGBoost) on the FULL training set.
    2. Use out-of-fold predictions on the training set as input features for
       a logistic-regression meta-model. The meta-model learns when to weight
       each base model.
    3. Pass the meta-model output through Platt scaling (logistic on
       held-out predictions) to fix overconfidence.

Why this matters:
    Raw XGBoost is overconfident (says 90% but realizes 30%). DC + Elo are
    better-calibrated but weaker on accuracy. Meta-stacking + Platt scaling
    typically fixes both: better accuracy AND better calibration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.models.base import MatchProbabilities, Model


class StackingEnsemble(Model):
    """Stacks Dixon-Coles + Elo + XGBoost outputs into a calibrated 1X2 model.

    For O/U and BTTS we run a similar but simpler stacking on those markets
    when the base model exposes those probabilities.
    """

    name = "stacking"

    def __init__(self, base_models: list[Model], n_splits: int = 5) -> None:
        self.base_models = base_models
        self.n_splits = n_splits
        self.meta_1x2: CalibratedClassifierCV | None = None
        self.meta_ou_2_5: CalibratedClassifierCV | None = None
        self.meta_btts: CalibratedClassifierCV | None = None
        self.fitted_at: datetime | None = None

    # ---------- helpers ----------

    @staticmethod
    def _outcome_1x2(home_goals: int, away_goals: int) -> int:
        if home_goals > away_goals:
            return 0
        if home_goals == away_goals:
            return 1
        return 2

    def _stack_features_for(
        self, predictions: list[MatchProbabilities], market: str,
    ) -> np.ndarray:
        """Pull the relevant probabilities from each base model into a row."""
        rows = []
        for p in predictions:
            if market == "1x2":
                rows.append([p.p_home_win, p.p_draw, p.p_away_win])
            elif market == "ou_2.5":
                rows.append([p.p_over_2_5])
            elif market == "btts":
                rows.append([p.p_btts_yes])
        return np.array(rows, dtype=np.float32)

    # ---------- fit ----------

    def fit(self, training_data: list[dict]) -> None:
        """OOF stacking: refit each base model n_splits times to get unbiased
        meta-features for the training set."""
        if not training_data:
            raise ValueError("no training data")
        sorted_data = sorted(training_data, key=lambda m: m["kickoff_date"])
        n = len(sorted_data)
        logger.info(f"Stacking: building OOF meta-features over {n} matches")

        kf = KFold(n_splits=self.n_splits, shuffle=False)  # NO shuffle = preserves time order
        oof_1x2 = np.zeros((n, len(self.base_models) * 3), dtype=np.float32)
        oof_ou25 = np.zeros((n, len(self.base_models)), dtype=np.float32)
        oof_btts = np.zeros((n, len(self.base_models)), dtype=np.float32)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n))):
            logger.info(f"  fold {fold_idx + 1}/{self.n_splits}: train={len(train_idx)} val={len(val_idx)}")
            train_subset = [sorted_data[i] for i in train_idx]
            val_subset = [sorted_data[i] for i in val_idx]

            for bm_idx, base_model in enumerate(self.base_models):
                # Re-instantiate fresh model state if possible. Most of our base
                # models accept .fit() repeatedly, but XGBoost can be re-fit.
                base_model.fit(train_subset)
                for j, vm in enumerate(val_subset):
                    try:
                        pred = base_model.predict_match(
                            vm["home_team_id"], vm["away_team_id"],
                            kickoff_date=vm["kickoff_date"],
                            league_id=vm.get("league_id"),
                            match_id=vm.get("match_id"),
                        )
                    except KeyError:
                        # Team unknown to this fold's base model; default to uniform
                        pred = MatchProbabilities(
                            p_home_win=1/3, p_draw=1/3, p_away_win=1/3,
                            p_over_2_5=0.5, p_under_2_5=0.5,
                            p_btts_yes=0.5, p_btts_no=0.5,
                            expected_home_goals=0, expected_away_goals=0,
                        )
                    row_idx = val_idx[j]
                    oof_1x2[row_idx, bm_idx * 3:(bm_idx + 1) * 3] = [
                        pred.p_home_win, pred.p_draw, pred.p_away_win
                    ]
                    oof_ou25[row_idx, bm_idx] = pred.p_over_2_5
                    oof_btts[row_idx, bm_idx] = pred.p_btts_yes

        # Now refit base models on FULL data so production prediction uses all data
        for bm in self.base_models:
            bm.fit(sorted_data)

        # Build labels
        y_1x2 = np.array([
            self._outcome_1x2(int(m["home_goals"]), int(m["away_goals"]))
            for m in sorted_data
        ])
        y_ou25 = np.array([
            1 if int(m["home_goals"]) + int(m["away_goals"]) > 2 else 0
            for m in sorted_data
        ])
        y_btts = np.array([
            1 if int(m["home_goals"]) > 0 and int(m["away_goals"]) > 0 else 0
            for m in sorted_data
        ])

        # Train meta-models with internal CV-based Platt calibration.
        # sklearn 1.8+ removed the multi_class kwarg (multinomial is default).
        self.meta_1x2 = CalibratedClassifierCV(
            LogisticRegression(max_iter=500),
            method="sigmoid", cv=3,
        )
        self.meta_1x2.fit(oof_1x2, y_1x2)

        self.meta_ou_2_5 = CalibratedClassifierCV(
            LogisticRegression(max_iter=500), method="sigmoid", cv=3,
        )
        self.meta_ou_2_5.fit(oof_ou25, y_ou25)

        self.meta_btts = CalibratedClassifierCV(
            LogisticRegression(max_iter=500), method="sigmoid", cv=3,
        )
        self.meta_btts.fit(oof_btts, y_btts)

        self.fitted_at = datetime.now()
        logger.info(f"Stacking ready: meta-models calibrated on {n} OOF rows.")

    # ---------- predict ----------

    def predict_match(
        self, home_team_id: int, away_team_id: int, **context
    ) -> MatchProbabilities:
        if self.meta_1x2 is None:
            raise RuntimeError("StackingEnsemble not fitted")

        base_preds: list[MatchProbabilities] = []
        for bm in self.base_models:
            try:
                p = bm.predict_match(home_team_id, away_team_id, **context)
                base_preds.append(p)
            except KeyError:
                base_preds.append(MatchProbabilities(
                    p_home_win=1/3, p_draw=1/3, p_away_win=1/3,
                    p_over_2_5=0.5, p_under_2_5=0.5,
                    p_btts_yes=0.5, p_btts_no=0.5,
                    expected_home_goals=0, expected_away_goals=0,
                ))

        feat_1x2 = self._stack_features_for(base_preds, "1x2").reshape(1, -1)
        feat_ou25 = self._stack_features_for(base_preds, "ou_2.5").reshape(1, -1)
        feat_btts = self._stack_features_for(base_preds, "btts").reshape(1, -1)

        p_1x2 = self.meta_1x2.predict_proba(feat_1x2)[0]
        p_over25 = float(self.meta_ou_2_5.predict_proba(feat_ou25)[0][1])
        p_btts = float(self.meta_btts.predict_proba(feat_btts)[0][1])

        # For O/U 1.5 / 3.5 / handicap, average the base models' individual outputs
        avg = lambda f: float(np.mean([f(p) for p in base_preds]))
        return MatchProbabilities(
            p_home_win=float(p_1x2[0]),
            p_draw=float(p_1x2[1]),
            p_away_win=float(p_1x2[2]),
            p_over_2_5=p_over25,
            p_under_2_5=1.0 - p_over25,
            p_over_1_5=avg(lambda p: p.p_over_1_5),
            p_under_1_5=1.0 - avg(lambda p: p.p_over_1_5),
            p_over_3_5=avg(lambda p: p.p_over_3_5),
            p_under_3_5=1.0 - avg(lambda p: p.p_over_3_5),
            p_btts_yes=p_btts,
            p_btts_no=1.0 - p_btts,
            p_home_minus_1_5=avg(lambda p: p.p_home_minus_1_5),
            p_away_plus_1_5=1.0 - avg(lambda p: p.p_home_minus_1_5),
            expected_home_goals=avg(lambda p: p.expected_home_goals),
            expected_away_goals=avg(lambda p: p.expected_away_goals),
            features={"model": "stacking", "n_base": len(self.base_models)},
        )
