"""
CreditRisk Analytics — Model Training Pipeline
==============================================

Orchestrates multi-model competitive benchmarking (Logistic Regression, Random Forest,
and Gradient Boosting), stratified cross-validation, hyperparameter selection,
probability calibration, and serialized model persistence using Joblib.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, ModelConfig, logger, model_config
from src.data.preprocessing import CreditDataPreprocessor, prepare_train_test_split
from src.features.engineering import CreditFeatureEngineer
from src.models.calibration import ProbabilityCalibrator
from src.models.evaluate import EvaluationMetrics, ModelEvaluator


@dataclass
class ModelArtifact:
    """Production artifact bundle containing the fitted pipeline, calibrator, and audit metadata."""
    model_name: str
    preprocessor: CreditDataPreprocessor
    feature_engineer: CreditFeatureEngineer
    feature_names: List[str]
    scaler: Optional[StandardScaler]
    classifier: Any
    calibrator: ProbabilityCalibrator
    test_metrics: EvaluationMetrics
    cv_summary: Dict[str, Any]
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def predict_proba(self, X: pd.DataFrame, calibrate: bool = True) -> np.ndarray:
        """End-to-end inference pipeline: preprocessing -> feature engineering -> scoring -> calibration."""
        X_clean = self.preprocessor.transform(X)
        X_feat = self.feature_engineer.transform(X_clean)
        # Select matching features
        X_aligned = X_feat[self.feature_names]

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_aligned)
            raw_prob = self.classifier.predict_proba(X_scaled)[:, 1]
        else:
            raw_prob = self.classifier.predict_proba(X_aligned)[:, 1]

        if calibrate and self.calibrator.is_fitted:
            return self.calibrator.predict_proba(raw_prob)
        return raw_prob


class CreditModelTrainer:
    """
    Executes reproducible credit risk model training and governance validation.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or model_config
        self.evaluator = ModelEvaluator()

    def train_and_benchmark(
        self,
        df: pd.DataFrame,
    ) -> Tuple[ModelArtifact, Dict[str, Dict[str, float]]]:
        """
        Split data, evaluate candidate models via cross-validation, select and calibrate
        the champion model, and evaluate on held-out test data.
        """
        logger.info("Initializing Credit Model Training and Benchmarking Pipeline...")
        X_train, X_test, y_train, y_test = prepare_train_test_split(df, self.config)

        # 1. Fit Preprocessor on Train only
        preprocessor = CreditDataPreprocessor()
        preprocessor.fit(X_train)
        X_train_clean = preprocessor.transform(X_train)
        X_test_clean = preprocessor.transform(X_test)

        # 2. Fit Feature Engineer on Train only
        feat_engineer = CreditFeatureEngineer()
        feat_engineer.fit(X_train_clean)
        X_train_eng = feat_engineer.transform(X_train_clean)
        X_test_eng = feat_engineer.transform(X_test_clean)

        # Align columns
        feature_names = [c for c in X_train_eng.columns if pd.api.types.is_numeric_dtype(X_train_eng[c])]
        X_tr = X_train_eng[feature_names].fillna(0.0)
        X_te = X_test_eng[feature_names].fillna(0.0)

        logger.info("Total Engineered Training Features: %d", len(feature_names))

        # 3. Define Candidate Classifiers
        candidates = {
            "Logistic_Regression": {
                "estimator": LogisticRegression(C=0.1, max_iter=1000, random_state=self.config.random_state),
                "scale": True,
            },
            "Random_Forest": {
                "estimator": RandomForestClassifier(
                    n_estimators=100, max_depth=6, min_samples_leaf=20, random_state=self.config.random_state, n_jobs=-1
                ),
                "scale": False,
            },
            "Hist_Gradient_Boosting": {
                "estimator": HistGradientBoostingClassifier(
                    max_iter=100, max_depth=5, l2_regularization=1.5, random_state=self.config.random_state
                ),
                "scale": False,
            },
        }

        # 4. Stratified K-Fold Cross Validation
        cv_summary = {}
        skf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)

        for name, spec in candidates.items():
            logger.info("Evaluating candidate model: %s ...", name)
            start_t = time.time()
            auc_scores = []
            brier_scores = []

            for train_idx, val_idx in skf.split(X_tr, y_train):
                X_f_tr, X_f_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
                y_f_tr, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                est = spec["estimator"]
                if spec["scale"]:
                    scaler = StandardScaler()
                    X_f_tr_s = scaler.fit_transform(X_f_tr)
                    X_f_val_s = scaler.transform(X_f_val)
                    est.fit(X_f_tr_s, y_f_tr)
                    p_val = est.predict_proba(X_f_val_s)[:, 1]
                else:
                    est.fit(X_f_tr, y_f_tr)
                    p_val = est.predict_proba(X_f_val)[:, 1]

                auc_scores.append(self.evaluator.evaluate(y_f_val, p_val, compute_ci=False).roc_auc)
                brier_scores.append(self.evaluator.evaluate(y_f_val, p_val, compute_ci=False).brier_score)

            elapsed = time.time() - start_t
            mean_auc = float(np.mean(auc_scores))
            mean_gini = 2.0 * mean_auc - 1.0
            mean_brier = float(np.mean(brier_scores))

            cv_summary[name] = {
                "cv_mean_auc": round(mean_auc, 4),
                "cv_std_auc": round(float(np.std(auc_scores)), 4),
                "cv_mean_gini": round(mean_gini, 4),
                "cv_mean_brier": round(mean_brier, 4),
                "training_time_sec": round(elapsed, 2),
            }
            logger.info(
                "[%s] CV ROC-AUC: %.4f (+/- %.4f) | Gini: %.4f | Time: %.2fs",
                name, mean_auc, float(np.std(auc_scores)), mean_gini, elapsed,
            )

        # 5. Select Champion Model based on CV AUC
        champion_name = max(cv_summary.keys(), key=lambda k: cv_summary[k]["cv_mean_auc"])
        logger.info("Selected Champion Model: %s (CV AUC: %.4f)", champion_name, cv_summary[champion_name]["cv_mean_auc"])

        champ_spec = candidates[champion_name]
        champion_clf = champ_spec["estimator"]

        # Split training set for final probability calibration
        calib_split_idx = int(len(X_tr) * 0.8)
        X_sub_tr, X_calib = X_tr.iloc[:calib_split_idx], X_tr.iloc[calib_split_idx:]
        y_sub_tr, y_calib = y_train.iloc[:calib_split_idx], y_train.iloc[calib_split_idx:]

        scaler = StandardScaler() if champ_spec["scale"] else None
        if scaler is not None:
            X_sub_tr_s = scaler.fit_transform(X_sub_tr)
            X_calib_s = scaler.transform(X_calib)
            X_te_s = scaler.transform(X_te)
            champion_clf.fit(X_sub_tr_s, y_sub_tr)
            p_calib_raw = champion_clf.predict_proba(X_calib_s)[:, 1]
            p_test_raw = champion_clf.predict_proba(X_te_s)[:, 1]
        else:
            champion_clf.fit(X_sub_tr, y_sub_tr)
            p_calib_raw = champion_clf.predict_proba(X_calib)[:, 1]
            p_test_raw = champion_clf.predict_proba(X_te)[:, 1]

        # 6. Fit Probability Calibrator
        calibrator = ProbabilityCalibrator(method=self.config.calibration_method)
        calibrator.fit(p_calib_raw, y_calib)
        p_test_calibrated = calibrator.predict_proba(p_test_raw)

        # 7. Evaluate on Held-out Out-of-Sample Test Set
        logger.info("Evaluating Champion model on held-out test set (%d records)...", len(y_test))
        test_metrics = self.evaluator.evaluate(
            y_test,
            p_test_calibrated,
            compute_ci=True,
            n_bootstrap=self.config.n_bootstrap,
            random_seed=self.config.random_state,
        )

        logger.info(
            "Test Set Results — ROC-AUC: %.4f [95%% CI: %.4f - %.4f] | Gini: %.4f | KS: %.4f (at threshold %.4f)",
            test_metrics.roc_auc,
            test_metrics.auc_ci_lower,
            test_metrics.auc_ci_upper,
            test_metrics.gini,
            test_metrics.ks_statistic,
            test_metrics.ks_threshold,
        )

        artifact = ModelArtifact(
            model_name=champion_name,
            preprocessor=preprocessor,
            feature_engineer=feat_engineer,
            feature_names=feature_names,
            scaler=scaler,
            classifier=champion_clf,
            calibrator=calibrator,
            test_metrics=test_metrics,
            cv_summary=cv_summary,
        )

        # 8. Save Artifact to Disk
        artifact_path = MODELS_DIR / "pd_model_best.joblib"
        joblib.dump(artifact, artifact_path)
        logger.info("Serialized model artifact saved to %s", artifact_path)

        return artifact, cv_summary
