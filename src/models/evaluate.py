"""
CreditRisk Analytics — Model Evaluation & Statistical Testing
=============================================================

Implements quantitative credit risk evaluation metrics aligned with Basel II/III
and regulatory model risk management standards (SR 11-7 / OCC 2011-12).
Calculates ROC-AUC, Gini (Somers' D), Kolmogorov-Smirnov (KS) statistic,
Brier score, PR-AUC, and non-parametric bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import logger


@dataclass
class EvaluationMetrics:
    """Comprehensive container for credit scoring performance metrics."""
    roc_auc: float
    gini: float
    ks_statistic: float
    ks_threshold: float
    brier_score: float
    log_loss_val: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_mat: Dict[str, int]
    auc_ci_lower: float = 0.0
    auc_ci_upper: float = 0.0
    gini_ci_lower: float = 0.0
    gini_ci_upper: float = 0.0
    ks_ci_lower: float = 0.0
    ks_ci_upper: float = 0.0
    hosmer_lemeshow_stat: Optional[float] = None
    hosmer_lemeshow_pvalue: Optional[float] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging and persistence."""
        return {
            "roc_auc": round(self.roc_auc, 4),
            "gini": round(self.gini, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_threshold": round(self.ks_threshold, 4),
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss_val, 4),
            "pr_auc": round(self.pr_auc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "auc_95_ci": [round(self.auc_ci_lower, 4), round(self.auc_ci_upper, 4)],
            "gini_95_ci": [round(self.gini_ci_lower, 4), round(self.gini_ci_upper, 4)],
            "ks_95_ci": [round(self.ks_ci_lower, 4), round(self.ks_ci_upper, 4)],
            "confusion_matrix": self.confusion_mat,
        }


class ModelEvaluator:
    """
    Computes statistical discriminatory power, accuracy, and calibration tests
    for credit default prediction models.
    """

    def __init__(self, cutoff: float = 0.5):
        self.cutoff = cutoff

    def calculate_ks(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """
        Compute the two-sample Kolmogorov-Smirnov (KS) statistic:
        max separation between empirical cumulative distributions of defaults and non-defaults.
        """
        # Sort by predicted probability descending
        desc_order = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_order]
        y_prob_sorted = y_prob[desc_order]

        n_defaults = np.sum(y_true_sorted == 1)
        n_non_defaults = np.sum(y_true_sorted == 0)

        if n_defaults == 0 or n_non_defaults == 0:
            return 0.0, 0.5

        cum_defaults = np.cumsum(y_true_sorted == 1) / n_defaults
        cum_non_defaults = np.cumsum(y_true_sorted == 0) / n_non_defaults

        diff = np.abs(cum_defaults - cum_non_defaults)
        ks_idx = np.argmax(diff)
        ks_stat = float(diff[ks_idx])
        ks_threshold = float(y_prob_sorted[ks_idx])

        return ks_stat, ks_threshold

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray,
        compute_ci: bool = True,
        n_bootstrap: int = 500,
        random_seed: int = 42,
    ) -> EvaluationMetrics:
        """
        Evaluate full set of credit scoring discriminatory and calibration metrics.
        """
        y_true_arr = np.asarray(y_true).astype(int)
        y_prob_arr = np.asarray(y_prob).astype(float)
        y_pred_arr = (y_prob_arr >= self.cutoff).astype(int)

        # 1. Discrimination Metrics
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        gini = float(2.0 * auc - 1.0)
        ks_stat, ks_thresh = self.calculate_ks(y_true_arr, y_prob_arr)
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

        # 2. Probability Calibration Error Metrics
        brier = float(brier_score_loss(y_true_arr, y_prob_arr))
        ll = float(log_loss(y_true_arr, np.clip(y_prob_arr, 1e-15, 1 - 1e-15)))

        # 3. Discrete Cutoff Metrics
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)
        conf_dict = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

        # 4. Hosmer-Lemeshow Goodness-of-Fit Test (Deciles of predicted risk)
        hl_stat, hl_pval = self._hosmer_lemeshow(y_true_arr, y_prob_arr)

        # 5. Bootstrap Confidence Intervals (95% CI)
        auc_ci = (auc, auc)
        gini_ci = (gini, gini)
        ks_ci = (ks_stat, ks_stat)

        if compute_ci and len(y_true_arr) > 50:
            auc_ci, gini_ci, ks_ci = self._bootstrap_ci(
                y_true_arr, y_prob_arr, n_bootstrap=n_bootstrap, seed=random_seed
            )

        metrics = EvaluationMetrics(
            roc_auc=auc,
            gini=gini,
            ks_statistic=ks_stat,
            ks_threshold=ks_thresh,
            brier_score=brier,
            log_loss_val=ll,
            pr_auc=pr_auc,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            confusion_mat=conf_dict,
            auc_ci_lower=auc_ci[0],
            auc_ci_upper=auc_ci[1],
            gini_ci_lower=gini_ci[0],
            gini_ci_upper=gini_ci[1],
            ks_ci_lower=ks_ci[0],
            ks_ci_upper=ks_ci[1],
            hosmer_lemeshow_stat=hl_stat,
            hosmer_lemeshow_pvalue=hl_pval,
        )

        return metrics

    def _hosmer_lemeshow(self, y_true: np.ndarray, y_prob: np.ndarray, g: int = 10) -> Tuple[float, float]:
        """Compute Hosmer-Lemeshow calibration chi-squared test statistic."""
        try:
            df = pd.DataFrame({"true": y_true, "prob": y_prob})
            df["bin"] = pd.qcut(df["prob"], q=g, duplicates="drop")
            grouped = df.groupby("bin", observed=False).agg(
                n=("true", "count"),
                obs_defaults=("true", "sum"),
                exp_defaults=("prob", "sum"),
            )
            obs_non = grouped["n"] - grouped["obs_defaults"]
            exp_non = grouped["n"] - grouped["exp_defaults"]

            # Chi-square contributions
            chi2_def = ((grouped["obs_defaults"] - grouped["exp_defaults"]) ** 2) / np.maximum(grouped["exp_defaults"], 1e-6)
            chi2_non = ((obs_non - exp_non) ** 2) / np.maximum(exp_non, 1e-6)
            hl_stat = float((chi2_def + chi2_non).sum())
            dof = max(1, len(grouped) - 2)
            p_val = float(1.0 - stats.chi2.cdf(hl_stat, dof))
            return hl_stat, p_val
        except Exception:
            return 0.0, 1.0

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Non-parametric bootstrapping for ROC-AUC, Gini, and KS confidence intervals."""
        rng = np.random.RandomState(seed)
        n = len(y_true)
        auc_boot = []
        ks_boot = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            y_b = y_true[idx]
            p_b = y_prob[idx]
            # Skip degenerate resamples
            if np.unique(y_b).size < 2:
                continue
            auc_b = roc_auc_score(y_b, p_b)
            ks_b, _ = self.calculate_ks(y_b, p_b)
            auc_boot.append(auc_b)
            ks_boot.append(ks_b)

        if not auc_boot:
            return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

        auc_low, auc_high = np.percentile(auc_boot, [2.5, 97.5])
        gini_low = 2.0 * auc_low - 1.0
        gini_high = 2.0 * auc_high - 1.0
        ks_low, ks_high = np.percentile(ks_boot, [2.5, 97.5])

        return (float(auc_low), float(auc_high)), (float(gini_low), float(gini_high)), (float(ks_low), float(ks_high))
