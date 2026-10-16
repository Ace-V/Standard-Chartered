"""
CreditRisk Analytics — Performance Decay & Model Monitoring
============================================================

Tracks discriminatory power erosion, Gini/KS degradation, and calibration drift
over time or across out-of-time (OOT) cohorts, generating compliance alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import logger
from src.models.evaluate import EvaluationMetrics, ModelEvaluator


@dataclass
class PerformanceDriftReport:
    """Summary of model performance decay between baseline and monitoring periods."""
    baseline_metrics: Optional[EvaluationMetrics]
    monitoring_metrics: Optional[EvaluationMetrics]
    auc_delta: float
    gini_delta: float
    gini_decay_pct: float
    ks_delta: float
    ks_decay_pct: float
    brier_delta: float
    log_loss_delta: float
    precision: float
    recall: float
    f1_score: float
    ece: float
    status: str  # "HEALTHY", "WARNING", "CRITICAL", "UNAVAILABLE_NO_LABELS", "INSUFFICIENT_SAMPLE"
    labels_available: bool = True
    sample_size: int = 0
    explanation: str = ""
    alerts: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert drift report to tabular overview."""
        if not self.labels_available or self.monitoring_metrics is None or self.baseline_metrics is None:
            return pd.DataFrame([{
                "Status": self.status,
                "Labels Available": self.labels_available,
                "Sample Size": self.sample_size,
                "Explanation": self.explanation,
            }])

        return pd.DataFrame([
            {"Metric": "ROC-AUC", "Baseline": round(self.baseline_metrics.roc_auc, 4), "Monitoring": round(self.monitoring_metrics.roc_auc, 4), "Delta": round(self.auc_delta, 4), "Decay Pct": "-"},
            {"Metric": "Gini (Somers' D)", "Baseline": round(self.baseline_metrics.gini, 4), "Monitoring": round(self.monitoring_metrics.gini, 4), "Delta": round(self.gini_delta, 4), "Decay Pct": f"{self.gini_decay_pct:.1f}%"},
            {"Metric": "KS Statistic", "Baseline": round(self.baseline_metrics.ks_statistic, 4), "Monitoring": round(self.monitoring_metrics.ks_statistic, 4), "Delta": round(self.ks_delta, 4), "Decay Pct": f"{self.ks_decay_pct:.1f}%"},
            {"Metric": "Brier Score", "Baseline": round(self.baseline_metrics.brier_score, 4), "Monitoring": round(self.monitoring_metrics.brier_score, 4), "Delta": round(self.brier_delta, 4), "Decay Pct": "-"},
            {"Metric": "Log Loss", "Baseline": round(self.baseline_metrics.log_loss_val, 4), "Monitoring": round(self.monitoring_metrics.log_loss_val, 4), "Delta": round(self.log_loss_delta, 4), "Decay Pct": "-"},
            {"Metric": "Precision", "Baseline": "-", "Monitoring": round(self.precision, 4), "Delta": "-", "Decay Pct": "-"},
            {"Metric": "Recall", "Baseline": "-", "Monitoring": round(self.recall, 4), "Delta": "-", "Decay Pct": "-"},
            {"Metric": "F1 Score", "Baseline": "-", "Monitoring": round(self.f1_score, 4), "Delta": "-", "Decay Pct": "-"},
            {"Metric": "ECE (Calibration)", "Baseline": "-", "Monitoring": round(self.ece, 4), "Delta": "-", "Decay Pct": "-"},
        ])


class PerformanceMonitor:
    """
    Evaluates out-of-time (OOT) model decay against historical validation benchmarks.

    IMPLEMENTATION RULES:
    1. Strictly separates feature drift from actual model performance deterioration.
    2. Never claims performance deterioration without ground-truth default outcomes.
    3. Handles delayed / missing labels and insufficient sample sizes gracefully.
    """

    def __init__(
        self,
        gini_alert_threshold_pct: float = 15.0,
        ks_alert_threshold_pct: float = 20.0,
        brier_alert_increase: float = 0.05,
    ):
        self.gini_alert_thresh = gini_alert_threshold_pct
        self.ks_alert_thresh = ks_alert_threshold_pct
        self.brier_alert_increase = brier_alert_increase
        self.evaluator = ModelEvaluator()

    def compare_performance(
        self,
        baseline_metrics: Optional[EvaluationMetrics],
        y_true_monitoring: Optional[np.ndarray | pd.Series],
        y_prob_monitoring: np.ndarray,
        threshold: float = 0.5,
    ) -> PerformanceDriftReport:
        """
        Compare current monitoring cohort against historical baseline metrics.
        """
        y_prob = np.asarray(y_prob_monitoring, dtype=float)
        n_obs = len(y_prob)

        # Case 1: Ground-truth outcome labels are unavailable
        if y_true_monitoring is None:
            return PerformanceDriftReport(
                baseline_metrics=baseline_metrics,
                monitoring_metrics=None,
                auc_delta=0.0,
                gini_delta=0.0,
                gini_decay_pct=0.0,
                ks_delta=0.0,
                ks_decay_pct=0.0,
                brier_delta=0.0,
                log_loss_delta=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                ece=0.0,
                status="UNAVAILABLE_NO_LABELS",
                labels_available=False,
                sample_size=n_obs,
                explanation=(
                    "Outcome labels are not available for this monitoring period. In credit risk, default "
                    "outcomes require a 12-month performance observation window (or 90+ DPD definition). "
                    "Actual model performance (ROC-AUC, Brier score, Log loss) cannot be evaluated or claimed "
                    "as deteriorated without ground-truth default labels. Only feature drift and prediction "
                    "distribution monitoring can be assessed."
                ),
                alerts=["INFO: Performance monitoring is dormant pending arrival of 12-month outcome maturity labels."],
            )

        y_true = np.asarray(y_true_monitoring, dtype=float)
        valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]

        if len(y_true) == 0:
            return PerformanceDriftReport(
                baseline_metrics=baseline_metrics,
                monitoring_metrics=None,
                auc_delta=0.0,
                gini_delta=0.0,
                gini_decay_pct=0.0,
                ks_delta=0.0,
                ks_decay_pct=0.0,
                brier_delta=0.0,
                log_loss_delta=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                ece=0.0,
                status="UNAVAILABLE_NO_LABELS",
                labels_available=False,
                sample_size=0,
                explanation="No valid non-null label pairs found in monitoring batch.",
                alerts=["INFO: Outcome labels are completely null."],
            )

        # Case 2: Insufficient sample size or lack of both classes
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        if len(y_true) < 30 or n_pos < 3 or n_neg < 3:
            return PerformanceDriftReport(
                baseline_metrics=baseline_metrics,
                monitoring_metrics=None,
                auc_delta=0.0,
                gini_delta=0.0,
                gini_decay_pct=0.0,
                ks_delta=0.0,
                ks_decay_pct=0.0,
                brier_delta=0.0,
                log_loss_delta=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                ece=0.0,
                status="INSUFFICIENT_SAMPLE",
                labels_available=True,
                sample_size=len(y_true),
                explanation=(
                    f"Sample size too small (n={len(y_true)}, defaults={n_pos}, non-defaults={n_neg}) "
                    "to compute statistically sound discriminatory performance metrics (minimum 30 accounts and 3 defaults required)."
                ),
                alerts=["WARNING: Insufficient sample size for reliable model performance evaluation."],
            )

        # Case 3: Labels available & sufficient sample size
        mon_metrics = self.evaluator.evaluate(y_true, y_prob, compute_ci=False)

        # Confusion matrix metrics at chosen threshold
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calibration ECE (10 quantile bins)
        from src.models.calibration import compute_calibration_curve
        _, _, _, ece, _ = compute_calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

        # Delta comparisons
        auc_delta = 0.0
        gini_delta = 0.0
        gini_decay_pct = 0.0
        ks_delta = 0.0
        ks_decay_pct = 0.0
        brier_delta = 0.0
        log_loss_delta = 0.0

        alerts = []
        status = "HEALTHY"

        if baseline_metrics is not None:
            auc_delta = mon_metrics.roc_auc - baseline_metrics.roc_auc
            gini_delta = mon_metrics.gini - baseline_metrics.gini
            ks_delta = mon_metrics.ks_statistic - baseline_metrics.ks_statistic
            brier_delta = mon_metrics.brier_score - baseline_metrics.brier_score
            log_loss_delta = mon_metrics.log_loss_val - baseline_metrics.log_loss_val

            base_gini = max(baseline_metrics.gini, 1e-4)
            base_ks = max(baseline_metrics.ks_statistic, 1e-4)

            gini_decay_pct = max(0.0, ((base_gini - mon_metrics.gini) / base_gini) * 100.0)
            ks_decay_pct = max(0.0, ((base_ks - mon_metrics.ks_statistic) / base_ks) * 100.0)

            if gini_decay_pct >= self.gini_alert_thresh:
                alerts.append(f"WARNING: Gini coefficient decayed by {gini_decay_pct:.1f}% (threshold: {self.gini_alert_thresh}%)")
                status = "WARNING"

            if ks_decay_pct >= self.ks_alert_thresh:
                alerts.append(f"WARNING: Kolmogorov-Smirnov statistic decayed by {ks_decay_pct:.1f}% (threshold: {self.ks_alert_thresh}%)")
                status = "WARNING"

            if brier_delta >= self.brier_alert_increase:
                alerts.append(f"WARNING: Brier score deteriorated by +{brier_delta:.4f} (probability error increased).")
                status = "WARNING"

            if gini_decay_pct >= 25.0 or ks_decay_pct >= 30.0:
                status = "CRITICAL"
                alerts.append("CRITICAL: Severe discriminatory power deterioration; model recalibration or retirement advised.")

        return PerformanceDriftReport(
            baseline_metrics=baseline_metrics,
            monitoring_metrics=mon_metrics,
            auc_delta=auc_delta,
            gini_delta=gini_delta,
            gini_decay_pct=gini_decay_pct,
            ks_delta=ks_delta,
            ks_decay_pct=ks_decay_pct,
            brier_delta=brier_delta,
            log_loss_delta=log_loss_delta,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            ece=ece,
            status=status,
            labels_available=True,
            sample_size=len(y_true),
            explanation=f"Performance evaluated successfully over mature cohort of {len(y_true)} labelled accounts.",
            alerts=alerts,
        )
