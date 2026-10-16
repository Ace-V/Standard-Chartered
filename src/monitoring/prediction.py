"""
CreditRisk Analytics — Monitoring Subsystem: Prediction Monitoring (Module D)
=============================================================================

Monitors model score and probability output distributions independently of labels:
1. Predicted default probability distribution statistics (mean, std, median, skewness, percentiles).
2. Risk segment / rating band distribution shifts.
3. Changes in average predicted probability (expected default rate shift).
4. Output Score / PD Population Stability Index (PSI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import skew

from src.monitoring.drift import PSIResult, calculate_psi
from src.risk.segmentation import MasterRatingScale


@dataclass
class ProbabilityDistributionStats:
    """Summary statistics for predicted probabilities."""
    count: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    skewness: float
    p10: float
    p25: float
    p75: float
    p90: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Count": self.count,
            "Mean PD": round(self.mean, 4),
            "Std Dev": round(self.std, 4),
            "Median PD": round(self.median, 4),
            "Min PD": round(self.min_val, 4),
            "Max PD": round(self.max_val, 4),
            "Skewness": round(self.skewness, 4),
            "P10": round(self.p10, 4),
            "P90": round(self.p90, 4),
        }


@dataclass
class PredictionMonitoringReport:
    """Consolidated Prediction & Risk Segment Distribution Report."""
    baseline_stats: ProbabilityDistributionStats
    monitoring_stats: ProbabilityDistributionStats
    mean_pd_delta: float
    mean_pd_shift_pct: float
    pd_psi_result: PSIResult
    risk_segment_migration: pd.DataFrame
    status: str  # "GREEN", "AMBER", "RED"
    alerts: List[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        """Compare baseline and monitoring probability statistics."""
        b = self.baseline_stats.to_dict()
        m = self.monitoring_stats.to_dict()
        rows = []
        for k in b.keys():
            if k == "Count":
                rows.append({"Metric": k, "Baseline": f"{int(b[k]):,}", "Monitoring": f"{int(m[k]):,}", "Delta": f"{int(m[k]) - int(b[k]):,}"})
            else:
                delta = m[k] - b[k]
                rows.append({"Metric": k, "Baseline": f"{b[k]:.4f}", "Monitoring": f"{m[k]:.4f}", "Delta": f"{delta:+.4f}"})
        return pd.DataFrame(rows)


class PredictionMonitor:
    """
    Surveillance service for model output probabilities and risk segment distributions.
    """

    def __init__(
        self,
        mean_shift_warning_pct: float = 15.0,
        mean_shift_critical_pct: float = 30.0,
        psi_green_thresh: float = 0.10,
        psi_amber_thresh: float = 0.25,
    ):
        self.mean_shift_warning_pct = mean_shift_warning_pct
        self.mean_shift_critical_pct = mean_shift_critical_pct
        self.psi_green_thresh = psi_green_thresh
        self.psi_amber_thresh = psi_amber_thresh
        self.rating_scale = MasterRatingScale()

    def _compute_stats(self, probs: np.ndarray) -> ProbabilityDistributionStats:
        p = np.asarray(probs, dtype=float)
        p = p[np.isfinite(p)]
        if len(p) == 0:
            return ProbabilityDistributionStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return ProbabilityDistributionStats(
            count=len(p),
            mean=float(np.mean(p)),
            std=float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
            median=float(np.median(p)),
            min_val=float(np.min(p)),
            max_val=float(np.max(p)),
            skewness=float(skew(p)) if len(p) > 2 else 0.0,
            p10=float(np.percentile(p, 10)),
            p25=float(np.percentile(p, 25)),
            p75=float(np.percentile(p, 75)),
            p90=float(np.percentile(p, 90)),
        )

    def monitor_predictions(
        self,
        baseline_probs: np.ndarray | pd.Series,
        monitoring_probs: np.ndarray | pd.Series,
    ) -> PredictionMonitoringReport:
        """
        Evaluate output probability distribution drift and risk rating migration.
        """
        b_p = np.asarray(baseline_probs, dtype=float)
        m_p = np.asarray(monitoring_probs, dtype=float)

        b_stats = self._compute_stats(b_p)
        m_stats = self._compute_stats(m_p)

        mean_delta = m_stats.mean - b_stats.mean
        mean_shift_pct = (mean_delta / b_stats.mean * 100.0) if b_stats.mean > 1e-6 else 0.0

        # Compute output score PSI
        pd_psi = calculate_psi(
            expected=b_p,
            actual=m_p,
            num_bins=10,
            metric_name="Calibrated_PD",
            green_thresh=self.psi_green_thresh,
            amber_thresh=self.psi_amber_thresh,
        )

        # Risk segment migration across Master Rating Scale
        b_grades = pd.Series(self.rating_scale.assign_grades_vectorized(b_p))
        m_grades = pd.Series(self.rating_scale.assign_grades_vectorized(m_p))

        b_share = (b_grades.value_counts(normalize=True) * 100.0)
        m_share = (m_grades.value_counts(normalize=True) * 100.0)

        all_grades = [tier["grade"] for tier in self.rating_scale.scale]
        migration_rows = []
        for g in all_grades:
            b_pct = b_share.get(g, 0.0)
            m_pct = m_share.get(g, 0.0)
            migration_rows.append({
                "Rating Grade": g,
                "Baseline Share (%)": round(b_pct, 2),
                "Monitoring Share (%)": round(m_pct, 2),
                "Net Shift (%)": round(m_pct - b_pct, 2),
            })
        df_migration = pd.DataFrame(migration_rows)

        # Determine alerts and status
        alerts: List[str] = []
        status = "GREEN"

        if pd_psi.traffic_light == "RED" or abs(mean_shift_pct) >= self.mean_shift_critical_pct:
            status = "RED"
            if pd_psi.traffic_light == "RED":
                alerts.append(f"CRITICAL: Output PD distribution has shifted severely (PSI={pd_psi.psi_value:.4f} >= {self.psi_amber_thresh}).")
            if abs(mean_shift_pct) >= self.mean_shift_critical_pct:
                alerts.append(f"CRITICAL: Mean predicted PD shifted by {mean_shift_pct:+.1f}% (critical threshold: ±{self.mean_shift_critical_pct}%).")
        elif pd_psi.traffic_light == "AMBER" or abs(mean_shift_pct) >= self.mean_shift_warning_pct:
            status = "AMBER"
            if pd_psi.traffic_light == "AMBER":
                alerts.append(f"WARNING: Output PD distribution exhibits moderate drift (PSI={pd_psi.psi_value:.4f}).")
            if abs(mean_shift_pct) >= self.mean_shift_warning_pct:
                alerts.append(f"WARNING: Mean predicted PD shifted by {mean_shift_pct:+.1f}% (warning threshold: ±{self.mean_shift_warning_pct}%).")

        return PredictionMonitoringReport(
            baseline_stats=b_stats,
            monitoring_stats=m_stats,
            mean_pd_delta=mean_delta,
            mean_pd_shift_pct=mean_shift_pct,
            pd_psi_result=pd_psi,
            risk_segment_migration=df_migration,
            status=status,
            alerts=alerts,
        )
