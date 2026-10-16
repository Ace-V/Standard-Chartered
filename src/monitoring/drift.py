"""
CreditRisk Analytics — Population & Characteristic Stability Monitoring (PSI / CSI)
===================================================================================

Implements regulatory Population Stability Index (PSI) and Characteristic Stability
Index (CSI) algorithms according to standard banking supervisory guidelines.
Computes traffic light compliance indicators:
  - Green (PSI < 0.10): Stable population
  - Amber (0.10 <= PSI < 0.25): Moderate shift / Investigate
  - Red (PSI >= 0.25): Significant shift / Recalibrate model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import MonitoringConfig, logger, monitoring_config


@dataclass
class PSIResult:
    """Detailed summary of a Population Stability Index computation."""
    metric_name: str
    psi_value: float
    traffic_light: str
    interpretation: str
    bin_details: pd.DataFrame = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "psi_value": round(self.psi_value, 4),
            "traffic_light": self.traffic_light,
            "interpretation": self.interpretation,
        }


def calculate_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_bins: int = 10,
    metric_name: str = "Score_or_PD",
    green_thresh: float = 0.10,
    amber_thresh: float = 0.25,
) -> PSIResult:
    """
    Computes the Population Stability Index (PSI) between baseline and monitoring arrays.

    Formula:
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)

    # Remove any NaN or infinite values
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]

    if len(exp) == 0 or len(act) == 0:
        return PSIResult(
            metric_name=metric_name,
            psi_value=0.0,
            traffic_light="GREEN",
            interpretation="Insufficient data to compute stability.",
            bin_details=pd.DataFrame(),
        )

    # Establish quantile bin edges based on EXPECTED (baseline) population
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp, quantiles)
    # Ensure strictly increasing bin edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Deduplicate edges if tied values occur
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) <= 2:
        # Fallback to linear spacing if distribution has zero variance
        unique_edges = np.linspace(exp.min() - 1e-4, exp.max() + 1e-4, num_bins + 1)
        unique_edges[0] = -np.inf
        unique_edges[-1] = np.inf

    # Digitize both populations into the reference bin edges
    exp_counts = np.histogram(exp, bins=unique_edges)[0]
    act_counts = np.histogram(act, bins=unique_edges)[0]

    n_exp = len(exp)
    n_act = len(act)

    # Proportions
    exp_pct = exp_counts / n_exp
    act_pct = act_counts / n_act

    # Regularization to prevent division by zero or ln(0)
    eps = 1e-4
    exp_pct_safe = np.where(exp_pct == 0, eps, exp_pct)
    act_pct_safe = np.where(act_pct == 0, eps, act_pct)

    # Calculate bin contributions
    psi_contributions = (act_pct_safe - exp_pct_safe) * np.log(act_pct_safe / exp_pct_safe)
    total_psi = float(np.sum(psi_contributions))

    # Traffic light determination
    if total_psi < green_thresh:
        traffic_light = "GREEN"
        interp = "Stable distribution. No significant drift detected."
    elif total_psi < amber_thresh:
        traffic_light = "AMBER"
        interp = "Moderate drift detected. Review loan origination policies and credit risk segments."
    else:
        traffic_light = "RED"
        interp = "Severe drift detected. Model discrimination is likely compromised; recalibration required."

    # Construct tabular bin diagnostics
    bin_labels = [f"Bin {i+1} [{unique_edges[i]:.2f}, {unique_edges[i+1]:.2f})" for i in range(len(unique_edges) - 1)]
    bin_df = pd.DataFrame({
        "bin": bin_labels,
        "expected_count": exp_counts,
        "actual_count": act_counts,
        "expected_pct": np.round(exp_pct * 100, 2),
        "actual_pct": np.round(act_pct * 100, 2),
        "pct_difference": np.round((act_pct - exp_pct) * 100, 2),
        "psi_contribution": np.round(psi_contributions, 4),
    })

    return PSIResult(
        metric_name=metric_name,
        psi_value=total_psi,
        traffic_light=traffic_light,
        interpretation=interp,
        bin_details=bin_df,
    )


def calculate_csi_for_features(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
    features: List[str],
    num_bins: int = 10,
) -> pd.DataFrame:
    """
    Computes Characteristic Stability Index (CSI) across key predictive features
    to diagnose which risk factors are driving population drift.
    """
    csi_records = []

    for feat in features:
        if feat in df_expected.columns and feat in df_actual.columns:
            res = calculate_psi(
                expected=df_expected[feat],
                actual=df_actual[feat],
                num_bins=num_bins,
                metric_name=feat,
            )
            csi_records.append({
                "Feature": feat,
                "CSI_Value": round(res.psi_value, 4),
                "Traffic_Light": res.traffic_light,
                "Interpretation": res.interpretation,
            })

    csi_df = pd.DataFrame(csi_records)
    if not csi_df.empty:
        csi_df = csi_df.sort_values(by="CSI_Value", ascending=False).reset_index(drop=True)
    return csi_df


class DriftMonitor:
    """
    Comprehensive drift surveillance service for production credit risk pipelines.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or monitoring_config

    def run_full_drift_audit(
        self,
        df_baseline: pd.DataFrame,
        df_target: pd.DataFrame,
        score_col: str = "credit_score",
        pd_col: str = "calibrated_pd",
    ) -> Dict[str, Any]:
        """Execute full suite of PSI and CSI drift analytics."""
        score_psi = None
        if score_col in df_baseline.columns and score_col in df_target.columns:
            score_psi = calculate_psi(
                expected=df_baseline[score_col],
                actual=df_target[score_col],
                num_bins=self.config.num_bins,
                metric_name=score_col,
            )

        pd_psi = None
        if pd_col in df_baseline.columns and pd_col in df_target.columns:
            pd_psi = calculate_psi(
                expected=df_baseline[pd_col],
                actual=df_target[pd_col],
                num_bins=self.config.num_bins,
                metric_name=pd_col,
            )

        csi_df = calculate_csi_for_features(
            df_expected=df_baseline,
            df_actual=df_target,
            features=self.config.monitored_features,
            num_bins=self.config.num_bins,
        )

        return {
            "score_psi": score_psi,
            "pd_psi": pd_psi,
            "csi_features": csi_df,
        }


@dataclass
class DistributionComparisonResult:
    """Statistical distribution comparison between baseline and monitoring features."""
    feature_name: str
    baseline_count: int
    monitoring_count: int
    baseline_mean: float
    monitoring_mean: float
    mean_delta: float
    mean_shift_pct: float
    baseline_std: float
    monitoring_std: float
    baseline_median: float
    monitoring_median: float
    ks_statistic: float
    ks_pvalue: float
    psi_result: PSIResult
    drift_detected: bool
    summary_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature_name,
            "baseline_mean": round(self.baseline_mean, 4),
            "monitoring_mean": round(self.monitoring_mean, 4),
            "mean_shift_pct": round(self.mean_shift_pct, 2),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "psi_value": round(self.psi_result.psi_value, 4),
            "traffic_light": self.psi_result.traffic_light,
            "drift_detected": self.drift_detected,
        }


def compare_feature_distributions(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    feature_name: str,
    num_bins: int = 10,
    ks_alpha: float = 0.01,
    green_thresh: float = 0.10,
    amber_thresh: float = 0.25,
) -> DistributionComparisonResult:
    """
    Perform formal statistical distribution comparison:
    1. Population Stability Index (PSI) with configurable thresholds
    2. Kolmogorov-Smirnov (KS) two-sample nonparametric test
    3. Parametric shift in central tendency (mean) and dispersion (std)
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]

    if len(exp) == 0 or len(act) == 0:
        empty_psi = PSIResult(
            metric_name=feature_name,
            psi_value=0.0,
            traffic_light="GREEN",
            interpretation="Insufficient valid observations.",
            bin_details=pd.DataFrame(),
        )
        return DistributionComparisonResult(
            feature_name=feature_name,
            baseline_count=len(exp),
            monitoring_count=len(act),
            baseline_mean=0.0,
            monitoring_mean=0.0,
            mean_delta=0.0,
            mean_shift_pct=0.0,
            baseline_std=0.0,
            monitoring_std=0.0,
            baseline_median=0.0,
            monitoring_median=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            psi_result=empty_psi,
            drift_detected=False,
            summary_message="Insufficient data to evaluate feature distribution.",
        )

    # 1. PSI Calculation
    psi_res = calculate_psi(
        expected=exp,
        actual=act,
        num_bins=num_bins,
        metric_name=feature_name,
        green_thresh=green_thresh,
        amber_thresh=amber_thresh,
    )

    # 2. Kolmogorov-Smirnov 2-sample test
    ks_stat, ks_pval = ks_2samp(exp, act)

    # 3. Summary statistics
    b_mean, m_mean = float(np.mean(exp)), float(np.mean(act))
    b_std, m_std = float(np.std(exp, ddof=1)) if len(exp) > 1 else 0.0, float(np.std(act, ddof=1)) if len(act) > 1 else 0.0
    b_med, m_med = float(np.median(exp)), float(np.median(act))

    mean_delta = m_mean - b_mean
    mean_shift_pct = (mean_delta / abs(b_mean) * 100.0) if abs(b_mean) > 1e-6 else 0.0

    # Drift decision: Flagged if PSI >= warning threshold OR statistically significant KS shift
    drift_detected = bool((psi_res.traffic_light in ("AMBER", "RED")) or (ks_pval < ks_alpha and abs(mean_shift_pct) > 5.0))

    msg = f"Feature '{feature_name}': PSI={psi_res.psi_value:.4f} ({psi_res.traffic_light}), KS stat={ks_stat:.4f} (p={ks_pval:.2e})."

    return DistributionComparisonResult(
        feature_name=feature_name,
        baseline_count=len(exp),
        monitoring_count=len(act),
        baseline_mean=b_mean,
        monitoring_mean=m_mean,
        mean_delta=mean_delta,
        mean_shift_pct=mean_shift_pct,
        baseline_std=b_std,
        monitoring_std=m_std,
        baseline_median=b_med,
        monitoring_median=m_med,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
        psi_result=psi_res,
        drift_detected=drift_detected,
        summary_message=msg,
    )


def get_psi_methodology_and_limitations() -> Dict[str, Any]:
    """
    Document PSI binning methodology and critical mathematical/business limitations.
    """
    return {
        "formula": "PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))",
        "binning_methodology": {
            "type": "Baseline Quantile Bins (Equal-Frequency on Expected Population)",
            "rationale": (
                "Decile binning based on the baseline training distribution ensures each reference bin "
                "contains ~10% of the population under stability, maximizing statistical resolution across "
                "the entire score or feature spectrum."
            ),
            "zero_handling": "A smoothing epsilon of 1e-4 is applied to zero-count bins to prevent division by zero or ln(0).",
        },
        "regulatory_thresholds": {
            "GREEN": "PSI < 0.10: Stable distribution, no model or portfolio intervention required.",
            "AMBER": "0.10 <= PSI < 0.25: Moderate shift detected, enhanced surveillance and credit policy review advised.",
            "RED": "PSI >= 0.25: Significant population drift, model discrimination may be impaired; formal recalibration required.",
        },
        "critical_limitations": [
            "Covariate Shift vs. Concept Drift: PSI measures shifts in feature inputs P(X). It does NOT prove that "
            "model performance P(Y|X) has degraded. A model can experience feature shift while maintaining perfect discriminatory rank-order.",
            "1D Marginal Invariance: PSI assesses single features in isolation; it cannot detect changes in multivariate correlation structures or copula dependencies.",
            "Tied Value Distortion: Highly discrete or zero-inflated variables (e.g. delinquency months = 0) can cause collapsed quantile bins.",
            "Sample Size Sensitivity: Small sample sizes (<500 records) produce wider sampling variance in PSI estimates.",
        ],
    }

