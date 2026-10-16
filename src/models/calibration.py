"""
CreditRisk Analytics — Probability Calibration Module
=====================================================

Implements regulatory probability calibration (Isotonic Regression & Platt Sigmoid Scaling)
ensuring that model-predicted Probability of Default (PD) closely reflects empirical default
frequencies across all risk tranches.

CONCEPTUAL FRAMEWORK — Three Distinct Performance Dimensions
-------------------------------------------------------------

1. RANKING QUALITY (Discrimination)
   Measures how well the model separates defaulters from non-defaulters in ordinal terms.
   Key metrics: ROC-AUC, Gini (Somers' D), Kolmogorov-Smirnov statistic.
   A model with ROC-AUC = 0.78 correctly ranks a randomly-chosen defaulter above a randomly-
   chosen non-defaulter 78% of the time. Ranking quality is insensitive to the absolute
   probability values — only their ordering matters.

2. PROBABILITY CALIBRATION (Accuracy of stated PD)
   Measures whether the predicted probability p(x) = 0.20 corresponds to an empirical
   default rate of ~20% across the sub-population of borrowers assigned that score.
   Key metrics: Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   Brier Score, Log Loss, Hosmer-Lemeshow chi-squared test.
   A model can discriminate well (high AUC) but be grossly miscalibrated (e.g., systematically
   underestimating default risk). For IFRS 9 and Basel, calibration is a regulatory requirement
   because PDs drive loan-loss provisions and capital consumption directly.

3. CLASSIFICATION THRESHOLD PERFORMANCE (Operational decision quality)
   Measures how well the model performs when a specific probability cut-off is applied to
   make binary decisions (approve/decline, flag for collections).
   Key metrics: Precision, Recall, F1-Score, Specificity, Accuracy.
   These metrics are only meaningful at a selected operating point. Credit decisions
   rarely use hard binary cutoffs — they use risk-tiered pricing and limits.
   Evaluation using accuracy alone is misleading in the presence of class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.config import logger


# ---------------------------------------------------------------------------
# Core calibration curve utility
# ---------------------------------------------------------------------------

def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Computes reliability curve coordinates and calibration error metrics.

    Parameters
    ----------
    y_true : array of binary labels (0/1)
    y_prob : array of predicted default probabilities
    n_bins : number of calibration bins
    strategy : 'quantile' (equal-frequency) or 'uniform' (equal-width)

    Returns
    -------
    prob_true  : Observed empirical default frequency in each bin
    prob_pred  : Average predicted default probability in each bin
    bin_counts : Number of observations in each bin
    ece        : Expected Calibration Error (population-weighted mean absolute error)
    mce        : Maximum Calibration Error (worst-bin absolute error)
    """
    y_t = np.asarray(y_true).astype(float)
    y_p = np.asarray(y_prob).astype(float)
    n_samples = len(y_t)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_p, quantiles * 100)
        bins[0] = 0.0
        bins[-1] = 1.0 + 1e-6
        bins = np.unique(bins)
    else:
        bins = np.linspace(0.0, 1.0 + 1e-6, n_bins + 1)

    bin_assignments = np.digitize(y_p, bins) - 1

    prob_true_list: List[float] = []
    prob_pred_list: List[float] = []
    bin_counts_list: List[int] = []
    errors: List[float] = []

    for b in range(len(bins) - 1):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        if count > 0:
            avg_pred = float(np.mean(y_p[mask]))
            avg_true = float(np.mean(y_t[mask]))
            prob_true_list.append(avg_true)
            prob_pred_list.append(avg_pred)
            bin_counts_list.append(count)
            errors.append(abs(avg_pred - avg_true))

    prob_true = np.array(prob_true_list)
    prob_pred = np.array(prob_pred_list)
    bin_counts = np.array(bin_counts_list)

    if len(bin_counts) > 0 and n_samples > 0:
        weights = bin_counts / n_samples
        ece = float(np.sum(weights * np.array(errors)))
        mce = float(np.max(errors))
    else:
        ece = 0.0
        mce = 0.0

    return prob_true, prob_pred, bin_counts, ece, mce


# ---------------------------------------------------------------------------
# Probability Calibrator
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """
    Fits monotonic transformation on raw model probabilities to produce
    actuarially accurate and Basel-compliant default probabilities.

    Calibration is performed on a held-out validation partition (separate
    from both training and test sets) to avoid overfitting or target leakage.

    Methods
    -------
    isotonic : Non-parametric monotone piecewise-constant mapping (preferred
               when calibration data is large enough, n > 1,000).
    sigmoid  : Platt scaling — logistic regression on log-odds space. More
               stable for smaller calibration sets.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Method must be 'isotonic' or 'sigmoid', got '{method}'")
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_raw_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration curve on validation partition (never on final test set)."""
        raw_p = np.asarray(y_raw_prob).ravel()
        y = np.asarray(y_true).ravel()

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0.0001, y_max=0.9999, out_of_bounds="clip")
            self.calibrator.fit(raw_p, y)
        else:
            # Platt Sigmoid: Logistic regression on log-odds
            log_odds = np.log(
                np.clip(raw_p, 1e-6, 1 - 1e-6) / (1.0 - np.clip(raw_p, 1e-6, 1 - 1e-6))
            ).reshape(-1, 1)
            self.calibrator = LogisticRegression(solver="lbfgs")
            self.calibrator.fit(log_odds, y)

        self.is_fitted = True
        logger.info("Fitted ProbabilityCalibrator using %s method.", self.method)
        return self

    def predict_proba(self, y_raw_prob: np.ndarray) -> np.ndarray:
        """Transform raw probability into calibrated default probability."""
        if not self.is_fitted or self.calibrator is None:
            raise RuntimeError("ProbabilityCalibrator is not fitted yet.")

        raw_p = np.asarray(y_raw_prob).ravel()

        if self.method == "isotonic":
            calibrated = self.calibrator.predict(raw_p)
        else:
            log_odds = np.log(
                np.clip(raw_p, 1e-6, 1 - 1e-6) / (1.0 - np.clip(raw_p, 1e-6, 1 - 1e-6))
            ).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(log_odds)[:, 1]

        # Ensure PD remains in valid probability range (0.0001 to 0.9999)
        return np.clip(calibrated, 0.0001, 0.9999)


# ---------------------------------------------------------------------------
# Bin-level calibration table
# ---------------------------------------------------------------------------

@dataclass
class BinCalibrationRow:
    """One row of a calibration analysis table for a single probability bin."""
    bin_label: str
    bin_low: float
    bin_high: float
    count: int
    mean_predicted_pd: float
    observed_default_rate: float
    calibration_error: float           # abs(mean_pred - observed)
    lower_ci_95: float                 # 95% Wilson score CI lower bound
    upper_ci_95: float                 # 95% Wilson score CI upper bound
    is_statistically_uncertain: bool   # True when count < 30


@dataclass
class CalibrationComparisonResult:
    """
    Comprehensive comparison of pre-calibration (raw) vs post-calibration metrics,
    including bin-level analysis and scalar diagnostic statistics.
    """
    # Scalar metrics — raw (uncalibrated)
    raw_brier: float
    raw_log_loss: float
    raw_ece: float
    raw_mce: float

    # Scalar metrics — post-calibration
    cal_brier: float
    cal_log_loss: float
    cal_ece: float
    cal_mce: float

    # Calibration method used
    method: str

    # Bin-level detail tables
    raw_bin_table: pd.DataFrame
    cal_bin_table: pd.DataFrame

    # Calibration curve arrays for plotting
    raw_prob_true: np.ndarray
    raw_prob_pred: np.ndarray
    raw_bin_counts: np.ndarray
    cal_prob_true: np.ndarray
    cal_prob_pred: np.ndarray
    cal_bin_counts: np.ndarray

    def improvement_pct(self, metric: str) -> float:
        """Return % improvement (positive = better) for 'brier', 'log_loss', 'ece', 'mce'."""
        raw_map = {"brier": self.raw_brier, "log_loss": self.raw_log_loss,
                   "ece": self.raw_ece, "mce": self.raw_mce}
        cal_map = {"brier": self.cal_brier, "log_loss": self.cal_log_loss,
                   "ece": self.cal_ece, "mce": self.cal_mce}
        raw_val = raw_map.get(metric, 0.0)
        cal_val = cal_map.get(metric, 0.0)
        if raw_val == 0.0:
            return 0.0
        return round((raw_val - cal_val) / raw_val * 100, 2)

    def scalar_summary_df(self) -> pd.DataFrame:
        """Return a side-by-side scalar metric comparison DataFrame."""
        return pd.DataFrame([
            {
                "Metric": "Brier Score",
                "Raw (Uncalibrated)": round(self.raw_brier, 5),
                "Post-Calibration": round(self.cal_brier, 5),
                "Improvement %": self.improvement_pct("brier"),
                "Direction": "↓ Lower is better",
            },
            {
                "Metric": "Log Loss",
                "Raw (Uncalibrated)": round(self.raw_log_loss, 5),
                "Post-Calibration": round(self.cal_log_loss, 5),
                "Improvement %": self.improvement_pct("log_loss"),
                "Direction": "↓ Lower is better",
            },
            {
                "Metric": "Expected Calibration Error (ECE)",
                "Raw (Uncalibrated)": round(self.raw_ece, 5),
                "Post-Calibration": round(self.cal_ece, 5),
                "Improvement %": self.improvement_pct("ece"),
                "Direction": "↓ Lower is better",
            },
            {
                "Metric": "Maximum Calibration Error (MCE)",
                "Raw (Uncalibrated)": round(self.raw_mce, 5),
                "Post-Calibration": round(self.cal_mce, 5),
                "Improvement %": self.improvement_pct("mce"),
                "Direction": "↓ Lower is better",
            },
        ])


def _wilson_score_ci(n_successes: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Preferred over normal approximation for small n and extreme proportions.
    Returns (lower, upper) bounds.
    """
    if n_total == 0:
        return 0.0, 1.0
    p_hat = n_successes / n_total
    denom = 1.0 + z ** 2 / n_total
    centre = (p_hat + z ** 2 / (2 * n_total)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_total + z ** 2 / (4 * n_total ** 2))) / denom
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def _build_bin_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Build detailed bin-level calibration table with Wilson CIs and uncertainty flags.
    Returns a DataFrame with one row per non-empty bin.
    """
    y_t = np.asarray(y_true).astype(float)
    y_p = np.asarray(y_prob).astype(float)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_p, quantiles * 100)
        bins[0] = 0.0
        bins[-1] = 1.0 + 1e-6
        bins = np.unique(bins)
    else:
        bins = np.linspace(0.0, 1.0 + 1e-6, n_bins + 1)

    bin_assignments = np.digitize(y_p, bins) - 1
    rows: List[BinCalibrationRow] = []

    for b in range(len(bins) - 1):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_low = float(bins[b])
        bin_high = float(bins[b + 1])
        mean_pred = float(np.mean(y_p[mask]))
        n_defaults = int(np.sum(y_t[mask]))
        obs_dr = n_defaults / count
        cal_err = abs(mean_pred - obs_dr)
        ci_lo, ci_hi = _wilson_score_ci(n_defaults, count)

        rows.append(BinCalibrationRow(
            bin_label=f"[{bin_low:.3f}, {bin_high:.3f})",
            bin_low=round(bin_low, 4),
            bin_high=round(bin_high, 4),
            count=count,
            mean_predicted_pd=round(mean_pred, 5),
            observed_default_rate=round(obs_dr, 5),
            calibration_error=round(cal_err, 5),
            lower_ci_95=round(ci_lo, 5),
            upper_ci_95=round(ci_hi, 5),
            is_statistically_uncertain=count < 30,
        ))

    return pd.DataFrame([
        {
            "Bin": r.bin_label,
            "Bin Low": r.bin_low,
            "Bin High": r.bin_high,
            "Count": r.count,
            "Mean Predicted PD": r.mean_predicted_pd,
            "Observed Default Rate": r.observed_default_rate,
            "Calibration Error": r.calibration_error,
            "95% CI Lower": r.lower_ci_95,
            "95% CI Upper": r.upper_ci_95,
            "Statistically Uncertain (n<30)": r.is_statistically_uncertain,
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# CalibrationAnalyser — main comparison engine
# ---------------------------------------------------------------------------

class CalibrationAnalyser:
    """
    Compares raw model output against post-calibration probabilities on the
    same evaluation data (held-out test set), computing bin-level and scalar
    diagnostic metrics.

    The calibrator MUST already be fitted on a held-out validation partition,
    never on the test set. This class only performs read-only analysis.

    Usage
    -----
    >>> analyser = CalibrationAnalyser(n_bins=10, strategy='quantile')
    >>> result = analyser.compare(
    ...     y_true=y_test,
    ...     y_raw_prob=raw_scores,
    ...     y_cal_prob=calibrated_scores,
    ...     method='isotonic'
    ... )
    >>> print(result.scalar_summary_df())
    """

    def __init__(self, n_bins: int = 10, strategy: str = "quantile"):
        self.n_bins = n_bins
        self.strategy = strategy

    def compare(
        self,
        y_true: np.ndarray,
        y_raw_prob: np.ndarray,
        y_cal_prob: np.ndarray,
        method: str = "isotonic",
    ) -> CalibrationComparisonResult:
        """
        Run side-by-side calibration analysis: raw vs calibrated probabilities.

        Parameters
        ----------
        y_true     : Binary labels from the held-out test set.
        y_raw_prob : Uncalibrated probabilities from the model's sigmoid/softmax output.
        y_cal_prob : Calibrated probabilities (output of ProbabilityCalibrator).
        method     : Name of calibration method applied ('isotonic' or 'sigmoid').

        Returns
        -------
        CalibrationComparisonResult with full diagnostic tables and curve arrays.
        """
        y_t = np.asarray(y_true).astype(float)
        y_r = np.asarray(y_raw_prob).astype(float)
        y_c = np.asarray(y_cal_prob).astype(float)

        # --- Scalar metrics ---
        raw_brier = float(brier_score_loss(y_t, y_r))
        cal_brier = float(brier_score_loss(y_t, y_c))

        raw_ll = float(log_loss(y_t, np.clip(y_r, 1e-15, 1 - 1e-15)))
        cal_ll = float(log_loss(y_t, np.clip(y_c, 1e-15, 1 - 1e-15)))

        raw_pt, raw_pp, raw_counts, raw_ece, raw_mce = compute_calibration_curve(
            y_t, y_r, n_bins=self.n_bins, strategy=self.strategy
        )
        cal_pt, cal_pp, cal_counts, cal_ece, cal_mce = compute_calibration_curve(
            y_t, y_c, n_bins=self.n_bins, strategy=self.strategy
        )

        # --- Bin-level tables ---
        raw_bin_table = _build_bin_table(y_t, y_r, n_bins=self.n_bins, strategy=self.strategy)
        cal_bin_table = _build_bin_table(y_t, y_c, n_bins=self.n_bins, strategy=self.strategy)

        logger.info(
            "CalibrationAnalyser — Raw: Brier=%.4f, ECE=%.4f | Calibrated: Brier=%.4f, ECE=%.4f [method=%s]",
            raw_brier, raw_ece, cal_brier, cal_ece, method,
        )

        return CalibrationComparisonResult(
            raw_brier=raw_brier,
            raw_log_loss=raw_ll,
            raw_ece=raw_ece,
            raw_mce=raw_mce,
            cal_brier=cal_brier,
            cal_log_loss=cal_ll,
            cal_ece=cal_ece,
            cal_mce=cal_mce,
            method=method,
            raw_bin_table=raw_bin_table,
            cal_bin_table=cal_bin_table,
            raw_prob_true=raw_pt,
            raw_prob_pred=raw_pp,
            raw_bin_counts=raw_counts,
            cal_prob_true=cal_pt,
            cal_prob_pred=cal_pp,
            cal_bin_counts=cal_counts,
        )
