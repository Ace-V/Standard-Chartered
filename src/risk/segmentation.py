"""
CreditRisk Analytics — Risk Segmentation, Scorecard Scaling & Decile Analysis
==============================================================================

Implements:
  - Banking bureau credit scorecard scaling (Base Score, PDO, Target Odds)
  - Master Rating Scale (AAA to CCC/Default) mapping continuous calibrated PDs
  - BorrowerRiskSegmenter: configurable probability-band risk segments with
    observed default rate, mean predicted PD, calibration error, and
    Wilson score confidence intervals
  - Risk decile analysis: monotonicity testing across predicted probability deciles

IMPORTANT DISCLOSURE
--------------------
All risk segment boundaries (Low Risk, Medium-Low, Medium-High, High, Very High)
are ILLUSTRATIVE thresholds chosen for demonstration purposes only. They are NOT
bank underwriting policy, regulatory minimums, or validated credit guidelines.
In practice, segment boundaries must be derived from:
  • Institution-specific risk appetite and capital allocation
  • Regulatory-approved PD model validation and backtesting outcomes
  • Supervisory review (Basel IRB approval process)
  • IFRS 9 / CECL model governance sign-off
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RATING_SCALE, ScorecardConfig, logger, scorecard_config


# ---------------------------------------------------------------------------
# Credit Scorecard Transformer
# ---------------------------------------------------------------------------

class CreditScorecard:
    """
    Transforms Probability of Default into a standard 300-850 credit bureau score.

    Mathematical formulation:
        Odds = (1 - PD) / PD  [Good:Bad odds]
        Factor = PDO / ln(2)
        Offset = BaseScore - Factor * ln(TargetOdds)
        Score = Offset + Factor * ln(Odds)
    """

    def __init__(self, config: Optional[ScorecardConfig] = None):
        self.config = config or scorecard_config
        self.factor = self.config.pdo / np.log(2.0)
        self.offset = self.config.base_score - (self.factor * np.log(self.config.target_odds))

    def pd_to_score(self, pd_values: np.ndarray | pd.Series | float) -> np.ndarray | float:
        """Convert default probability to credit score."""
        is_scalar = isinstance(pd_values, (float, int, np.floating))
        p = np.asarray(pd_values, dtype=float)

        # Clip p to avoid log(0) or division by zero
        p_safe = np.clip(p, 1e-6, 1.0 - 1e-6)
        odds = (1.0 - p_safe) / p_safe

        score = self.offset + (self.factor * np.log(odds))
        score_bounded = np.clip(score, self.config.min_score, self.config.max_score)

        if is_scalar:
            return float(round(float(score_bounded), 1))
        return np.round(score_bounded, 1)

    def score_to_pd(self, scores: np.ndarray | pd.Series | float) -> np.ndarray | float:
        """Inverse transformation: calculate implied default probability from credit score."""
        is_scalar = isinstance(scores, (float, int, np.floating))
        s = np.asarray(scores, dtype=float)

        log_odds = (s - self.offset) / self.factor
        odds = np.exp(log_odds)
        pd_val = 1.0 / (1.0 + odds)

        if is_scalar:
            return float(pd_val)
        return pd_val


# ---------------------------------------------------------------------------
# Master Rating Scale
# ---------------------------------------------------------------------------

class MasterRatingScale:
    """
    Institutional Master Rating Scale categorizing default risk into granular grades.
    """

    def __init__(self, scale_definitions: Optional[List[Dict[str, Any]]] = None):
        self.scale = scale_definitions or RATING_SCALE

    def assign_grade(self, pd_value: float) -> str:
        """Assign letter rating grade to an individual default probability."""
        for tier in self.scale:
            if tier["pd_min"] <= pd_value < tier["pd_max"]:
                return tier["grade"]
        return self.scale[-1]["grade"]

    def assign_grades_vectorized(self, pd_series: pd.Series | np.ndarray) -> np.ndarray:
        """Assign letter grades to array of probabilities."""
        p = np.asarray(pd_series, dtype=float)
        grades = np.full(p.shape, fill_value=self.scale[-1]["grade"], dtype=object)

        for tier in self.scale:
            mask = (p >= tier["pd_min"]) & (p < tier["pd_max"])
            grades[mask] = tier["grade"]

        # Ensure values exactly at 1.0 are assigned default tier
        grades[p >= 1.0] = self.scale[-1]["grade"]
        return grades

    def summarize_portfolio(self, df: pd.DataFrame, pd_col: str, target_col: Optional[str] = None) -> pd.DataFrame:
        """Generate rating grade distribution and observed default statistics."""
        data = df.copy()
        if "rating_grade" not in data.columns:
            data["rating_grade"] = self.assign_grades_vectorized(data[pd_col])

        summary_rows = []
        for tier in self.scale:
            grade = tier["grade"]
            sub = data[data["rating_grade"] == grade]
            count = len(sub)
            pct = (count / len(data)) * 100 if len(data) > 0 else 0.0
            avg_pd = sub[pd_col].mean() if count > 0 else 0.0
            obs_dr = sub[target_col].mean() if (target_col and target_col in sub.columns and count > 0) else None

            row = {
                "Grade": grade,
                "Description": tier["description"],
                "PD_Band_Min": tier["pd_min"],
                "PD_Band_Max": tier["pd_max"],
                "Account_Count": count,
                "Portfolio_Share_Pct": round(pct, 2),
                "Average_Predicted_PD": round(avg_pd, 4),
            }
            if obs_dr is not None:
                row["Observed_Default_Rate"] = round(obs_dr, 4)
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Basic RiskSegmenter (scorecard + rating)
# ---------------------------------------------------------------------------

class RiskSegmenter:
    """
    High-level orchestration combining scorecard scoring and rating grade assignment.
    """

    def __init__(self, scorecard: Optional[CreditScorecard] = None, rating_scale: Optional[MasterRatingScale] = None):
        self.scorecard = scorecard or CreditScorecard()
        self.rating_scale = rating_scale or MasterRatingScale()

    def segment(self, df: pd.DataFrame, pd_col: str = "calibrated_pd") -> pd.DataFrame:
        """Enrich portfolio DataFrame with credit score and master rating grade."""
        out = df.copy()
        if pd_col not in out.columns:
            raise KeyError(f"Probability column '{pd_col}' not found in DataFrame.")

        out["credit_score"] = self.scorecard.pd_to_score(out[pd_col].values)
        out["rating_grade"] = self.rating_scale.assign_grades_vectorized(out[pd_col].values)
        return out


# ---------------------------------------------------------------------------
# Illustrative Risk Segment definitions
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE ONLY — not bank policy. See module docstring for full disclaimer.
ILLUSTRATIVE_RISK_BANDS: List[Dict[str, Any]] = [
    {
        "segment": "Low Risk",
        "pd_low": 0.0,
        "pd_high": 0.05,
        "color": "#0D9488",
    },
    {
        "segment": "Medium-Low Risk",
        "pd_low": 0.05,
        "pd_high": 0.15,
        "color": "#0284C7",
    },
    {
        "segment": "Medium-High Risk",
        "pd_low": 0.15,
        "pd_high": 0.30,
        "color": "#F59E0B",
    },
    {
        "segment": "High Risk",
        "pd_low": 0.30,
        "pd_high": 0.60,
        "color": "#EF4444",
    },
    {
        "segment": "Very High Risk",
        "pd_low": 0.60,
        "pd_high": 1.01,
        "color": "#7C3AED",
    },
]


# ---------------------------------------------------------------------------
# Dataclass for one segment row
# ---------------------------------------------------------------------------

@dataclass
class SegmentRow:
    """Descriptive statistics for a single borrower risk segment."""
    segment: str
    pd_low: float
    pd_high: float
    count: int
    portfolio_share_pct: float
    mean_predicted_pd: float
    observed_default_rate: float
    calibration_error: float          # |mean_pred - obs_dr|
    ci_lower_95: float                # Wilson score 95% CI lower
    ci_upper_95: float                # Wilson score 95% CI upper
    is_statistically_uncertain: bool  # True when count < 30


@dataclass
class DecileRow:
    """Descriptive statistics for one risk decile (D1 = lowest PD, D10 = highest PD)."""
    decile: int
    pd_low: float
    pd_high: float
    count: int
    mean_predicted_pd: float
    observed_default_rate: float
    calibration_error: float
    ci_lower_95: float
    ci_upper_95: float
    is_statistically_uncertain: bool
    is_monotone_with_previous: Optional[bool]  # None for decile 1 (no predecessor)


@dataclass
class BorrowerSegmentationResult:
    """
    Full output of BorrowerRiskSegmenter.run(), combining segment-level
    and decile-level analyses plus portfolio-wide monotonicity diagnostics.

    Attributes
    ----------
    segment_table : pd.DataFrame
        One row per configurable risk band with counts, default rates, CIs,
        and calibration error. Bands with insufficient data are flagged.

    decile_table : pd.DataFrame
        Ten equal-frequency deciles ranked by predicted PD (D1 lowest, D10 highest).
        Includes monotonicity check — whether observed default rate increases
        across consecutive deciles as would be expected from a well-ordered model.

    monotonicity_violations : int
        Number of decile transitions where the observed default rate *decreases*
        contrary to the predicted PD ordering. Zero violations is ideal.

    monotonicity_spearman_r : float
        Spearman rank correlation between decile rank and observed default rate.
        Values close to +1.0 indicate strong monotone risk ordering.

    monotonicity_spearman_p : float
        Two-sided p-value for the Spearman correlation. Small p-values indicate
        statistically significant monotone ordering.

    overall_calibration_error : float
        Portfolio-wide mean absolute calibration error (unweighted by bin size).
    """
    segment_table: pd.DataFrame
    decile_table: pd.DataFrame
    monotonicity_violations: int
    monotonicity_spearman_r: float
    monotonicity_spearman_p: float
    overall_calibration_error: float
    scored_df: pd.DataFrame  # full portfolio df enriched with segment + decile columns


def _wilson_ci(n_successes: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 1.0
    p_hat = n_successes / n_total
    denom = 1.0 + z ** 2 / n_total
    centre = (p_hat + z ** 2 / (2 * n_total)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_total + z ** 2 / (4 * n_total ** 2))) / denom
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


# ---------------------------------------------------------------------------
# BorrowerRiskSegmenter — main analysis engine
# ---------------------------------------------------------------------------

class BorrowerRiskSegmenter:
    """
    Rigorous borrower risk segmentation using predicted default probabilities.

    All segment boundaries are illustrative and configurable via `risk_bands`.
    Results include:
    - Borrower count and portfolio share per segment
    - Observed default rate (empirical, not assumed) with 95% Wilson score CIs
    - Mean predicted probability per segment
    - Calibration error per segment (|predicted - observed|)
    - Statistical uncertainty flag when segment contains fewer than 30 observations
    - Ten-decile risk ordering analysis with monotonicity testing

    The analyser does NOT assume that risk ordering is correct or that calibration
    is good — it measures and reports both explicitly, including violations.
    """

    #: Minimum borrowers in a segment to consider it statistically reliable.
    UNCERTAINTY_THRESHOLD = 30

    def __init__(
        self,
        risk_bands: Optional[List[Dict[str, Any]]] = None,
        n_deciles: int = 10,
    ):
        self.risk_bands = risk_bands or ILLUSTRATIVE_RISK_BANDS
        self.n_deciles = n_deciles

    def run(
        self,
        df: pd.DataFrame,
        pd_col: str,
        target_col: str,
    ) -> BorrowerSegmentationResult:
        """
        Execute full segmentation and decile analysis.

        Parameters
        ----------
        df         : DataFrame containing calibrated PD column and true default labels.
        pd_col     : Column name of calibrated predicted default probabilities.
        target_col : Column name of binary default indicator (0/1).

        Returns
        -------
        BorrowerSegmentationResult with segment table, decile table, and
        portfolio-level monotonicity diagnostics.
        """
        if pd_col not in df.columns:
            raise KeyError(f"PD column '{pd_col}' not found in DataFrame.")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

        data = df[[pd_col, target_col]].copy().dropna()
        n_total = len(data)

        # ------------------------------------------------------------------
        # 1. Segment-level analysis
        # ------------------------------------------------------------------
        segment_rows: List[SegmentRow] = []

        for band in self.risk_bands:
            mask = (data[pd_col] >= band["pd_low"]) & (data[pd_col] < band["pd_high"])
            sub = data[mask]
            count = len(sub)
            pct = (count / n_total * 100) if n_total > 0 else 0.0

            if count > 0:
                mean_pred = float(sub[pd_col].mean())
                n_defaults = int(sub[target_col].sum())
                obs_dr = n_defaults / count
                ci_lo, ci_hi = _wilson_ci(n_defaults, count)
                cal_err = abs(mean_pred - obs_dr)
            else:
                mean_pred = 0.0
                obs_dr = 0.0
                ci_lo, ci_hi = 0.0, 1.0
                cal_err = 0.0

            segment_rows.append(SegmentRow(
                segment=band["segment"],
                pd_low=band["pd_low"],
                pd_high=band["pd_high"],
                count=count,
                portfolio_share_pct=round(pct, 2),
                mean_predicted_pd=round(mean_pred, 5),
                observed_default_rate=round(obs_dr, 5),
                calibration_error=round(cal_err, 5),
                ci_lower_95=round(ci_lo, 5),
                ci_upper_95=round(ci_hi, 5),
                is_statistically_uncertain=count < self.UNCERTAINTY_THRESHOLD,
            ))

        segment_table = pd.DataFrame([
            {
                "Segment (Illustrative)": r.segment,
                "PD Band Low": r.pd_low,
                "PD Band High": r.pd_high,
                "Borrower Count": r.count,
                "Portfolio Share %": r.portfolio_share_pct,
                "Mean Predicted PD": r.mean_predicted_pd,
                "Observed Default Rate": r.observed_default_rate,
                "Calibration Error": r.calibration_error,
                "95% CI Lower": r.ci_lower_95,
                "95% CI Upper": r.ci_upper_95,
                "Statistically Uncertain (n<30)": r.is_statistically_uncertain,
            }
            for r in segment_rows
        ])

        # ------------------------------------------------------------------
        # 2. Decile analysis
        # ------------------------------------------------------------------
        data = data.copy()
        data["_decile"] = pd.qcut(
            data[pd_col], q=self.n_deciles, labels=False, duplicates="drop"
        ) + 1  # 1-indexed (D1 = lowest PD, D10 = highest PD)

        decile_rows: List[DecileRow] = []
        prev_obs_dr: Optional[float] = None

        for d in sorted(data["_decile"].dropna().unique()):
            sub = data[data["_decile"] == d]
            count = len(sub)
            mean_pred = float(sub[pd_col].mean())
            pd_lo = float(sub[pd_col].min())
            pd_hi = float(sub[pd_col].max())
            n_defaults = int(sub[target_col].sum())
            obs_dr = n_defaults / count if count > 0 else 0.0
            ci_lo, ci_hi = _wilson_ci(n_defaults, count)
            cal_err = abs(mean_pred - obs_dr)

            if prev_obs_dr is None:
                is_monotone = None
            else:
                is_monotone = obs_dr >= prev_obs_dr

            decile_rows.append(DecileRow(
                decile=int(d),
                pd_low=round(pd_lo, 5),
                pd_high=round(pd_hi, 5),
                count=count,
                mean_predicted_pd=round(mean_pred, 5),
                observed_default_rate=round(obs_dr, 5),
                calibration_error=round(cal_err, 5),
                ci_lower_95=round(ci_lo, 5),
                ci_upper_95=round(ci_hi, 5),
                is_statistically_uncertain=count < self.UNCERTAINTY_THRESHOLD,
                is_monotone_with_previous=is_monotone,
            ))
            prev_obs_dr = obs_dr

        decile_table = pd.DataFrame([
            {
                "Decile": r.decile,
                "PD Low": r.pd_low,
                "PD High": r.pd_high,
                "Count": r.count,
                "Mean Predicted PD": r.mean_predicted_pd,
                "Observed Default Rate": r.observed_default_rate,
                "Calibration Error": r.calibration_error,
                "95% CI Lower": r.ci_lower_95,
                "95% CI Upper": r.ci_upper_95,
                "Monotone (↑ from prev)": r.is_monotone_with_previous,
                "Statistically Uncertain": r.is_statistically_uncertain,
            }
            for r in decile_rows
        ])

        # ------------------------------------------------------------------
        # 3. Monotonicity statistics
        # ------------------------------------------------------------------
        violations = sum(
            1 for r in decile_rows[1:]
            if r.is_monotone_with_previous is False
        )

        obs_drs = [r.observed_default_rate for r in decile_rows]
        decile_ranks = list(range(1, len(obs_drs) + 1))
        spearman_r, spearman_p = stats.spearmanr(decile_ranks, obs_drs)

        overall_cal_err = float(np.mean([r.calibration_error for r in decile_rows]))

        # ------------------------------------------------------------------
        # 4. Enrich full DataFrame with segment + decile labels
        # ------------------------------------------------------------------
        scored_df = df.copy()
        if pd_col in scored_df.columns:
            # Assign risk segments
            segment_labels = pd.Series(index=scored_df.index, data="Unassigned", dtype=object)
            for band in self.risk_bands:
                mask = (scored_df[pd_col] >= band["pd_low"]) & (scored_df[pd_col] < band["pd_high"])
                segment_labels[mask] = band["segment"]
            scored_df["risk_segment"] = segment_labels

            # Assign deciles (recompute on full df for consistency)
            try:
                scored_df["risk_decile"] = pd.qcut(
                    scored_df[pd_col], q=self.n_deciles, labels=False, duplicates="drop"
                ) + 1
            except Exception:
                scored_df["risk_decile"] = np.nan

        logger.info(
            "BorrowerRiskSegmenter — %d segments, %d deciles. "
            "Monotonicity violations: %d/%d. Spearman r=%.4f (p=%.4f).",
            len(segment_rows), len(decile_rows), violations, len(decile_rows) - 1,
            spearman_r, spearman_p,
        )

        return BorrowerSegmentationResult(
            segment_table=segment_table,
            decile_table=decile_table,
            monotonicity_violations=violations,
            monotonicity_spearman_r=float(spearman_r),
            monotonicity_spearman_p=float(spearman_p),
            overall_calibration_error=overall_cal_err,
            scored_df=scored_df,
        )
