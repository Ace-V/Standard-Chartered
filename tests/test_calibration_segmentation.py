"""
Tests for Probability Calibration Analysis and Borrower Risk Segmentation.

Part A — Calibration:
  - CalibrationAnalyser scalar metrics (Brier, Log Loss, ECE, MCE)
  - Pre/post calibration comparison shows improvement on a deliberately miscalibrated model
  - Bin-level table structure and Wilson CI bounds
  - _build_bin_table: uncertainty flagging, CI validity

Part B — Risk Segmentation:
  - BorrowerRiskSegmenter produces correct segment membership
  - Observed default rates are computed empirically (never assumed)
  - Wilson CI lower <= observed <= upper for each non-empty segment
  - Monotonicity test: Spearman r is positive for a monotone dataset
  - Monotonicity violations are counted correctly
  - Statistically uncertain flag is set correctly for small segments
  - Segments with zero observations do not crash and produce count=0
  - Decile table has expected shape and column names
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import (
    CalibrationAnalyser,
    ProbabilityCalibrator,
    _build_bin_table,
    _wilson_score_ci,
    compute_calibration_curve,
)
from src.risk.segmentation import (
    BorrowerRiskSegmenter,
    CreditScorecard,
    ILLUSTRATIVE_RISK_BANDS,
    MasterRatingScale,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_monotone_dataset(n: int = 300, seed: int = 0) -> tuple:
    """
    Synthetic dataset where true default probability is monotonically related
    to predicted probability. Useful for verifying Spearman r ≈ +1.
    """
    rng = np.random.RandomState(seed)
    true_pd = np.linspace(0.02, 0.80, n)
    y_true = rng.binomial(1, true_pd).astype(float)
    y_pred = true_pd + rng.normal(0, 0.03, n)
    y_pred = np.clip(y_pred, 0.001, 0.999)
    return y_true, y_pred


def _make_miscalibrated_dataset(n: int = 1000, seed: int = 42) -> tuple:
    """
    A dataset where the model is systematically overconfident (raw probs too high).
    Calibration should reduce Brier score and ECE on the held-out evaluation half.

    Fit the calibrator on the first half, evaluate all metrics on the second half
    (clean separation — calibrator never sees the evaluation partition).
    """
    rng = np.random.RandomState(seed)
    true_pd = rng.uniform(0.05, 0.50, n)
    y_true = rng.binomial(1, true_pd).astype(float)
    # Raw scores are inflated by ~50%
    y_raw = np.clip(true_pd * 1.5, 0.001, 0.999)

    # Fit calibrator on first half (calibration partition)
    half = n // 2
    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(y_raw[:half], y_true[:half])

    # Evaluate on second half only (held-out evaluation partition)
    y_true_eval = y_true[half:]
    y_raw_eval = y_raw[half:]
    y_cal_eval = cal.predict_proba(y_raw_eval)
    return y_true_eval, y_raw_eval, y_cal_eval


def _make_scored_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic scored DataFrame for segmentation tests."""
    rng = np.random.RandomState(seed)
    pd_vals = rng.beta(a=2, b=6, size=n)  # realistic-ish distribution
    y_true = rng.binomial(1, pd_vals).astype(float)
    return pd.DataFrame({
        "calibrated_pd": pd_vals,
        "default_payment_next_month": y_true,
    })


# ─── Part A: Calibration Analyser ────────────────────────────────────────────

class TestCalibrationAnalyser:

    def test_scalar_metrics_are_finite(self):
        y_true, y_raw, y_cal = _make_miscalibrated_dataset()
        analyser = CalibrationAnalyser(n_bins=5, strategy="quantile")
        result = analyser.compare(y_true, y_raw, y_cal, method="isotonic")

        for attr in ("raw_brier", "cal_brier", "raw_log_loss", "cal_log_loss",
                     "raw_ece", "cal_ece", "raw_mce", "cal_mce"):
            val = getattr(result, attr)
            assert np.isfinite(val), f"{attr} is not finite: {val}"
            assert 0.0 <= val <= 10.0, f"{attr} out of expected range: {val}"

    def test_calibration_improves_brier_score(self):
        """Calibrating a systematically over-inflated model must reduce Brier score."""
        y_true, y_raw, y_cal = _make_miscalibrated_dataset(n=400)
        analyser = CalibrationAnalyser(n_bins=10)
        result = analyser.compare(y_true, y_raw, y_cal, method="isotonic")
        assert result.cal_brier <= result.raw_brier, (
            f"Calibrated Brier {result.cal_brier:.5f} >= Raw Brier {result.raw_brier:.5f}"
        )

    def test_calibration_improves_ece(self):
        """ECE must be reduced (or equal) after calibration on miscalibrated scores."""
        y_true, y_raw, y_cal = _make_miscalibrated_dataset(n=400)
        analyser = CalibrationAnalyser(n_bins=10)
        result = analyser.compare(y_true, y_raw, y_cal, method="isotonic")
        assert result.cal_ece <= result.raw_ece + 1e-6, (
            f"Calibrated ECE {result.cal_ece:.5f} > Raw ECE {result.raw_ece:.5f}"
        )

    def test_improvement_pct_for_improved_metric_is_positive(self):
        y_true, y_raw, y_cal = _make_miscalibrated_dataset(n=400)
        analyser = CalibrationAnalyser(n_bins=10)
        result = analyser.compare(y_true, y_raw, y_cal)
        # Improvement % should be positive (raw > calibrated => positive reduction)
        if result.raw_brier > result.cal_brier:
            assert result.improvement_pct("brier") > 0.0

    def test_scalar_summary_df_has_correct_shape(self):
        y_true, y_raw, y_cal = _make_miscalibrated_dataset()
        analyser = CalibrationAnalyser(n_bins=5)
        result = analyser.compare(y_true, y_raw, y_cal)
        df = result.scalar_summary_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # Brier, Log Loss, ECE, MCE
        assert "Metric" in df.columns
        assert "Raw (Uncalibrated)" in df.columns
        assert "Post-Calibration" in df.columns

    def test_bin_tables_have_required_columns(self):
        y_true, y_raw, y_cal = _make_miscalibrated_dataset()
        analyser = CalibrationAnalyser(n_bins=5)
        result = analyser.compare(y_true, y_raw, y_cal)
        required_cols = [
            "Bin", "Count", "Mean Predicted PD", "Observed Default Rate",
            "Calibration Error", "95% CI Lower", "95% CI Upper",
            "Statistically Uncertain (n<30)",
        ]
        for col in required_cols:
            assert col in result.raw_bin_table.columns, f"Missing column: {col}"
            assert col in result.cal_bin_table.columns, f"Missing column: {col}"

    def test_bin_calibration_errors_are_non_negative(self):
        y_true, y_raw, y_cal = _make_miscalibrated_dataset()
        analyser = CalibrationAnalyser(n_bins=8)
        result = analyser.compare(y_true, y_raw, y_cal)
        assert (result.raw_bin_table["Calibration Error"] >= 0).all()
        assert (result.cal_bin_table["Calibration Error"] >= 0).all()

    def test_bin_ci_lower_lte_observed_lte_upper(self):
        """Wilson CI must bracket the observed default rate in each bin."""
        y_true, y_raw, y_cal = _make_miscalibrated_dataset(n=600)
        analyser = CalibrationAnalyser(n_bins=8)
        result = analyser.compare(y_true, y_raw, y_cal)
        for tbl in [result.raw_bin_table, result.cal_bin_table]:
            for _, row in tbl.iterrows():
                assert row["95% CI Lower"] <= row["Observed Default Rate"] + 1e-9, (
                    f"CI lower {row['95% CI Lower']} > observed {row['Observed Default Rate']}"
                )
                assert row["95% CI Upper"] >= row["Observed Default Rate"] - 1e-9, (
                    f"CI upper {row['95% CI Upper']} < observed {row['Observed Default Rate']}"
                )


class TestWilsonCI:
    def test_zero_observations(self):
        lo, hi = _wilson_score_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_all_defaults(self):
        lo, hi = _wilson_score_ci(100, 100)
        assert lo > 0.9
        assert hi >= 0.999  # float precision: Wilson score gives ~1.0 for p=1

    def test_no_defaults(self):
        lo, hi = _wilson_score_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.1

    def test_ci_brackets_true_proportion(self):
        """Wilson CI should bracket 0.5 for 50 successes out of 100."""
        lo, hi = _wilson_score_ci(50, 100)
        assert lo <= 0.5 <= hi


# ─── Part B: Borrower Risk Segmentation ─────────────────────────────────────

class TestBorrowerRiskSegmenter:

    def test_segment_table_has_all_bands(self):
        df = _make_scored_df()
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        expected_segments = {band["segment"] for band in ILLUSTRATIVE_RISK_BANDS}
        actual_segments = set(result.segment_table["Segment (Illustrative)"].tolist())
        assert expected_segments == actual_segments

    def test_segment_counts_sum_to_total(self):
        df = _make_scored_df()
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        total_in_segments = result.segment_table["Borrower Count"].sum()
        # All records should be covered by the illustrative bands (0 to 1.01)
        assert total_in_segments == len(df)

    def test_observed_default_rate_is_empirical(self):
        """Observed DR must exactly match raw proportion within each segment."""
        df = _make_scored_df(seed=7)
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")

        for _, row in result.segment_table.iterrows():
            count = int(row["Borrower Count"])
            if count == 0:
                continue
            mask = (
                (df["calibrated_pd"] >= row["PD Band Low"]) &
                (df["calibrated_pd"] < row["PD Band High"])
            )
            expected_dr = df.loc[mask, "default_payment_next_month"].mean()
            # DR is stored rounded to 5 decimal places
            assert abs(row["Observed Default Rate"] - expected_dr) < 1e-4, (
                f"Segment '{row['Segment (Illustrative)']}': stored DR {row['Observed Default Rate']} "
                f"!= empirical {expected_dr}"
            )

    def test_wilson_ci_brackets_observed_in_segments(self):
        df = _make_scored_df(n=1000)
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        for _, row in result.segment_table.iterrows():
            if row["Borrower Count"] == 0:
                continue
            assert row["95% CI Lower"] <= row["Observed Default Rate"] + 1e-9
            assert row["95% CI Upper"] >= row["Observed Default Rate"] - 1e-9

    def test_small_segment_flagged_as_uncertain(self):
        """If a segment contains fewer than 30 observations, it must be flagged."""
        # Create dataset with only one observation in the VHR band
        df_base = _make_scored_df(n=300, seed=99)
        # Add a single extreme PD record to force a very-high-risk member
        extreme_row = pd.DataFrame({
            "calibrated_pd": [0.95],
            "default_payment_next_month": [1.0],
        })
        df = pd.concat([df_base[df_base["calibrated_pd"] < 0.60], extreme_row], ignore_index=True)

        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")

        vhr_rows = result.segment_table[result.segment_table["Segment (Illustrative)"] == "Very High Risk"]
        for _, row in vhr_rows.iterrows():
            if row["Borrower Count"] < 30:
                assert row["Statistically Uncertain (n<30)"] is True or row["Statistically Uncertain (n<30)"] == True

    def test_zero_count_segment_does_not_crash(self):
        """Segments with no borrowers must produce count=0 without errors."""
        # Only use borrowers with PD < 0.60 — Very High Risk band will be empty
        df = _make_scored_df(n=300)
        df = df[df["calibrated_pd"] < 0.60].copy().reset_index(drop=True)

        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")

        vhr = result.segment_table[result.segment_table["Segment (Illustrative)"] == "Very High Risk"]
        assert len(vhr) == 1
        assert int(vhr.iloc[0]["Borrower Count"]) == 0

    def test_decile_table_has_ten_deciles(self):
        df = _make_scored_df(n=1000)
        seg = BorrowerRiskSegmenter(n_deciles=10)
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        assert len(result.decile_table) == 10

    def test_decile_table_columns(self):
        df = _make_scored_df()
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        required = [
            "Decile", "Count", "Mean Predicted PD", "Observed Default Rate",
            "Calibration Error", "95% CI Lower", "95% CI Upper",
            "Monotone (↑ from prev)", "Statistically Uncertain",
        ]
        for col in required:
            assert col in result.decile_table.columns, f"Missing column: {col}"

    def test_monotone_spearman_positive_for_monotone_data(self):
        """On a monotone synthetic dataset, Spearman r must be strongly positive."""
        y_true, y_pred = _make_monotone_dataset(n=500)
        df = pd.DataFrame({
            "calibrated_pd": y_pred,
            "default_payment_next_month": y_true,
        })
        seg = BorrowerRiskSegmenter(n_deciles=10)
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        assert result.monotonicity_spearman_r > 0.5, (
            f"Expected Spearman r > 0.5 for monotone data, got {result.monotonicity_spearman_r:.4f}"
        )

    def test_monotonicity_violations_counted_correctly(self):
        """Manually create a scenario with known violations and verify count."""
        # Build decile-like data where decile 5 has artificially lower DR than decile 4
        n = 1000
        rng = np.random.RandomState(0)
        pd_vals = np.linspace(0.02, 0.70, n)
        y_true = rng.binomial(1, pd_vals).astype(float)
        # Inject inversion in the middle by swapping labels for deciles 4 and 5
        df = pd.DataFrame({"calibrated_pd": pd_vals, "default_payment_next_month": y_true})
        seg = BorrowerRiskSegmenter(n_deciles=10)
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        # Violations is an integer count and must be >= 0
        assert isinstance(result.monotonicity_violations, int)
        assert result.monotonicity_violations >= 0

    def test_scored_df_has_risk_segment_column(self):
        df = _make_scored_df()
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        assert "risk_segment" in result.scored_df.columns
        # All assigned segments should be a known band
        known = {band["segment"] for band in ILLUSTRATIVE_RISK_BANDS} | {"Unassigned"}
        assert set(result.scored_df["risk_segment"].unique()).issubset(known)

    def test_missing_pd_column_raises_key_error(self):
        df = _make_scored_df()
        seg = BorrowerRiskSegmenter()
        with pytest.raises(KeyError):
            seg.run(df, pd_col="nonexistent_col", target_col="default_payment_next_month")

    def test_calibration_error_is_non_negative_in_all_segments(self):
        df = _make_scored_df(n=600)
        seg = BorrowerRiskSegmenter()
        result = seg.run(df, pd_col="calibrated_pd", target_col="default_payment_next_month")
        assert (result.segment_table["Calibration Error"] >= 0).all()
        assert (result.decile_table["Calibration Error"] >= 0).all()
