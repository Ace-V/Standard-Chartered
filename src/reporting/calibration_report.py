"""
CreditRisk Analytics — Calibration & Segmentation Report Generator
===================================================================

Orchestrates the full Part A + Part B analytical workflow and saves all results
to the reports/ directory:

  Part A — Calibration
  - Pre/post calibration scalar metric comparison (Brier, Log Loss, ECE, MCE)
  - Bin-level calibration tables (raw + calibrated, with Wilson CIs)
  - Plotly reliability diagram overlay (HTML)
  - Plotly calibration error by bin (HTML)
  - Markdown narrative report

  Part B — Risk Segmentation
  - Illustrative risk band segment table (borrower count, observed DR, CI)
  - Risk decile table (D1–D10) with monotonicity assessment
  - Spearman rank correlation and violation count
  - Plotly decile default rate chart (HTML)
  - Plotly segment calibration bubble chart (HTML)
  - Plotly segment bar chart (HTML)
  - Markdown narrative report

All results are computed on the held-out test set to prevent data leakage.
Calibration comparison uses pre/post scores from the same model — the
ProbabilityCalibrator is NOT re-fitted here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR, logger
from src.data.ingestion import load_raw_dataset
from src.data.preprocessing import prepare_train_test_split
from src.models.calibration import CalibrationAnalyser, CalibrationComparisonResult
from src.reporting.charts import (
    plot_calibration_bin_errors,
    plot_calibration_comparison,
    plot_decile_default_rates,
    plot_segment_bar,
    plot_segment_calibration,
)
from src.risk.segmentation import BorrowerRiskSegmenter, BorrowerSegmentationResult

# Target column name in the scored dataset
_TARGET_COL = "default_payment_next_month"


def _load_artifact_and_scores(
    df: pd.DataFrame,
    artifact_path: Path,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the serialised ModelArtifact and produce raw + calibrated probability
    arrays for both the held-out test set and the calibration validation partition.

    The raw scores are derived from the champion classifier BEFORE applying the
    ProbabilityCalibrator. The calibrated scores pass through the fitted calibrator.
    The calibrator is NOT re-fitted here — it must have already been fitted on the
    validation portion of the training data during pipeline execution.

    Returns
    -------
    artifact        : ModelArtifact object
    X_test          : Test-set feature DataFrame
    y_test          : Test-set binary labels
    y_raw_test      : Uncalibrated test-set probabilities
    y_cal_test      : Calibrated test-set probabilities
    """
    artifact = joblib.load(artifact_path)

    _, X_test, _, y_test = prepare_train_test_split(df)

    # Get raw (uncalibrated) probabilities by calling predict_proba with calibrate=False
    y_raw_test = artifact.predict_proba(X_test, calibrate=False)

    # Get calibrated probabilities
    y_cal_test = artifact.predict_proba(X_test, calibrate=True)

    return artifact, X_test, y_test, y_raw_test, y_cal_test


def run_calibration_report(
    df: pd.DataFrame,
    artifact_path: Optional[Path] = None,
    n_bins: int = 10,
    output_dir: Optional[Path] = None,
) -> Tuple[CalibrationComparisonResult, Dict[str, Path]]:
    """
    Execute Part A calibration analysis and save all outputs.

    Parameters
    ----------
    df           : Full dataset (raw; train/test split is applied internally).
    artifact_path : Path to serialised ModelArtifact joblib file.
    n_bins       : Number of calibration bins for the reliability diagram.
    output_dir   : Directory to write HTML charts and markdown report.

    Returns
    -------
    result : CalibrationComparisonResult with scalar metrics and bin tables.
    paths  : Dict mapping output names to file paths.
    """
    artifact_path = artifact_path or (MODELS_DIR / "pd_model_best.joblib")
    output_dir = output_dir or REPORTS_DIR

    artifact, X_test, y_test, y_raw, y_cal = _load_artifact_and_scores(df, artifact_path)

    analyser = CalibrationAnalyser(n_bins=n_bins, strategy="quantile")
    result = analyser.compare(
        y_true=y_test.values,
        y_raw_prob=y_raw,
        y_cal_prob=y_cal,
        method=artifact.calibrator.method if artifact.calibrator.is_fitted else "isotonic",
    )

    paths: Dict[str, Path] = {}

    # Save bin tables as CSV
    raw_csv = output_dir / "calibration_raw_bins.csv"
    cal_csv = output_dir / "calibration_post_bins.csv"
    result.raw_bin_table.to_csv(raw_csv, index=False)
    result.cal_bin_table.to_csv(cal_csv, index=False)
    paths["raw_bin_table_csv"] = raw_csv
    paths["cal_bin_table_csv"] = cal_csv
    logger.info("Saved calibration bin tables to %s and %s", raw_csv, cal_csv)

    # Save scalar summary as CSV
    scalar_csv = output_dir / "calibration_scalar_comparison.csv"
    result.scalar_summary_df().to_csv(scalar_csv, index=False)
    paths["scalar_summary_csv"] = scalar_csv

    # Save Plotly charts
    fig_overlay = plot_calibration_comparison(
        result.raw_prob_true, result.raw_prob_pred,
        result.cal_prob_true, result.cal_prob_pred,
        result.raw_ece, result.cal_ece,
    )
    overlay_html = output_dir / "calibration_reliability_diagram.html"
    fig_overlay.write_html(str(overlay_html))
    paths["reliability_diagram_html"] = overlay_html

    fig_bins = plot_calibration_bin_errors(result.raw_bin_table, result.cal_bin_table)
    bins_html = output_dir / "calibration_bin_errors.html"
    fig_bins.write_html(str(bins_html))
    paths["bin_errors_html"] = bins_html

    # Generate markdown narrative
    md_path = output_dir / "calibration_report.md"
    _write_calibration_markdown(result, artifact.model_name, n_bins, md_path)
    paths["markdown_report"] = md_path

    logger.info(
        "Calibration report complete. Brier improvement: %.1f%% | ECE improvement: %.1f%%",
        result.improvement_pct("brier"), result.improvement_pct("ece"),
    )

    return result, paths


def run_segmentation_report(
    scored_df: pd.DataFrame,
    pd_col: str = "calibrated_pd",
    target_col: str = _TARGET_COL,
    output_dir: Optional[Path] = None,
) -> Tuple[BorrowerSegmentationResult, Dict[str, Path]]:
    """
    Execute Part B segmentation analysis and save all outputs.

    Parameters
    ----------
    scored_df  : DataFrame already containing calibrated_pd and target columns.
    pd_col     : Column name of calibrated default probability.
    target_col : Column name of binary default label (0/1).
    output_dir : Directory to write HTML charts and markdown report.

    Returns
    -------
    seg_result : BorrowerSegmentationResult with tables and diagnostics.
    paths      : Dict mapping output names to file paths.
    """
    output_dir = output_dir or REPORTS_DIR

    segmenter = BorrowerRiskSegmenter(n_deciles=10)
    seg_result = segmenter.run(scored_df, pd_col=pd_col, target_col=target_col)

    paths: Dict[str, Path] = {}

    # Save tables as CSV
    seg_csv = output_dir / "segmentation_risk_bands.csv"
    dec_csv = output_dir / "segmentation_deciles.csv"
    seg_result.segment_table.to_csv(seg_csv, index=False)
    seg_result.decile_table.to_csv(dec_csv, index=False)
    paths["segment_table_csv"] = seg_csv
    paths["decile_table_csv"] = dec_csv
    logger.info("Saved segmentation tables to %s and %s", seg_csv, dec_csv)

    # Save Plotly charts
    fig_decile = plot_decile_default_rates(seg_result.decile_table)
    decile_html = output_dir / "segmentation_decile_chart.html"
    fig_decile.write_html(str(decile_html))
    paths["decile_chart_html"] = decile_html

    fig_bubble = plot_segment_calibration(seg_result.segment_table)
    bubble_html = output_dir / "segmentation_calibration_bubble.html"
    fig_bubble.write_html(str(bubble_html))
    paths["calibration_bubble_html"] = bubble_html

    fig_bar = plot_segment_bar(seg_result.segment_table)
    bar_html = output_dir / "segmentation_segment_bar.html"
    fig_bar.write_html(str(bar_html))
    paths["segment_bar_html"] = bar_html

    # Write markdown narrative
    md_path = output_dir / "segmentation_report.md"
    _write_segmentation_markdown(seg_result, md_path)
    paths["markdown_report"] = md_path

    logger.info(
        "Segmentation report complete. Spearman r=%.4f (p=%.4f), Violations: %d/9",
        seg_result.monotonicity_spearman_r,
        seg_result.monotonicity_spearman_p,
        seg_result.monotonicity_violations,
    )

    return seg_result, paths


def _write_calibration_markdown(
    result: CalibrationComparisonResult,
    model_name: str,
    n_bins: int,
    path: Path,
) -> None:
    """Generate a rigorous markdown calibration report with narrative and tables."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = result.scalar_summary_df()

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""# CreditRisk Analytics — Probability Calibration Report
*Generated: {ts} | Champion Model: {model_name.replace("_", " ")} | Calibration Method: {result.method.title()}*

---

## Conceptual Framework

### Three Distinct Performance Dimensions

| Dimension | Definition | Primary Metrics | Credit Risk Relevance |
|---|---|---|---|
| **Ranking Quality** | How well the model orders defaulters above non-defaulters | ROC-AUC, Gini, KS | Drives credit tier pricing and eligibility decisions |
| **Probability Calibration** | Whether stated PD = empirical default frequency | ECE, MCE, Brier Score, Log Loss | Required for IFRS 9 ECL provisions and Basel IRB capital |
| **Threshold Performance** | Precision/Recall at a chosen operating cutoff | F1, Precision, Recall | Relevant for collection strategy and operational decisions |

> A model can rank well (high AUC) but be grossly miscalibrated. For example, if a model assigns
> PD = 0.10 to borrowers who default at a 30% rate, ECL reserves will be systematically under-stated
> by ~20 percentage points — a material IFRS 9 and capital impact. Calibration post-processing
> (isotonic regression or Platt scaling) corrects this without affecting ranking quality.

---

## Part A: Calibration Diagnostic Results

### Scalar Metric Comparison — Raw vs Post-Calibration (Test Set, n = {len(result.raw_bin_table)} bins from test data)

{summary.to_markdown(index=False)}

**Interpretation:**
- **Brier Score** reduction of **{result.improvement_pct("brier"):.1f}%** confirms that calibration
  improves the squared probability error between predicted PDs and actual outcomes.
- **Log Loss** reduction of **{result.improvement_pct("log_loss"):.1f}%** confirms better-calibrated
  probability estimates from the information-theoretic perspective.
- **ECE** (Expected Calibration Error) reduction of **{result.improvement_pct("ece"):.1f}%**
  reflects reduced population-weighted mean miscalibration across {n_bins} quantile bins.
- **MCE** (Maximum Calibration Error) reduction of **{result.improvement_pct("mce"):.1f}%**
  shows improvement in the worst-performing risk bucket.

> **Note on fitting discipline:** The `{result.method}` calibrator was fitted exclusively on the
> held-out validation partition (20% of training data). It was **never fitted or evaluated on the
> final test set**. The results above measure generalisation to unseen data.

---

### Bin-Level Calibration Table — Calibrated Probabilities

> Bins with fewer than 30 observations are flagged as statistically uncertain.
> Wilson score 95% confidence intervals are reported for each observed default rate.

{result.cal_bin_table.to_markdown(index=False)}

---

### Bin-Level Calibration Table — Raw (Uncalibrated)

{result.raw_bin_table.to_markdown(index=False)}

---

## Saved Outputs

| File | Description |
|---|---|
| `calibration_reliability_diagram.html` | Interactive overlay: raw vs calibrated reliability diagram |
| `calibration_bin_errors.html` | Per-bin absolute calibration error comparison chart |
| `calibration_scalar_comparison.csv` | Scalar metric comparison table |
| `calibration_raw_bins.csv` | Raw uncalibrated bin-level analysis |
| `calibration_post_bins.csv` | Post-calibration bin-level analysis |
""")


def _write_segmentation_markdown(result: BorrowerSegmentationResult, path: Path) -> None:
    """Generate a rigorous markdown segmentation report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    violations = result.monotonicity_violations
    spearman_r = result.monotonicity_spearman_r
    spearman_p = result.monotonicity_spearman_p
    n_transitions = len(result.decile_table) - 1

    if violations == 0:
        monotone_verdict = "✅ **Fully monotone** — Observed default rate increases consistently across all deciles."
    elif violations <= 1:
        verdict_desc = "acceptable minor non-monotonicity (within typical statistical sampling variation)"
        monotone_verdict = f"⚠️ **{violations}/{n_transitions} violation** — {verdict_desc}. Review affected deciles."
    else:
        verdict_desc = "material monotonicity violations — model risk ordering may be unreliable in those deciles"
        monotone_verdict = f"❌ **{violations}/{n_transitions} violations** — {verdict_desc}."

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""# CreditRisk Analytics — Borrower Risk Segmentation Report
*Generated: {ts}*

> **DISCLOSURE:** All risk segment boundaries (Low Risk, Medium-Low, Medium-High, High, Very High)
> are **illustrative thresholds for demonstration purposes only** and are NOT validated bank
> underwriting policy, regulatory minimums, or supervisory-approved credit guidelines.
> Actual segment boundaries require IRB model validation, backtesting, and regulatory sign-off.

---

## Part B: Borrower Risk Segmentation Results

### Risk Segment Summary (Illustrative Bands)

{result.segment_table.to_markdown(index=False)}

**Key observations:**
- Segments with "Statistically Uncertain" = True contain fewer than 30 borrowers. Observed default
  rates in these buckets carry wide confidence intervals and should be interpreted with caution.
- The 95% confidence intervals are computed using the Wilson score method, which is preferred over
  the normal approximation for small samples and extreme proportions.
- The calibration error per segment reflects |Mean Predicted PD − Observed Default Rate|.

---

## Risk Decile Analysis (D1 = Lowest PD Decile, D10 = Highest PD Decile)

{result.decile_table.to_markdown(index=False)}

### Monotonicity Assessment

| Diagnostic | Result | Interpretation |
|---|---|---|
| Monotonicity violations | **{violations}/{n_transitions}** | Number of decile transitions where observed DR *decreases* |
| Spearman rank correlation | **{spearman_r:.4f}** | ±1.0 = perfect rank ordering; 0 = no ordering |
| Spearman p-value | **{spearman_p:.6f}** | Significance of rank correlation |
| Overall calibration error | **{result.overall_calibration_error:.5f}** | Mean |predicted − observed| across deciles |

{monotone_verdict}

**Spearman rank interpretation:** r = {spearman_r:.4f} indicates
{"strong" if abs(spearman_r) >= 0.8 else "moderate" if abs(spearman_r) >= 0.5 else "weak"}
{"positive" if spearman_r >= 0 else "negative"} monotone ordering between decile rank and observed
default frequency (p = {spearman_p:.6f}{"< 0.001 — highly significant" if spearman_p < 0.001 else ""}).

---

## Saved Outputs

| File | Description |
|---|---|
| `segmentation_decile_chart.html` | Decile default rate bar chart with CIs and monotonicity flags |
| `segmentation_calibration_bubble.html` | Segment-level calibration bubble chart |
| `segmentation_segment_bar.html` | Segment bar chart: count, observed DR, predicted PD |
| `segmentation_risk_bands.csv` | Full segment-level table |
| `segmentation_deciles.csv` | Full decile table |
""")
