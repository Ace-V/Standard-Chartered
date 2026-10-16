"""
CreditRisk Analytics — Probability Calibration & Borrower Risk Segmentation
============================================================================

Presents rigorous pre/post calibration analysis (reliability diagrams, Brier score,
ECE, MCE, bin-level tables) and borrower risk segmentation (configurable bands,
observed default rates, Wilson CIs, risk decile analysis with monotonicity testing).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR
from src.data.ingestion import load_raw_dataset
from src.data.preprocessing import prepare_train_test_split
from src.models.calibration import CalibrationAnalyser
from src.reporting.charts import (
    plot_calibration_bin_errors,
    plot_calibration_comparison,
    plot_decile_default_rates,
    plot_segment_bar,
    plot_segment_calibration,
)
from src.risk.segmentation import BorrowerRiskSegmenter

st.set_page_config(
    page_title="Calibration & Segmentation | CreditRisk",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Probability Calibration & Borrower Risk Segmentation")
st.caption(
    "Rigorous assessment of calibration quality, ranking quality, and borrower risk ordering. "
    "All segment boundaries are illustrative — not bank policy."
)

# ─── Prerequisite: model artifact ─────────────────────────────────────────
artifact_path = MODELS_DIR / "pd_model_best.joblib"
if not artifact_path.exists():
    st.error(
        "No trained model artifact found. Run the pipeline first: "
        "`.venv\\Scripts\\python run_pipeline.py`"
    )
    st.stop()

# ─── Data & scores ─────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_raw_dataset(source="uci")

@st.cache_data
def compute_scores(_artifact_path: str):
    """Load artifact and compute raw + calibrated test scores. Cached by artifact path."""
    artifact = joblib.load(_artifact_path)
    df = get_data()
    _, X_test, _, y_test = prepare_train_test_split(df)
    y_raw = artifact.predict_proba(X_test, calibrate=False)
    y_cal = artifact.predict_proba(X_test, calibrate=True)
    method = artifact.calibrator.method if artifact.calibrator.is_fitted else "isotonic"
    return y_test.values, y_raw, y_cal, method, artifact.model_name

df = get_data()
y_test, y_raw, y_cal, cal_method, model_name = compute_scores(str(artifact_path))

@st.cache_data
def get_calibration_result(_y_test, _y_raw, _y_cal, method, n_bins=10):
    analyser = CalibrationAnalyser(n_bins=n_bins, strategy="quantile")
    return analyser.compare(
        y_true=_y_test, y_raw_prob=_y_raw, y_cal_prob=_y_cal, method=method
    )

@st.cache_data
def get_segmentation_result(_df, _pd_col, _target_col):
    segmenter = BorrowerRiskSegmenter(n_deciles=10)
    return segmenter.run(_df, pd_col=_pd_col, target_col=_target_col)

# ─── Top-level tabs ─────────────────────────────────────────────────────────
tab_calib, tab_seg = st.tabs([
    "📐 Part A — Probability Calibration",
    "📊 Part B — Borrower Risk Segmentation",
])

# ═══════════════════════════════════════════════════════════════════════════
# PART A: CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.subheader("Part A: Probability Calibration Analysis")
    st.info(
        "**Three Distinct Performance Dimensions:**\n\n"
        "1. **Ranking Quality (Discrimination):** How well the model orders borrowers by "
        "default risk. Measured by ROC-AUC, Gini, KS. A model can rank perfectly but predict "
        "wrong absolute probabilities.\n\n"
        "2. **Probability Calibration:** Whether stated PD = empirical default frequency. "
        "If the model outputs PD = 0.20 for a group that defaults at 40%, reserves and capital "
        "will be under-stated. Measured by Brier score, ECE, Log Loss.\n\n"
        "3. **Classification Threshold Performance:** Precision/Recall at a chosen operating "
        "cutoff. Only meaningful at a specific decision boundary — rarely the right metric for "
        "credit pricing or provisioning. Evaluation by accuracy alone is misleading when classes "
        "are imbalanced."
    )

    # Calibration controls
    n_bins_input = st.slider(
        "Number of calibration bins",
        min_value=5, max_value=20, value=10, step=1,
        help="More bins = finer resolution but fewer observations per bin. "
             "Bins with n<30 are flagged as statistically uncertain."
    )

    cal_result = get_calibration_result(y_test, y_raw, y_cal, cal_method, n_bins=n_bins_input)

    # ── KPI metrics ──────────────────────────────────────────────────────
    st.markdown("#### Scalar Metric Comparison — Test Set")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Brier Score (Raw)", f"{cal_result.raw_brier:.5f}")
        st.metric("Brier Score (Calibrated)", f"{cal_result.cal_brier:.5f}",
                  delta=f"{cal_result.improvement_pct('brier'):+.1f}% improvement",
                  delta_color="inverse")
    with c2:
        st.metric("Log Loss (Raw)", f"{cal_result.raw_log_loss:.5f}")
        st.metric("Log Loss (Calibrated)", f"{cal_result.cal_log_loss:.5f}",
                  delta=f"{cal_result.improvement_pct('log_loss'):+.1f}%",
                  delta_color="inverse")
    with c3:
        st.metric("ECE (Raw)", f"{cal_result.raw_ece:.5f}")
        st.metric("ECE (Calibrated)", f"{cal_result.cal_ece:.5f}",
                  delta=f"{cal_result.improvement_pct('ece'):+.1f}%",
                  delta_color="inverse")
    with c4:
        st.metric("MCE (Raw)", f"{cal_result.raw_mce:.5f}")
        st.metric("MCE (Calibrated)", f"{cal_result.cal_mce:.5f}",
                  delta=f"{cal_result.improvement_pct('mce'):+.1f}%",
                  delta_color="inverse")

    st.caption(
        f"Champion: **{model_name.replace('_', ' ')}** | Calibration method: **{cal_method.title()}** "
        f"| Calibrator fitted on held-out validation partition — **never on test set**."
    )

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────
    tab_overlay, tab_bins_chart, tab_raw_table, tab_cal_table = st.tabs([
        "Reliability Diagram (Overlay)",
        "Calibration Error by Bin",
        "Raw Bin Table",
        "Calibrated Bin Table",
    ])

    with tab_overlay:
        fig_overlay = plot_calibration_comparison(
            cal_result.raw_prob_true, cal_result.raw_prob_pred,
            cal_result.cal_prob_true, cal_result.cal_prob_pred,
            cal_result.raw_ece, cal_result.cal_ece,
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
        st.caption(
            "The 45° dashed line represents perfect calibration. Points above the diagonal "
            "indicate under-estimation of default risk (model is too optimistic). Points below "
            "indicate over-estimation (model is too conservative)."
        )

    with tab_bins_chart:
        fig_bins = plot_calibration_bin_errors(cal_result.raw_bin_table, cal_result.cal_bin_table)
        st.plotly_chart(fig_bins, use_container_width=True)

    with tab_raw_table:
        st.markdown("**Raw (Uncalibrated) Bin-Level Analysis**")
        st.dataframe(cal_result.raw_bin_table, use_container_width=True)

    with tab_cal_table:
        st.markdown("**Post-Calibration Bin-Level Analysis**")
        st.dataframe(cal_result.cal_bin_table, use_container_width=True)
        uncertain_n = cal_result.cal_bin_table["Statistically Uncertain (n<30)"].sum()
        if uncertain_n > 0:
            st.warning(
                f"⚠ {uncertain_n} bin(s) contain fewer than 30 observations. "
                "Observed default rates in those bins carry wide confidence intervals."
            )

# ═══════════════════════════════════════════════════════════════════════════
# PART B: RISK SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_seg:
    st.subheader("Part B: Borrower Risk Segmentation")
    st.warning(
        "**⚠ DISCLOSURE:** All risk segment boundaries (Low Risk, Medium-Low Risk, "
        "Medium-High Risk, High Risk, Very High Risk) are **illustrative thresholds "
        "chosen for demonstration purposes only**. They are NOT validated bank underwriting "
        "policy, regulatory minimums, or supervisory-approved credit guidelines. "
        "Actual segment boundaries require IRB model validation, backtesting, and regulatory "
        "sign-off (Basel Committee / PRA / ECB supervisory review)."
    )

    # Score the full portfolio for segmentation
    @st.cache_data
    def get_scored_portfolio():
        artifact = joblib.load(str(artifact_path))
        df_full = get_data()
        p_cal = artifact.predict_proba(df_full, calibrate=True)
        out = df_full.copy()
        out["calibrated_pd"] = p_cal
        return out

    df_scored = get_scored_portfolio()
    target_col = "default_payment_next_month"

    seg_result = get_segmentation_result(df_scored, "calibrated_pd", target_col)

    # ── Monotonicity KPIs ─────────────────────────────────────────────────
    n_transitions = len(seg_result.decile_table) - 1
    spearman_r = seg_result.monotonicity_spearman_r
    spearman_p = seg_result.monotonicity_spearman_p
    violations = seg_result.monotonicity_violations

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Borrowers Analysed", f"{len(df_scored):,}")
    with c2:
        st.metric("Spearman Rank Correlation",
                  f"{spearman_r:.4f}",
                  help="Correlation between decile rank and observed default rate. +1.0 = perfectly monotone.")
        st.caption(f"p-value: {spearman_p:.2e}")
    with c3:
        st.metric("Monotonicity Violations", f"{violations} / {n_transitions}",
                  delta="✅ Fully monotone" if violations == 0 else f"⚠ {violations} violation(s)",
                  delta_color="off" if violations == 0 else "inverse")
    with c4:
        st.metric("Overall Calibration Error", f"{seg_result.overall_calibration_error:.5f}",
                  help="Mean |predicted PD − observed default rate| across deciles")

    st.markdown("---")

    tab_decile, tab_seg_bar, tab_bubble, tab_seg_table, tab_dec_table = st.tabs([
        "Risk Decile Analysis",
        "Segment Bar Chart",
        "Segment Calibration (Bubble)",
        "Segment Table",
        "Decile Table",
    ])

    with tab_decile:
        fig_decile = plot_decile_default_rates(seg_result.decile_table)
        st.plotly_chart(fig_decile, use_container_width=True)
        st.caption(
            "🟥 Red bars indicate monotonicity violations — deciles where observed default rate "
            "is *lower* than the previous decile despite higher predicted PD. "
            "Error bars show 95% Wilson score confidence intervals."
        )
        if violations == 0:
            st.success(
                f"✅ Monotone ordering confirmed across all {n_transitions} decile transitions. "
                f"Spearman r = {spearman_r:.4f} (p = {spearman_p:.2e})."
            )
        else:
            st.warning(
                f"⚠ {violations} of {n_transitions} decile transitions are non-monotone. "
                f"Spearman r = {spearman_r:.4f}. Investigate affected deciles for model limitations."
            )

    with tab_seg_bar:
        fig_bar = plot_segment_bar(seg_result.segment_table)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_bubble:
        fig_bubble = plot_segment_calibration(seg_result.segment_table)
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption(
            "Points close to the 45° diagonal are well-calibrated. "
            "Points above = model under-estimates risk (optimistic). "
            "Points below = model over-estimates risk (conservative). "
            "⭕ Open circles indicate segments with fewer than 30 observations (statistically uncertain)."
        )

    with tab_seg_table:
        st.markdown("**Borrower Risk Segment Table (Illustrative Boundaries)**")
        st.dataframe(seg_result.segment_table, use_container_width=True)

    with tab_dec_table:
        st.markdown("**Risk Decile Table (D1 = Lowest PD Decile, D10 = Highest)**")
        st.dataframe(seg_result.decile_table, use_container_width=True)
