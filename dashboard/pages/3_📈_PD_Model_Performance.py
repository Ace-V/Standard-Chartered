"""
CreditRisk Analytics — Probability of Default (PD) Model Performance
=====================================================================

Displays multi-model cross-validation benchmarking, champion model evaluation,
discrimination curves (ROC, PR, KS), probability calibration, and bootstrap CIs.
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

from src.config import MODELS_DIR
from src.data.ingestion import load_raw_dataset
from src.data.preprocessing import prepare_train_test_split
from src.models.evaluate import ModelEvaluator
from src.models.train import CreditModelTrainer
from src.reporting.charts import (
    plot_calibration_curve,
    plot_ks_curve,
    plot_roc_curve,
    plot_score_distribution,
)
from src.risk.segmentation import CreditScorecard

st.set_page_config(page_title="PD Model Performance | CreditRisk", page_icon="📈", layout="wide")

st.title("📈 Probability of Default (PD) Model Evaluation & Calibration")
st.caption("Quantitative discrimination metrics, calibration curves, and supervisory model validation.")

artifact_path = MODELS_DIR / "pd_model_best.joblib"

@st.cache_data
def get_data():
    return load_raw_dataset(source="uci")

df = get_data()

# Model Training Trigger if artifact doesn't exist
if not artifact_path.exists():
    st.warning("No pre-trained model artifact found in `models/saved/`. Click below to train and benchmark models.")
    if st.button("🚀 Train & Benchmark PD Models Now", type="primary"):
        with st.spinner("Training Logistic Regression, Random Forest, and Gradient Boosting with 5-Fold CV..."):
            trainer = CreditModelTrainer()
            artifact, cv_summary = trainer.train_and_benchmark(df)
            st.success("Training and calibration completed successfully!")
            st.rerun()
    st.stop()

# Load Artifact
artifact = joblib.load(artifact_path)
test_metrics = artifact.test_metrics

# 1. Headline Model Performance Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Champion Model", artifact.model_name.replace("_", " "))
with col2:
    st.metric("Test ROC-AUC", f"{test_metrics.roc_auc:.4f}", help="Area Under the Receiver Operating Characteristic Curve")
with col3:
    st.metric("Gini (Somers' D)", f"{test_metrics.gini:.4f}", help="2 * AUC - 1 (Regulatory discrimination index)")
with col4:
    st.metric("KS Statistic", f"{test_metrics.ks_statistic:.4f}", help="Maximum separation between default and non-default distributions")
with col5:
    st.metric("Brier Score", f"{test_metrics.brier_score:.4f}", help="Mean squared calibration error (lower is superior)")

# 95% Confidence Intervals
st.info(
    f"**Statistical Rigor & 95% Bootstrap Confidence Intervals (N=1,000 resamples):**\n"
    f"- **ROC-AUC (95% CI):** [{test_metrics.auc_ci_lower:.4f}, {test_metrics.auc_ci_upper:.4f}]\n"
    f"- **Gini Coefficient (95% CI):** [{test_metrics.gini_ci_lower:.4f}, {test_metrics.gini_ci_upper:.4f}]\n"
    f"- **KS Separation (95% CI):** [{test_metrics.ks_ci_lower:.4f}, {test_metrics.ks_ci_upper:.4f}]"
)

st.markdown("---")

# 2. Multi-Model Cross-Validation Leaderboard
st.subheader("Cross-Validation Leaderboard (Stratified 5-Fold)")
if artifact.cv_summary:
    cv_rows = []
    for m_name, metrics in artifact.cv_summary.items():
        cv_rows.append({
            "Candidate Model": m_name.replace("_", " "),
            "Mean CV ROC-AUC": metrics.get("cv_mean_auc"),
            "Std CV AUC": metrics.get("cv_std_auc"),
            "Mean CV Gini": metrics.get("cv_mean_gini"),
            "Mean CV Brier Score": metrics.get("cv_mean_brier"),
            "Training Time (s)": metrics.get("training_time_sec"),
            "Status": "CHAMPION" if m_name == artifact.model_name else "BENCHMARK",
        })
    cv_table = pd.DataFrame(cv_rows)
    st.dataframe(cv_table, use_container_width=True)

st.markdown("---")

# 3. Interactive Discriminatory & Calibration Charts
st.subheader("Discrimination Power & Reliability Curves")

_, X_test, _, y_test = prepare_train_test_split(df)
y_prob_calibrated = artifact.predict_proba(X_test, calibrate=True)

tab_roc, tab_ks, tab_calib, tab_score = st.tabs([
    "ROC Curve",
    "Kolmogorov-Smirnov Separation",
    "Probability Calibration",
    "Score Distribution",
])

with tab_roc:
    fig_roc = plot_roc_curve(y_test.values, y_prob_calibrated, test_metrics.roc_auc, test_metrics.gini)
    st.plotly_chart(fig_roc, use_container_width=True)

with tab_ks:
    fig_ks = plot_ks_curve(y_test.values, y_prob_calibrated, test_metrics.ks_statistic, test_metrics.ks_threshold)
    st.plotly_chart(fig_ks, use_container_width=True)

with tab_calib:
    fig_calib = plot_calibration_curve(y_test.values, y_prob_calibrated)
    st.plotly_chart(fig_calib, use_container_width=True)

with tab_score:
    scorecard = CreditScorecard()
    scores = scorecard.pd_to_score(y_prob_calibrated)
    eval_df = pd.DataFrame({"credit_score": scores, "default_payment_next_month": y_test.values})
    fig_score = plot_score_distribution(eval_df)
    st.plotly_chart(fig_score, use_container_width=True)

# 4. Interactive Cutoff Decision Matrix
st.markdown("---")
st.subheader("Interactive Decision Cutoff Sensitivity")

cutoff = st.slider("Select Probability Decision Cutoff Threshold", min_value=0.05, max_value=0.90, value=0.50, step=0.05)
evaluator = ModelEvaluator(cutoff=cutoff)
custom_metrics = evaluator.evaluate(y_test.values, y_prob_calibrated, compute_ci=False)

c_m1, c_m2, c_m3, c_m4 = st.columns(4)
with c_m1:
    st.metric("Accuracy at Cutoff", f"{custom_metrics.accuracy:.2%}")
with c_m2:
    st.metric("Precision (Default)", f"{custom_metrics.precision:.2%}")
with c_m3:
    st.metric("Recall (Sensitivity)", f"{custom_metrics.recall:.2%}")
with c_m4:
    st.metric("F1-Score", f"{custom_metrics.f1:.4f}")

# Confusion matrix display
cm = custom_metrics.confusion_mat
col_cm1, col_cm2 = st.columns(2)
with col_cm1:
    cm_df = pd.DataFrame({
        "Predicted Non-Default": [f"TN: {cm['TN']:,}", f"FN: {cm['FN']:,}"],
        "Predicted Default": [f"FP: {cm['FP']:,}", f"TP: {cm['TP']:,}"],
    }, index=["Actual Non-Default", "Actual Default"])
    st.table(cm_df)

with col_cm2:
    st.caption(
        f"At threshold {cutoff:.2f}, the model detects **{cm['TP']:,}** out of **{cm['TP'] + cm['FN']:,}** defaults "
        f"({custom_metrics.recall:.1%} default capture rate), while generating **{cm['FP']:,}** false alarms."
    )
