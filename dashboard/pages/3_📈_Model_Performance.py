"""
CreditRisk Analytics — Page 3: Model Performance
================================================

Comprehensive diagnostic evaluation of probability of default models:
- ROC curves with Gini annotations
- Precision-Recall curves
- Reliability calibration curves (Raw vs Calibrated)
- Brier score and Log loss comparisons
- Benchmark model comparisons (Logistic Regression vs Random Forest vs Gradient Boosting)
- Interactive operational threshold explorer
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
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import MODELS_DIR
from src.data.ingestion import load_raw_dataset
from src.data.preprocessing import prepare_train_test_split
from src.reporting.charts import plot_calibration_comparison, plot_ks_separation, plot_roc_curve

st.set_page_config(page_title="Model Performance | CreditRisk", page_icon="📈", layout="wide")

st.title("📈 Model Performance & Diagnostic Evaluation")
st.caption("Benchmarking discriminatory power, probability calibration, and operational cutoffs.")

artifact_path = MODELS_DIR / "pd_model_best.joblib"
if not artifact_path.exists():
    st.error("Champion model artifact missing. Please run `python run_pipeline.py` first.")
    st.stop()

@st.cache_data
def get_model_and_test_eval():
    model_art = joblib.load(artifact_path)
    df = load_raw_dataset(source="uci")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    p_test_cal = model_art.predict_proba(X_test, calibrate=True)
    p_test_raw = model_art.predict_proba(X_test, calibrate=False)
    return model_art, y_test.values, p_test_cal, p_test_raw

with st.spinner("Loading model evaluations..."):
    model_art, y_test, p_test_cal, p_test_raw = get_model_and_test_eval()

tm = model_art.test_metrics

# ---------------------------------------------------------------------------
# Section 1: Multi-Model Benchmark Comparison Table
# ---------------------------------------------------------------------------
st.subheader("🏆 Model Benchmark Comparison (5-Fold Stratified CV)")

cv_rows = []
for m_name, metrics in model_art.cv_summary.items():
    is_champ = m_name == model_art.model_name
    cv_rows.append({
        "Algorithm": m_name.replace("_", " ") + (" ⭐ (Champion)" if is_champ else ""),
        "CV ROC-AUC": f"{metrics['cv_mean_auc']:.4f} ± {metrics['cv_std_auc']:.4f}",
        "CV Gini": f"{metrics['cv_mean_gini']:.4f}",
        "CV Brier Score": f"{metrics['cv_mean_brier']:.4f}",
        "Training Latency": f"{metrics['training_time_sec']:.2f}s",
    })
st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Core Diagnostic Curves (ROC, PR, Calibration, KS)
# ---------------------------------------------------------------------------
st.subheader("📊 Diagnostic Discrimination & Calibration Curves")

col_c1, col_c2 = st.columns(2)

with col_c1:
    # 1. ROC Curve
    fig_roc = plot_roc_curve(y_test, p_test_cal, roc_auc=tm.roc_auc, gini=tm.gini)
    st.plotly_chart(fig_roc, use_container_width=True)

with col_c2:
    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, p_test_cal)
    prevalence = float(np.mean(y_test))

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall_vals, y=precision_vals,
        mode="lines",
        name=f"Champion Model (PR-AUC = {tm.pr_auc:.4f})",
        line={"color": "#0284C7", "width": 3},
    ))
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[prevalence, prevalence],
        mode="lines",
        name=f"No-Skill Baseline ({prevalence:.1%})",
        line={"color": "#94A3B8", "width": 1.5, "dash": "dash"},
    ))
    fig_pr.update_layout(
        title={"text": "<b>Precision-Recall (PR) Curve</b>", "x": 0.05},
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        template="plotly_white",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    st.plotly_chart(fig_pr, use_container_width=True)

col_c3, col_c4 = st.columns(2)

with col_c3:
    # 3. Reliability Calibration Plot
    from src.models.calibration import CalibrationAnalyser
    cal_analyser = CalibrationAnalyser(n_bins=10, strategy="quantile")
    cal_comp = cal_analyser.compare(y_test, p_test_raw, p_test_cal, method="isotonic")
    fig_cal = plot_calibration_comparison(cal_comp)
    st.plotly_chart(fig_cal, use_container_width=True)

with col_c4:
    # 4. Kolmogorov-Smirnov (KS) Cumulative Distribution Plot
    fig_ks = plot_ks_separation(y_test, p_test_cal, ks_stat=tm.ks_statistic, ks_threshold=tm.ks_threshold)
    st.plotly_chart(fig_ks, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Brier Score & Log Loss Breakdown
# ---------------------------------------------------------------------------
st.subheader("📉 Probabilistic Error Metrics: Brier Score & Log Loss")

brier_raw = float(np.mean((p_test_raw - y_test) ** 2))
brier_cal = float(np.mean((p_test_cal - y_test) ** 2))
logloss_raw = float(cal_comp.raw_log_loss)
logloss_cal = float(cal_comp.cal_log_loss)

err_col1, err_col2 = st.columns([1, 1])
with err_col1:
    st.markdown("#### Probabilistic Loss Comparison (Raw vs. Calibrated)")
    err_df = pd.DataFrame([
        {
            "Metric": "Brier Score Loss",
            "Formula": "Mean Squared Probability Error",
            "Raw (Uncalibrated)": f"{brier_raw:.4f}",
            "Calibrated": f"{brier_cal:.4f}",
            "Improvement": f"{((brier_raw - brier_cal) / brier_raw * 100):+.2f}%",
        },
        {
            "Metric": "Log Loss (Cross-Entropy)",
            "Formula": "-[y*ln(p) + (1-y)*ln(1-p)]",
            "Raw (Uncalibrated)": f"{logloss_raw:.4f}",
            "Calibrated": f"{logloss_cal:.4f}",
            "Improvement": f"{((logloss_raw - logloss_cal) / logloss_raw * 100):+.2f}%",
        },
        {
            "Metric": "Expected Calibration Error (ECE)",
            "Formula": "Population-Weighted |Pred - Obs|",
            "Raw (Uncalibrated)": f"{cal_comp.raw_ece:.4f}",
            "Calibrated": f"{cal_comp.cal_ece:.4f}",
            "Improvement": f"{((cal_comp.raw_ece - cal_comp.cal_ece) / cal_comp.raw_ece * 100):+.2f}%",
        },
    ])
    st.dataframe(err_df, use_container_width=True)

with err_col2:
    st.info(
        "**💡 Methodological Distinction:**\n\n"
        "- **Ranking Quality (AUC / Gini):** Measures whether borrowers who default receive higher scores than non-defaulters. "
        "Monotonic probability transformations do not change ranking quality.\n"
        "- **Calibration (Brier / Log Loss / ECE):** Measures whether a predicted 20% PD truly reflects a 20% empirical default frequency. "
        "Calibration is mandatory under IFRS 9 because probability inputs drive loss reserves directly."
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Interactive Operational Threshold Explorer
# ---------------------------------------------------------------------------
st.subheader("🎛️ Operational Decision Cutoff & Confusion Matrix Explorer")
st.markdown("Slide the probability cutoff to simulate real-world underwriting approval and decline thresholds.")

cutoff = st.slider("Probability Decision Cutoff (Approve if PD < Cutoff)", min_value=0.05, max_value=0.90, value=0.2154, step=0.01)

y_pred_cutoff = (p_test_cal >= cutoff).astype(int)
tp = int(np.sum((y_test == 1) & (y_pred_cutoff == 1)))
fp = int(np.sum((y_test == 0) & (y_pred_cutoff == 1)))
tn = int(np.sum((y_test == 0) & (y_pred_cutoff == 0)))
fn = int(np.sum((y_test == 1) & (y_pred_cutoff == 0)))

total_n = len(y_test)
precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
approval_rate = (tn + fn) / total_n * 100.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Selected Cutoff", f"{cutoff:.2%}")
m2.metric("Approval Rate", f"{approval_rate:.1f}%")
m3.metric("Precision (PPV)", f"{precision:.2%}")
m4.metric("Recall (Sensitivity)", f"{recall:.2%}")
m5.metric("F1 Score", f"{f1:.4f}")

# Display Confusion Matrix
cm_df = pd.DataFrame(
    [[f"TN: {tn:,}", f"FP (False Alarm): {fp:,}"],
     [f"FN (Missed Default): {fn:,}", f"TP (Caught Default): {tp:,}"]],
    index=["Actual Non-Default (y=0)", "Actual Default (y=1)"],
    columns=["Predicted Non-Default (Approved)", "Predicted Default (Declined)"],
)
st.dataframe(cm_df, use_container_width=True)
