"""
CreditRisk Analytics — Page 1: Executive Overview
=================================================

Presents high-level portfolio KPIs, model evaluation benchmarks, and borrower risk distribution.
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

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, UCI_DATASET_ATTRIBUTION
from src.data.ingestion import load_raw_dataset
from src.risk.pd_analysis import backtest_pd_by_grade
from src.risk.segmentation import MasterRatingScale, RiskSegmenter

st.set_page_config(page_title="Executive Overview | CreditRisk", page_icon="🏛️", layout="wide")

st.title("🏛️ CreditRisk Analytics — Executive Overview")
st.caption("Quantitative Credit Risk Modeling, Portfolio Diagnostics & Supervisory Governance")

# Prototype & Governance Attribution Banner
st.warning(
    f"**⚖️ GOVERNANCE NOTICE & DATASET ATTRIBUTION:**\n\n"
    f"{UCI_DATASET_ATTRIBUTION.disclaimer}\n\n"
    f"*Citation:* {UCI_DATASET_ATTRIBUTION.citation} (License: {UCI_DATASET_ATTRIBUTION.license_type})"
)

# Load Model Artifact & Data Safely
artifact_path = MODELS_DIR / "pd_model_best.joblib"
csv_path = PROCESSED_DATA_DIR / "uci_credit_card_default.csv"

if not artifact_path.exists():
    st.error(
        f"**Champion Model Artifact Missing:** Could not locate `{artifact_path}`.\n\n"
        "Please run `python run_pipeline.py` or trigger model training from the pipeline to generate model artifacts."
    )
    st.stop()

@st.cache_data
def load_portfolio_data():
    df = load_raw_dataset(source="uci")
    model_art = joblib.load(artifact_path)
    probs = model_art.predict_proba(df, calibrate=True)
    df["calibrated_pd"] = probs

    segmenter = RiskSegmenter()
    df = segmenter.segment(df, pd_col="calibrated_pd")
    return df, model_art

with st.spinner("Loading scored portfolio from disk..."):
    df_scored, model_art = load_portfolio_data()

# ---------------------------------------------------------------------------
# 1. Headline Executive KPI Cards
# ---------------------------------------------------------------------------
st.subheader("📊 Portfolio & Risk Summary KPIs")

total_accounts = len(df_scored)
target_col = "default_payment_next_month"
obs_defaults = int(df_scored[target_col].sum()) if target_col in df_scored.columns else 0
obs_default_rate = (obs_defaults / total_accounts * 100.0) if total_accounts > 0 else 0.0
avg_predicted_pd = float(df_scored["calibrated_pd"].mean() * 100.0)
test_auc = model_art.test_metrics.roc_auc
test_gini = model_art.test_metrics.gini
test_ks = model_art.test_metrics.ks_statistic

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Accounts", f"{total_accounts:,}", "Retail Portfolio")
with kpi2:
    st.metric("Observed Default Rate", f"{obs_default_rate:.2f}%", f"{obs_defaults:,} defaults")
with kpi3:
    st.metric("Mean Predicted PD", f"{avg_predicted_pd:.2f}%", f"Δ: {(avg_predicted_pd - obs_default_rate):+.2f}%")
with kpi4:
    st.metric("Champion Model", model_art.model_name.replace("_", " "), f"AUC: {test_auc:.4f}")
with kpi5:
    st.metric("Regulatory Gini", f"{test_gini:.4f}", f"KS: {test_ks:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# 2. Borrower Risk Distribution across Master Rating Scale
# ---------------------------------------------------------------------------
st.subheader("🎯 Borrower Risk Distribution (Master Rating Scale: AAA to CCC)")

scale = MasterRatingScale()
rating_summary = scale.summarize_portfolio(df_scored, pd_col="calibrated_pd", target_col=target_col)
backtest_res = backtest_pd_by_grade(df_scored, grade_col="rating_grade", pd_col="calibrated_pd", target_col=target_col)

# Merge backtest traffic lights into rating summary
merged_rating = rating_summary.merge(
    backtest_res[["Rating_Grade", "Traffic_Light", "P_Value_One_Sided"]],
    left_on="Grade", right_on="Rating_Grade", how="left"
).drop(columns=["Rating_Grade"])

row_chart, row_table = st.columns([1, 1])

with row_chart:
    # Plot distribution bar chart
    fig_dist = go.Figure()
    grade_colors = {
        "AAA": "#10B981", "AA": "#34D399", "A": "#0284C7", "BBB": "#38BDF8",
        "BB": "#F59E0B", "B": "#F97316", "CCC": "#EF4444"
    }
    bar_colors = [grade_colors.get(g, "#64748B") for g in merged_rating["Grade"]]

    fig_dist.add_trace(go.Bar(
        x=merged_rating["Grade"],
        y=merged_rating["Portfolio_Pct"],
        marker_color=bar_colors,
        text=[f"{pct:.1f}%<br>({cnt:,})" for pct, cnt in zip(merged_rating["Portfolio_Pct"], merged_rating["Account_Count"])],
        textposition="outside",
        hovertemplate="<b>Grade %{x}</b><br>Share: %{y:.2f}%<extra></extra>",
    ))

    fig_dist.update_layout(
        title={"text": "<b>Account Share by Master Rating Grade</b>", "x": 0.05, "font": {"size": 15}},
        xaxis_title="Internal Rating Grade",
        yaxis_title="Portfolio Share (%)",
        template="plotly_white",
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        font={"family": "Inter, Arial, sans-serif", "size": 12},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with row_table:
    st.markdown("#### Master Rating Scale Ledger")
    def highlight_tl(val):
        colors = {
            "GREEN": "background-color: #DCFCE7; color: #166534; font-weight: bold",
            "AMBER": "background-color: #FEF3C7; color: #92400E; font-weight: bold",
            "RED": "background-color: #FEE2E2; color: #991B1B; font-weight: bold",
        }
        return colors.get(val, "")

    display_df = merged_rating.rename(columns={
        "Grade": "Rating Grade",
        "Description": "Credit Quality Description",
        "Account_Count": "Accounts",
        "Portfolio_Pct": "Share (%)",
        "Average_PD": "Model PD (%)",
        "Realized_Default_Rate": "Realized DR (%)",
        "Traffic_Light": "Basel Traffic Light",
    })
    styled_df = display_df.style.map(highlight_tl, subset=["Basel Traffic Light"])
    st.dataframe(styled_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 3. Model Evaluation Summary
# ---------------------------------------------------------------------------
st.subheader("📈 Model Benchmarking & Evaluation Summary")

col_cv, col_test = st.columns([1, 1])

with col_cv:
    st.markdown("#### 5-Fold Stratified Cross-Validation Benchmarks")
    cv_records = []
    for m_name, metrics in model_art.cv_summary.items():
        is_champ = m_name == model_art.model_name
        cv_records.append({
            "Model Name": m_name.replace("_", " ") + (" ⭐ (Champion)" if is_champ else ""),
            "CV ROC-AUC": f"{metrics['cv_mean_auc']:.4f} ± {metrics['cv_std_auc']:.4f}",
            "CV Gini": f"{metrics['cv_mean_gini']:.4f}",
            "CV Brier": f"{metrics['cv_mean_brier']:.4f}",
            "Train Time": f"{metrics['training_time_sec']:.2f}s",
        })
    st.dataframe(pd.DataFrame(cv_records), use_container_width=True)

with col_test:
    st.markdown("#### Out-of-Sample Test Set Performance (6,000 Accounts)")
    tm = model_art.test_metrics
    test_df = pd.DataFrame([
        {"Metric": "ROC-AUC", "Estimate": f"{tm.roc_auc:.4f}", "95% Bootstrap CI": f"[{tm.auc_ci_lower:.4f}, {tm.auc_ci_upper:.4f}]"},
        {"Metric": "Gini (Somers' D)", "Estimate": f"{tm.gini:.4f}", "95% Bootstrap CI": f"[{tm.gini_ci_lower:.4f}, {tm.gini_ci_upper:.4f}]"},
        {"Metric": "KS Statistic", "Estimate": f"{tm.ks_statistic:.4f}", "Cutoff": f"PD = {tm.ks_threshold:.4f}"},
        {"Metric": "Brier Score", "Estimate": f"{tm.brier_score:.4f}", "Benchmark": "Lower is better"},
        {"Metric": "Log Loss", "Estimate": f"{tm.log_loss_val:.4f}", "Benchmark": "Cross-entropy"},
        {"Metric": "PR-AUC", "Estimate": f"{tm.pr_auc:.4f}", "Baseline Prevalence": f"{obs_default_rate:.2f}%"},
    ])
    st.dataframe(test_df, use_container_width=True)
