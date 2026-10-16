"""
CreditRisk Analytics — Page 2: Portfolio Analytics
==================================================

Interactive exploratory risk analysis, borrower distributions, default rates by feature,
and risk-segment analysis with dynamic filters.
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR
from src.data.ingestion import load_raw_dataset
from src.reporting.charts import plot_decile_default_rates, plot_segment_bar
from src.risk.segmentation import BorrowerRiskSegmenter, MasterRatingScale, RiskSegmenter

st.set_page_config(page_title="Portfolio Analytics | CreditRisk", page_icon="📊", layout="wide")

st.title("📊 Credit Risk Portfolio Analytics")
st.caption("Deep-dive demographic distributions, delinquency behavior, and segment default rates.")

artifact_path = MODELS_DIR / "pd_model_best.joblib"
if not artifact_path.exists():
    st.error("Champion model artifact missing. Please run `python run_pipeline.py` first.")
    st.stop()

@st.cache_data
def get_dataset():
    df = load_raw_dataset(source="uci")
    model_art = joblib.load(artifact_path)
    probs = model_art.predict_proba(df, calibrate=True)
    df["calibrated_pd"] = probs

    segmenter = RiskSegmenter()
    df = segmenter.segment(df, pd_col="calibrated_pd")
    return df

df_full = get_dataset()

# ===========================================================================
# Sidebar Interactive Filters
# ===========================================================================
st.sidebar.header("🔍 Interactive Portfolio Filters")

# Age Filter
min_age = int(df_full["age"].min())
max_age = int(df_full["age"].max())
age_range = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Education Filter
edu_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others", 0: "Unknown", 5: "Unknown", 6: "Unknown"}
edu_options = ["All"] + list(dict.fromkeys(edu_map.values()))
edu_choice = st.sidebar.selectbox("Education Level", options=edu_options, index=0)

# Marriage Filter
marriage_map = {1: "Married", 2: "Single", 3: "Others", 0: "Unknown"}
marriage_options = ["All"] + list(dict.fromkeys(marriage_map.values()))
marriage_choice = st.sidebar.selectbox("Marital Status", options=marriage_options, index=0)

# Credit Limit Filter
min_limit = int(df_full["limit_bal"].min())
max_limit = int(df_full["limit_bal"].max())
limit_range = st.sidebar.slider("Credit Limit ($ NTD)", min_value=min_limit, max_value=max_limit, value=(min_limit, max_limit), step=10000)

# Apply filters
filtered_df = df_full[
    (df_full["age"] >= age_range[0]) & (df_full["age"] <= age_range[1]) &
    (df_full["limit_bal"] >= limit_range[0]) & (df_full["limit_bal"] <= limit_range[1])
].copy()

if edu_choice != "All":
    filtered_df = filtered_df[filtered_df["education"].map(edu_map) == edu_choice]

if marriage_choice != "All":
    filtered_df = filtered_df[filtered_df["marriage"].map(marriage_map) == marriage_choice]

st.info(f"**Filtered Cohort:** Displaying **{len(filtered_df):,}** of {len(df_full):,} accounts ({(len(filtered_df)/len(df_full)*100):.1f}%).")

target_col = "default_payment_next_month"

# ===========================================================================
# Tabs: 1. Feature Default Rates, 2. Risk Segments, 3. Borrower Distribution
# ===========================================================================
tab_features, tab_segments, tab_distributions = st.tabs([
    "📈 Default Rates by Feature",
    "🎯 Risk-Segment Analysis",
    "👥 Borrower Distributions",
])

# ---------------------------------------------------------------------------
# Tab 1: Default Rates by Feature
# ---------------------------------------------------------------------------
with tab_features:
    st.subheader("Default Rates Across Borrower Attributes")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Default Rate by Repayment Status (pay_0)")
        pay_labels = {
            -2: "No consumption", -1: "Paid in full", 0: "Revolving",
            1: "30 DPD (1 mo)", 2: "60 DPD (2 mo)", 3: "90 DPD (3 mo)",
            4: "120 DPD", 5: "150 DPD", 6: "180 DPD", 7: "210 DPD", 8: "240+ DPD"
        }
        pay_summary = filtered_df.groupby("pay_0").agg(
            Accounts=("id", "count"),
            Defaults=(target_col, "sum"),
        ).reset_index()
        pay_summary["Default_Rate_Pct"] = (pay_summary["Defaults"] / pay_summary["Accounts"] * 100.0)
        pay_summary["Status_Label"] = pay_summary["pay_0"].map(pay_labels).fillna("Unknown")

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Bar(
            x=[f"{r.Status_Label} ({r.pay_0})" for _, r in pay_summary.iterrows()],
            y=pay_summary["Default_Rate_Pct"],
            marker_color="#DC2626",
            text=[f"{v:.1f}%<br>(n={a:,})" for v, a in zip(pay_summary["Default_Rate_Pct"], pay_summary["Accounts"])],
            textposition="outside",
        ))
        fig_pay.update_layout(
            title={"text": "<b>Default Rate by Recent Payment Status (pay_0)</b>", "x": 0.05},
            xaxis_title="Repayment Status Code",
            yaxis_title="Default Rate (%)",
            template="plotly_white",
            margin={"l": 50, "r": 30, "t": 50, "b": 100},
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    with c2:
        st.markdown("#### Default Rate by Credit Limit Decile")
        filtered_df["limit_decile"] = pd.qcut(filtered_df["limit_bal"], q=10, duplicates="drop")
        lim_summary = filtered_df.groupby("limit_decile", observed=True).agg(
            Accounts=("id", "count"),
            Defaults=(target_col, "sum"),
        ).reset_index()
        lim_summary["Default_Rate_Pct"] = (lim_summary["Defaults"] / lim_summary["Accounts"] * 100.0)
        lim_summary["Decile_Label"] = [f"Decile {i+1}" for i in range(len(lim_summary))]

        fig_lim = go.Figure()
        fig_lim.add_trace(go.Bar(
            x=lim_summary["Decile_Label"],
            y=lim_summary["Default_Rate_Pct"],
            marker_color="#0284C7",
            text=[f"{v:.1f}%" for v in lim_summary["Default_Rate_Pct"]],
            textposition="outside",
        ))
        fig_lim.update_layout(
            title={"text": "<b>Default Rate Across Credit Limit Deciles</b>", "x": 0.05},
            xaxis_title="Credit Limit Decile (Low to High)",
            yaxis_title="Default Rate (%)",
            template="plotly_white",
            margin={"l": 50, "r": 30, "t": 50, "b": 50},
        )
        st.plotly_chart(fig_lim, use_container_width=True)

    st.markdown("#### Demographic Interaction: Education Level vs. Marital Status")
    demo_pivot = filtered_df.pivot_table(
        index=filtered_df["education"].map(edu_map),
        columns=filtered_df["marriage"].map(marriage_map),
        values=target_col,
        aggfunc=lambda x: np.mean(x) * 100.0,
    ).round(2)
    st.dataframe(demo_pivot.style.background_gradient(cmap="Reds"), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Risk-Segment Analysis
# ---------------------------------------------------------------------------
with tab_segments:
    st.subheader("Borrower Risk Segmentation & Monotonicity Verification")
    st.caption("Verifying whether observed default rate increases monotonically across predicted risk bands.")

    borrower_segmenter = BorrowerRiskSegmenter()
    seg_res = borrower_segmenter.segment(filtered_df, pd_col="calibrated_pd", target_col=target_col)

    fig_deciles = plot_decile_default_rates(seg_res.decile_table)
    st.plotly_chart(fig_deciles, use_container_width=True)

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("#### Risk Decile Analysis Ledger")
        st.dataframe(seg_res.decile_table, use_container_width=True)
    with c_s2:
        st.markdown("#### Risk Band Segments Ledger (Illustrative Bands)")
        st.dataframe(seg_res.segment_table, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Borrower Distributions
# ---------------------------------------------------------------------------
with tab_distributions:
    st.subheader("Borrower Distribution & Balance Characteristics")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("#### Credit Limit Distribution by Default Outcome")
        fig_kde = px.histogram(
            filtered_df,
            x="limit_bal",
            color=target_col,
            barmode="overlay",
            nbins=40,
            color_discrete_map={0: "#0284C7", 1: "#DC2626"},
            labels={target_col: "Default Status", "limit_bal": "Credit Limit ($ NTD)"},
        )
        fig_kde.update_layout(template="plotly_white", margin={"l": 50, "r": 30, "t": 50, "b": 50})
        st.plotly_chart(fig_kde, use_container_width=True)

    with d2:
        st.markdown("#### Age Distribution Box Plot by Default Outcome")
        fig_box = px.box(
            filtered_df,
            x=target_col,
            y="age",
            color=target_col,
            color_discrete_map={0: "#0284C7", 1: "#DC2626"},
            labels={target_col: "Default Status (0=Non-default, 1=Default)", "age": "Borrower Age"},
        )
        fig_box.update_layout(template="plotly_white", margin={"l": 50, "r": 30, "t": 50, "b": 50}, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
