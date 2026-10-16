"""
CreditRisk Analytics — Portfolio Overview & Exploratory Data Analysis (EDA)
============================================================================

Visualizes credit exposures, borrower demographics, target class distribution,
repayment delinquency gradients, credit limit profiles, and borrower risk segments.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.eda import (
    analyze_default_rate_by_credit_limit,
    analyze_default_rate_by_repayment_status,
    compute_dataset_summary,
    identify_and_verify_target,
    plot_credit_limit_distribution_by_target,
    plot_default_rate_by_repayment_status,
    plot_default_rates_across_segments,
    plot_target_distribution,
)
from src.data.ingestion import UCI_DATA_DICTIONARY, UCICardDataIngestor, load_raw_dataset

st.set_page_config(page_title="Portfolio Overview & EDA | CreditRisk", page_icon="📊", layout="wide")

st.title("📊 Loan Portfolio Overview & Exploratory Data Analysis")
st.caption("Quantitative exploratory analysis of borrower risk factors, repayment statuses, and class imbalance.")

@st.cache_data
def get_portfolio_data():
    return load_raw_dataset(source="uci")

try:
    df = get_portfolio_data()
except Exception as e:
    st.error(f"Error loading portfolio dataset: {e}")
    st.stop()

target_col, target_summary = identify_and_verify_target(df)

# Headline Portfolio Metrics
col1, col2, col3, col4 = st.columns(4)
total_limit = df["limit_bal"].sum()
total_balance = df["bill_amt1"].sum()
default_rate = target_summary.default_rate_pct
avg_age = df["age"].mean()

with col1:
    st.metric("Total Credit Exposure (Limits)", f"${total_limit:,.0f}")
with col2:
    st.metric("Current Outstanding Balance", f"${total_balance:,.0f}")
with col3:
    st.metric("Realized Default Rate", f"{default_rate:.2f}%", help=f"Class Imbalance: {target_summary.imbalance_ratio:.2f}:1")
with col4:
    st.metric("Average Borrower Age", f"{avg_age:.1f} years")

# Disclaimers & Statistical Guidance Callout
st.info(
    "**STATISTICAL & METHODOLOGICAL GUARDRAILS:**\n"
    f"- **Class Imbalance:** Observed default prevalence is **{default_rate:.2f}%** ({target_summary.default_count:,} defaults vs. {target_summary.non_default_count:,} non-defaults, **{target_summary.imbalance_ratio:.2f}:1 ratio**). Models must be evaluated on discrimination (ROC-AUC/Gini) and calibration (Brier score) rather than raw accuracy.\n"
    "- **Association vs. Causation:** Demographic and balance factors are empirical correlative markers, not causal mechanisms. Lower assigned limits reflect past underwriting discretion.\n"
    "- **Zero Target Leakage:** Feature pipelines do not utilize target information.\n"
    "- **Asset Class Limitation:** Consumer credit card delinquency patterns do not directly generalize to corporate borrowers."
)

st.markdown("---")

tab_target, tab_repay, tab_limit, tab_segments, tab_summary, tab_dict = st.tabs([
    "Target Distribution",
    "Repayment Status Gradient",
    "Credit Limit Distribution",
    "Cross-Sectional Segments",
    "Statistical Summaries",
    "Variable Data Dictionary",
])

with tab_target:
    st.subheader("Target Class Distribution & Imbalance Audit")
    fig_target = plot_target_distribution(df, target_col=target_col)
    st.plotly_chart(fig_target, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Target Summary Table:**")
        target_tbl = pd.DataFrame([{
            "Target Column": target_summary.target_column_name,
            "Total Borrowers": f"{target_summary.total_records:,}",
            "Performing (Class 0)": f"{target_summary.non_default_count:,} ({100 - target_summary.default_rate_pct:.1f}%)",
            "Defaults (Class 1)": f"{target_summary.default_count:,} ({target_summary.default_rate_pct:.1f}%)",
            "Imbalance Ratio": f"{target_summary.imbalance_ratio:.2f} : 1",
            "Binary Integrity": "PASSED [0, 1]" if target_summary.is_valid else "FAILED",
        }])
        st.table(target_tbl)
    with col_t2:
        st.caption(
            "In retail banking portfolios, moderate class imbalance (~20-25%) reflects natural credit selection. "
            "Downstream models use stratified k-fold cross-validation and probability calibration to prevent majority-class bias."
        )

with tab_repay:
    st.subheader("Default Rate by Repayment Status (PAY_0)")
    st.caption("Monotonic escalation in default risk as repayment delay increases from 0 to 8 months.")

    fig_repay = plot_default_rate_by_repayment_status(df, pay_col="pay_0", target_col=target_col)
    st.plotly_chart(fig_repay, use_container_width=True)

    repay_df = analyze_default_rate_by_repayment_status(df, pay_col="pay_0", target_col=target_col)
    st.write("**Repayment Delay Risk Gradient Table (with 95% Wilson Score Confidence Intervals):**")
    st.dataframe(repay_df, use_container_width=True)

with tab_limit:
    st.subheader("Credit Limit Underwriting Distribution by Default Outcome")
    fig_limit = plot_credit_limit_distribution_by_target(df, limit_col="limit_bal", target_col=target_col)
    st.plotly_chart(fig_limit, use_container_width=True)

    limit_df = analyze_default_rate_by_credit_limit(df, limit_col="limit_bal", target_col=target_col)
    st.write("**Default Rates by Credit Limit Bracket:**")
    st.dataframe(limit_df, use_container_width=True)

with tab_segments:
    st.subheader("Multi-Factor Risk Segmentation")
    fig_segments = plot_default_rates_across_segments(df)
    st.plotly_chart(fig_segments, use_container_width=True)

with tab_summary:
    st.subheader("Parametric & Non-Parametric Distribution Summaries")
    summaries = compute_dataset_summary(df)

    st.write("**Numerical Feature Summary (Mean, Std, Median, IQR, Skewness):**")
    st.dataframe(summaries["numerical_summary"], use_container_width=True)

    st.write("**Categorical Feature Summary:**")
    st.dataframe(summaries["categorical_summary"], use_container_width=True)

with tab_dict:
    st.subheader("Preserved Data Lineage & Variable Dictionary")
    st.caption("Original UCI raw attributes mapped to standardized snake_case schema.")

    dict_rows = []
    for orig_name, meta in UCI_DATA_DICTIONARY.items():
        dict_rows.append({
            "Original Column": orig_name,
            "Standardized Name": meta["standard_name"],
            "Data Type": meta["data_type"],
            "Description": meta["description"],
            "Valid Values / Bounds": meta["valid_values"],
        })
    dict_df = pd.DataFrame(dict_rows)
    st.dataframe(dict_df, use_container_width=True)

    with st.expander("🔍 View Raw Cached Data Sample (Preserved Original Headers)"):
        ingestor = UCICardDataIngestor()
        raw_df = ingestor.load_raw()
        st.dataframe(raw_df.head(15), use_container_width=True)
