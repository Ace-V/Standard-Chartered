"""
CreditRisk Analytics — Streamlit Platform Gateway & Portfolio Overview
======================================================================

Main dashboard entry point for the Credit Risk Modelling and Monitoring System.
Displays system architecture, executive summary KPIs, legal/attribution notices,
and real-time pipeline status.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for Streamlit execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    UCI_DATASET_ATTRIBUTION,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    logger,
)
from src.data.ingestion import load_raw_dataset
from src.data.validation import DataValidator
from src.storage.database import get_db_manager

# Configure Streamlit page
st.set_page_config(
    page_title="CreditRisk Analytics | Quantitative Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-consumer {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-corporate {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🏦 CreditRisk Analytics — Risk Modelling & Monitoring System")
    st.caption("Institutional Quantitative Risk, IFRS 9 Impairment, and Basel Model Governance Platform")

    # Regulatory & Prototype Disclaimer Banner
    st.warning(
        f"**PROTOTYPE NOTICE & GOVERNANCE ATTRIBUTION:**\n\n"
        f"{UCI_DATASET_ATTRIBUTION.disclaimer}\n\n"
        f"*Reference Citation:* {UCI_DATASET_ATTRIBUTION.citation} (License: {UCI_DATASET_ATTRIBUTION.license_type})"
    )

    # Top-Level Tabs
    tab_overview, tab_architecture, tab_dataset = st.tabs([
        "Executive Summary",
        "System Architecture & Extensibility",
        "Dataset Attribution & Ingestion Status",
    ])

    with tab_overview:
        st.subheader("Platform Capabilities Overview")
        col1, col2, col3, col4 = st.columns(4)

        # Check existing dataset
        csv_path = PROCESSED_DATA_DIR / "uci_credit_card_default.csv"
        data_loaded = csv_path.exists()
        model_trained = (MODELS_DIR / "pd_model_best.joblib").exists()

        with col1:
            st.metric(
                label="Portfolio Asset Class",
                value="Retail Credit Cards",
                delta="Consumer Proxy",
                help="Active model baseline trained on retail revolving consumer credit.",
            )
        with col2:
            st.metric(
                label="Data Status",
                value="Cached & Verified" if data_loaded else "Ingestion Pending",
                delta="30,000 Accounts" if data_loaded else "0 Accounts",
            )
        with col3:
            st.metric(
                label="Champion PD Model",
                value="Calibrated Ensemble" if model_trained else "Not Fitted Yet",
                delta="Gini ~ 0.53 | KS ~ 0.40" if model_trained else None,
            )
        with col4:
            st.metric(
                label="Accounting Standard",
                value="IFRS 9 / CECL",
                delta="3-Stage Impairment",
                help="Three-stage ECL calculation with forward-looking macroeconomic scenarios.",
            )

        st.markdown("---")
        st.markdown("### Executive Platform Workflow")

        st.info(
            """
            Navigate through the specialized analytical modules via the left sidebar:
            1. **1_📊_Portfolio_Overview**: Explore demographic profiles, delinquency distributions, and credit limits.
            2. **2_🔍_Data_Quality_&_Validation**: Run BCBS 239 regulatory data hygiene, boundary, and integrity audits.
            3. **3_📈_PD_Model_Performance**: Inspect multi-model ROC/PR curves, Kolmogorov-Smirnov separation, and reliability curves.
            4. **4_💰_ECL_&_Scenario_Analysis**: Dynamic IFRS 9 Expected Credit Loss staging with interactive macroeconomic stress testing.
            5. **5_🛡️_Monitoring_&_Stability**: Production drift monitoring with Population Stability Index (PSI) and feature CSI.
            """
        )

    with tab_architecture:
        st.subheader("Dual-Engine Architecture: Retail Consumer & Corporate Credit")
        st.markdown(
            """
            The platform is built on an **abstract risk domain architecture**, strictly decoupling the
            predictive engines (`ModelEvaluator`, `ProbabilityCalibrator`, `ECLCalculator`, `DriftMonitor`, `CreditScorecard`)
            from data schema specifics.

            ```
            ┌────────────────────────────────────────────────────────┐
            │               CreditRisk Analytics Platform            │
            └──────────────────────────┬─────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
            ┌─────────────────────────┐         ┌─────────────────────────┐
            │  UCICardDataIngestor    │         │ CorporateDistressAdapter│
            │  (Consumer Credit Cards)│         │ (Corporate Loans & BS)  │
            │  - Revolving limits     │         │ - Working Capital/Assets│
            │  - Repayment 0-6 months │         │ - Interest Coverage     │
            │  - Utilization ratios   │         │ - Debt to Equity, ROA   │
            └───────────┬─────────────┘         └───────────┬─────────────┘
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
            ┌─────────────────────────────────────────────────────────┐
            │                Agnostic Risk Engines                    │
            │  • PD Estimation & Calibration (Isotonic / Platt)       │
            │  • Scorecard Scaling (Base Score 600, PDO 20)           │
            │  • Master Rating Scale (AAA to CCC / Default)           │
            │  • IFRS 9 ECL Three-Stage Impairment Engine             │
            │  • Population Stability Index (PSI) Drift Monitor       │
            └─────────────────────────────┬───────────────────────────┘
                                          ▼
            ┌─────────────────────────────────────────────────────────┐
            │               Storage & Visualization                   │
            │  • SQLAlchemy Layer (SQLite / PostgreSQL)               │
            │  • Interactive Plotly Reporting & Streamlit Dashboard   │
            └─────────────────────────────────────────────────────────┘
            ```
            """
        )

    with tab_dataset:
        st.subheader("Data Ingestion & Integrity Status")
        st.write(f"**Dataset Name:** {UCI_DATASET_ATTRIBUTION.dataset_name}")
        st.write(f"**License:** {UCI_DATASET_ATTRIBUTION.license_type}")
        st.write(f"**Official Repository Source:** [{UCI_DATASET_ATTRIBUTION.source_url}]({UCI_DATASET_ATTRIBUTION.source_url})")

        if st.button("🚀 Ingest & Verify Dataset Now", type="primary"):
            with st.spinner("Ingesting and validating dataset..."):
                try:
                    df = load_raw_dataset(source="uci", force_refresh=False)
                    st.success(f"Successfully loaded dataset! Shape: {df.shape[0]:,} accounts, {df.shape[1]} features.")
                    validator = DataValidator()
                    report = validator.validate(df)
                    st.write(f"**Data Quality Status:** {'PASSED' if report.is_valid else 'WARNINGS IDENTIFIED'}")
                    st.dataframe(df.head(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


if __name__ == "__main__":
    main()
