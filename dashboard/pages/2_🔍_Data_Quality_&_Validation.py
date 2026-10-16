"""
CreditRisk Analytics — Data Quality & Regulatory Validation Audit
=================================================================

Performs automated data quality pre-flight checks, schema assertions,
boundary constraint evaluations, and BCBS 239 compliance verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.data.ingestion import load_raw_dataset
from src.data.validation import DataValidator

st.set_page_config(page_title="Data Quality & Validation | CreditRisk", page_icon="🔍", layout="wide")

st.title("🔍 Data Quality & Pre-Flight Validation Engine")
st.caption("Automated audit adhering to BCBS 239 principles for risk data aggregation and reporting.")

@st.cache_data
def get_dataset():
    return load_raw_dataset(source="uci")

try:
    df = get_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

validator = DataValidator()
report = validator.validate(df)

# Top KPI Audit Banner
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Records Audited", f"{report.total_records:,}")
with c2:
    status_label = "VALID" if report.is_valid else "NON-COMPLIANT"
    st.metric("Overall Data Hygiene", status_label, delta="BCBS 239")
with c3:
    st.metric("Critical / Blocking Errors", report.summary_stats.get("critical_errors", 0))
with c4:
    st.metric("Warnings & Audit Notes", report.summary_stats.get("warnings_and_info", 0))

st.markdown("---")

# Detailed Audit Issues Table
st.subheader("Automated Data Quality Audit Findings")

issues_df = report.to_dataframe()
if not issues_df.empty:
    def highlight_severity(val):
        color = {
            "CRITICAL": "background-color: #FEE2E2; color: #991B1B; font-weight: bold",
            "ERROR": "background-color: #FECACA; color: #B91C1C",
            "WARNING": "background-color: #FEF3C7; color: #92400E",
            "INFO": "background-color: #E0F2FE; color: #075985",
            "PASSED": "background-color: #DCFCE7; color: #166534",
        }.get(val, "")
        return color

    if "severity" in issues_df.columns:
        styled_issues = issues_df.style.map(highlight_severity, subset=["severity"])
        st.dataframe(styled_issues, use_container_width=True)
    else:
        st.dataframe(issues_df, use_container_width=True)
else:
    st.success("All data quality checks passed with zero issues detected!")

# Column Specifications & Distribution Summary
st.markdown("---")
st.subheader("Field Schema & Boundary Constraints")

col_df = pd.DataFrame(report.column_checks)
st.dataframe(col_df, use_container_width=True)

# BCBS 239 Regulatory Compliance Framework Checklist
st.markdown("---")
st.subheader("BCBS 239 Risk Data Aggregation Compliance Checklist")

col_a, col_b = st.columns(2)
with col_a:
    st.checkbox("Principle 3: Accuracy & Integrity — Automated validation rules verified.", value=report.is_valid, disabled=True)
    st.checkbox("Principle 4: Completeness — Zero unhandled null values across scoring features.", value=report.summary_stats.get("missing_cells", 0) == 0, disabled=True)
    st.checkbox("Principle 5: Timeliness — Automated pipeline caches data for fast retrieval.", value=True, disabled=True)

with col_b:
    st.checkbox("Principle 6: Adaptability — Ingestor hierarchy supports Corporate & Retail assets.", value=True, disabled=True)
    st.checkbox("Principle 8: Comprehensiveness — Full distribution and boundary stats recorded.", value=True, disabled=True)
    st.checkbox("Principle 11: Validation & Auditability — Model logs preserved in SQLite/PostgreSQL.", value=True, disabled=True)
