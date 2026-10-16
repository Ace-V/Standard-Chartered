"""
CreditRisk Analytics — Model Monitoring, Governance & Drift Surveillance
========================================================================

Monitors production credit models against regulatory drift standards:
- Module A: Data Quality (missing-value rates, invalid ranges, unexpected categories, schema drift, duplicates)
- Module B: Feature Drift (Population Stability Index - PSI, CSI, distribution comparisons)
- Module C: Model Performance (ROC-AUC, Brier score, log loss, precision/recall, calibration)
- Module D: Prediction Monitoring (probability distributions, score PSI, risk segment migration)
- Module E: Alerting Engine (configurable thresholds, warning/critical severity, alert history)
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
from src.monitoring.alerting import AlertManager, AlertSeverity
from src.monitoring.data_quality import DataQualityMonitor
from src.monitoring.drift import (
    DriftMonitor,
    calculate_csi_for_features,
    calculate_psi,
    compare_feature_distributions,
    get_psi_methodology_and_limitations,
)
from src.monitoring.performance import PerformanceMonitor
from src.monitoring.prediction import PredictionMonitor
from src.monitoring.simulation import simulate_monitoring_cohort
from src.monitoring.subsystem import ModelMonitoringSubsystem
from src.monitoring.visualizations import (
    plot_data_quality_issues,
    plot_feature_csi_ranking,
    plot_prediction_distribution_shift,
    plot_risk_segment_migration,
)
from src.reporting.charts import plot_psi_bins
from src.risk.segmentation import MasterRatingScale, RiskSegmenter
from src.storage.database import get_db_manager

st.set_page_config(page_title="Model Monitoring & Stability | CreditRisk", page_icon="🛡️", layout="wide")

st.title("🛡️ Credit Risk Model Monitoring & Governance Subsystem")
st.caption("Supervisory tracking of Data Quality, Feature Drift (PSI/CSI), Prediction Stability, and Performance Decay.")

artifact_path = MODELS_DIR / "pd_model_best.joblib"
if not artifact_path.exists():
    st.warning("Please visit '3_📈_PD_Model_Performance' first to train the baseline PD model.")
    st.stop()

model_art = joblib.load(artifact_path)
df_raw = load_raw_dataset(source="uci")

# Partition into Baseline (Train) and Default Out-of-Time Test Cohort
X_train, X_test, y_train, y_test = prepare_train_test_split(df_raw)

p_train = model_art.predict_proba(X_train, calibrate=True)
p_test = model_art.predict_proba(X_test, calibrate=True)

segmenter = RiskSegmenter()
df_baseline = X_train.copy()
df_baseline["calibrated_pd"] = p_train
df_baseline = segmenter.segment(df_baseline, pd_col="calibrated_pd")

# ===========================================================================
# Top Control Panel: Select Monitoring Cohort (Real vs. Explicit Simulation)
# ===========================================================================

st.sidebar.header("🕹️ Monitoring Cohort Selector")
cohort_choice = st.sidebar.selectbox(
    "Choose Monitoring Intake Batch",
    options=[
        "Out-of-Time Test Cohort (Real Split)",
        "[SIMULATION: Stable Production]",
        "[SIMULATION: Macroeconomic Covariate Shift]",
        "[SIMULATION: Upstream Data Pipeline Defect]",
        "[SIMULATION: Recent Unlabelled Originations]",
        "[SIMULATION: Small Sample Pilot (n=20)]",
    ],
    index=0,
    help="Select an empirical out-of-time cohort or a synthetic simulation scenario.",
)

# Initialize cohort based on selection
is_sim = "[SIMULATION:" in cohort_choice
sim_meta = cohort_choice if is_sim else None

if cohort_choice == "Out-of-Time Test Cohort (Real Split)":
    monitoring_df = X_test.copy()
    monitoring_labels = y_test.copy()
    monitoring_probs = p_test
elif cohort_choice == "[SIMULATION: Stable Production]":
    sim = simulate_monitoring_cohort(df_raw, scenario_type="stable")
    monitoring_df = sim.data
    monitoring_labels = sim.labels
    monitoring_probs = model_art.predict_proba(monitoring_df, calibrate=True)
elif cohort_choice == "[SIMULATION: Macroeconomic Covariate Shift]":
    sim = simulate_monitoring_cohort(df_raw, scenario_type="covariate_shift")
    monitoring_df = sim.data
    monitoring_labels = sim.labels
    monitoring_probs = model_art.predict_proba(monitoring_df, calibrate=True)
elif cohort_choice == "[SIMULATION: Upstream Data Pipeline Defect]":
    sim = simulate_monitoring_cohort(df_raw, scenario_type="data_quality_defect")
    monitoring_df = sim.data
    monitoring_labels = sim.labels
    # Predict with missing handling / fillna for simulation demonstration
    clean_copy = monitoring_df.fillna(monitoring_df.median(numeric_only=True))
    monitoring_probs = model_art.predict_proba(clean_copy, calibrate=True)
elif cohort_choice == "[SIMULATION: Recent Unlabelled Originations]":
    sim = simulate_monitoring_cohort(df_raw, scenario_type="unlabelled_recent")
    monitoring_df = sim.data
    monitoring_labels = None  # Explicitly unlabelled!
    monitoring_probs = model_art.predict_proba(monitoring_df, calibrate=True)
else:  # Small sample pilot
    sim = simulate_monitoring_cohort(df_raw, scenario_type="small_sample")
    monitoring_df = sim.data
    monitoring_labels = sim.labels
    monitoring_probs = model_art.predict_proba(monitoring_df, calibrate=True)

# Run full surveillance cycle
subsystem = ModelMonitoringSubsystem()
monitoring_report = subsystem.run_surveillance_cycle(
    df_baseline=df_baseline,
    df_monitoring=monitoring_df,
    baseline_probs=p_train,
    monitoring_probs=monitoring_probs,
    monitoring_labels=monitoring_labels,
    baseline_metrics=model_art.test_metrics,
    is_simulation=is_sim,
    simulation_name=sim_meta,
)

# Mandatory Synthetic Simulation Warning Banner
if is_sim:
    st.warning(
        f"⚠️ **PROVABLY SYNTHETIC SIMULATION ACTIVE:** You are evaluating `{cohort_choice}`.\n\n"
        "This dataset partition was generated synthetically to test and demonstrate model surveillance behavior. "
        "It does **NOT** represent real production metrics."
    )

# Executive Governance Summary Bar
gov_col1, gov_col2, gov_col3, gov_col4, gov_col5 = st.columns(5)
with gov_col1:
    status_icon = "🟢 GREEN" if monitoring_report.overall_governance_status == "GREEN" else (
        "🟡 AMBER" if monitoring_report.overall_governance_status == "AMBER" else "🔴 RED"
    )
    st.metric("Governance Status", status_icon)
with gov_col2:
    dq_icon = "🟢 OK" if monitoring_report.data_quality_report.overall_status == "HEALTHY" else (
        "🟡 WARN" if monitoring_report.data_quality_report.overall_status == "WARNING" else "🔴 CRIT"
    )
    st.metric("Data Quality", dq_icon, f"{monitoring_report.data_quality_report.total_records:,} rows")
with gov_col3:
    st.metric("Score PSI", f"{monitoring_report.prediction_report.pd_psi_result.psi_value:.4f}", monitoring_report.prediction_report.pd_psi_result.traffic_light)
with gov_col4:
    st.metric("Mean PD Shift", f"{monitoring_report.prediction_report.mean_pd_shift_pct:+.1f}%")
with gov_col5:
    lbl_stat = "Available" if monitoring_report.performance_report.labels_available else "Dormant (No Labels)"
    st.metric("Performance Audit", lbl_stat, f"{monitoring_report.performance_report.status}")

# Regulatory Guidance Callout
st.info(
    "**Supervisory Benchmark Standards (OCC / EBA / Basel):**\n"
    "- **PSI < 0.10 (Green):** Stable population. No model intervention required.\n"
    "- **0.10 <= PSI < 0.25 (Amber):** Moderate shift detected. Enhanced portfolio monitoring required.\n"
    "- **PSI >= 0.25 (Red):** Significant shift. Model discrimination compromised; formal recalibration required.\n"
    "- **Governance Rule:** Feature drift (covariate shift) must be monitored continuously; model performance decay can **only** be evaluated once 12-month default maturity labels arrive."
)

st.markdown("---")

# ===========================================================================
# Surveillance Modules Tabs
# ===========================================================================

tab_dq, tab_drift, tab_pred, tab_perf, tab_alerts, tab_audit = st.tabs([
    "📋 Data Quality (Module A)",
    "🌊 Feature Drift & CSI (Module B)",
    "📈 Prediction Monitoring (Module D)",
    "🎯 Model Performance (Module C)",
    "🚨 Alerting Engine (Module E)",
    "📝 Compliance Audit Log",
])

# ---------------------------------------------------------------------------
# Tab 1: Module A — Data Quality
# ---------------------------------------------------------------------------
with tab_dq:
    st.subheader("Data Quality Surveillance (Module A)")
    dq = monitoring_report.data_quality_report

    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("Total Batch Records", f"{dq.total_records:,}")
    dq2.metric("Missing Columns", f"{len(dq.missing_columns)}")
    dq3.metric("Duplicate IDs", f"{dq.duplicate_ids_count}")
    dq4.metric("Duplicate Rows", f"{dq.duplicate_rows_count}")

    fig_dq = plot_data_quality_issues(dq)
    st.plotly_chart(fig_dq, use_container_width=True)

    c_dq_t1, c_dq_t2 = st.columns(2)
    with c_dq_t1:
        st.markdown("#### Missing-Value Rates per Column")
        st.dataframe(dq.missingness_summary(), use_container_width=True)
    with c_dq_t2:
        st.markdown("#### Schema Drift & Integrity Issues")
        st.dataframe(dq.schema_audit_summary(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Module B — Feature Drift & CSI
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Population & Characteristic Stability Monitoring (Module B)")
    st.caption("Tracking Population Stability Index (PSI) and Kolmogorov-Smirnov distribution shifts across predictive factors.")

    fig_csi = plot_feature_csi_ranking(monitoring_report.csi_summary_table)
    st.plotly_chart(fig_csi, use_container_width=True)

    st.markdown("#### Feature-Level Characteristic Stability Index (CSI) Summary")
    def highlight_csi(val):
        colors = {
            "GREEN": "background-color: #DCFCE7; color: #166534; font-weight: bold",
            "AMBER": "background-color: #FEF3C7; color: #92400E; font-weight: bold",
            "RED": "background-color: #FEE2E2; color: #991B1B; font-weight: bold",
        }
        return colors.get(val, "")

    if not monitoring_report.csi_summary_table.empty:
        styled_csi = monitoring_report.csi_summary_table.style.map(highlight_csi, subset=["Traffic_Light"])
        st.dataframe(styled_csi, use_container_width=True)

    with st.expander("📚 PSI Binning Methodology & Mathematical Limitations"):
        docs = get_psi_methodology_and_limitations()
        st.markdown(f"**Formula:** `{docs['formula']}`")
        st.markdown(f"**Binning Approach:** {docs['binning_methodology']['type']} ({docs['binning_methodology']['rationale']})")
        st.markdown("**Critical Limitations:**")
        for lim in docs["critical_limitations"]:
            st.markdown(f"- {lim}")

# ---------------------------------------------------------------------------
# Tab 3: Module D — Prediction Monitoring
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader("Predicted Probability & Risk Segment Monitoring (Module D)")
    st.caption("Surveillance of model outputs and risk tier migrations independent of ground-truth labels.")

    pred_rep = monitoring_report.prediction_report

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fig_kde = plot_prediction_distribution_shift(p_train, monitoring_probs)
        st.plotly_chart(fig_kde, use_container_width=True)
    with p_col2:
        fig_mig = plot_risk_segment_migration(pred_rep.risk_segment_migration)
        st.plotly_chart(fig_mig, use_container_width=True)

    col_ps1, col_ps2 = st.columns(2)
    with col_ps1:
        st.markdown("#### Probability Distribution Statistics Comparison")
        st.dataframe(pred_rep.summary_table(), use_container_width=True)
    with col_ps2:
        st.markdown("#### Risk Rating Grade Migration")
        st.dataframe(pred_rep.risk_segment_migration, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Module C — Model Performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Model Performance Surveillance & Degradation Tracking (Module C)")

    perf_rep = monitoring_report.performance_report

    if not perf_rep.labels_available:
        st.info(
            "**ℹ️ Performance Monitoring Dormant:**\n\n"
            f"{perf_rep.explanation}\n\n"
            "*Per supervisory risk principles, performance metrics cannot be manufactured or claimed as degraded without actual ground-truth defaults.*"
        )
    else:
        if perf_rep.status == "INSUFFICIENT_SAMPLE":
            st.warning(f"⚠️ {perf_rep.explanation}")
        else:
            pf1, pf2, pf3, pf4 = st.columns(4)
            pf1.metric("ROC-AUC", f"{perf_rep.monitoring_metrics.roc_auc:.4f}", f"{perf_rep.auc_delta:+.4f}")
            pf2.metric("Gini (Somers' D)", f"{perf_rep.monitoring_metrics.gini:.4f}", f"-{perf_rep.gini_decay_pct:.1f}% decay")
            pf3.metric("KS Statistic", f"{perf_rep.monitoring_metrics.ks_statistic:.4f}", f"-{perf_rep.ks_decay_pct:.1f}% decay")
            pf4.metric("Brier Score", f"{perf_rep.monitoring_metrics.brier_score:.4f}", f"{perf_rep.brier_delta:+.4f}")

            st.markdown("#### Performance Metric Ledger (Mature Outcome Cohort)")
            st.dataframe(perf_rep.to_dataframe(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Module E — Alerting Engine
# ---------------------------------------------------------------------------
with tab_alerts:
    st.subheader("Monitoring Alerting Engine & Incident Management (Module E)")
    st.caption("Supervisory warnings, critical thresholds, and recommended remediation actions.")

    active_alerts = monitoring_report.active_alerts

    if active_alerts:
        for alert in active_alerts:
            sev_badge = "🔴 CRITICAL" if alert.severity == AlertSeverity.CRITICAL else (
                "🟡 WARNING" if alert.severity == AlertSeverity.WARNING else "ℹ️ INFO"
            )
            with st.container():
                st.markdown(f"**{sev_badge} — {alert.metric_name}** (`{alert.alert_id}`)")
                st.markdown(f"- **Message:** {alert.message}")
                st.markdown(f"- **Recommended Governance Action:** {alert.recommended_action}")
                if st.button(f"Mark Alert Acknowledged: {alert.alert_id}", key=f"btn_ack_{alert.alert_id}"):
                    subsystem.alert_manager.acknowledge_alert(alert.alert_id)
                    st.success(f"Alert {alert.alert_id} acknowledged.")
                st.divider()
    else:
        st.success("🟢 No active supervisory alerts. All parameters are within regulatory limits.")

    st.markdown("#### Complete Monitoring Alert History")
    st.dataframe(subsystem.alert_manager.history_dataframe(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 6: Compliance Audit Log
# ---------------------------------------------------------------------------
with tab_audit:
    st.subheader("Model Governance & Regulatory Compliance Audit Trail")
    db = get_db_manager()

    if st.button("📝 Record Current Surveillance Cycle to Database"):
        db.log_monitoring_event(
            metric_type="SURVEILLANCE_CYCLE",
            feature_or_score="full_monitoring_report",
            metric_value=monitoring_report.prediction_report.pd_psi_result.psi_value,
            traffic_light=monitoring_report.overall_governance_status,
            interpretation=monitoring_report.executive_summary[:200],
        )
        st.success("Successfully recorded monitoring audit log!")

    audit_logs = db.get_monitoring_audit_logs(limit=25)
    if not audit_logs.empty:
        st.dataframe(audit_logs, use_container_width=True)
    else:
        st.info("No monitoring audit events recorded in database yet.")
