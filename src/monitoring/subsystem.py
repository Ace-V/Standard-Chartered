"""
CreditRisk Analytics — Model Monitoring Subsystem Orchestrator
==============================================================

Glues together Modules A through E into a unified production surveillance engine:
- Module A: Data Quality Monitor (missing rates, range violations, category drift, schema checks)
- Module B: Feature Drift Monitor (PSI, CSI, KS distribution tests, covariate shift)
- Module C: Model Performance Monitor (ROC-AUC, Brier score, log loss, precision, recall, ECE)
- Module D: Prediction Monitor (predicted probability distribution, risk segment migration)
- Module E: Alerting Engine (configurable thresholds, human-readable explanations, alert history)

CRITICAL GOVERNANCE PRINCIPLES:
1. Feature drift is strictly distinguished from model performance deterioration.
2. Performance decay is NEVER claimed without verified ground-truth default outcomes.
3. Unlabelled intake is handled gracefully with clear dormancy disclosures.
4. All simulation runs are visibly tagged with synthetic disclosures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.evaluate import EvaluationMetrics
from src.monitoring.alerting import Alert, AlertCategory, AlertManager, AlertSeverity
from src.monitoring.data_quality import DataQualityMonitor, DataQualityReport
from src.monitoring.drift import (
    DistributionComparisonResult,
    DriftMonitor,
    PSIResult,
    calculate_csi_for_features,
    calculate_psi,
    compare_feature_distributions,
    get_psi_methodology_and_limitations,
)
from src.monitoring.performance import PerformanceDriftReport, PerformanceMonitor
from src.monitoring.prediction import PredictionMonitor, PredictionMonitoringReport


@dataclass
class ComprehensiveMonitoringReport:
    """Full enterprise monitoring cycle audit report."""
    cycle_timestamp: str
    overall_governance_status: str  # "GREEN", "AMBER", "RED"
    data_quality_report: DataQualityReport
    feature_drift_results: Dict[str, DistributionComparisonResult]
    csi_summary_table: pd.DataFrame
    prediction_report: PredictionMonitoringReport
    performance_report: PerformanceDriftReport
    active_alerts: List[Alert]
    executive_summary: str
    is_simulation: bool = False
    simulation_metadata: Optional[str] = None


class ModelMonitoringSubsystem:
    """
    Enterprise credit risk model surveillance subsystem.
    """

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        data_quality_monitor: Optional[DataQualityMonitor] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        prediction_monitor: Optional[PredictionMonitor] = None,
    ):
        self.alert_manager = alert_manager or AlertManager()
        self.dq_monitor = data_quality_monitor or DataQualityMonitor()
        self.perf_monitor = performance_monitor or PerformanceMonitor()
        self.pred_monitor = prediction_monitor or PredictionMonitor()
        self.drift_monitor = DriftMonitor()

    def run_surveillance_cycle(
        self,
        df_baseline: pd.DataFrame,
        df_monitoring: pd.DataFrame,
        baseline_probs: np.ndarray,
        monitoring_probs: np.ndarray,
        monitoring_labels: Optional[np.ndarray | pd.Series] = None,
        baseline_metrics: Optional[EvaluationMetrics] = None,
        monitored_features: Optional[List[str]] = None,
        expected_columns: Optional[List[str]] = None,
        range_rules: Optional[Dict[str, Tuple[float, float]]] = None,
        category_rules: Optional[Dict[str, Any]] = None,
        is_simulation: bool = False,
        simulation_name: Optional[str] = None,
    ) -> ComprehensiveMonitoringReport:
        """
        Execute an integrated surveillance audit cycle across all 5 modules.
        """
        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

        # ------------------------------------------------------------------
        # Module A: Data Quality Audit
        # ------------------------------------------------------------------
        dq_report = self.dq_monitor.audit_batch(
            df=df_monitoring,
            expected_columns=expected_columns or list(df_baseline.columns),
            range_rules=range_rules,
            category_rules=category_rules,
        )

        # Trigger Data Quality alerts if critical issues found
        for alert_msg in dq_report.alerts:
            sev = AlertSeverity.CRITICAL if "CRITICAL" in alert_msg else AlertSeverity.WARNING
            self.alert_manager.record_custom_alert(
                category=AlertCategory.DATA_QUALITY,
                severity=sev,
                metric_name="Data Quality Check",
                current_value=1.0,
                threshold_warning=1.0,
                threshold_critical=1.0,
                message=alert_msg,
                recommended_action="Inspect upstream data ingestion pipelines and ETL schema mappings.",
            )

        # ------------------------------------------------------------------
        # Module B: Feature Drift & Distribution Shift
        # ------------------------------------------------------------------
        features_to_check = monitored_features or [
            c for c in df_baseline.columns
            if c in df_monitoring.columns and np.issubdtype(df_baseline[c].dtype, np.number)
        ][:8]

        feat_drift_results: Dict[str, DistributionComparisonResult] = {}
        for feat in features_to_check:
            comp_res = compare_feature_distributions(
                expected=df_baseline[feat],
                actual=df_monitoring[feat],
                feature_name=feat,
                num_bins=10,
            )
            feat_drift_results[feat] = comp_res

            # Trigger alert if PSI crossed threshold
            self.alert_manager.evaluate_metric(
                rule_name="csi_feature",
                current_value=comp_res.psi_result.psi_value,
                metric_display_name=f"CSI ({feat})",
                context_details=f"KS stat={comp_res.ks_statistic:.4f}, mean shift={comp_res.mean_shift_pct:+.1f}%.",
            )

        csi_df = calculate_csi_for_features(
            df_expected=df_baseline,
            df_actual=df_monitoring,
            features=features_to_check,
            num_bins=10,
        )

        # ------------------------------------------------------------------
        # Module C: Model Performance Decay
        # ------------------------------------------------------------------
        perf_report = self.perf_monitor.compare_performance(
            baseline_metrics=baseline_metrics,
            y_true_monitoring=monitoring_labels,
            y_prob_monitoring=monitoring_probs,
        )

        if perf_report.labels_available and perf_report.status in ("WARNING", "CRITICAL"):
            self.alert_manager.evaluate_metric(
                rule_name="gini_decay_pct",
                current_value=perf_report.gini_decay_pct,
                metric_display_name="Gini Decay Pct",
                context_details=f"AUC delta={perf_report.auc_delta:+.4f}, KS delta={perf_report.ks_delta:+.4f}.",
            )

        # ------------------------------------------------------------------
        # Module D: Prediction Output & Risk Segment Migration
        # ------------------------------------------------------------------
        pred_report = self.pred_monitor.monitor_predictions(
            baseline_probs=baseline_probs,
            monitoring_probs=monitoring_probs,
        )

        # Evaluate prediction alerts
        self.alert_manager.evaluate_metric(
            rule_name="psi_score",
            current_value=pred_report.pd_psi_result.psi_value,
            metric_display_name="Score PSI",
            context_details="Predicted probability distribution shift.",
        )
        self.alert_manager.evaluate_metric(
            rule_name="mean_pd_shift_pct",
            current_value=abs(pred_report.mean_pd_shift_pct),
            metric_display_name="Mean PD Shift Pct",
            context_details=f"Baseline mean={pred_report.baseline_stats.mean:.4f}, Monitoring mean={pred_report.monitoring_stats.mean:.4f}.",
        )

        # ------------------------------------------------------------------
        # Module E: Overall Status & Executive Synthesis
        # ------------------------------------------------------------------
        active_alerts = self.alert_manager.get_active_alerts(unacknowledged_only=True)
        critical_count = sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL)
        warning_count = sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING)

        if critical_count > 0:
            overall_status = "RED"
        elif warning_count > 0:
            overall_status = "AMBER"
        else:
            overall_status = "GREEN"

        # Formulate executive governance narrative
        exec_lines = [
            f"Surveillance cycle completed at {now_str} (Governance Status: {overall_status}).",
            f"- Data Quality: {dq_report.overall_status} (Records: {dq_report.total_records:,}, Duplicate IDs: {dq_report.duplicate_ids_count}).",
            f"- Feature Drift: Evaluated {len(features_to_check)} key predictive risk factors.",
            f"- Prediction Stability: Score PSI = {pred_report.pd_psi_result.psi_value:.4f} ({pred_report.status}), Mean PD shift = {pred_report.mean_pd_shift_pct:+.1f}%.",
        ]

        if perf_report.labels_available:
            exec_lines.append(
                f"- Model Performance: {perf_report.status} over {perf_report.sample_size:,} labelled accounts "
                f"(Gini decay: {perf_report.gini_decay_pct:.1f}%, AUC delta: {perf_report.auc_delta:+.4f})."
            )
        else:
            exec_lines.append(
                "- Model Performance: DORMANT (Labels unavailable). Performance decay is NOT claimed without ground-truth mature outcomes."
            )

        exec_lines.append(f"- Active Supervisory Alerts: {len(active_alerts)} ({critical_count} Critical, {warning_count} Warning).")

        return ComprehensiveMonitoringReport(
            cycle_timestamp=now_str,
            overall_governance_status=overall_status,
            data_quality_report=dq_report,
            feature_drift_results=feat_drift_results,
            csi_summary_table=csi_df,
            prediction_report=pred_report,
            performance_report=perf_report,
            active_alerts=active_alerts,
            executive_summary="\n".join(exec_lines),
            is_simulation=is_simulation,
            simulation_metadata=simulation_name,
        )
