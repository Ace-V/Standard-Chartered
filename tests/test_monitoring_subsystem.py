"""
CreditRisk Analytics — Unit & Integration Tests for Monitoring Subsystem
========================================================================

Comprehensive tests for:
- Module A: Data Quality (missing values, invalid ranges, unexpected categories, schema drift, duplicates)
- Module B: Feature Drift (PSI, CSI, KS tests, distribution comparison, stable vs shifted)
- Module C: Model Performance (ROC-AUC, Brier score, log loss, precision/recall, calibration,
                               handling missing labels, handling insufficient sample sizes)
- Module D: Prediction Monitoring (probability distributions, score PSI, risk segment migration)
- Module E: Alerting Engine (configurable thresholds, warning/critical severity, alert history)
- Simulation Engine (mandatory synthetic disclaimers, labelled scenarios)
- Subsystem Orchestration (integrated surveillance cycle)
- Visualizations (Plotly chart generation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.models.evaluate import EvaluationMetrics
from src.monitoring.alerting import (
    Alert,
    AlertCategory,
    AlertManager,
    AlertRule,
    AlertSeverity,
    get_default_alert_rules,
)
from src.monitoring.data_quality import (
    ColumnQualityMetric,
    DataQualityMonitor,
    DataQualityReport,
)
from src.monitoring.drift import (
    DistributionComparisonResult,
    calculate_csi_for_features,
    calculate_psi,
    compare_feature_distributions,
    get_psi_methodology_and_limitations,
)
from src.monitoring.performance import PerformanceDriftReport, PerformanceMonitor
from src.monitoring.prediction import (
    PredictionMonitor,
    PredictionMonitoringReport,
    ProbabilityDistributionStats,
)
from src.monitoring.simulation import SimulatedCohort, simulate_monitoring_cohort
from src.monitoring.subsystem import (
    ComprehensiveMonitoringReport,
    ModelMonitoringSubsystem,
)
from src.monitoring.visualizations import (
    plot_data_quality_issues,
    plot_feature_csi_ranking,
    plot_prediction_distribution_shift,
    plot_risk_segment_migration,
)


# ===========================================================================
# 1. Module A: Data Quality Tests
# ===========================================================================

class TestDataQualityMonitor:
    """Tests for missing-value rates, invalid ranges, unexpected categories, schema drift, and duplicates."""

    def test_clean_data_passes_with_healthy_status(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [25, 35, 45, 55, 65],
            "education": [1, 2, 2, 1, 3],
            "limit_bal": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0],
        })
        monitor = DataQualityMonitor()
        report = monitor.audit_batch(
            df=df,
            expected_columns=["id", "age", "education", "limit_bal"],
            range_rules={"age": (18, 100), "limit_bal": (1000, 1_000_000)},
            category_rules={"education": [1, 2, 3, 4]},
            id_column="id",
        )
        assert report.overall_status == "HEALTHY"
        assert len(report.missing_columns) == 0
        assert report.duplicate_ids_count == 0

    def test_missing_values_triggers_warning_and_critical(self):
        # 100 rows: 3% missing in col_warn, 8% missing in col_crit
        n = 100
        df = pd.DataFrame({
            "id": list(range(n)),
            "col_warn": [np.nan if i < 3 else 10.0 for i in range(n)],
            "col_crit": [np.nan if i < 8 else 20.0 for i in range(n)],
        })
        monitor = DataQualityMonitor(warning_missing_pct=1.0, critical_missing_pct=5.0)
        report = monitor.audit_batch(df=df, id_column="id")

        assert report.overall_status == "CRITICAL"
        assert report.column_metrics["col_warn"].status == "WARNING"
        assert report.column_metrics["col_crit"].status == "CRITICAL"

        df_missing = report.missingness_summary()
        assert len(df_missing) == 2
        assert df_missing.iloc[0]["Column"] == "col_crit"

    def test_invalid_numerical_ranges_flagged(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "age": [25, -99, 45, 150],  # -99 and 150 are invalid
        })
        monitor = DataQualityMonitor()
        report = monitor.audit_batch(df=df, range_rules={"age": (18, 100)})

        assert report.column_metrics["age"].invalid_range_count == 2
        assert report.column_metrics["age"].invalid_range_pct == 50.0
        assert report.overall_status == "CRITICAL"

    def test_unexpected_categories_flagged(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "education": [1, 2, 999, 2],  # 999 is unexpected
        })
        monitor = DataQualityMonitor()
        report = monitor.audit_batch(df=df, category_rules={"education": [1, 2, 3, 4]})

        assert report.column_metrics["education"].unexpected_categories_count == 1
        assert "unexpected categories" in report.column_metrics["education"].issues[0].lower()

    def test_schema_drift_missing_and_unexpected_columns(self):
        df = pd.DataFrame({
            "id": [1, 2],
            "extra_col": [10, 20],
        })
        monitor = DataQualityMonitor()
        report = monitor.audit_batch(
            df=df,
            expected_columns=["id", "required_col_1", "required_col_2"],
        )
        assert "required_col_1" in report.missing_columns
        assert "required_col_2" in report.missing_columns
        assert "extra_col" in report.unexpected_columns
        assert report.overall_status == "CRITICAL"

    def test_duplicate_primary_keys_and_rows(self):
        df = pd.DataFrame({
            "id": [1, 1, 2],  # duplicate id 1
            "val": [10, 10, 20],  # row 0 and 1 are complete duplicates
        })
        monitor = DataQualityMonitor()
        report = monitor.audit_batch(df=df, id_column="id")

        assert report.duplicate_ids_count == 1
        assert report.duplicate_rows_count == 1
        assert report.overall_status == "CRITICAL"

    def test_empty_dataframe_handled_gracefully(self):
        df = pd.DataFrame()
        monitor = DataQualityMonitor()
        report = monitor.audit_batch(df=df)
        assert report.overall_status == "CRITICAL"
        assert report.total_records == 0


# ===========================================================================
# 2. Module B: Feature Drift Tests
# ===========================================================================

class TestFeatureDriftMonitor:
    """Tests for PSI, CSI, distribution comparisons, and methodology disclosures."""

    def test_stable_feature_distribution_produces_green_psi(self):
        rng = np.random.RandomState(42)
        base = rng.normal(50000, 10000, size=3000)
        mon = rng.normal(50000, 10000, size=3000)

        comp = compare_feature_distributions(base, mon, feature_name="limit_bal")
        assert comp.psi_result.psi_value < 0.10
        assert comp.psi_result.traffic_light == "GREEN"
        assert comp.drift_detected is False

    def test_shifted_feature_distribution_produces_red_psi(self):
        rng = np.random.RandomState(42)
        base = rng.normal(50000, 10000, size=3000)
        # Shift distribution significantly downward (severe credit contraction)
        mon = rng.normal(25000, 5000, size=3000)

        comp = compare_feature_distributions(base, mon, feature_name="limit_bal")
        assert comp.psi_result.psi_value >= 0.25
        assert comp.psi_result.traffic_light == "RED"
        assert comp.drift_detected is True
        assert comp.ks_statistic > 0.30

    def test_csi_ranking_identifies_most_drifted_feature(self):
        rng = np.random.RandomState(42)
        df_base = pd.DataFrame({
            "stable_feat": rng.normal(100, 10, size=2000),
            "drifted_feat": rng.normal(50, 5, size=2000),
        })
        df_mon = pd.DataFrame({
            "stable_feat": rng.normal(100, 10, size=2000),
            "drifted_feat": rng.normal(80, 15, size=2000),  # shifted
        })

        csi_df = calculate_csi_for_features(df_base, df_mon, features=["stable_feat", "drifted_feat"])
        assert csi_df.iloc[0]["Feature"] == "drifted_feat"
        assert csi_df.iloc[0]["CSI_Value"] > csi_df.iloc[1]["CSI_Value"]

    def test_psi_methodology_and_limitations_completeness(self):
        docs = get_psi_methodology_and_limitations()
        assert "PSI" in docs["formula"]
        assert "GREEN" in docs["regulatory_thresholds"]
        assert "AMBER" in docs["regulatory_thresholds"]
        assert "RED" in docs["regulatory_thresholds"]
        assert any("Covariate Shift vs. Concept Drift" in lim for lim in docs["critical_limitations"])
        assert any("1D Marginal Invariance" in lim for lim in docs["critical_limitations"])


# ===========================================================================
# 3. Module C: Model Performance Tests (Labels Available vs. Unavailable)
# ===========================================================================

class TestModelPerformanceMonitor:
    """Tests for performance evaluation, graceful handling of missing labels and small samples."""

    @pytest.fixture
    def baseline_metrics(self) -> EvaluationMetrics:
        return EvaluationMetrics(
            roc_auc=0.78,
            gini=0.56,
            ks_statistic=0.42,
            ks_threshold=0.22,
            brier_score=0.13,
            log_loss_val=0.45,
            pr_auc=0.52,
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1=0.72,
            confusion_mat={"tp": 100, "fp": 20, "fn": 30, "tn": 350},
        )

    def test_unavailable_labels_handled_gracefully(self, baseline_metrics):
        """Rule: Never claim performance deterioration without ground-truth labels."""
        monitor = PerformanceMonitor()
        y_prob = np.linspace(0.05, 0.85, 500)

        # Labels are None
        report = monitor.compare_performance(
            baseline_metrics=baseline_metrics,
            y_true_monitoring=None,
            y_prob_monitoring=y_prob,
        )

        assert report.status == "UNAVAILABLE_NO_LABELS"
        assert report.labels_available is False
        assert "Outcome labels are not available" in report.explanation
        assert "dormant" in report.alerts[0].lower()
        assert report.monitoring_metrics is None

        # Verify tabular export presents explanation
        df = report.to_dataframe()
        assert "Labels Available" in df.columns
        assert bool(df.iloc[0]["Labels Available"]) is False

    def test_all_null_labels_handled_gracefully(self, baseline_metrics):
        monitor = PerformanceMonitor()
        y_prob = np.linspace(0.05, 0.85, 200)
        y_true = np.array([np.nan] * 200)

        report = monitor.compare_performance(
            baseline_metrics=baseline_metrics,
            y_true_monitoring=y_true,
            y_prob_monitoring=y_prob,
        )
        assert report.status == "UNAVAILABLE_NO_LABELS"
        assert report.labels_available is False

    def test_insufficient_sample_size_flagged(self, baseline_metrics):
        monitor = PerformanceMonitor()
        # Only 15 accounts with 1 default
        y_true = np.array([0] * 14 + [1])
        y_prob = np.linspace(0.1, 0.4, 15)

        report = monitor.compare_performance(
            baseline_metrics=baseline_metrics,
            y_true_monitoring=y_true,
            y_prob_monitoring=y_prob,
        )
        assert report.status == "INSUFFICIENT_SAMPLE"
        assert report.labels_available is True
        assert "Sample size too small" in report.explanation

    def test_mature_labelled_cohort_evaluates_all_metrics(self, baseline_metrics):
        rng = np.random.RandomState(42)
        n = 500
        y_prob = rng.beta(2, 5, size=n)
        y_true = rng.binomial(1, y_prob).astype(float)

        monitor = PerformanceMonitor()
        report = monitor.compare_performance(
            baseline_metrics=baseline_metrics,
            y_true_monitoring=y_true,
            y_prob_monitoring=y_prob,
        )

        assert report.labels_available is True
        assert report.monitoring_metrics is not None
        assert report.monitoring_metrics.roc_auc > 0.50
        assert report.brier_delta is not None
        assert report.precision >= 0.0
        assert report.recall >= 0.0
        assert report.ece >= 0.0

        df = report.to_dataframe()
        assert "ROC-AUC" in df["Metric"].values
        assert "Brier Score" in df["Metric"].values
        assert "Log Loss" in df["Metric"].values
        assert "Precision" in df["Metric"].values
        assert "Recall" in df["Metric"].values
        assert "ECE (Calibration)" in df["Metric"].values


# ===========================================================================
# 4. Module D: Prediction Monitoring Tests
# ===========================================================================

class TestPredictionMonitor:
    """Tests for output probability statistics and rating segment migration."""

    def test_stable_predictions_produce_green_status(self):
        rng = np.random.RandomState(42)
        b_p = rng.beta(2, 6, size=2000)
        m_p = rng.beta(2, 6, size=2000)

        monitor = PredictionMonitor()
        report = monitor.monitor_predictions(b_p, m_p)

        assert report.status == "GREEN"
        assert report.pd_psi_result.psi_value < 0.10
        assert abs(report.mean_pd_shift_pct) < 10.0

    def test_severe_prediction_shift_triggers_red_status(self):
        rng = np.random.RandomState(42)
        b_p = rng.beta(2, 8, size=2000)  # low mean PD
        m_p = rng.beta(5, 3, size=2000)  # much higher mean PD (severe worsening)

        monitor = PredictionMonitor()
        report = monitor.monitor_predictions(b_p, m_p)

        assert report.status == "RED"
        assert report.pd_psi_result.psi_value >= 0.25
        assert len(report.alerts) > 0

    def test_risk_segment_migration_table_structure(self):
        rng = np.random.RandomState(42)
        b_p = rng.beta(2, 6, size=1000)
        m_p = rng.beta(3, 5, size=1000)

        monitor = PredictionMonitor()
        report = monitor.monitor_predictions(b_p, m_p)

        df_mig = report.risk_segment_migration
        assert "Rating Grade" in df_mig.columns
        assert "Baseline Share (%)" in df_mig.columns
        assert "Monitoring Share (%)" in df_mig.columns
        assert "Net Shift (%)" in df_mig.columns
        assert len(df_mig) >= 7  # AAA to CCC


# ===========================================================================
# 5. Module E: Alerting Engine Tests
# ===========================================================================

class TestAlertingEngine:
    """Tests for alert threshold evaluation, severity states, and history ledger."""

    def test_alert_thresholds_warning_and_critical(self):
        manager = AlertManager()

        # psi_score rule: warning at 0.10, critical at 0.25
        a_none = manager.evaluate_metric("psi_score", 0.05)
        assert a_none is None

        a_warn = manager.evaluate_metric("psi_score", 0.15)
        assert a_warn is not None
        assert a_warn.severity == AlertSeverity.WARNING

        a_crit = manager.evaluate_metric("psi_score", 0.35)
        assert a_crit is not None
        assert a_crit.severity == AlertSeverity.CRITICAL

    def test_alert_acknowledgement_and_history(self):
        manager = AlertManager()
        a1 = manager.record_custom_alert(
            category=AlertCategory.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            metric_name="Missingness",
            current_value=2.5,
            threshold_warning=1.0,
            threshold_critical=5.0,
            message="2.5% missing values",
            recommended_action="Inspect pipeline",
        )
        assert len(manager.get_active_alerts(unacknowledged_only=True)) == 1

        # Acknowledge alert
        success = manager.acknowledge_alert(a1.alert_id)
        assert success is True
        assert len(manager.get_active_alerts(unacknowledged_only=True)) == 0
        assert len(manager.get_active_alerts(unacknowledged_only=False)) == 1

        df_hist = manager.history_dataframe()
        assert len(df_hist) == 1
        assert bool(df_hist.iloc[0]["Acknowledged"]) is True


# ===========================================================================
# 6. Simulation Engine Tests
# ===========================================================================

class TestSimulationEngine:
    """Tests for simulated monitoring cohorts and mandatory synthetic disclaimers."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        n = 500
        return pd.DataFrame({
            "id": list(range(n)),
            "limit_bal": rng.normal(50000, 10000, size=n),
            "age": rng.randint(20, 70, size=n),
            "education": rng.choice([1, 2, 3], size=n),
            "bill_amt1": rng.normal(10000, 5000, size=n),
            "pay_0": rng.choice([0, 1, 2], size=n),
            "default_payment_next_month": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        })

    def test_simulated_cohort_has_synthetic_disclaimer(self, sample_data):
        sim = simulate_monitoring_cohort(sample_data, scenario_type="stable")
        assert sim.is_simulation is True
        assert "PROVABLY SYNTHETIC" in sim.disclaimer
        assert "[SIMULATION:" in sim.name

    def test_unlabelled_recent_scenario_has_none_labels(self, sample_data):
        sim = simulate_monitoring_cohort(sample_data, scenario_type="unlabelled_recent")
        assert sim.labels is None
        assert sim.labels_available is False

    def test_covariate_shift_scenario_shifts_features(self, sample_data):
        sim = simulate_monitoring_cohort(sample_data, scenario_type="covariate_shift")
        # limit_bal is contracted
        assert sim.data["limit_bal"].mean() < sample_data["limit_bal"].mean()

    def test_data_quality_defect_scenario_injects_nulls(self, sample_data):
        sim = simulate_monitoring_cohort(sample_data, scenario_type="data_quality_defect")
        assert sim.data["limit_bal"].isna().sum() > 0


# ===========================================================================
# 7. Subsystem Orchestrator Tests
# ===========================================================================

class TestModelMonitoringSubsystem:
    """End-to-end integration tests for ModelMonitoringSubsystem."""

    def test_full_surveillance_cycle_runs_successfully(self):
        rng = np.random.RandomState(42)
        n = 1000

        df_base = pd.DataFrame({
            "limit_bal": rng.normal(50000, 10000, size=n),
            "age": rng.randint(20, 65, size=n),
        })
        df_mon = pd.DataFrame({
            "limit_bal": rng.normal(50000, 10000, size=n),
            "age": rng.randint(20, 65, size=n),
        })
        b_p = rng.beta(2, 6, size=n)
        m_p = rng.beta(2, 6, size=n)

        subsystem = ModelMonitoringSubsystem()
        report = subsystem.run_surveillance_cycle(
            df_baseline=df_base,
            df_monitoring=df_mon,
            baseline_probs=b_p,
            monitoring_probs=m_p,
            monitoring_labels=None,  # test unlabelled
        )

        assert isinstance(report, ComprehensiveMonitoringReport)
        assert report.overall_governance_status in ("GREEN", "AMBER", "RED")
        assert report.performance_report.status == "UNAVAILABLE_NO_LABELS"
        assert len(report.feature_drift_results) == 2
        assert "Surveillance cycle completed" in report.executive_summary


# ===========================================================================
# 8. Visualization Tests
# ===========================================================================

class TestMonitoringVisualizations:
    """Verify Plotly monitoring figures."""

    def test_plot_feature_csi_ranking(self):
        csi_df = pd.DataFrame({
            "Feature": ["limit_bal", "age"],
            "CSI_Value": [0.045, 0.120],
        })
        fig = plot_feature_csi_ranking(csi_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_plot_prediction_distribution_shift(self):
        p1 = np.linspace(0.01, 0.50, 100)
        p2 = np.linspace(0.05, 0.60, 100)
        fig = plot_prediction_distribution_shift(p1, p2)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_risk_segment_migration(self):
        mig_df = pd.DataFrame({
            "Rating Grade": ["AAA", "AA", "A", "BBB"],
            "Baseline Share (%)": [5.0, 15.0, 30.0, 50.0],
            "Monitoring Share (%)": [4.0, 14.0, 32.0, 50.0],
        })
        fig = plot_risk_segment_migration(mig_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
