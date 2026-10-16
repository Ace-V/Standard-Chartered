"""
Tests for Model Monitoring, Drift Detection (PSI/CSI), and Performance Decay.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import EvaluationMetrics
from src.monitoring.drift import calculate_csi_for_features, calculate_psi
from src.monitoring.performance import PerformanceMonitor


def test_psi_identical_distributions():
    np.random.seed(42)
    expected = np.random.normal(loc=600, scale=50, size=5000)
    actual = np.random.normal(loc=600, scale=50, size=5000)

    res = calculate_psi(expected, actual, num_bins=10)
    assert res.psi_value < 0.05
    assert res.traffic_light == "GREEN"


def test_psi_severe_drift():
    np.random.seed(42)
    expected = np.random.normal(loc=650, scale=40, size=5000)
    # Significant shift downwards (severe deterioration)
    actual = np.random.normal(loc=520, scale=40, size=5000)

    res = calculate_psi(expected, actual, num_bins=10)
    assert res.psi_value >= 0.25
    assert res.traffic_light == "RED"


def test_csi_feature_calculation():
    df_exp = pd.DataFrame({
        "age": np.random.randint(20, 60, size=1000),
        "limit_bal": np.random.normal(50000, 10000, size=1000),
    })
    df_act = pd.DataFrame({
        "age": np.random.randint(20, 60, size=1000),
        "limit_bal": np.random.normal(20000, 5000, size=1000),  # shifted limit_bal
    })

    csi_df = calculate_csi_for_features(df_exp, df_act, features=["age", "limit_bal"])
    assert len(csi_df) == 2
    limit_row = csi_df[csi_df["Feature"] == "limit_bal"].iloc[0]
    assert limit_row["CSI_Value"] > 0.10


def test_performance_monitor_decay():
    base_metrics = EvaluationMetrics(
        roc_auc=0.80,
        gini=0.60,
        ks_statistic=0.50,
        ks_threshold=0.30,
        brier_score=0.10,
        log_loss_val=0.35,
        pr_auc=0.55,
        accuracy=0.82,
        precision=0.65,
        recall=0.70,
        f1=0.67,
        confusion_mat={"TN": 80, "FP": 10, "FN": 5, "TP": 25},
    )

    perf_monitor = PerformanceMonitor(gini_alert_threshold_pct=15.0)

    # Degraded cohort (random guessing)
    y_true_mon = np.array([0] * 500 + [1] * 500)
    y_prob_mon = np.random.uniform(0.1, 0.9, size=1000)

    report = perf_monitor.compare_performance(base_metrics, y_true_mon, y_prob_mon)

    assert report.gini_decay_pct > 15.0
    assert report.status in ("WARNING", "CRITICAL")
    assert len(report.alerts) > 0
