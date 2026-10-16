"""
Tests for Model Evaluation & Calibration Modules.
"""

import numpy as np
import pytest

from src.models.calibration import ProbabilityCalibrator, compute_calibration_curve
from src.models.evaluate import ModelEvaluator


def test_model_evaluator_perfect_discrimination():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    evaluator = ModelEvaluator(cutoff=0.5)
    metrics = evaluator.evaluate(y_true, y_prob, compute_ci=False)

    assert metrics.roc_auc == 1.0
    assert metrics.gini == 1.0
    assert metrics.ks_statistic == 1.0
    assert metrics.accuracy == 1.0


def test_model_evaluator_random_classifier():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_prob, compute_ci=False)

    assert metrics.roc_auc == 0.5
    assert metrics.gini == 0.0


def test_probability_calibrator_isotonic():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Distorted/uncalibrated probabilities
    raw_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.70, 0.75, 0.80, 0.85, 0.90])

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(raw_prob, y_true)
    calibrated = calibrator.predict_proba(raw_prob)

    assert len(calibrated) == len(raw_prob)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Monotonicity check
    assert np.all(np.diff(calibrated) >= 0)


def test_compute_calibration_curve_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    prob_true, prob_pred, counts, ece, mce = compute_calibration_curve(y_true, y_prob, n_bins=2)

    assert len(prob_true) == 2
    assert len(prob_pred) == 2
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
