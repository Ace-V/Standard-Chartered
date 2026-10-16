"""
Tests for Credit Risk Feature Engineering Module.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import CreditFeatureEngineer


@pytest.fixture
def sample_feature_input():
    return pd.DataFrame({
        "id": [1, 2],
        "limit_bal": [10000.0, 50000.0],
        "pay_0": [0, 2],
        "pay_2": [0, 1],
        "pay_3": [-1, 0],
        "pay_4": [0, 0],
        "pay_5": [0, 0],
        "pay_6": [0, 0],
        "bill_amt1": [5000.0, 40000.0],
        "bill_amt2": [4000.0, 35000.0],
        "bill_amt3": [3000.0, 30000.0],
        "bill_amt4": [2000.0, 25000.0],
        "bill_amt5": [1000.0, 20000.0],
        "bill_amt6": [500.0, 15000.0],
        "pay_amt1": [1000.0, 2000.0],
        "pay_amt2": [1000.0, 2000.0],
        "pay_amt3": [1000.0, 2000.0],
        "pay_amt4": [1000.0, 2000.0],
        "pay_amt5": [1000.0, 2000.0],
        "pay_amt6": [500.0, 1000.0],
    })


def test_feature_engineer_generates_utilization(sample_feature_input):
    engineer = CreditFeatureEngineer()
    engineer.fit(sample_feature_input)
    transformed = engineer.transform(sample_feature_input)

    assert "utilization_1" in transformed.columns
    assert "mean_utilization" in transformed.columns
    # Row 0: 5000 / 10000 = 0.5
    assert np.isclose(transformed.loc[0, "utilization_1"], 0.5)
    # Row 1: 40000 / 50000 = 0.8
    assert np.isclose(transformed.loc[1, "utilization_1"], 0.8)


def test_feature_engineer_delinquency_metrics(sample_feature_input):
    engineer = CreditFeatureEngineer()
    transformed = engineer.transform(sample_feature_input)

    assert "max_delinquency_recent" in transformed.columns
    assert "severe_delinquency_flag" in transformed.columns
    # Row 0 max pay status is 0
    assert transformed.loc[0, "max_delinquency_recent"] == 0
    assert transformed.loc[0, "severe_delinquency_flag"] == 0
    # Row 1 has pay_0 = 2 (severe delinquency)
    assert transformed.loc[1, "max_delinquency_recent"] == 2
    assert transformed.loc[1, "severe_delinquency_flag"] == 1


def test_feature_engineer_payment_ratios(sample_feature_input):
    engineer = CreditFeatureEngineer()
    transformed = engineer.transform(sample_feature_input)

    assert "pay_ratio_1" in transformed.columns
    assert "mean_pay_ratio" in transformed.columns
    # Row 0 pay_amt1 / bill_amt2 = 1000 / 4000 = 0.25
    assert np.isclose(transformed.loc[0, "pay_ratio_1"], 0.25)
