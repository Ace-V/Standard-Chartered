"""
Tests for Data Ingestion, Programmatic Target Verification, and EDA Modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.eda import (
    analyze_default_rate_by_credit_limit,
    analyze_default_rate_by_repayment_status,
    analyze_default_rate_by_segments,
    compute_dataset_summary,
    generate_and_save_eda_reports,
    identify_and_verify_target,
    plot_credit_limit_distribution_by_target,
    plot_default_rate_by_repayment_status,
    plot_target_distribution,
)
from src.data.ingestion import (
    UCI_DATA_DICTIONARY,
    UCICardDataIngestor,
    load_raw_dataset,
)
from src.features.engineering import CreditFeatureEngineer


@pytest.fixture
def sample_eda_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "limit_bal": np.random.choice([20000, 50000, 150000, 300000, 600000], size=n),
        "sex": np.random.choice([1, 2], size=n),
        "education": np.random.choice([1, 2, 3, 4], size=n),
        "marriage": np.random.choice([1, 2, 3], size=n),
        "age": np.random.randint(20, 65, size=n),
        "pay_0": np.random.choice([-1, 0, 1, 2, 3], size=n, p=[0.2, 0.5, 0.15, 0.1, 0.05]),
        "pay_2": np.random.choice([-1, 0, 1, 2], size=n),
        "pay_3": [0] * n,
        "pay_4": [0] * n,
        "pay_5": [0] * n,
        "pay_6": [0] * n,
        "bill_amt1": np.random.uniform(1000, 50000, size=n),
        "bill_amt2": np.random.uniform(1000, 50000, size=n),
        "bill_amt3": np.random.uniform(1000, 50000, size=n),
        "bill_amt4": np.random.uniform(1000, 50000, size=n),
        "bill_amt5": np.random.uniform(1000, 50000, size=n),
        "bill_amt6": np.random.uniform(1000, 50000, size=n),
        "pay_amt1": np.random.uniform(500, 10000, size=n),
        "pay_amt2": np.random.uniform(500, 10000, size=n),
        "pay_amt3": np.random.uniform(500, 10000, size=n),
        "pay_amt4": np.random.uniform(500, 10000, size=n),
        "pay_amt5": np.random.uniform(500, 10000, size=n),
        "pay_amt6": np.random.uniform(500, 10000, size=n),
        "default_payment_next_month": np.random.choice([0, 1], size=n, p=[0.78, 0.22]),
    })


def test_target_identification_and_verification(sample_eda_df):
    target_col, summary = identify_and_verify_target(sample_eda_df)
    assert target_col == "default_payment_next_month"
    assert summary.is_valid is True
    assert summary.total_records == 200
    assert summary.unique_values == [0, 1]
    assert summary.imbalance_ratio > 1.0


def test_target_identification_ucimlrepo_alias(sample_eda_df):
    # Rename target column to 'Y' as returned in ucimlrepo
    df_y = sample_eda_df.rename(columns={"default_payment_next_month": "Y"})
    target_col, summary = identify_and_verify_target(df_y)
    assert target_col == "Y"
    assert summary.is_valid is True


def test_target_identification_rejects_non_binary():
    df_invalid = pd.DataFrame({"default_payment_next_month": [0, 1, 2, 0, 1]})
    target_col, summary = identify_and_verify_target(df_invalid)
    assert summary.is_valid is False
    assert "Invalid target values" in summary.diagnostic_message


def test_compute_dataset_summary(sample_eda_df):
    summaries = compute_dataset_summary(sample_eda_df)
    assert "numerical_summary" in summaries
    assert "categorical_summary" in summaries

    num_df = summaries["numerical_summary"]
    assert "limit_bal" in num_df["Variable"].values
    assert "age" in num_df["Variable"].values


def test_repayment_status_default_rate_analysis(sample_eda_df):
    dr_df = analyze_default_rate_by_repayment_status(sample_eda_df, pay_col="pay_0")
    assert "default_rate_pct" in dr_df.columns
    assert "ci_95_low" in dr_df.columns
    assert "ci_95_high" in dr_df.columns
    # CIs should be valid percentages
    assert (dr_df["ci_95_low"] >= 0).all()
    assert (dr_df["ci_95_high"] <= 100).all()


def test_credit_limit_default_rate_analysis(sample_eda_df):
    limit_df = analyze_default_rate_by_credit_limit(sample_eda_df)
    assert "limit_tier" in limit_df.columns
    assert "default_rate_pct" in limit_df.columns
    assert len(limit_df) > 0


def test_zero_target_leakage_in_feature_engineering(sample_eda_df):
    """
    CRITICAL STATISTICAL REQUIREMENT: Verify that feature engineering
    operates purely on predictor features (X) without using or leaking the target.
    """
    X = sample_eda_df.drop(columns=["default_payment_next_month"])
    engineer = CreditFeatureEngineer()
    engineer.fit(X)
    X_transformed = engineer.transform(X)

    # Feature engineering must NOT create or include the target variable
    assert "default_payment_next_month" not in X_transformed.columns
    assert "Y" not in X_transformed.columns


def test_generate_and_save_eda_reports_creates_files(sample_eda_df, tmp_path):
    out = generate_and_save_eda_reports(sample_eda_df, output_dir=tmp_path)

    assert out["report_markdown"].exists()
    assert out["fig_target"].exists()
    assert out["fig_repay"].exists()
    assert out["fig_limit"].exists()
    assert out["fig_segments"].exists()

    # Verify report content contains regulatory disclaimers
    content = out["report_markdown"].read_text(encoding="utf-8")
    assert "Class Imbalance Disclosure" in content
    assert "Association vs. Causation Principle" in content
    assert "Zero Target Leakage Rule" in content
    assert "Consumer Credit Proxy Only" in content


def test_data_dictionary_completeness():
    ingestor = UCICardDataIngestor()
    data_dict = ingestor.get_data_dictionary()
    assert "LIMIT_BAL" in data_dict
    assert "PAY_0" in data_dict
    assert "BILL_AMT1" in data_dict
    assert "PAY_AMT1" in data_dict
    assert "default.payment.next.month" in data_dict
