"""
Tests for Data Quality Validation Engine (BCBS 239 checks).
"""

import pandas as pd
import pytest

from src.data.validation import DataValidator, ValidationReport


@pytest.fixture
def sample_valid_data():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "limit_bal": [50000.0, 120000.0, 20000.0, 300000.0, 80000.0],
        "sex": [1, 2, 1, 2, 1],
        "education": [1, 2, 3, 4, 2],
        "marriage": [1, 2, 1, 2, 1],
        "age": [25, 34, 45, 50, 28],
        "pay_0": [0, -1, 1, 2, 0],
        "bill_amt1": [3000.0, 15000.0, 500.0, 45000.0, 1200.0],
        "pay_amt1": [1000.0, 15000.0, 500.0, 5000.0, 1200.0],
        "default_payment_next_month": [0, 0, 1, 1, 0],
    })


def test_validation_passes_on_valid_data(sample_valid_data):
    validator = DataValidator()
    report = validator.validate(sample_valid_data)
    assert report.is_valid is True
    assert report.total_records == 5
    assert report.summary_stats["critical_errors"] == 0


def test_validation_flags_empty_dataframe():
    validator = DataValidator()
    empty_df = pd.DataFrame()
    report = validator.validate(empty_df)
    assert report.is_valid is False
    assert report.total_records == 0


def test_validation_flags_duplicate_ids(sample_valid_data):
    df_dup = sample_valid_data.copy()
    df_dup.loc[1, "id"] = 1  # Duplicate ID 1
    validator = DataValidator()
    report = validator.validate(df_dup)
    assert report.is_valid is False
    issues = [i["check"] for i in report.issues_found]
    assert "Duplicate Primary Keys" in issues


def test_validation_flags_invalid_age(sample_valid_data):
    df_bad_age = sample_valid_data.copy()
    df_bad_age.loc[0, "age"] = 10  # Age < 18
    validator = DataValidator()
    report = validator.validate(df_bad_age)
    assert report.is_valid is False
    issues = [i["check"] for i in report.issues_found]
    assert "Age Boundary Violation" in issues


def test_validation_flags_negative_limit(sample_valid_data):
    df_bad_limit = sample_valid_data.copy()
    df_bad_limit.loc[0, "limit_bal"] = -500.0
    validator = DataValidator()
    report = validator.validate(df_bad_limit)
    assert report.is_valid is False
    issues = [i["check"] for i in report.issues_found]
    assert "Negative or Zero Credit Limit" in issues


def test_validation_flags_missing_values(sample_valid_data):
    df_missing = sample_valid_data.copy()
    df_missing.loc[0, "age"] = None
    validator = DataValidator()
    report = validator.validate(df_missing)
    issues = [i["check"] for i in report.issues_found]
    assert "Missing Values" in issues


def test_validation_flags_non_binary_target(sample_valid_data):
    df_bad_target = sample_valid_data.copy()
    df_bad_target.loc[0, "default_payment_next_month"] = 5
    validator = DataValidator()
    report = validator.validate(df_bad_target)
    assert report.is_valid is False
    issues = [i["check"] for i in report.issues_found]
    assert "Target Variable Integrity" in issues


def test_validation_audits_unmapped_categories(sample_valid_data):
    df_unmapped = sample_valid_data.copy()
    df_unmapped.loc[0, "education"] = 5  # Unmapped education code (0, 5, 6)
    df_unmapped.loc[1, "marriage"] = 0   # Unmapped marriage code (0)
    validator = DataValidator()
    report = validator.validate(df_unmapped)
    issues = [i["check"] for i in report.issues_found]
    assert "Non-Standard Education Levels" in issues
    assert "Non-Standard Marriage Levels" in issues
