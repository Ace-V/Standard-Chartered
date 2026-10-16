"""
CreditRisk Analytics — Feature Engineering Module
=================================================

Constructs quantitative credit risk signals from consumer transaction, balance,
and repayment histories. Captures revolving credit utilization, delinquency trajectories,
payment ratios, and balance volatility aligned with Basel / CECL risk drivers.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import logger


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw credit and balance histories into predictive credit risk metrics.
    Operates statelessly on input matrices to ensure zero test leakage.
    """

    def __init__(self, include_raw_features: bool = True):
        self.include_raw_features = include_raw_features
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditFeatureEngineer":
        # Feature generation is deterministic; we discover output feature names on sample fit
        transformed = self.transform(X.iloc[:5])
        self.engineered_feature_names = list(transformed.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer credit risk features from raw credit accounts."""
        df = X.copy()
        new_cols: dict[str, np.ndarray | pd.Series] = {}

        limit = df["limit_bal"] if "limit_bal" in df.columns else pd.Series(10000.0, index=df.index)
        limit_safe = np.where(limit <= 0, 1000.0, limit)

        # ----------------------------------------------------------------------
        # 1. Utilization Ratios (Bill Amount / Credit Limit)
        # ----------------------------------------------------------------------
        utilization_cols = []
        for i in range(1, 7):
            bill_col = f"bill_amt{i}"
            if bill_col in df.columns:
                util_col_name = f"utilization_{i}"
                # Clip utilization to [-0.1, 2.0] to control extreme outliers while preserving signals
                new_cols[util_col_name] = np.clip(df[bill_col] / limit_safe, -0.1, 2.0)
                utilization_cols.append(util_col_name)

        if utilization_cols:
            util_df = pd.DataFrame({c: new_cols[c] for c in utilization_cols}, index=df.index)
            new_cols["mean_utilization"] = util_df.mean(axis=1)
            new_cols["max_utilization"] = util_df.max(axis=1)
            if "utilization_1" in new_cols and "utilization_6" in new_cols:
                new_cols["utilization_trend"] = new_cols["utilization_1"] - new_cols["utilization_6"]

        # ----------------------------------------------------------------------
        # 2. Delinquency Trajectories (Payment Status History)
        # ----------------------------------------------------------------------
        pay_cols = [c for c in df.columns if c.startswith("pay_") and not c.startswith("pay_amt")]
        # Sort paying attention to pay_0, pay_2, pay_3, ...
        pay_cols = sorted(pay_cols)
        if pay_cols:
            pay_sub = df[pay_cols]
            new_cols["max_delinquency_recent"] = pay_sub.max(axis=1)
            new_cols["sum_delinquency"] = pay_sub.apply(lambda row: sum(max(0, val) for val in row), axis=1)
            # Delinquency flag: 1 if borrower experienced 60+ DPD (status >= 2)
            new_cols["severe_delinquency_flag"] = (pay_sub >= 2).any(axis=1).astype(int)
            # Current delinquency flag
            if "pay_0" in df.columns:
                new_cols["current_delinquency_flag"] = (df["pay_0"] > 0).astype(int)

        # ----------------------------------------------------------------------
        # 3. Payment-to-Bill Ratios
        # ----------------------------------------------------------------------
        payment_ratios = []
        for i in range(1, 6):
            pay_col = f"pay_amt{i}"
            next_bill = f"bill_amt{i + 1}"
            if pay_col in df.columns and next_bill in df.columns:
                denom = np.where(df[next_bill] <= 0, 100.0, df[next_bill])
                ratio_name = f"pay_ratio_{i}"
                new_cols[ratio_name] = np.clip(df[pay_col] / denom, 0.0, 2.0)
                payment_ratios.append(ratio_name)

        if payment_ratios:
            pay_ratio_df = pd.DataFrame({c: new_cols[c] for c in payment_ratios}, index=df.index)
            new_cols["mean_pay_ratio"] = pay_ratio_df.mean(axis=1)

        # ----------------------------------------------------------------------
        # 4. Bill Amount Volatility & Net Growth
        # ----------------------------------------------------------------------
        bill_cols = [f"bill_amt{i}" for i in range(1, 7) if f"bill_amt{i}" in df.columns]
        if bill_cols:
            bill_sub = df[bill_cols]
            new_cols["mean_bill_amt"] = bill_sub.mean(axis=1)
            new_cols["bill_amt_std"] = bill_sub.std(axis=1).fillna(0.0)
            if "bill_amt1" in df.columns and "bill_amt6" in df.columns:
                new_cols["net_bill_change"] = df["bill_amt1"] - df["bill_amt6"]

        # ----------------------------------------------------------------------
        # 5. Payment Amount Statistics
        # ----------------------------------------------------------------------
        pay_amt_cols = [f"pay_amt{i}" for i in range(1, 7) if f"pay_amt{i}" in df.columns]
        if pay_amt_cols:
            pay_amt_sub = df[pay_amt_cols]
            new_cols["mean_pay_amt"] = pay_amt_sub.mean(axis=1)
            # Count of zero payments
            new_cols["zero_payment_count"] = (pay_amt_sub == 0).sum(axis=1)

        # Assemble transformed DataFrame
        engineered_df = pd.DataFrame(new_cols, index=df.index)

        if self.include_raw_features:
            out_df = pd.concat([df, engineered_df], axis=1)
        else:
            out_df = engineered_df

        return out_df


def get_feature_names(engineer: CreditFeatureEngineer, X_sample: pd.DataFrame) -> List[str]:
    """Retrieve full list of output feature names after engineering."""
    transformed = engineer.transform(X_sample.iloc[:2])
    return list(transformed.columns)
