"""
CreditRisk Analytics — Data Preprocessing Module
================================================

Handles data cleaning, categorical normalization, missing data treatment,
and strict training/test splitting without data leakage, adhering to
banking regulatory standards (BCBS / SR 11-7).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.config import ModelConfig, ValidationConfig, logger, model_config, validation_config


class CreditDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for Credit Card Risk data.
    Fits solely on the training partition to avoid target and statistical leakage.
    """

    def __init__(
        self,
        validation_cfg: Optional[ValidationConfig] = None,
        model_cfg: Optional[ModelConfig] = None,
    ):
        self.val_config = validation_cfg or validation_config
        self.mod_config = model_cfg or model_config
        self.is_fitted = False
        self.numeric_impute_values: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditDataPreprocessor":
        """Compute training medians for numeric imputation."""
        X_df = X.copy()
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.numeric_impute_values[col] = float(X_df[col].median(skipna=True))

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical recoding, clipping, and imputation."""
        df = X.copy()

        # 1. Clean Education: map undocumented (0, 5, 6) to 4 (Others)
        if "education" in df.columns:
            df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})

        # 2. Clean Marriage: map undocumented (0) to 3 (Others)
        if "marriage" in df.columns:
            df["marriage"] = df["marriage"].replace({0: 3})

        # 3. Handle Missing Values using training medians
        for col, median_val in self.numeric_impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)

        return df


def prepare_train_test_split(
    df: pd.DataFrame,
    cfg: Optional[ModelConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition dataset into training and out-of-sample testing sets using
    stratified sampling on the default target variable.
    """
    config = cfg or model_config
    target_col = config.target_column

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Remove non-feature identifier from X
    id_col = config.id_column
    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    stratify_target = y if config.stratify else None

    logger.info(
        "Performing Stratified Train-Test Split (test_size=%.2f, random_state=%d)",
        config.test_size,
        config.random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_target,
    )

    logger.info(
        "Dataset Split: Train rows=%d (Default rate: %.2f%%), Test rows=%d (Default rate: %.2f%%)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )

    return X_train, X_test, y_train, y_test
