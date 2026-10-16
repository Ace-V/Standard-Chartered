"""
CreditRisk Analytics — Data Quality & Validation Engine
=======================================================

Implements institutional Data Quality (DQ) and governance principles aligned with
BCBS 239 (Principles for Effective Risk Data Aggregation and Risk Reporting).
Provides pre-flight sanity checks, schema validation, boundary checks, and audit logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import ValidationConfig, logger, validation_config


@dataclass
class ValidationReport:
    """Encapsulates the audit output of the Data Quality engine."""
    is_valid: bool
    total_records: int
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    column_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert validation issue records to a tabular format for reporting."""
        if not self.issues_found:
            return pd.DataFrame([{"check": "All Data Quality Checks", "status": "PASSED", "details": "No anomalies"}])
        return pd.DataFrame(self.issues_found)

    def print_summary(self) -> None:
        """Print audit summary to standard logging stream."""
        logger.info("=== Data Quality Audit Report ===")
        logger.info("Status: %s", "PASSED" if self.is_valid else "FAILED")
        logger.info("Total Rows Audited: %d", self.total_records)
        logger.info("Total Quality Issues / Warnings: %d", len(self.issues_found))
        for issue in self.issues_found:
            logger.info("[%s] %s: %s", issue.get("severity", "INFO"), issue.get("check"), issue.get("details"))


class DataValidator:
    """
    Validates credit datasets against banking compliance, data integrity,
    and distributional assumptions.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or validation_config

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Execute comprehensive suite of data quality checks."""
        issues: List[Dict[str, Any]] = []
        col_checks: List[Dict[str, Any]] = []
        n_rows = len(df)

        if n_rows == 0:
            return ValidationReport(
                is_valid=False,
                total_records=0,
                issues_found=[{"check": "Empty Dataset", "severity": "CRITICAL", "details": "Dataset contains 0 rows"}],
            )

        # 1. Missing Value Check
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            for col, count in missing_counts[missing_counts > 0].items():
                pct = (count / n_rows) * 100
                issues.append({
                    "check": "Missing Values",
                    "column": col,
                    "severity": "ERROR" if pct > 5.0 else "WARNING",
                    "details": f"{count} missing values ({pct:.2f}%)",
                })

        # 2. Duplicate ID Check
        if self.config.id_col in df.columns:
            dup_count = df.duplicated(subset=[self.config.id_col]).sum()
            if dup_count > 0:
                issues.append({
                    "check": "Duplicate Primary Keys",
                    "column": self.config.id_col,
                    "severity": "CRITICAL",
                    "details": f"Found {dup_count} duplicate '{self.config.id_col}' values.",
                })

        # 3. Target Distribution Check
        if self.config.target_col in df.columns:
            target_series = df[self.config.target_col]
            unique_targets = set(target_series.dropna().unique())
            if not unique_targets.issubset({0, 1}):
                issues.append({
                    "check": "Target Variable Integrity",
                    "column": self.config.target_col,
                    "severity": "CRITICAL",
                    "details": f"Target must be binary [0, 1], found: {unique_targets}",
                })
            else:
                default_rate = target_series.mean()
                if default_rate < 0.01 or default_rate > 0.50:
                    issues.append({
                        "check": "Target Imbalance Warning",
                        "column": self.config.target_col,
                        "severity": "WARNING",
                        "details": f"Extreme default rate observed: {default_rate:.2%}",
                    })

        # 4. Limit Balance Bounds
        if "limit_bal" in df.columns:
            negative_limits = (df["limit_bal"] <= 0).sum()
            if negative_limits > 0:
                issues.append({
                    "check": "Negative or Zero Credit Limit",
                    "column": "limit_bal",
                    "severity": "ERROR",
                    "details": f"{negative_limits} accounts have limit_bal <= 0",
                })

        # 5. Age Boundaries
        if "age" in df.columns:
            out_of_bounds_age = ((df["age"] < self.config.min_age) | (df["age"] > self.config.max_age)).sum()
            if out_of_bounds_age > 0:
                issues.append({
                    "check": "Age Boundary Violation",
                    "column": "age",
                    "severity": "ERROR",
                    "details": f"{out_of_bounds_age} records with age outside [{self.config.min_age}, {self.config.max_age}]",
                })

        # 6. Education Categories Audit
        if "education" in df.columns:
            unmapped_edu = (~df["education"].isin(self.config.valid_education_levels)).sum()
            if unmapped_edu > 0:
                issues.append({
                    "check": "Non-Standard Education Levels",
                    "column": "education",
                    "severity": "INFO",
                    "details": f"{unmapped_edu} records have non-standard education values (0, 5, 6 mapped during preprocessing)",
                })

        # 7. Marriage Categories Audit
        if "marriage" in df.columns:
            unmapped_mar = (~df["marriage"].isin(self.config.valid_marriage_levels)).sum()
            if unmapped_mar > 0:
                issues.append({
                    "check": "Non-Standard Marriage Levels",
                    "column": "marriage",
                    "severity": "INFO",
                    "details": f"{unmapped_mar} records have marriage code 0 (mapped to 3 during preprocessing)",
                })

        # 8. Over-limit / Excessive Utilization Check
        if "bill_amt1" in df.columns and "limit_bal" in df.columns:
            over_limit = (df["bill_amt1"] > (df["limit_bal"] * 1.5)).sum()
            if over_limit > 0:
                issues.append({
                    "check": "Over-Limit Exposure (>150% Utilization)",
                    "column": "bill_amt1",
                    "severity": "INFO",
                    "details": f"{over_limit} accounts show bill amount exceeding 150% of assigned credit limit",
                })

        # Summarize Column Metrics
        for col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            col_checks.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique()),
                "min": float(df[col].min()) if is_num and not df[col].empty else None,
                "max": float(df[col].max()) if is_num and not df[col].empty else None,
            })

        # Determine overall validity (CRITICAL or ERROR severities make is_valid False)
        critical_errors = [i for i in issues if i["severity"] in ("CRITICAL", "ERROR")]
        is_valid = len(critical_errors) == 0

        summary_stats = {
            "total_records": n_rows,
            "total_columns": len(df.columns),
            "missing_cells": int(total_missing),
            "default_rate": float(df[self.config.target_col].mean()) if self.config.target_col in df.columns else None,
            "critical_errors": len(critical_errors),
            "warnings_and_info": len(issues) - len(critical_errors),
        }

        report = ValidationReport(
            is_valid=is_valid,
            total_records=n_rows,
            issues_found=issues,
            summary_stats=summary_stats,
            column_checks=col_checks,
        )
        return report
