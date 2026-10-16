"""
CreditRisk Analytics — Monitoring Subsystem: Data Quality Module (Module A)
==========================================================================

Monitors incoming production batches for:
1. Missing-value rates (per feature and dataset total)
2. Invalid numerical ranges (out-of-bounds, negative amounts)
3. Unexpected categorical values (unseen codes, category drift)
4. Schema drift (missing columns, extra columns, datatype shifts)
5. Duplicate records (primary key duplicates, full-row duplicates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnQualityMetric:
    """Quality metrics for an individual feature."""
    column_name: str
    dtype: str
    missing_count: int
    missing_pct: float
    invalid_range_count: int
    invalid_range_pct: float
    unexpected_categories_count: int
    unexpected_categories_pct: float
    status: str  # "OK", "WARNING", "CRITICAL"
    issues: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Consolidated Data Quality Audit Report."""
    total_records: int
    total_columns: int
    missing_columns: List[str]
    unexpected_columns: List[str]
    dtype_mismatches: Dict[str, Tuple[str, str]]  # col -> (expected, actual)
    duplicate_rows_count: int
    duplicate_ids_count: int
    column_metrics: Dict[str, ColumnQualityMetric]
    overall_status: str  # "HEALTHY", "WARNING", "CRITICAL"
    alerts: List[str] = field(default_factory=list)

    def missingness_summary(self) -> pd.DataFrame:
        """Tabular summary of missing-value rates per column."""
        rows = []
        for col, m in self.column_metrics.items():
            if m.missing_count > 0:
                rows.append({
                    "Column": col,
                    "Missing Count": m.missing_count,
                    "Missing (%)": round(m.missing_pct, 2),
                    "Status": m.status,
                })
        df = pd.DataFrame(rows)
        return df.sort_values(by="Missing (%)", ascending=False) if not df.empty else pd.DataFrame(
            columns=["Column", "Missing Count", "Missing (%)", "Status"]
        )

    def invalid_values_summary(self) -> pd.DataFrame:
        """Tabular summary of invalid ranges and unexpected categories."""
        rows = []
        for col, m in self.column_metrics.items():
            if m.invalid_range_count > 0 or m.unexpected_categories_count > 0:
                rows.append({
                    "Column": col,
                    "Invalid Range Count": m.invalid_range_count,
                    "Invalid Range (%)": round(m.invalid_range_pct, 2),
                    "Unexpected Cat Count": m.unexpected_categories_count,
                    "Unexpected Cat (%)": round(m.unexpected_categories_pct, 2),
                    "Issues": "; ".join(m.issues),
                    "Status": m.status,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Column", "Invalid Range Count", "Invalid Range (%)", "Unexpected Cat Count", "Unexpected Cat (%)", "Issues", "Status"]
        )

    def schema_audit_summary(self) -> pd.DataFrame:
        """Overview of schema modifications."""
        rows = []
        for col in self.missing_columns:
            rows.append({"Type": "Missing Column", "Field": col, "Details": "Expected column missing from batch", "Severity": "CRITICAL"})
        for col in self.unexpected_columns:
            rows.append({"Type": "Unexpected Column", "Field": col, "Details": "Unrecognized column present in batch", "Severity": "INFO"})
        for col, (exp_t, act_t) in self.dtype_mismatches.items():
            rows.append({"Type": "Data Type Mismatch", "Field": col, "Details": f"Expected {exp_t}, got {act_t}", "Severity": "WARNING"})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Type", "Field", "Details", "Severity"])


class DataQualityMonitor:
    """
    Evaluates incoming batches against standard quality thresholds and reference schema.
    """

    def __init__(
        self,
        warning_missing_pct: float = 1.0,
        critical_missing_pct: float = 5.0,
        warning_invalid_pct: float = 1.0,
        critical_invalid_pct: float = 5.0,
    ):
        self.warning_missing_pct = warning_missing_pct
        self.critical_missing_pct = critical_missing_pct
        self.warning_invalid_pct = warning_invalid_pct
        self.critical_invalid_pct = critical_invalid_pct

    def audit_batch(
        self,
        df: pd.DataFrame,
        reference_schema: Optional[Dict[str, str]] = None,
        expected_columns: Optional[List[str]] = None,
        range_rules: Optional[Dict[str, Tuple[float, float]]] = None,
        category_rules: Optional[Dict[str, List[Any] | Set[Any]]] = None,
        id_column: Optional[str] = "id",
    ) -> DataQualityReport:
        """
        Execute an end-to-end data quality audit on an incoming data batch.
        """
        n_rows, n_cols = df.shape
        alerts: List[str] = []
        overall_status = "HEALTHY"

        if n_rows == 0:
            alerts.append("CRITICAL: Incoming batch is empty (0 records).")
            return DataQualityReport(
                total_records=0,
                total_columns=n_cols,
                missing_columns=[],
                unexpected_columns=[],
                dtype_mismatches={},
                duplicate_rows_count=0,
                duplicate_ids_count=0,
                column_metrics={},
                overall_status="CRITICAL",
                alerts=alerts,
            )

        # 1. Schema check
        missing_cols: List[str] = []
        unexpected_cols: List[str] = []
        dtype_mismatches: Dict[str, Tuple[str, str]] = {}

        if expected_columns is not None:
            missing_cols = [c for c in expected_columns if c not in df.columns]
            unexpected_cols = [c for c in df.columns if c not in expected_columns]
            if missing_cols:
                alerts.append(f"CRITICAL: {len(missing_cols)} expected column(s) missing from batch: {missing_cols}")
                overall_status = "CRITICAL"
            if unexpected_cols:
                alerts.append(f"INFO: {len(unexpected_cols)} unexpected column(s) present in batch: {unexpected_cols}")

        if reference_schema is not None:
            for col, expected_type in reference_schema.items():
                if col in df.columns:
                    act_type = str(df[col].dtype)
                    # Normalize simple int/float checks
                    if expected_type == "numeric" and not np.issubdtype(df[col].dtype, np.number):
                        dtype_mismatches[col] = (expected_type, act_type)
                    elif expected_type not in ("numeric", act_type) and expected_type not in act_type:
                        dtype_mismatches[col] = (expected_type, act_type)

            if dtype_mismatches:
                alerts.append(f"WARNING: Data type mismatches detected in {len(dtype_mismatches)} column(s).")
                if overall_status != "CRITICAL":
                    overall_status = "WARNING"

        # 2. Duplicate checks
        dup_rows = int(df.duplicated().sum())
        dup_ids = 0
        if id_column and id_column in df.columns:
            dup_ids = int(df.duplicated(subset=[id_column]).sum())
            if dup_ids > 0:
                alerts.append(f"CRITICAL: {dup_ids} duplicate primary key(s) detected in '{id_column}'.")
                overall_status = "CRITICAL"

        if dup_rows > 0:
            alerts.append(f"WARNING: {dup_rows} fully duplicate row(s) detected.")
            if overall_status != "CRITICAL":
                overall_status = "WARNING"

        # 3. Column-level quality audit
        col_metrics: Dict[str, ColumnQualityMetric] = {}
        range_rules = range_rules or {}
        category_rules = category_rules or {}

        for col in df.columns:
            series = df[col]
            missing_cnt = int(series.isna().sum())
            missing_pct = (missing_cnt / n_rows) * 100.0

            # Range check
            invalid_range_cnt = 0
            if col in range_rules and np.issubdtype(series.dtype, np.number):
                min_val, max_val = range_rules[col]
                valid_num = series.dropna()
                out_of_bounds = (valid_num < min_val) | (valid_num > max_val)
                invalid_range_cnt = int(out_of_bounds.sum())

            invalid_range_pct = (invalid_range_cnt / n_rows) * 100.0

            # Categorical check
            unexpected_cat_cnt = 0
            if col in category_rules:
                allowed = set(category_rules[col])
                valid_cat = series.dropna()
                unexpected_mask = ~valid_cat.isin(allowed)
                unexpected_cat_cnt = int(unexpected_mask.sum())

            unexpected_cat_pct = (unexpected_cat_cnt / n_rows) * 100.0

            # Column status & issues
            issues: List[str] = []
            col_status = "OK"

            if missing_pct >= self.critical_missing_pct:
                col_status = "CRITICAL"
                issues.append(f"High missingness: {missing_pct:.1f}% >= {self.critical_missing_pct}%")
            elif missing_pct >= self.warning_missing_pct:
                col_status = "WARNING"
                issues.append(f"Moderate missingness: {missing_pct:.1f}%")

            if invalid_range_pct >= self.critical_invalid_pct:
                col_status = "CRITICAL"
                issues.append(f"High invalid range rate: {invalid_range_pct:.1f}%")
            elif invalid_range_pct >= self.warning_invalid_pct:
                col_status = "WARNING" if col_status != "CRITICAL" else col_status
                issues.append(f"Out of range values: {invalid_range_pct:.1f}%")

            if unexpected_cat_pct >= self.critical_invalid_pct:
                col_status = "CRITICAL"
                issues.append(f"High unexpected categories: {unexpected_cat_pct:.1f}%")
            elif unexpected_cat_pct > 0:
                col_status = "WARNING" if col_status != "CRITICAL" else col_status
                issues.append(f"Unseen categories: {unexpected_cat_pct:.1f}%")

            if col_status == "CRITICAL" and overall_status != "CRITICAL":
                overall_status = "CRITICAL"
                alerts.append(f"CRITICAL: Column '{col}' failed quality checks: {'; '.join(issues)}")
            elif col_status == "WARNING" and overall_status == "HEALTHY":
                overall_status = "WARNING"
                alerts.append(f"WARNING: Column '{col}' flagged: {'; '.join(issues)}")

            col_metrics[col] = ColumnQualityMetric(
                column_name=col,
                dtype=str(series.dtype),
                missing_count=missing_cnt,
                missing_pct=missing_pct,
                invalid_range_count=invalid_range_cnt,
                invalid_range_pct=invalid_range_pct,
                unexpected_categories_count=unexpected_cat_cnt,
                unexpected_categories_pct=unexpected_cat_pct,
                status=col_status,
                issues=issues,
            )

        return DataQualityReport(
            total_records=n_rows,
            total_columns=n_cols,
            missing_columns=missing_cols,
            unexpected_columns=unexpected_cols,
            dtype_mismatches=dtype_mismatches,
            duplicate_rows_count=dup_rows,
            duplicate_ids_count=dup_ids,
            column_metrics=col_metrics,
            overall_status=overall_status,
            alerts=alerts,
        )
