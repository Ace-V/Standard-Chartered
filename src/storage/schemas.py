"""
CreditRisk Analytics — Data Storage Pydantic Validation Schemas
==============================================================

Provides strict data validation for persistent database records:
- Dataset metadata and provenance
- Machine learning experiment tracking
- Model versioning and registry
- Performance evaluation metrics
- Scoring and inference outputs
- Model stability & drift monitoring
- Operational risk alerts and audit trails

Prevents ingestion of corrupted records, enforces range constraints,
and guarantees no PII is inadvertently accepted into storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetMetadataSchema(BaseModel):
    """Validation schema for dataset registry and provenance tracking."""
    model_config = ConfigDict(from_attributes=True)

    dataset_name: str = Field(..., min_length=1, max_length=128, description="Dataset identifier")
    version: str = Field(default="1.0.0", max_length=32, description="Semantic version")
    row_count: int = Field(..., ge=0, description="Total number of observations")
    column_count: int = Field(..., ge=1, description="Total number of features/attributes")
    data_hash: str = Field(..., min_length=8, max_length=128, description="SHA-256 data integrity hash")
    source_url: Optional[str] = Field(default=None, max_length=512)
    license_type: Optional[str] = Field(default=None, max_length=128)
    target_column: Optional[str] = Field(default=None, max_length=64)
    features_list: Optional[List[str]] = Field(default=None)
    ingested_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentMetadataSchema(BaseModel):
    """Validation schema for ML experiment tracking and reproducibility."""
    model_config = ConfigDict(from_attributes=True)

    experiment_id: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = Field(default=None, max_length=1000)
    random_seed: int = Field(default=42)
    train_size: int = Field(..., ge=0)
    test_size: int = Field(..., ge=0)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelVersionSchema(BaseModel):
    """Validation schema for registered credit risk model versions."""
    model_config = ConfigDict(from_attributes=True)

    model_id: str = Field(..., min_length=1, max_length=64)
    model_name: str = Field(..., min_length=1, max_length=64)
    version_tag: str = Field(..., min_length=1, max_length=32)
    algorithm: str = Field(..., min_length=1, max_length=64)
    is_champion: bool = Field(default=False)
    artifact_path: Optional[str] = Field(default=None, max_length=512)
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    trained_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationResultSchema(BaseModel):
    """Validation schema for out-of-sample and cross-validation performance metrics."""
    model_config = ConfigDict(from_attributes=True)

    model_id: str = Field(..., min_length=1, max_length=64)
    dataset_split: Literal["train", "validation", "test", "oot"] = Field(default="test")
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    pr_auc: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gini: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    brier_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    log_loss_value: Optional[float] = Field(default=None, ge=0.0)
    ks_statistic: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    precision_val: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recall_val: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    f1_val: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metrics_json: Optional[Dict[str, Any]] = Field(default_factory=dict)
    evaluated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictionRecordSchema(BaseModel):
    """Validation schema for individual borrower scoring outputs."""
    model_config = ConfigDict(from_attributes=True)

    account_id: str = Field(..., min_length=1, max_length=64)
    model_name: str = Field(..., min_length=1, max_length=64)
    raw_pd: float = Field(..., ge=0.0, le=1.0)
    calibrated_pd: float = Field(..., ge=0.0, le=1.0)
    credit_score: float = Field(..., ge=0.0, le=1000.0)
    rating_grade: str = Field(..., min_length=1, max_length=16)
    scored_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class MonitoringResultSchema(BaseModel):
    """Validation schema for population stability, feature drift, and data quality logs."""
    model_config = ConfigDict(from_attributes=True)

    metric_type: str = Field(..., min_length=1, max_length=32)
    feature_or_score: str = Field(..., min_length=1, max_length=64)
    metric_value: float = Field(...)
    traffic_light: Literal["GREEN", "AMBER", "RED"] = Field(...)
    interpretation: Optional[str] = Field(default=None)
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    monitored_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertRecordSchema(BaseModel):
    """Validation schema for real-time risk, drift, and data quality alerts."""
    model_config = ConfigDict(from_attributes=True)

    alert_id: str = Field(..., min_length=1, max_length=64)
    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(...)
    category: str = Field(..., min_length=1, max_length=64)
    metric_name: str = Field(..., min_length=1, max_length=64)
    metric_value: float = Field(...)
    threshold: float = Field(...)
    message: str = Field(..., min_length=1, max_length=1000)
    acknowledged: bool = Field(default=False)
    raised_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class ECLRunSchema(BaseModel):
    """Validation schema for IFRS 9 ECL runs."""
    model_config = ConfigDict(from_attributes=True)

    run_id: str = Field(..., min_length=1, max_length=64)
    portfolio_size: int = Field(..., ge=0)
    total_ead: float = Field(..., ge=0.0)
    total_ecl_weighted: float = Field(..., ge=0.0)
    total_ecl_base: float = Field(..., ge=0.0)
    total_ecl_downturn: float = Field(..., ge=0.0)
    total_ecl_upturn: float = Field(..., ge=0.0)
    coverage_ratio_pct: float = Field(..., ge=0.0)
    stage_1_ecl: float = Field(..., ge=0.0)
    stage_2_ecl: float = Field(..., ge=0.0)
    stage_3_ecl: float = Field(..., ge=0.0)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
