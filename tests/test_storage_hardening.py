"""
CreditRisk Analytics — Hardened Storage & Database Layer Test Suite
==================================================================

Comprehensive tests for:
- Database schema validation via Pydantic models
- ACID transaction boundaries and automatic rollback safety
- Dataset metadata registry & provenance tracking
- Machine learning experiment tracking and hyperparameter persistence
- Model versioning, champion promotion, and history queries
- Model evaluation metric storage and foreign key constraints
- Prediction storage, bounds clamping, and PII avoidance
- Operational risk, drift, and performance alert logging and acknowledgment
- Environment variable configuration parsing (SQLite & PostgreSQL)
- Alembic database migration execution on synthetic database

All tests use in-memory or temporary local SQLite fixtures — zero paid services or external APIs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import inspect

from src.config import DatabaseConfig
from src.storage.database import (
    AlertRecord,
    Base,
    DatabaseManager,
    DatasetMetadata,
    ECLRunRecord,
    EvaluationResult,
    ExperimentMetadata,
    ModelPrediction,
    ModelVersion,
    MonitoringAuditLog,
    PortfolioLoan,
)
from src.storage.schemas import (
    AlertRecordSchema,
    DatasetMetadataSchema,
    ECLRunSchema,
    EvaluationResultSchema,
    ExperimentMetadataSchema,
    ModelVersionSchema,
    MonitoringResultSchema,
    PredictionRecordSchema,
)


@pytest.fixture
def memory_db(tmp_path):
    """Provide a fresh isolated SQLite database instance for each test."""
    db_file = tmp_path / "hardened_test.db"
    cfg = DatabaseConfig(db_url=f"sqlite:///{db_file.as_posix()}", echo=False)
    manager = DatabaseManager(config=cfg)
    return manager


# ==============================================================================
# 1. SCHEMA VALIDATION TESTS (PYDANTIC)
# ==============================================================================

def test_dataset_metadata_schema_valid():
    """Valid dataset metadata passes schema validation."""
    data = {
        "dataset_name": "UCI_Credit_Card",
        "version": "1.0.0",
        "row_count": 30000,
        "column_count": 25,
        "data_hash": "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
        "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "license_type": "CC BY 4.0",
        "target_column": "default_payment_next_month",
        "features_list": ["limit_bal", "age", "pay_0"],
    }
    validated = DatasetMetadataSchema(**data)
    assert validated.dataset_name == "UCI_Credit_Card"
    assert validated.row_count == 30000


def test_dataset_metadata_schema_invalid():
    """Negative row count or missing data hash raises ValidationError."""
    with pytest.raises(ValidationError):
        DatasetMetadataSchema(
            dataset_name="UCI_Credit_Card",
            version="1.0.0",
            row_count=-10,  # Invalid: negative
            column_count=25,
            data_hash="abc",
        )


def test_model_version_schema_valid():
    """Valid model version metadata passes validation."""
    schema = ModelVersionSchema(
        model_id="lr_v1_0",
        model_name="Logistic_Regression",
        version_tag="v1.0.0",
        algorithm="LogisticRegression",
        is_champion=True,
        hyperparameters={"C": 0.1, "max_iter": 1000},
    )
    assert schema.algorithm == "LogisticRegression"
    assert schema.is_champion is True


def test_evaluation_result_schema_bounds():
    """Evaluation metrics out of probabilistic bounds raise ValidationError."""
    # Valid
    valid = EvaluationResultSchema(
        model_id="lr_v1",
        dataset_split="test",
        roc_auc=0.785,
        gini=0.570,
        brier_score=0.135,
        log_loss_value=0.432,
    )
    assert valid.roc_auc == 0.785

    # Invalid ROC-AUC > 1.0
    with pytest.raises(ValidationError):
        EvaluationResultSchema(
            model_id="lr_v1",
            dataset_split="test",
            roc_auc=1.25,  # Invalid
        )

    # Invalid Gini < -1.0
    with pytest.raises(ValidationError):
        EvaluationResultSchema(
            model_id="lr_v1",
            dataset_split="test",
            roc_auc=0.75,
            gini=-1.5,  # Invalid
        )


def test_prediction_record_schema_bounds():
    """Borrower predictions must respect probability [0, 1] and score ranges."""
    valid = PredictionRecordSchema(
        account_id="ACC_001",
        model_name="LightGBM_Classifier",
        raw_pd=0.154,
        calibrated_pd=0.142,
        credit_score=685.0,
        rating_grade="BBB",
    )
    assert valid.credit_score == 685.0

    # Invalid calibrated PD > 1.0
    with pytest.raises(ValidationError):
        PredictionRecordSchema(
            account_id="ACC_001",
            model_name="TestModel",
            raw_pd=0.2,
            calibrated_pd=1.5,  # Invalid
            credit_score=600.0,
            rating_grade="BBB",
        )


def test_alert_record_schema_severity():
    """Alert severity must be one of INFO, WARNING, or CRITICAL."""
    valid = AlertRecordSchema(
        alert_id="ALT_001",
        severity="WARNING",
        category="FEATURE_DRIFT",
        metric_name="PSI_limit_bal",
        metric_value=0.18,
        threshold=0.10,
        message="Moderate population shift detected on limit_bal.",
    )
    assert valid.severity == "WARNING"

    with pytest.raises(ValidationError):
        AlertRecordSchema(
            alert_id="ALT_002",
            severity="FATAL",  # Invalid literal
            category="FEATURE_DRIFT",
            metric_name="PSI",
            metric_value=0.5,
            threshold=0.25,
            message="Invalid severity test.",
        )


# ==============================================================================
# 2. ACID TRANSACTION INTEGRITY & ROLLBACK SAFETY
# ==============================================================================

def test_transaction_commit_success(memory_db):
    """Successful transaction persists changes."""
    with memory_db.transaction() as session:
        session.add(DatasetMetadata(
            dataset_name="TestDS",
            version="1.0",
            row_count=100,
            column_count=5,
            data_hash="dummy_hash_12345678",
        ))

    items = memory_db.get_dataset_metadata("TestDS")
    assert len(items) == 1
    assert items[0].row_count == 100


def test_transaction_rollback_on_exception(memory_db):
    """Exception inside transaction block triggers automatic rollback."""
    with pytest.raises(RuntimeError, match="Simulated database failure"):
        with memory_db.transaction() as session:
            session.add(DatasetMetadata(
                dataset_name="ShouldBeRolledBack",
                version="1.0",
                row_count=500,
                column_count=10,
                data_hash="rollback_hash_12345678",
            ))
            # Force an unexpected failure mid-transaction
            raise RuntimeError("Simulated database failure")

    # Assert that nothing was written
    items = memory_db.get_dataset_metadata("ShouldBeRolledBack")
    assert len(items) == 0


# ==============================================================================
# 3. DATASET & EXPERIMENT METADATA PERSISTENCE
# ==============================================================================

def test_dataset_metadata_crud(memory_db):
    """Save and query dataset provenance records."""
    rec1 = memory_db.save_dataset_metadata({
        "dataset_name": "Credit_Portfolio_2024",
        "version": "1.0.0",
        "row_count": 25000,
        "column_count": 24,
        "data_hash": "sha256_hash_value_abcdef123456",
        "source_url": "https://example.com/data",
        "license_type": "Proprietary",
        "target_column": "default_flag",
        "features_list": ["feat1", "feat2", "feat3"],
    })
    assert rec1.id is not None

    records = memory_db.get_dataset_metadata("Credit_Portfolio_2024")
    assert len(records) == 1
    assert records[0].row_count == 25000
    features = json.loads(records[0].features_json)
    assert features == ["feat1", "feat2", "feat3"]


def test_experiment_metadata_crud(memory_db):
    """Save and retrieve machine learning experiment tracking metadata."""
    exp = memory_db.save_experiment({
        "experiment_id": "EXP_2026_001",
        "description": "Baseline logistic regression vs gradient boosting",
        "random_seed": 42,
        "train_size": 24000,
        "test_size": 6000,
        "parameters": {"cv_folds": 5, "stratified": True, "scaling": "RobustScaler"},
    })
    assert exp.experiment_id == "EXP_2026_001"

    retrieved = memory_db.get_experiment("EXP_2026_001")
    assert retrieved is not None
    assert retrieved.random_seed == 42
    params = json.loads(retrieved.parameters_json)
    assert params["scaling"] == "RobustScaler"


# ==============================================================================
# 4. MODEL REGISTRY, CHAMPION PROMOTION & EVALUATION RESULTS
# ==============================================================================

def test_model_versioning_and_champion_promotion(memory_db):
    """Test registering models and promoting champion models within algorithm family."""
    # Register model 1 (initial champion)
    m1 = memory_db.save_model_version({
        "model_id": "lr_v1",
        "model_name": "Logistic_Regression",
        "version_tag": "1.0.0",
        "algorithm": "LogisticRegression",
        "is_champion": True,
        "hyperparameters": {"C": 0.01},
    })
    assert m1.is_champion is True

    champ = memory_db.get_champion_model(algorithm="LogisticRegression")
    assert champ is not None
    assert champ.model_id == "lr_v1"

    # Register model 2 and promote to champion (should unseat model 1)
    m2 = memory_db.save_model_version({
        "model_id": "lr_v2",
        "model_name": "Logistic_Regression",
        "version_tag": "2.0.0",
        "algorithm": "LogisticRegression",
        "is_champion": True,
        "hyperparameters": {"C": 1.0},
    })
    assert m2.is_champion is True

    # Check that lr_v2 is the new champion
    champ_updated = memory_db.get_champion_model(algorithm="LogisticRegression")
    assert champ_updated.model_id == "lr_v2"

    # Explicit promotion method
    memory_db.set_champion_model("lr_v1")
    champ_reverted = memory_db.get_champion_model(algorithm="LogisticRegression")
    assert champ_reverted.model_id == "lr_v1"

    # Query history
    history = memory_db.get_model_history("Logistic_Regression")
    assert len(history) == 2


def test_evaluation_results_storage(memory_db):
    """Store and retrieve out-of-sample evaluation metrics linked to model."""
    memory_db.save_model_version({
        "model_id": "rf_v1",
        "model_name": "Random_Forest",
        "version_tag": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "is_champion": False,
    })

    eval_rec = memory_db.save_evaluation_result({
        "model_id": "rf_v1",
        "dataset_split": "test",
        "roc_auc": 0.778,
        "pr_auc": 0.542,
        "gini": 0.556,
        "brier_score": 0.138,
        "log_loss_value": 0.445,
        "ks_statistic": 0.435,
        "precision_val": 0.672,
        "recall_val": 0.354,
        "f1_val": 0.464,
        "metrics_json": {"confusion_matrix": [[4300, 380], [850, 470]]},
    })
    assert eval_rec.id is not None

    results = memory_db.get_evaluation_results("rf_v1")
    assert len(results) == 1
    assert results[0].roc_auc == 0.778
    cm = json.loads(results[0].metrics_json)["confusion_matrix"]
    assert cm[0][0] == 4300


# ==============================================================================
# 5. PREDICTION STORAGE & CLAMPING TESTS
# ==============================================================================

def test_predictions_persistence_and_clamping(memory_db):
    """Verify batch predictions are clamped to valid ranges and queryable."""
    df_preds = pd.DataFrame([
        {"id": "ACC_101", "raw_pd": -0.05, "calibrated_pd": 0.02, "credit_score": 750, "rating_grade": "AA"},
        {"id": "ACC_102", "raw_pd": 1.25, "calibrated_pd": 0.85, "credit_score": 420, "rating_grade": "CCC"},
    ])

    count = memory_db.save_predictions(df_preds, model_name="Prod_PD_Model")
    assert count == 2

    stored = memory_db.get_predictions("Prod_PD_Model")
    assert len(stored) == 2
    # Ensure -0.05 was clamped to 0.0
    acc_101 = stored[stored["account_id"] == "ACC_101"].iloc[0]
    assert acc_101["raw_pd"] == 0.0
    # Ensure 1.25 was clamped to 1.0
    acc_102 = stored[stored["account_id"] == "ACC_102"].iloc[0]
    assert acc_102["raw_pd"] == 1.0


# ==============================================================================
# 6. OPERATIONAL RISK ALERTS & ACKNOWLEDGMENT
# ==============================================================================

def test_alerts_logging_and_acknowledgment(memory_db):
    """Alerts can be logged, filtered by severity, and acknowledged."""
    memory_db.save_alert({
        "alert_id": "ALT_DRIFT_01",
        "severity": "WARNING",
        "category": "FEATURE_DRIFT",
        "metric_name": "PSI_age",
        "metric_value": 0.145,
        "threshold": 0.10,
        "message": "Moderate population shift on borrower age.",
    })
    memory_db.save_alert({
        "alert_id": "ALT_DECAY_01",
        "severity": "CRITICAL",
        "category": "PERFORMANCE_DECAY",
        "metric_name": "ROC_AUC_DROP",
        "metric_value": 0.640,
        "threshold": 0.70,
        "message": "OOS ROC-AUC dropped below minimum acceptable barrier.",
    })

    # Retrieve all
    all_alerts = memory_db.get_alerts()
    assert len(all_alerts) == 2

    # Filter critical
    crit_alerts = memory_db.get_alerts(severity="CRITICAL")
    assert len(crit_alerts) == 1
    assert crit_alerts[0].alert_id == "ALT_DECAY_01"

    # Filter unacknowledged
    unack = memory_db.get_alerts(unacknowledged_only=True)
    assert len(unack) == 2

    # Acknowledge one
    success = memory_db.acknowledge_alert("ALT_DECAY_01")
    assert success is True

    # Check unacknowledged now has 1
    unack_after = memory_db.get_alerts(unacknowledged_only=True)
    assert len(unack_after) == 1
    assert unack_after[0].alert_id == "ALT_DRIFT_01"


# ==============================================================================
# 7. PRIVACY & ANTI-PII CONSTRAINTS VERIFICATION
# ==============================================================================

def test_privacy_and_anti_pii_assurance(memory_db):
    """
    Ensure no PII attributes (names, emails, SSN, street addresses, phone numbers)
    exist in the database schema tables.
    """
    inspector = inspect(memory_db.engine)
    pii_keywords = ["name", "email", "ssn", "phone", "address", "street", "zip", "dob"]

    tables = inspector.get_table_names()
    assert "portfolio_loans" in tables
    assert "model_predictions" in tables

    for table in tables:
        columns = [col["name"].lower() for col in inspector.get_columns(table)]
        for col_name in columns:
            # dataset_name, model_name, and metric_name are model metadata, not human names
            if col_name in ["dataset_name", "model_name", "metric_name"]:
                continue
            for pii in pii_keywords:
                assert pii not in col_name, f"Potential PII column '{col_name}' detected in table '{table}'"


# ==============================================================================
# 8. ENVIRONMENT VARIABLE CONFIGURATION PARSING
# ==============================================================================

def test_environment_variable_database_config():
    """Verify DatabaseConfig properly resolves SQLite and PostgreSQL connection strings."""
    # Custom SQLite
    with patch.dict(os.environ, {"CREDIT_RISK_DB_URL": "sqlite:///custom_path.db"}):
        cfg = DatabaseConfig()
        assert cfg.db_url == "sqlite:///custom_path.db"

    # PostgreSQL configuration string parsing
    pg_url = "postgresql+psycopg2://risk_analyst:SecretPass123@db.internal:5432/credit_risk"
    with patch.dict(os.environ, {"CREDIT_RISK_DB_URL": pg_url}):
        cfg_pg = DatabaseConfig()
        assert cfg_pg.db_url == pg_url
        assert cfg_pg.db_url.startswith("postgresql")


# ==============================================================================
# 9. IFRS 9 ECL RUN PERSISTENCE & SCHEMA VALIDATION
# ==============================================================================

def test_ecl_run_schema_and_persistence(memory_db):
    """Verify IFRS 9 ECL execution records with schema validation."""
    memory_db.save_ecl_run(
        run_id="ECL_STRESS_2026_Q1",
        total_accounts=5000,
        total_ead=12500000.0,
        total_ecl_weighted=680000.0,
        total_ecl_base=590000.0,
        total_ecl_downturn=890000.0,
        total_ecl_upturn=470000.0,
        coverage_ratio_pct=5.44,
        stage_1_ecl=140000.0,
        stage_2_ecl=320000.0,
        stage_3_ecl=220000.0,
        parameters={"lgd_base": 0.45, "sicr_multiplier": 2.5},
    )

    df_runs = memory_db.get_latest_ecl_runs(limit=5)
    assert len(df_runs) == 1
    assert df_runs.iloc[0]["run_id"] == "ECL_STRESS_2026_Q1"
    assert df_runs.iloc[0]["total_ead"] == 12500000.0


# ==============================================================================
# 10. DATABASE MIGRATIONS (ALEMBIC) PROGRAMMATIC VERIFICATION
# ==============================================================================

def test_alembic_migration_execution(tmp_path):
    """Verify Alembic migration upgrade runs cleanly on an isolated test SQLite DB."""
    from alembic import command
    from alembic.config import Config

    test_db = tmp_path / "alembic_target.db"
    test_url = f"sqlite:///{test_db.as_posix()}"

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", test_url)

    with patch.dict(os.environ, {"CREDIT_RISK_DB_URL": test_url}):
        command.upgrade(alembic_cfg, "head")

    assert test_db.exists()

