"""
CreditRisk Analytics — Database Storage Layer
============================================

Implements a hardened storage architecture supporting SQLite and PostgreSQL
via SQLAlchemy ORM. Manages dataset provenance, experiment runs, model versions,
evaluation results, predictions, IFRS 9 ECL runs, drift audit logs, and operational alerts.

Guarantees transactional integrity with rollback on error, schema validation,
and structured logging while strictly excluding personal identifiable information (PII).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
    desc,
    inspect,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from src.config import DatabaseConfig, db_config, logger
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

Base = declarative_base()


# ==============================================================================
# 1. ORM TABLE MODELS
# ==============================================================================

class DatasetMetadata(Base):
    """Dataset registry, checksum integrity, and source licensing attribution."""
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(128), nullable=False, index=True)
    version = Column(String(32), default="1.0.0", nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    data_hash = Column(String(128), nullable=False)
    source_url = Column(String(512), nullable=True)
    license_type = Column(String(128), nullable=True)
    target_column = Column(String(64), nullable=True)
    features_json = Column(Text, nullable=True)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ExperimentMetadata(Base):
    """Machine learning experiment parameters, reproducibility seed, and split configurations."""
    __tablename__ = "experiment_metadata"

    experiment_id = Column(String(64), primary_key=True)
    description = Column(Text, nullable=True)
    random_seed = Column(Integer, default=42, nullable=False)
    train_size = Column(Integer, nullable=False)
    test_size = Column(Integer, nullable=False)
    parameters_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ModelVersion(Base):
    """Model registry for candidate and champion credit risk estimation engines."""
    __tablename__ = "model_versions"

    model_id = Column(String(64), primary_key=True)
    model_name = Column(String(64), nullable=False, index=True)
    version_tag = Column(String(32), nullable=False)
    algorithm = Column(String(64), nullable=False)
    is_champion = Column(Boolean, default=False, nullable=False, index=True)
    artifact_path = Column(String(512), nullable=True)
    hyperparameters_json = Column(Text, nullable=True)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    evaluations = relationship("EvaluationResult", back_populates="model", cascade="all, delete-orphan")


class EvaluationResult(Base):
    """Out-of-sample and cross-validation performance metrics linked to model versions."""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(64), ForeignKey("model_versions.model_id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_split = Column(String(32), default="test", nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=True)
    gini = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    log_loss_value = Column(Float, nullable=True)
    ks_statistic = Column(Float, nullable=True)
    precision_val = Column(Float, nullable=True)
    recall_val = Column(Float, nullable=True)
    f1_val = Column(Float, nullable=True)
    metrics_json = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    model = relationship("ModelVersion", back_populates="evaluations")


class PortfolioLoan(Base):
    """Raw loan account and borrower profile attributes (strictly anonymous, no PII)."""
    __tablename__ = "portfolio_loans"

    id = Column(String(64), primary_key=True)
    limit_bal = Column(Float, nullable=True)
    sex = Column(Integer, nullable=True)
    education = Column(Integer, nullable=True)
    marriage = Column(Integer, nullable=True)
    age = Column(Integer, nullable=True)
    pay_0 = Column(Integer, nullable=True)
    bill_amt1 = Column(Float, nullable=True)
    pay_amt1 = Column(Float, nullable=True)
    default_payment_next_month = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelPrediction(Base):
    """Scored predictions, calibrated default probabilities, and credit ratings."""
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(64), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    raw_pd = Column(Float, nullable=False)
    calibrated_pd = Column(Float, nullable=False)
    credit_score = Column(Float, nullable=False)
    rating_grade = Column(String(16), nullable=False)
    scored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ECLRunRecord(Base):
    """IFRS 9 Expected Credit Loss portfolio execution run metadata."""
    __tablename__ = "ecl_runs"

    run_id = Column(String(64), primary_key=True)
    run_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    portfolio_size = Column(Integer, nullable=False)
    total_ead = Column(Float, nullable=False)
    total_ecl_weighted = Column(Float, nullable=False)
    total_ecl_base = Column(Float, nullable=False)
    total_ecl_downturn = Column(Float, nullable=False)
    total_ecl_upturn = Column(Float, nullable=False)
    coverage_ratio_pct = Column(Float, nullable=False)
    stage_1_ecl = Column(Float, nullable=False)
    stage_2_ecl = Column(Float, nullable=False)
    stage_3_ecl = Column(Float, nullable=False)
    parameters_json = Column(Text, nullable=True)


class MonitoringAuditLog(Base):
    """Population stability and model performance decay compliance log."""
    __tablename__ = "monitoring_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    metric_type = Column(String(32), nullable=False)  # PSI, CSI, GINI_DECAY, KS_DECAY
    feature_or_score = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    traffic_light = Column(String(16), nullable=False)  # GREEN, AMBER, RED
    interpretation = Column(Text, nullable=True)


class AlertRecord(Base):
    """Operational risk, distribution shift, and critical performance degradation alerts."""
    __tablename__ = "alert_records"

    alert_id = Column(String(64), primary_key=True)
    severity = Column(String(16), nullable=False, index=True)  # INFO, WARNING, CRITICAL
    category = Column(String(64), nullable=False, index=True)  # FEATURE_DRIFT, PERFORMANCE_DECAY, DATA_QUALITY
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False, nullable=False, index=True)
    raised_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


# ==============================================================================
# 2. DATABASE MANAGER WITH TRANSACTION SAFETY
# ==============================================================================

class DatabaseManager:
    """
    Manages database lifecycle, connection pooling, transactions, and persistence.
    Supports SQLite file/in-memory storage and enterprise PostgreSQL without code alterations.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or db_config
        is_sqlite = "sqlite" in self.config.db_url.lower()
        self.engine = create_engine(
            self.config.db_url,
            echo=self.config.echo,
            connect_args={"check_same_thread": False} if is_sqlite else {},
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.init_db()

    def init_db(self) -> None:
        """Create database tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Initialized CreditRisk database tables at %s", self.config.db_url)

    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """Context manager providing ACID transaction boundary with automatic rollback."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Transaction rolled back due to error: %s", str(e))
            raise
        finally:
            session.close()

    # --------------------------------------------------------------------------
    # Dataset Metadata Operations
    # --------------------------------------------------------------------------
    def save_dataset_metadata(
        self,
        metadata: Union[DatasetMetadataSchema, Dict[str, Any]],
    ) -> DatasetMetadata:
        """Validate and persist dataset provenance and checksum metadata."""
        if isinstance(metadata, dict):
            validated = DatasetMetadataSchema(**metadata)
        else:
            validated = metadata

        with self.transaction() as session:
            record = DatasetMetadata(
                dataset_name=validated.dataset_name,
                version=validated.version,
                row_count=validated.row_count,
                column_count=validated.column_count,
                data_hash=validated.data_hash,
                source_url=validated.source_url,
                license_type=validated.license_type,
                target_column=validated.target_column,
                features_json=json.dumps(validated.features_list or []),
                ingested_at=validated.ingested_at or datetime.now(timezone.utc),
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            logger.info("Saved dataset metadata for '%s' (v%s, rows=%d)", record.dataset_name, record.version, record.row_count)
            # Detach for return outside session
            session.expunge(record)
            return record

    def get_dataset_metadata(self, dataset_name: Optional[str] = None) -> List[DatasetMetadata]:
        """Retrieve dataset metadata records ordered by ingestion date descending."""
        with self.transaction() as session:
            query = session.query(DatasetMetadata)
            if dataset_name:
                query = query.filter(DatasetMetadata.dataset_name == dataset_name)
            results = query.order_by(desc(DatasetMetadata.ingested_at)).all()
            for r in results:
                session.expunge(r)
            return results

    # --------------------------------------------------------------------------
    # Experiment Tracking Operations
    # --------------------------------------------------------------------------
    def save_experiment(
        self,
        experiment: Union[ExperimentMetadataSchema, Dict[str, Any]],
    ) -> ExperimentMetadata:
        """Validate and persist ML experiment run configuration."""
        if isinstance(experiment, dict):
            validated = ExperimentMetadataSchema(**experiment)
        else:
            validated = experiment

        with self.transaction() as session:
            record = ExperimentMetadata(
                experiment_id=validated.experiment_id,
                description=validated.description,
                random_seed=validated.random_seed,
                train_size=validated.train_size,
                test_size=validated.test_size,
                parameters_json=json.dumps(validated.parameters or {}),
                created_at=validated.created_at or datetime.now(timezone.utc),
            )
            record = session.merge(record)
            session.flush()
            session.refresh(record)
            logger.info("Saved experiment metadata '%s' (seed=%d)", record.experiment_id, record.random_seed)
            session.expunge(record)
            return record

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Retrieve experiment metadata by experiment ID."""
        with self.transaction() as session:
            record = session.query(ExperimentMetadata).filter(ExperimentMetadata.experiment_id == experiment_id).first()
            if record:
                session.expunge(record)
            return record

    # --------------------------------------------------------------------------
    # Model Registry & Versioning
    # --------------------------------------------------------------------------
    def save_model_version(
        self,
        model_data: Union[ModelVersionSchema, Dict[str, Any]],
    ) -> ModelVersion:
        """Register or update a model version in the registry."""
        if isinstance(model_data, dict):
            validated = ModelVersionSchema(**model_data)
        else:
            validated = model_data

        with self.transaction() as session:
            if validated.is_champion:
                # If setting this model as champion, unset existing champions for this algorithm
                session.query(ModelVersion).filter(
                    ModelVersion.algorithm == validated.algorithm,
                    ModelVersion.is_champion == True,  # noqa: E712
                ).update({"is_champion": False})

            record = ModelVersion(
                model_id=validated.model_id,
                model_name=validated.model_name,
                version_tag=validated.version_tag,
                algorithm=validated.algorithm,
                is_champion=validated.is_champion,
                artifact_path=validated.artifact_path,
                hyperparameters_json=json.dumps(validated.hyperparameters or {}),
                trained_at=validated.trained_at or datetime.now(timezone.utc),
            )
            record = session.merge(record)
            session.flush()
            session.refresh(record)
            logger.info("Registered model version '%s' (algo=%s, champion=%s)", record.model_id, record.algorithm, record.is_champion)
            session.expunge(record)
            return record

    def set_champion_model(self, model_id: str) -> None:
        """Promote a specific model version to champion status within its algorithm family."""
        with self.transaction() as session:
            target = session.query(ModelVersion).filter(ModelVersion.model_id == model_id).first()
            if not target:
                raise ValueError(f"Model ID '{model_id}' not found in registry.")
            session.query(ModelVersion).filter(
                ModelVersion.algorithm == target.algorithm,
                ModelVersion.is_champion == True,  # noqa: E712
            ).update({"is_champion": False})
            target.is_champion = True
            logger.info("Promoted model '%s' to champion status.", model_id)

    def get_champion_model(self, algorithm: Optional[str] = None) -> Optional[ModelVersion]:
        """Fetch current champion model, optionally filtered by algorithm."""
        with self.transaction() as session:
            query = session.query(ModelVersion).filter(ModelVersion.is_champion == True)  # noqa: E712
            if algorithm:
                query = query.filter(ModelVersion.algorithm == algorithm)
            record = query.first()
            if record:
                session.expunge(record)
            return record

    def get_model_history(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """Fetch registered model history ordered by trained timestamp descending."""
        with self.transaction() as session:
            query = session.query(ModelVersion)
            if model_name:
                query = query.filter(ModelVersion.model_name == model_name)
            results = query.order_by(desc(ModelVersion.trained_at)).all()
            for r in results:
                session.expunge(r)
            return results

    # --------------------------------------------------------------------------
    # Model Evaluation Metrics
    # --------------------------------------------------------------------------
    def save_evaluation_result(
        self,
        eval_data: Union[EvaluationResultSchema, Dict[str, Any]],
    ) -> EvaluationResult:
        """Persist validated out-of-sample or cross-validation evaluation results."""
        if isinstance(eval_data, dict):
            validated = EvaluationResultSchema(**eval_data)
        else:
            validated = eval_data

        with self.transaction() as session:
            record = EvaluationResult(
                model_id=validated.model_id,
                dataset_split=validated.dataset_split,
                roc_auc=validated.roc_auc,
                pr_auc=validated.pr_auc,
                gini=validated.gini,
                brier_score=validated.brier_score,
                log_loss_value=validated.log_loss_value,
                ks_statistic=validated.ks_statistic,
                precision_val=validated.precision_val,
                recall_val=validated.recall_val,
                f1_val=validated.f1_val,
                metrics_json=json.dumps(validated.metrics_json or {}),
                evaluated_at=validated.evaluated_at or datetime.now(timezone.utc),
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            logger.info("Saved evaluation for model '%s' on %s split (ROC-AUC=%.4f)", record.model_id, record.dataset_split, record.roc_auc)
            session.expunge(record)
            return record

    def get_evaluation_results(self, model_id: str) -> List[EvaluationResult]:
        """Fetch all evaluation records associated with a specific model ID."""
        with self.transaction() as session:
            results = session.query(EvaluationResult).filter(
                EvaluationResult.model_id == model_id
            ).order_by(desc(EvaluationResult.evaluated_at)).all()
            for r in results:
                session.expunge(r)
            return results

    # --------------------------------------------------------------------------
    # Operational Risk & Drift Alerts
    # --------------------------------------------------------------------------
    def save_alert(
        self,
        alert_data: Union[AlertRecordSchema, Dict[str, Any]],
    ) -> AlertRecord:
        """Record an operational, data quality, or drift alert."""
        if isinstance(alert_data, dict):
            validated = AlertRecordSchema(**alert_data)
        else:
            validated = alert_data

        with self.transaction() as session:
            record = AlertRecord(
                alert_id=validated.alert_id,
                severity=validated.severity,
                category=validated.category,
                metric_name=validated.metric_name,
                metric_value=validated.metric_value,
                threshold=validated.threshold,
                message=validated.message,
                acknowledged=validated.acknowledged,
                raised_at=validated.raised_at or datetime.now(timezone.utc),
            )
            record = session.merge(record)
            session.flush()
            session.refresh(record)
            logger.warning("Raised %s Alert [%s]: %s", record.severity, record.category, record.message)
            session.expunge(record)
            return record

    def get_alerts(
        self,
        severity: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[AlertRecord]:
        """Query alert records filtered by severity or acknowledgment status."""
        with self.transaction() as session:
            query = session.query(AlertRecord)
            if severity:
                query = query.filter(AlertRecord.severity == severity.upper())
            if unacknowledged_only:
                query = query.filter(AlertRecord.acknowledged == False)  # noqa: E712
            results = query.order_by(desc(AlertRecord.raised_at)).limit(limit).all()
            for r in results:
                session.expunge(r)
            return results

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by a risk analyst or operator."""
        with self.transaction() as session:
            alert = session.query(AlertRecord).filter(AlertRecord.alert_id == alert_id).first()
            if not alert:
                return False
            alert.acknowledged = True
            logger.info("Acknowledged alert '%s'", alert_id)
            return True

    # --------------------------------------------------------------------------
    # Existing Operations (Preserved & Hardened)
    # --------------------------------------------------------------------------
    def save_loans(self, df: pd.DataFrame, if_exists: str = "replace") -> int:
        """Persist or update loan portfolio records."""
        cols = ["id", "limit_bal", "sex", "education", "marriage", "age", "pay_0", "bill_amt1", "pay_amt1", "default_payment_next_month"]
        present_cols = [c for c in cols if c in df.columns]
        sub_df = df[present_cols].copy()

        sub_df.to_sql(
            PortfolioLoan.__tablename__,
            con=self.engine,
            if_exists=if_exists,
            index=False,
            chunksize=5000,
        )
        logger.info("Persisted %d loan records into %s", len(sub_df), PortfolioLoan.__tablename__)
        return len(sub_df)

    def save_predictions(self, df: pd.DataFrame, model_name: str) -> int:
        """Batch insert model predictions, scores, and rating grades with range validation."""
        records = []
        now = datetime.now(timezone.utc)
        for _, row in df.iterrows():
            raw_pd = float(row.get("raw_pd", row.get("calibrated_pd", 0.0)))
            calibrated_pd = float(row.get("calibrated_pd", 0.0))
            score = float(row.get("credit_score", 600.0))
            grade = str(row.get("rating_grade", "BBB"))
            # Validate bounds
            raw_pd = max(0.0, min(1.0, raw_pd))
            calibrated_pd = max(0.0, min(1.0, calibrated_pd))
            score = max(0.0, min(1000.0, score))

            records.append({
                "account_id": str(row.get("id", "0")),
                "model_name": model_name,
                "raw_pd": raw_pd,
                "calibrated_pd": calibrated_pd,
                "credit_score": score,
                "rating_grade": grade,
                "scored_at": now,
            })

        pred_df = pd.DataFrame(records)
        pred_df.to_sql(
            ModelPrediction.__tablename__,
            con=self.engine,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        logger.info("Persisted %d prediction records for model '%s'", len(pred_df), model_name)
        return len(pred_df)

    def get_predictions(self, model_name: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent prediction records."""
        query = f"SELECT * FROM {ModelPrediction.__tablename__}"
        if model_name:
            query += f" WHERE model_name = '{model_name}'"
        query += f" ORDER BY scored_at DESC LIMIT {limit}"
        return pd.read_sql(query, con=self.engine)

    def save_ecl_run(
        self,
        run_id: str,
        total_accounts: int,
        total_ead: float,
        total_ecl_weighted: float,
        total_ecl_base: float,
        total_ecl_downturn: float,
        total_ecl_upturn: float,
        coverage_ratio_pct: float,
        stage_1_ecl: float,
        stage_2_ecl: float,
        stage_3_ecl: float,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log full ECL run metrics and scenario parameter snapshot with transaction safety."""
        # Validate schema
        ECLRunSchema(
            run_id=run_id,
            portfolio_size=total_accounts,
            total_ead=total_ead,
            total_ecl_weighted=total_ecl_weighted,
            total_ecl_base=total_ecl_base,
            total_ecl_downturn=total_ecl_downturn,
            total_ecl_upturn=total_ecl_upturn,
            coverage_ratio_pct=coverage_ratio_pct,
            stage_1_ecl=stage_1_ecl,
            stage_2_ecl=stage_2_ecl,
            stage_3_ecl=stage_3_ecl,
            parameters=parameters or {},
        )

        with self.transaction() as session:
            record = ECLRunRecord(
                run_id=run_id,
                portfolio_size=total_accounts,
                total_ead=total_ead,
                total_ecl_weighted=total_ecl_weighted,
                total_ecl_base=total_ecl_base,
                total_ecl_downturn=total_ecl_downturn,
                total_ecl_upturn=total_ecl_upturn,
                coverage_ratio_pct=coverage_ratio_pct,
                stage_1_ecl=stage_1_ecl,
                stage_2_ecl=stage_2_ecl,
                stage_3_ecl=stage_3_ecl,
                parameters_json=json.dumps(parameters or {}),
            )
            session.merge(record)
            logger.info("Recorded ECL Run: %s (Weighted ECL: $%.2f)", run_id, total_ecl_weighted)

    def log_monitoring_event(
        self,
        metric_type: str,
        feature_or_score: str,
        metric_value: float,
        traffic_light: str,
        interpretation: str,
    ) -> None:
        """Record model drift or stability event in the compliance audit trail with transaction safety."""
        with self.transaction() as session:
            log_entry = MonitoringAuditLog(
                metric_type=metric_type,
                feature_or_score=feature_or_score,
                metric_value=float(metric_value),
                traffic_light=traffic_light,
                interpretation=interpretation,
            )
            session.add(log_entry)
            logger.info("Logged monitoring event: %s on %s = %.4f [%s]", metric_type, feature_or_score, float(metric_value), traffic_light)

    def get_latest_ecl_runs(self, limit: int = 10) -> pd.DataFrame:
        """Fetch historical ECL valuation runs."""
        query = f"SELECT * FROM {ECLRunRecord.__tablename__} ORDER BY run_timestamp DESC LIMIT {limit}"
        return pd.read_sql(query, con=self.engine)

    def get_monitoring_audit_logs(self, limit: int = 50) -> pd.DataFrame:
        """Fetch recent monitoring compliance records."""
        query = f"SELECT * FROM {MonitoringAuditLog.__tablename__} ORDER BY audit_timestamp DESC LIMIT {limit}"
        return pd.read_sql(query, con=self.engine)


def get_db_manager() -> DatabaseManager:
    """Singleton-style database manager access."""
    return DatabaseManager()
