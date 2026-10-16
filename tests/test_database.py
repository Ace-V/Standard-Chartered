"""
Tests for Database Storage Layer (SQLAlchemy SQLite & PostgreSQL).
"""

import pandas as pd
import pytest

from src.config import DatabaseConfig
from src.storage.database import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_credit_risk.db"
    cfg = DatabaseConfig(db_url=f"sqlite:///{db_file.as_posix()}", echo=False)
    manager = DatabaseManager(config=cfg)
    return manager


def test_database_initialization(temp_db):
    assert temp_db.engine is not None


def test_database_save_and_retrieve_ecl_run(temp_db):
    temp_db.save_ecl_run(
        run_id="TEST_RUN_001",
        total_accounts=1000,
        total_ead=5000000.0,
        total_ecl_weighted=250000.0,
        total_ecl_base=220000.0,
        total_ecl_downturn=310000.0,
        total_ecl_upturn=180000.0,
        coverage_ratio_pct=5.0,
        stage_1_ecl=50000.0,
        stage_2_ecl=120000.0,
        stage_3_ecl=80000.0,
        parameters={"test": True},
    )

    df_runs = temp_db.get_latest_ecl_runs(limit=5)
    assert len(df_runs) == 1
    assert df_runs.iloc[0]["run_id"] == "TEST_RUN_001"
    assert df_runs.iloc[0]["portfolio_size"] == 1000


def test_database_log_monitoring_event(temp_db):
    temp_db.log_monitoring_event(
        metric_type="PSI",
        feature_or_score="credit_score",
        metric_value=0.045,
        traffic_light="GREEN",
        interpretation="Stable distribution",
    )

    logs = temp_db.get_monitoring_audit_logs(limit=5)
    assert len(logs) == 1
    assert logs.iloc[0]["traffic_light"] == "GREEN"
    assert logs.iloc[0]["metric_value"] == 0.045
