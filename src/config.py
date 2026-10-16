"""
CreditRisk Analytics — Configuration Management & System Settings
================================================================

Centralized configuration adhering to 12-factor application design,
supporting reproducible risk analytics, environment variable overrides,
and multi-asset credit risk extensibility (Retail / Consumer & Corporate).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = BASE_DIR / "src"
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models" / "saved"
REPORTS_DIR: Path = BASE_DIR / "reports"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure all essential directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. LOGGING CONFIGURATION
# ==============================================================================
def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure structured, production-grade logging."""
    if log_file is None:
        log_file = LOGS_DIR / "credit_risk.log"

    logger = logging.getLogger("CreditRisk")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on re-initialization
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except OSError:
            # Fallback if file system restricts writing
            pass

    return logger


logger = setup_logging()


# ==============================================================================
# 3. DATA INGESTION & ATTRIBUTION SETTINGS
# ==============================================================================
@dataclass(frozen=True)
class DatasetAttribution:
    """Metadata, provenance, and legal licensing for ingested datasets."""
    dataset_name: str
    citation: str
    source_url: str
    license_type: str
    disclaimer: str


UCI_DATASET_ATTRIBUTION = DatasetAttribution(
    dataset_name="Default of Credit Card Clients Dataset",
    citation="Yeh, I-Cheng, and Che-hui Lien. (2009). The comparisons of data mining techniques "
             "for the predictive accuracy of probability of default of credit card clients. "
             "Expert Systems with Applications, 36(2), 2473-2480.",
    source_url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    license_type="Creative Commons Attribution 4.0 International (CC BY 4.0)",
    disclaimer="PROTOTYPE NOTICE: This platform utilizes retail consumer credit card delinquency data "
               "to demonstrate quantitative modelling methodologies (PD estimation, scorecard scaling, "
               "IFRS 9 Expected Credit Loss staging, and population stability monitoring). "
               "This configuration represents a consumer credit portfolio and is NOT a validated corporate "
               "credit rating system. A corporate distress adapter is provided for institutional balance sheet assets."
)


@dataclass
class IngestionConfig:
    """Data source locations and file caching configuration."""
    uci_zip_url: str = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    raw_excel_filename: str = "default of credit card clients.xls"
    raw_zip_filename: str = "default_of_credit_card_clients.zip"
    processed_csv_filename: str = "uci_credit_card_default.csv"
    download_timeout_seconds: int = 45


# ==============================================================================
# 4. DATA VALIDATION & COLUMN SPECIFICATIONS
# ==============================================================================
@dataclass
class ValidationConfig:
    """Boundary conditions and business logic assertions for data quality."""
    id_col: str = "id"
    target_col: str = "default_payment_next_month"

    min_age: int = 18
    max_age: int = 100
    min_limit_bal: float = 1000.0

    valid_education_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    valid_marriage_levels: List[int] = field(default_factory=lambda: [1, 2, 3])
    valid_sex_levels: List[int] = field(default_factory=lambda: [1, 2])

    # Payment status range: -2 (no consumption), -1 (duly paid), 0 (revolving), 1..8 (months delay)
    pay_status_min: int = -2
    pay_status_max: int = 8


# Column mapping from raw UCI table to standardized names
RAW_UCI_COLUMN_MAPPING: Dict[str, str] = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_1": "pay_0",  # In some versions of the UCI dataset
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default.payment.next.month": "default_payment_next_month",
    "default payment next month": "default_payment_next_month",
}


# ==============================================================================
# 5. MODELLING & CALIBRATION CONFIGURATION
# ==============================================================================
@dataclass
class ModelConfig:
    """Model training hyperparameters, cross-validation, and splitting specs."""
    target_column: str = "default_payment_next_month"
    id_column: str = "id"
    test_size: float = 0.20
    random_state: int = 42
    n_splits: int = 5
    stratify: bool = True

    # Calibration parameters
    calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
    calibration_cv: str = "prefit"

    # Evaluation confidence interval bootstrapping
    n_bootstrap: int = 1000
    bootstrap_ci_level: float = 0.95


# ==============================================================================
# 6. RISK SEGMENTATION & SCORECARD CONFIGURATION
# ==============================================================================
@dataclass
class ScorecardConfig:
    """Credit bureau-style scorecard scaling parameters."""
    base_score: float = 600.0
    target_odds: float = 50.0  # Good:Bad odds at base_score
    pdo: float = 20.0          # Points to Double Odds
    min_score: int = 300
    max_score: int = 850


# Master Rating Scale mapping PD thresholds to Credit Grades
RATING_SCALE: List[Dict[str, Any]] = [
    {"grade": "AAA", "pd_min": 0.0000, "pd_max": 0.0050, "description": "Prime Investment Grade (Negligible Risk)"},
    {"grade": "AA",  "pd_min": 0.0050, "pd_max": 0.0150, "description": "High Quality (Very Low Risk)"},
    {"grade": "A",   "pd_min": 0.0150, "pd_max": 0.0350, "description": "Upper Medium Grade (Low Risk)"},
    {"grade": "BBB", "pd_min": 0.0350, "pd_max": 0.0750, "description": "Lower Medium Grade (Moderate Risk)"},
    {"grade": "BB",  "pd_min": 0.0750, "pd_max": 0.1500, "description": "Speculative Grade (Substantial Risk)"},
    {"grade": "B",   "pd_min": 0.1500, "pd_max": 0.3000, "description": "Highly Speculative (High Default Risk)"},
    {"grade": "CCC", "pd_min": 0.3000, "pd_max": 1.0000, "description": "Substandard / Impaired / Default Warning"},
]


# ==============================================================================
# 7. IFRS 9 EXPECTED CREDIT LOSS (ECL) CONFIGURATION
# ==============================================================================
@dataclass
class ECLConfig:
    """IFRS 9 Impairment and Expected Credit Loss calculation rules."""
    baseline_lgd: float = 0.45       # Loss Given Default (unsecured retail standard)
    ccf: float = 0.75                # Credit Conversion Factor for undrawn facility
    sicr_pd_multiplier: float = 2.5  # Relative PD increase threshold triggering Stage 2
    sicr_dpd_threshold: int = 30     # 30+ Days Past Due triggers Stage 2 SICR
    stage_3_dpd_threshold: int = 90  # 90+ DPD triggers Stage 3 Default
    lifetime_horizon_years: int = 3  # Horizon for lifetime ECL in revolving facility

    # Forward-looking macroeconomic scenarios (IFRS 9 compliance)
    scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Base": {
            "weight": 0.50,
            "pd_multiplier": 1.00,
            "lgd_multiplier": 1.00,
        },
        "Downturn": {
            "weight": 0.30,
            "pd_multiplier": 1.35,
            "lgd_multiplier": 1.15,
        },
        "Upturn": {
            "weight": 0.20,
            "pd_multiplier": 0.80,
            "lgd_multiplier": 0.90,
        },
    })


# ==============================================================================
# 8. MODEL MONITORING & DRIFT CONFIGURATION
# ==============================================================================
@dataclass
class MonitoringConfig:
    """Population Stability Index (PSI) and performance tracking parameters."""
    num_bins: int = 10
    psi_green_threshold: float = 0.10
    psi_amber_threshold: float = 0.25

    # Characteristic Stability Index (CSI) monitored variables
    monitored_features: List[str] = field(default_factory=lambda: [
        "limit_bal",
        "age",
        "bill_amt1",
        "pay_amt1",
        "utilization_1",
        "max_delinquency_recent",
    ])


# ==============================================================================
# 9. DATABASE STORAGE CONFIGURATION
# ==============================================================================
@dataclass
class DatabaseConfig:
    """Database connection string supporting SQLite and PostgreSQL."""
    db_url: str = field(
        default_factory=lambda: os.getenv(
            "CREDIT_RISK_DB_URL",
            f"sqlite:///{(DATA_DIR / 'credit_risk.db').as_posix()}"
        )
    )
    echo: bool = False


# Default Global Configuration Instances
ingestion_config = IngestionConfig()
validation_config = ValidationConfig()
model_config = ModelConfig()
scorecard_config = ScorecardConfig()
ecl_config = ECLConfig()
monitoring_config = MonitoringConfig()
db_config = DatabaseConfig()
