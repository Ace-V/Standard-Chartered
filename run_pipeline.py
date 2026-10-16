"""
CreditRisk Analytics — Command-Line Pipeline Runner
===================================================

Orchestrates the entire credit risk workflow end-to-end:
1. Data Ingestion (UCI Credit Card Clients)
2. BCBS 239 Data Quality Validation
3. Feature Engineering & Stratified Splitting
4. Multi-Model 5-Fold Cross-Validation Benchmarking
5. Champion Model Probability Calibration
6. Out-of-Sample Statistical Evaluation (ROC-AUC, Gini, KS)
7. Scorecard Scaling (Base Score 600, PDO 20) & Rating Segmentation (AAA to CCC)
8. IFRS 9 Expected Credit Loss (ECL) Calculation with Macroeconomic Scenarios
9. Population Stability Index (PSI) Drift Surveillance
10. Database Persistence (SQLite / PostgreSQL)
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, UCI_DATASET_ATTRIBUTION, logger
from src.data.ingestion import load_raw_dataset
from src.data.preprocessing import prepare_train_test_split
from src.data.validation import DataValidator
from src.models.train import CreditModelTrainer
from src.monitoring.drift import calculate_psi
from src.reporting.calibration_report import run_calibration_report, run_segmentation_report
from src.risk.expected_loss import ECLCalculator
from src.risk.pd_analysis import backtest_pd_by_grade
from src.risk.segmentation import MasterRatingScale, RiskSegmenter
from src.storage.database import get_db_manager


def run_pipeline(force_refresh: bool = False, dataset_source: str = "uci"):
    logger.info("=================================================================")
    logger.info("CreditRisk Analytics — Credit Risk Modelling & Monitoring System")
    logger.info("=================================================================")
    logger.info("Dataset Citation: %s", UCI_DATASET_ATTRIBUTION.citation)
    logger.info("Dataset License: %s", UCI_DATASET_ATTRIBUTION.license_type)
    logger.info("PROTOTYPE DISCLAIMER: %s", UCI_DATASET_ATTRIBUTION.disclaimer)
    logger.info("-----------------------------------------------------------------")

    # Step 1: Ingestion
    logger.info("[Step 1/8] Ingesting dataset from source '%s'...", dataset_source)
    df = load_raw_dataset(source=dataset_source, force_refresh=force_refresh)
    logger.info("Successfully loaded %d records and %d columns.", len(df), len(df.columns))

    # Step 2: BCBS 239 Data Quality Audit
    logger.info("[Step 2/8] Executing BCBS 239 Data Quality Audit...")
    validator = DataValidator()
    report = validator.validate(df)
    report.print_summary()
    if not report.is_valid:
        logger.warning("Data quality audit encountered critical validation flags!")

    # Step 2b: Exploratory Data Analysis & Plotly Reports
    if dataset_source == "uci":
        logger.info("[Step 2b] Executing Exploratory Data Analysis & Saving Plotly Charts to reports/...")
        from src.data.eda import generate_and_save_eda_reports
        eda_paths = generate_and_save_eda_reports(df)
        logger.info("Saved 4 interactive Plotly charts and markdown synthesis report to %s", eda_paths["report_markdown"])

    # Step 3: Model Training & Competitive Benchmarking
    logger.info("[Step 3/8] Initiating 5-Fold Cross-Validation Benchmarking...")
    trainer = CreditModelTrainer()
    artifact, cv_summary = trainer.train_and_benchmark(df)

    logger.info("-----------------------------------------------------------------")
    logger.info("Cross-Validation Leaderboard:")
    for m_name, metrics in cv_summary.items():
        logger.info(
            "  * %-25s | CV AUC: %.4f (+/- %.4f) | CV Gini: %.4f | Time: %.2fs",
            m_name, metrics["cv_mean_auc"], metrics["cv_std_auc"], metrics["cv_mean_gini"], metrics["training_time_sec"]
        )
    logger.info("Selected Champion: %s", artifact.model_name)
    logger.info(
        "Test Performance: ROC-AUC: %.4f [95%% CI: %.4f - %.4f] | Gini: %.4f | KS: %.4f",
        artifact.test_metrics.roc_auc,
        artifact.test_metrics.auc_ci_lower,
        artifact.test_metrics.auc_ci_upper,
        artifact.test_metrics.gini,
        artifact.test_metrics.ks_statistic,
    )
    logger.info("-----------------------------------------------------------------")

    # Step 3b: Calibration Assessment & Borrower Risk Segmentation Reports
    logger.info("[Step 3b] Generating Calibration & Segmentation reports...")
    try:
        cal_result, cal_paths = run_calibration_report(df, artifact_path=MODELS_DIR / "pd_model_best.joblib")
        logger.info(
            "Calibration Report: Brier improvement %.1f%% | ECE improvement %.1f%% | Saved to %s",
            cal_result.improvement_pct("brier"), cal_result.improvement_pct("ece"),
            cal_paths.get("markdown_report", "N/A"),
        )
    except Exception as exc:
        logger.warning("Calibration report step failed (non-fatal): %s", exc)

    # Step 4: Full Portfolio Scoring & Rating Segmentation
    logger.info("[Step 4/8] Scoring portfolio and scaling credit bureau scores...")
    p_calibrated = artifact.predict_proba(df, calibrate=True)
    df_scored = df.copy()
    df_scored["calibrated_pd"] = p_calibrated

    segmenter = RiskSegmenter()
    df_scored = segmenter.segment(df_scored, pd_col="calibrated_pd")

    scale = MasterRatingScale()
    rating_summary = scale.summarize_portfolio(df_scored, pd_col="calibrated_pd", target_col="default_payment_next_month")
    logger.info("Master Rating Scale Breakdown:\n%s", rating_summary.to_string(index=False))

    # Step 4b: Borrower Risk Segmentation Report
    try:
        seg_result, seg_paths = run_segmentation_report(df_scored, pd_col="calibrated_pd", target_col="default_payment_next_month")
        logger.info(
            "Segmentation Report: Spearman r=%.4f | Violations=%d/9 | Saved to %s",
            seg_result.monotonicity_spearman_r, seg_result.monotonicity_violations,
            seg_paths.get("markdown_report", "N/A"),
        )
    except Exception as exc:
        logger.warning("Segmentation report step failed (non-fatal): %s", exc)

    # Step 5: Basel Binomial Traffic Light Backtest
    logger.info("[Step 5/8] Running Basel Binomial Traffic Light Test...")
    backtest_df = backtest_pd_by_grade(df_scored, grade_col="rating_grade", pd_col="calibrated_pd", target_col="default_payment_next_month")
    logger.info("Basel Backtesting Results:\n%s", backtest_df[["Grade", "Accounts", "Observed_Default_Rate", "Predicted_PD", "Traffic_Light"]].to_string(index=False))

    # Step 6: IFRS 9 Expected Credit Loss (ECL) Calculation
    logger.info("[Step 6/8] Executing IFRS 9 Three-Stage ECL Valuation...")
    ecl_calc = ECLCalculator()
    ecl_result = ecl_calc.compute_portfolio_ecl(df_scored)

    logger.info(
        f"ECL Summary: Total EAD: ${ecl_result.total_ead:,.2f} | Weighted ECL: ${ecl_result.total_ecl_weighted:,.2f} | Coverage: {ecl_result.ecl_coverage_ratio_pct:.2f}%"
    )
    logger.info("Stage Breakdown:\n%s", ecl_result.summary_table().to_string(index=False))

    # Step 7: Population Stability Surveillance (PSI)
    logger.info("[Step 7/8] Evaluating Population Stability Index (PSI)...")
    X_train, X_test, _, _ = prepare_train_test_split(df)
    p_tr = artifact.predict_proba(X_train, calibrate=True)
    p_te = artifact.predict_proba(X_test, calibrate=True)

    psi_res = calculate_psi(expected=p_tr, actual=p_te, num_bins=10, metric_name="Calibrated PD")
    logger.info("PD Drift PSI: %.4f | Traffic Light: %s (%s)", psi_res.psi_value, psi_res.traffic_light, psi_res.interpretation)

    # Step 8: Database Storage & Audit Trail Persistence
    logger.info("[Step 8/8] Persisting run outputs to database...")
    db = get_db_manager()
    import hashlib
    df_bytes = df.to_csv(index=False).encode("utf-8")
    data_hash = hashlib.sha256(df_bytes).hexdigest()

    # 1. Dataset Metadata
    db.save_dataset_metadata({
        "dataset_name": "UCI_Credit_Card_Clients",
        "version": "1.0.0",
        "row_count": len(df),
        "column_count": len(df.columns),
        "data_hash": data_hash,
        "source_url": UCI_DATASET_ATTRIBUTION.source_url,
        "license_type": UCI_DATASET_ATTRIBUTION.license_type,
        "target_column": "default_payment_next_month",
        "features_list": [c for c in df.columns if c not in ["id", "default_payment_next_month"]],
    })

    # 2. Experiment Metadata
    exp_id = f"EXP_{uuid.uuid4().hex[:8].upper()}"
    db.save_experiment({
        "experiment_id": exp_id,
        "description": "5-Fold CV benchmark with isotonic calibration and IFRS 9 ECL",
        "random_seed": 42,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "parameters": {
            "models": list(cv_summary.keys()),
            "champion": artifact.model_name,
            "scaling": "Base 600, PDO 20",
        },
    })

    # 3. Model Version
    model_ver_id = f"{artifact.model_name.lower()}_v1"
    db.save_model_version({
        "model_id": model_ver_id,
        "model_name": artifact.model_name,
        "version_tag": "1.0.0",
        "algorithm": artifact.model_name,
        "is_champion": True,
        "artifact_path": str(MODELS_DIR / "pd_model_best.joblib"),
        "hyperparameters": {
            k: v for k, v in getattr(artifact.classifier, "get_params", lambda: {})().items()
            if isinstance(v, (int, float, str, bool, list, dict, type(None)))
        },
    })

    # 4. Evaluation Result
    db.save_evaluation_result({
        "model_id": model_ver_id,
        "dataset_split": "test",
        "roc_auc": float(artifact.test_metrics.roc_auc),
        "pr_auc": float(artifact.test_metrics.pr_auc),
        "gini": float(artifact.test_metrics.gini),
        "brier_score": float(artifact.test_metrics.brier_score),
        "log_loss_value": float(artifact.test_metrics.log_loss_val),
        "ks_statistic": float(artifact.test_metrics.ks_statistic),
        "precision_val": float(artifact.test_metrics.precision),
        "recall_val": float(artifact.test_metrics.recall),
        "f1_val": float(artifact.test_metrics.f1),
        "metrics_json": {
            "auc_ci_lower": float(artifact.test_metrics.auc_ci_lower),
            "auc_ci_upper": float(artifact.test_metrics.auc_ci_upper),
        },
    })

    db.save_loans(df, if_exists="replace")
    db.save_predictions(df_scored.head(1000), model_name=artifact.model_name)

    run_id = f"RUN_{uuid.uuid4().hex[:8].upper()}"
    b = ecl_result.stage_breakdown
    st1 = b.get("Stage 1 (Performing)", {}).get("total_ecl_weighted", 0.0)
    st2 = b.get("Stage 2 (SICR)", {}).get("total_ecl_weighted", 0.0)
    st3 = b.get("Stage 3 (Impaired)", {}).get("total_ecl_weighted", 0.0)

    db.save_ecl_run(
        run_id=run_id,
        total_accounts=ecl_result.total_accounts,
        total_ead=ecl_result.total_ead,
        total_ecl_weighted=ecl_result.total_ecl_weighted,
        total_ecl_base=ecl_result.total_ecl_base,
        total_ecl_downturn=ecl_result.total_ecl_downturn,
        total_ecl_upturn=ecl_result.total_ecl_upturn,
        coverage_ratio_pct=ecl_result.ecl_coverage_ratio_pct,
        stage_1_ecl=st1,
        stage_2_ecl=st2,
        stage_3_ecl=st3,
    )

    db.log_monitoring_event(
        metric_type="PSI",
        feature_or_score="calibrated_pd",
        metric_value=psi_res.psi_value,
        traffic_light=psi_res.traffic_light,
        interpretation=psi_res.interpretation,
    )

    logger.info("=================================================================")
    logger.info("Pipeline Execution Completed Successfully! Run ID: %s", run_id)
    logger.info("Launch dashboard with: streamlit run dashboard/app.py")
    logger.info("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CreditRisk Analytics Pipeline Runner")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download of raw dataset")
    parser.add_argument("--source", type=str, default="uci", choices=["uci", "corporate"], help="Dataset source ('uci' or 'corporate')")
    args = parser.parse_args()

    run_pipeline(force_refresh=args.force_refresh, dataset_source=args.source)
