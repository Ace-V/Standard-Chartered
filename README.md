# Standard Chartered — Quantitative Risk Analytics & Financial Forecasting

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework-IFRS9](https://img.shields.io/badge/Impairment-IFRS%209%20%2F%20CECL-darkgreen.svg)]()
[![Standard-BCBS239](https://img.shields.io/badge/Data%20Hygiene-BCBS%20239-navy.svg)]()
[![Database-SQLAlchemy_Alembic](https://img.shields.io/badge/Storage-SQLAlchemy%20%7C%20Alembic-purple.svg)]()
[![Tests-Pytest](https://img.shields.io/badge/Tests-147%20Passed%20(97%25%20Cov)-brightgreen.svg)]()
[![License-MIT_CC_BY](https://img.shields.io/badge/License-MIT%20%7C%20CC%20BY%204.0-lightgrey.svg)]()

This repository contains two production-grade quantitative financial modelling suites:
1. **Corporate Credit Risk Analytics (CCRA):** Institutional-grade Probability of Default (PD) estimation, credit scorecard scaling, IFRS 9 Expected Credit Loss (ECL) three-stage impairment, BCBS 239 data quality validation, population stability monitoring (PSI/CSI), and hardened SQLAlchemy/Alembic persistence with an interactive Streamlit dashboard.
2. **Time Series Forecasting & Asset Prediction (`forecasting-and-prediction/`):** Multi-model equity and asset price forecasting combining statistical ARIMA, Facebook Prophet, and a Hybrid ARIMA+LSTM neural network with Monte Carlo simulation.

---

## 🏛️ Repository Architecture

```
.
├── src/                                   # CCRA: Core Risk Modeling & Engineering Engine
│   ├── config.py                          # Centralized dataclass configurations & paths
│   ├── data/                              # BCBS 239 ingestion, validation, and splitters
│   ├── features/                          # Financial & revolving delinquency engineering
│   ├── models/                            # 5-Fold CV training, calibration, and metrics
│   ├── risk/                              # Scorecard scaling, Master Rating Scale, IFRS 9 ECL
│   ├── monitoring/                        # PSI, CSI, performance decay, and alerting
│   └── storage/                           # SQLAlchemy ORM (9 tables) & Pydantic v2 schemas
├── dashboard/                             # CCRA: 6-Page Streamlit Banking Analytics Portal
│   ├── app.py                             # Executive overview & KPI cards
│   └── pages/                             # Deep-dive analytics, data quality, PD, ECL, drift
├── forecasting-and-prediction/            # Project 2: Time Series Forecasting Engine
│   ├── TIME-Series Forecasting.py         # ARIMA, Prophet, and Hybrid ARIMA+LSTM
│   ├── MonteCarlo.py                      # Monte Carlo asset price trajectory simulator
│   ├── Timeseries.ipynb                   # Interactive exploration notebook
│   ├── requirements.txt                   # Dedicated forecasting dependencies
│   └── README.md                          # Dedicated forecasting documentation
├── alembic/                               # Database migration scripts (SQLite & PostgreSQL)
├── reports/                               # Generated model development & monitoring reports
├── tests/                                 # CCRA test suite (147 tests, 100% pass)
├── run_pipeline.py                        # Automated end-to-end CCRA pipeline runner
├── requirements.txt                       # CCRA dependencies
├── .env.example                           # Environment configuration template
└── README.md                              # Institutional system documentation
```

---

## 📊 Project 1: Corporate Credit Risk Analytics (CCRA)

### Overview & Capabilities
* **PD Estimation & Competitive Benchmarking:** Evaluates L2-regularized Logistic Regression, Random Forest, and Histogram Gradient Boosting with 5-fold stratified cross-validation on 30,000 credit records.
* **Champion Model Performance:** `Hist_Gradient_Boosting` selected as champion with **0.7860 CV ROC-AUC**, **0.7804 Test ROC-AUC** (95% CI: $[0.7653, 0.7958]$), **0.5608 Gini**, and **0.4278 KS statistic**.
* **Probability Calibration:** Isotonic regression and Platt sigmoid calibration fitted exclusively on held-out validation partitions, separating rank discrimination from probability calibration.
* **Credit Scorecard Scaling:** Standard logarithmic mapping ($\text{Base Score} = 600$, $\text{PDO} = 20$, $\text{Target Odds} = 50:1$) with an 8-grade Master Scale ($\text{AAA}$ to $\text{CCC}$) and **zero decile monotonicity violations** ($r_s = 1.0000$).
* **IFRS 9 Three-Stage ECL Engine:** Calculates Stage 1 (12m ECL), Stage 2 (Lifetime SICR), and Stage 3 (Default) provisions across probability-weighted macroeconomic scenarios (Base 50%, Downturn 30%, Upturn 20%). Evaluated portfolio: **$4.17B Total EAD**, **$490.5M Weighted ECL**, **11.76% Coverage Ratio**.
* **Ongoing Model Surveillance:** Real-time Population Stability Index ($\text{PSI} = 0.0010$, 🟢 GREEN) and Characteristic Stability Index (CSI) with operational alert workflows (`INFO`, `WARNING`, `CRITICAL`).
* **Hardened Storage:** 9 normalized SQLAlchemy tables with Pydantic v2 validation, automatic transaction rollback on error, and anti-PII schema assurance.

### CCRA Quickstart
```powershell
# 1. Activate environment and install dependencies
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run database migrations (Alembic)
alembic upgrade head

# 3. Run complete test suite (147 tests)
pytest tests\ -v

# 4. Execute end-to-end pipeline
python run_pipeline.py

# 5. Launch banking analytics dashboard
streamlit run dashboard/app.py
```

---

## 📈 Project 2: Time Series Forecasting & Asset Prediction

### Overview & Models (`forecasting-and-prediction/`)
A multi-paradigm time-series forecasting engine designed for equity price trajectory modelling and residual volatility forecasting:
1. **ARIMA (5,1,0):** Classical linear autoregressive integrated moving-average model capturing macro trends and autocorrelation.
2. **Facebook Prophet:** Additive regression model handling multi-period seasonality (weekly, yearly) and structural change points.
3. **Hybrid ARIMA + LSTM:** Decomposes the time series by fitting ARIMA on trend and leveraging an LSTM deep neural network (PyTorch) to model non-linear residual errors for superior predictive accuracy.
4. **Monte Carlo Simulation (`MonteCarlo.py`):** Simulates thousands of stochastic geometric Brownian motion asset price paths for risk stress testing.

### Sample Model Performance
```text
Model Evaluation Metrics (AAPL Test Horizon):
  * ARIMA (5,1,0):             MAE: 2.4536 | RMSE: 3.1242
  * Facebook Prophet:          MAE: 3.1847 | RMSE: 4.2156
  * Hybrid (ARIMA + LSTM):     MAE: 2.1834 | RMSE: 2.8932 (Best Performance)
```

### Forecasting Quickstart
```powershell
# Navigate to forecasting module
cd forecasting-and-prediction

# Install requirements
pip install -r requirements.txt

# Run forecasting pipeline (Default: AAPL)
python "TIME-Series Forecasting.py"

# Run custom ticker with 60-day horizon
python "TIME-Series Forecasting.py" --symbol MSFT --start 2020-01-01 --end 2024-01-01 --forecast_horizon 60 --epochs 20

# Run Monte Carlo trajectory simulation
python MonteCarlo.py
```

---

## 📑 Governance & Compliance Documentation

* [Model Development Report](reports/model_development_report.md): Quantitative methodology, CV leaderboards, scorecard scaling, and IFRS 9 staging.
* [Model Monitoring Report](reports/model_monitoring_report.md): BCBS 239 data hygiene, PSI/CSI mathematical formulations, and drift escalation SLAs.
* [Probability Calibration Report](reports/calibration_report.md): Reliability diagrams, Brier score decomposition, and ECE/MCE diagnostics.
* [Borrower Risk Segmentation Report](reports/segmentation_report.md): Risk decile monotonicity and Basel binomial traffic light tests.

---

## ⚠️ Regulatory Notice
*This software is an analytical research prototype. It has not received formal regulatory approval or supervisory sign-off for statutory Basel IRB capital calculations or live underwriting. All credit models are demonstrated on benchmark data and should be validated on internal portfolio data before deployment.*
