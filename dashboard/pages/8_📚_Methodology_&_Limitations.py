"""
CreditRisk Analytics -- Page 8: Methodology & Limitations
=========================================================

Comprehensive technical documentation covering:
  - Data provenance and dataset description
  - Feature engineering methodology
  - PD model development (LR, RF, HGB) and selection
  - Probability calibration (isotonic regression)
  - Master Rating Scale construction
  - IFRS 9 ECL staging methodology
  - Model monitoring: PSI / CSI drift detection
  - Fundamental limitations and governance warnings
  - Regulatory framework references (Basel III, IFRS 9, SR 11-7)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

st.set_page_config(
    page_title="Methodology & Limitations | CreditRisk",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Methodology, Governance & Limitations")
st.caption(
    "Technical documentation for the CreditRisk Analytics prototype: "
    "modelling choices, assumptions, validation approach, and fundamental limitations."
)

st.error(
    "**⚠️ MANDATORY PROTOTYPE DISCLAIMER — READ BEFORE INTERPRETING ANY OUTPUT:**\n\n"
    "This platform is an **academic and technical demonstration** of quantitative credit risk "
    "modelling methodologies. It is built on the **UCI Default of Credit Card Clients** dataset "
    "(consumer revolving credit, Taiwan, 2005). "
    "All model outputs, KPIs, rating grades, ECL provisions, and monitoring alerts are "
    "**illustrative and NOT validated for production use**. "
    "They must NOT be used for lending decisions, regulatory capital calculations, IFRS 9 "
    "financial reporting, or any other regulated purpose."
)

# ── Navigation tabs ──────────────────────────────────────────────────────────
tab_data, tab_feat, tab_model, tab_calib, tab_seg, tab_ecl, tab_monitor, tab_limits, tab_refs = st.tabs([
    "📂 Data & Provenance",
    "⚙️ Feature Engineering",
    "🤖 PD Modelling",
    "🎯 Calibration",
    "🏷️ Risk Segmentation",
    "💰 ECL Methodology",
    "🛡️ Model Monitoring",
    "⚠️ Limitations",
    "📖 References",
])

# ── TAB 1: DATA & PROVENANCE ─────────────────────────────────────────────────
with tab_data:
    st.subheader("Dataset Description & Provenance")

    st.markdown("""
### UCI Default of Credit Card Clients Dataset

**Citation:** Yeh, I-Cheng, and Che-hui Lien (2009). "The comparisons of data mining techniques
for the predictive accuracy of probability of default of credit card clients."
*Expert Systems with Applications*, 36(2), 2473–2480.

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Access Method:** `ucimlrepo` Python package (official UCI API). The dataset is downloaded
programmatically and cached locally in `data/processed/`. A failed download is **never silently
substituted** — an explicit error is raised and surfaced to the analyst.
""")

    st.markdown("### Dataset Characteristics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accounts", "30,000", "Taiwan credit cards")
    with col2:
        st.metric("Features (Raw)", "23", "Excluding ID & target")
    with col3:
        st.metric("Observed Default Rate", "~22.1%", "Next-month default")
    with col4:
        st.metric("Observation Window", "April–Sept 2005", "6-month panel")

    st.markdown("""
### Column Schema & Renaming

Raw UCI column names are mapped to `snake_case` to comply with Python naming conventions.
All original names are documented in `src/config.py → RAW_UCI_COLUMN_MAPPING`.

| Raw UCI Name | Standardised Name | Description |
|---|---|---|
| `ID` | `id` | Account identifier (excluded from modelling) |
| `LIMIT_BAL` | `limit_bal` | Credit limit (NT$) |
| `SEX` | `sex` | 1 = Male, 2 = Female |
| `EDUCATION` | `education` | 1 = Graduate, 2 = University, 3 = High School, 4 = Others |
| `MARRIAGE` | `marriage` | 1 = Married, 2 = Single, 3 = Others |
| `AGE` | `age` | Age in years |
| `PAY_0` | `pay_0` | Repayment status, September 2005 (most recent) |
| `PAY_2–PAY_6` | `pay_2–pay_6` | Repayment status, April–August 2005 |
| `BILL_AMT1–6` | `bill_amt1–6` | Bill statement amounts, April–September 2005 |
| `PAY_AMT1–6` | `pay_amt1–6` | Payment amounts, April–September 2005 |
| `default.payment.next.month` | `default_payment_next_month` | **Target**: 1 = Default |

### Repayment Status Coding
`pay_0` through `pay_6` are ordinal indicators:
- **−2** = No consumption that month
- **−1** = Paid duly (full payment)
- **0** = Use of revolving credit (minimum payment made)
- **1–8** = Payment delayed by 1 to 8 months
""")

    st.markdown("""
### Data Quality Controls (BCBS 239 Aligned)

The `DataValidator` class performs comprehensive pre-modelling checks:

| Check Type | Logic |
|---|---|
| Missing values | Count and rate by column |
| Duplicate records | Row-level exact deduplication |
| ID column integrity | Uniqueness enforcement |
| Age boundary | 18 ≤ age ≤ 100 |
| Credit limit floor | limit_bal ≥ NT$1,000 |
| Education categories | Undocumented codes {0, 5, 6} → remapped to 4 (Others) |
| Marriage categories | Undocumented code {0} → remapped to 3 (Others) |
| Payment status range | −2 ≤ pay_k ≤ 8 |
| Target domain | Binary {0, 1} verified |
""")

# ── TAB 2: FEATURE ENGINEERING ───────────────────────────────────────────────
with tab_feat:
    st.subheader("Feature Engineering Methodology")

    st.markdown("""
Feature engineering is implemented in `src/features/engineering.py` using
the `CreditFeatureEngineer` class (scikit-learn `BaseEstimator` / `TransformerMixin`).

**Critical rule:** All feature transformations are fit on the training partition **only**.
The fitted transformer is applied to validation and test data without re-fitting,
preventing statistical leakage.

### Derived Features

| Feature Name | Formula | Rationale |
|---|---|---|
| `utilization_1–6` | `bill_amt_k / limit_bal` (clipped [0, 1]) | Utilization rate: key default predictor |
| `avg_utilization` | mean of utilization_1–6 | Overall revolving credit usage pattern |
| `max_delinquency_recent` | max(pay_0, pay_2, pay_3) | Captures worst recent payment behaviour |
| `max_delinquency_all` | max(pay_0 through pay_6) | Full delinquency history |
| `delinquency_trend` | pay_0 − pay_6 | Improvement or deterioration in payment |
| `total_bill_amt` | sum of bill_amt_1–6 | Total outstanding obligation |
| `total_pay_amt` | sum of pay_amt_1–6 | Total payment volume made |
| `pay_to_bill_ratio` | total_pay_amt / (total_bill_amt + 1) | Repayment capacity indicator |
| `avg_bill_amt` | mean of bill_amt_1–6 | Average monthly statement |
| `avg_pay_amt` | mean of pay_amt_1–6 | Average monthly payment |
| `bill_trend` | bill_amt1 − bill_amt6 | Debt balance growth direction |
| `pay_trend` | pay_amt1 − pay_amt6 | Payment effort trajectory |
| `consecutive_delays` | count of pay_k ≥ 1 | Number of months with late payment |
| `ever_severely_delinquent` | indicator: any pay_k ≥ 3 | 90+ DPD flag |

All raw features (after cleaning) are retained alongside the engineered features.
""")

# ── TAB 3: PD MODELLING ──────────────────────────────────────────────────────
with tab_model:
    st.subheader("Probability of Default (PD) Model Development")

    st.markdown("""
### Data Partitioning

The dataset is split into three partitions using **stratified sampling** on the target label
to preserve class distribution:

| Partition | Share | Records | Purpose |
|---|---|---|---|
| Training (subtrain) | 64% | 19,200 | Model fitting |
| Calibration | 16% | 4,800 | Isotonic calibrator fitting |
| Out-of-Sample Test | 20% | 6,000 | Final held-out evaluation |

Random seed: 42 (fixed throughout for full reproducibility).

### Candidate Models

Three candidate models are evaluated via **5-fold stratified cross-validation** on the
combined training+calibration partition before the test set is touched.

| Model | Implementation | Key Hyperparameters |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | C=0.1, max_iter=1000, L2 penalty |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | n_estimators=100, max_depth=6, min_samples_leaf=20 |
| Hist Gradient Boosting | `sklearn.ensemble.HistGradientBoostingClassifier` | max_iter=100, max_depth=5, l2_reg=1.5 |

**Champion selection criterion:** Highest mean CV ROC-AUC across 5 folds.

### Champion Model: Hist Gradient Boosting (isotonic calibration)

The `HistGradientBoostingClassifier` consistently achieves the highest discriminatory power
on this dataset because it:
- Handles the ordinal repayment status features efficiently without explicit encoding
- Captures non-linear interactions between utilization, delinquency, and credit limits
- Includes native L2 regularisation reducing overfitting on the 30k-record dataset

### Evaluation Metrics

The final test evaluation uses **only the held-out test set** (never seen during training or calibration):

| Metric | Rationale |
|---|---|
| **ROC-AUC** | Overall discriminatory power (threshold-free) |
| **Gini coefficient** | 2 × AUC − 1; Basel IRB preferred ranking quality measure |
| **KS Statistic** | Maximum separation between default and non-default score distributions |
| **Brier Score** | Proper scoring rule for probability calibration quality |
| **Log Loss** | Cross-entropy; penalises confident wrong predictions |
| **PR-AUC** | Area under Precision-Recall curve; informative under class imbalance |
| **Precision / Recall** | At operationally selected thresholds |
| **95% Bootstrap CI on AUC** | Uncertainty quantification (n=1,000 bootstrap resamples) |

**Accuracy is NOT used as a primary evaluation metric** due to class imbalance (~22% positive rate).
""")

# ── TAB 4: CALIBRATION ───────────────────────────────────────────────────────
with tab_calib:
    st.subheader("Probability Calibration Methodology")

    st.markdown("""
### Why Calibration Matters

A model's **ranking quality** (AUC, Gini, KS) and its **probability calibration** are
fundamentally distinct properties:

| Property | Measures | Method |
|---|---|---|
| Ranking quality | Is borrower A correctly ordered relative to borrower B? | ROC-AUC, Gini, KS |
| Probability calibration | If the model predicts 20% PD, do ~20% of such borrowers actually default? | Brier Score, Reliability Diagram |
| Classification threshold performance | At a chosen cutoff, how many true positives vs. false positives? | Precision, Recall, F1 |

For **IFRS 9 ECL calculations**, probability calibration is critical: a perfectly ranking
but poorly calibrated model will systematically over- or under-provision loan loss reserves.

### Isotonic Regression Calibration

**Method:** Isotonic regression (non-parametric monotone function) fitted on a held-out
calibration set (16% of data, never used for model training).

**Process:**
1. Champion model trained on subtrain partition (64%).
2. Raw scores (uncalibrated probabilities) generated for calibration set (16%).
3. `IsotonicRegression` fitted on raw scores vs. true labels — this is a pool-adjacent-violators
   algorithm that returns a step-function mapping raw scores to calibrated probabilities.
4. Calibrated probabilities evaluated on test set only.

**Why isotonic (not Platt/sigmoid)?**
- Isotonic regression makes no parametric shape assumptions.
- It is more flexible for ensemble and gradient boosting outputs which often require
  non-monotone corrections beyond a sigmoid.
- Sufficient calibration set size (4,800 records) supports non-parametric estimation.

### Calibration Assessment

| Diagnostic | Description |
|---|---|
| **Reliability Diagram** | Plots mean predicted PD vs. observed default rate by probability bin |
| **Expected Calibration Error (ECE)** | Weighted mean bin-level absolute error |
| **Maximum Calibration Error (MCE)** | Worst-case single-bin miscalibration |
| **Brier Score Decomposition** | Reliability + Resolution + Uncertainty components |
""")

# ── TAB 5: RISK SEGMENTATION ─────────────────────────────────────────────────
with tab_seg:
    st.subheader("Risk Segmentation & Master Rating Scale")

    st.markdown("""
### Credit Bureau Scorecard Scaling

The calibrated PD is converted to a bureau-style credit score (range 300–850)
using the standard banking scorecard formula:

```
Factor = PDO / ln(2)
Offset = BaseScore − Factor × ln(TargetOdds)
Score  = Offset + Factor × ln((1 − PD) / PD)
Score  = clipped to [300, 850]
```

**Parameters (illustrative):**
| Parameter | Value | Meaning |
|---|---|---|
| Base Score | 600 | Score at target odds ratio |
| Target Odds | 50:1 | Good:Bad ratio at base score |
| PDO | 20 | Points to double the odds |

⚠️ These parameters are illustrative. In production, scorecard parameters are derived
from institution-specific calibration, regulatory approval, and business risk appetite.

### Master Rating Scale (MRS)

The MRS maps continuous calibrated PDs to discrete credit grades:

| Grade | PD Lower | PD Upper | Description |
|---|---|---|---|
| AAA | 0.00% | 0.50% | Prime Investment Grade — Negligible Risk |
| AA  | 0.50% | 1.50% | High Quality — Very Low Risk |
| A   | 1.50% | 3.50% | Upper Medium Grade — Low Risk |
| BBB | 3.50% | 7.50% | Lower Medium Grade — Moderate Risk |
| BB  | 7.50% | 15.00% | Speculative Grade — Substantial Risk |
| B   | 15.00% | 30.00% | Highly Speculative — High Default Risk |
| CCC | 30.00% | 100.00% | Substandard / Impaired / Default Warning |

⚠️ These grade thresholds are **illustrative reference points** loosely aligned with
rating agency conventions. They are NOT an approved internal ratings-based (IRB) scale.

### Borrower Risk Segments

Five operational risk bands applied for portfolio management reporting:

| Segment | Illustrative PD Range | Portfolio Management Implication |
|---|---|---|
| Low Risk | PD < 5% | Preferential pricing, low LGD assumptions |
| Medium-Low | 5% ≤ PD < 10% | Standard underwriting, monitoring |
| Medium-High | 10% ≤ PD < 20% | Enhanced monitoring, covenant review |
| High | 20% ≤ PD < 35% | Watchlist, remediation consideration |
| Very High | PD ≥ 35% | Default watch, Stage 3 / specific provision |

Segment boundaries must be derived from institution-specific risk appetite in practice.

### Basel Binomial Traffic Light Backtest

A binomial significance test is applied at the grade level:
- H₀: Model PD = Realised default rate
- H₁: Model PD < Realised default rate (model underestimates risk)
- One-tailed binomial p-value computed per grade
- Traffic lights: GREEN (p > 5%), AMBER (1% < p ≤ 5%), RED (p ≤ 1%)
""")

# ── TAB 6: ECL METHODOLOGY ───────────────────────────────────────────────────
with tab_ecl:
    st.subheader("Expected Credit Loss (ECL) Methodology — IFRS 9 Framework")

    st.markdown("""
### Core ECL Formula

```
ECL = PD × LGD × EAD × DF
```

Where:
- **PD** (Probability of Default) — Calibrated model output
- **LGD** (Loss Given Default) — Fraction of EAD lost at default
- **EAD** (Exposure at Default) — Outstanding balance at time of default
- **DF** (Discount Factor) — Present-value adjustment (set to 1.0 in simplified calculations)

### IFRS 9 Three-Stage Impairment Framework

| Stage | Trigger Condition | ECL Measurement | Interest Revenue |
|---|---|---|---|
| Stage 1 | No significant deterioration | 12-month ECL | Gross carrying amount |
| Stage 2 | Significant Increase in Credit Risk (SICR) | Lifetime ECL | Gross carrying amount |
| Stage 3 | Credit-Impaired (defaulted) | Lifetime ECL | Net of credit loss allowance |

**SICR Criteria (illustrative thresholds in this prototype):**
- Absolute PD ≥ 10% (Stage 2) or ≥ 30% (Stage 3)
- Delinquency: 1+ months delay triggers Stage 2; 3+ months or 90 DPD triggers Stage 3

### Forward-Looking Macroeconomic Scenarios

IFRS 9 requires **probability-weighted multiple economic scenarios**:

| Scenario | Weight | PD Multiplier | LGD Multiplier |
|---|---|---|---|
| Base | 50% | 1.00× | 1.00× |
| Downturn | 30% | 1.35× | 1.15× |
| Upturn | 20% | 0.80× | 0.90× |

**Weighted ECL = Σ (Scenario Weight × Scenario ECL)**

⚠️ Scenario weights, PD multipliers, and LGD multipliers are **illustrative**.
Production IFRS 9 models require macro-econometric models linking GDP growth,
unemployment, and interest rates to PD/LGD parameters.

### Corporate ECL Module

The corporate ECL tab demonstrates the ECL formula applied to stylised corporate loan
portfolios using hypothetical financial ratios (Interest Coverage, Debt-to-Equity, ROA).
These are **NOT derived from the UCI consumer credit model** and serve only to illustrate
the PD × LGD × EAD formula structure in a corporate context.

**Key LGD Conventions:**
| Asset Type | Typical LGD Range | Source |
|---|---|---|
| Unsecured consumer revolving credit | 40–75% | Basel II IRB floor: 45% |
| Senior secured corporate | 25–45% | Basel II IRB |
| Senior unsecured corporate | 45–65% | Basel II IRB |
| Subordinated debt | 65–85% | Basel II IRB |
""")

# ── TAB 7: MODEL MONITORING ──────────────────────────────────────────────────
with tab_monitor:
    st.subheader("Model Monitoring & Governance Subsystem")

    st.markdown("""
### Monitoring Architecture

The monitoring subsystem (`src/monitoring/`) implements five independent modules:

| Module | Class | Purpose |
|---|---|---|
| A. Data Quality | `DataQualityMonitor` | Missing values, invalid ranges, schema drift, duplicates |
| B. Feature Drift | `DriftMonitor` | PSI, CSI, distributional comparisons |
| C. Model Performance | `PerformanceMonitor` | AUC, Brier, log loss, precision/recall |
| D. Prediction Monitoring | `PredictionMonitor` | Score PSI, risk segment migration |
| E. Alerting | `AlertManager` | Configurable thresholds, warning/critical states |

### Population Stability Index (PSI)

PSI measures distributional shift between a baseline (training) and monitoring (current) population:

```
PSI = Σ [(Actual% − Expected%) × ln(Actual% / Expected%)]
```

**Standard Interpretation Thresholds:**

| PSI Value | Status | Regulatory Implication |
|---|---|---|
| PSI < 0.10 | ✅ GREEN — Stable | No action required |
| 0.10 ≤ PSI < 0.25 | ⚠️ AMBER — Monitor | Investigate root cause |
| PSI ≥ 0.25 | 🔴 RED — Significant Shift | Model redevelopment review |

⚠️ PSI thresholds are conventions, NOT statutory requirements. Different institutions
and regulators apply varying thresholds.

### Characteristic Stability Index (CSI)

CSI applies the PSI formula to individual input features (rather than the score distribution).
High CSI on a key feature (e.g., `utilization_1`, `max_delinquency_recent`) indicates that
the input data distribution has shifted — a leading indicator of potential model degradation
even before performance metrics deteriorate.

### Performance Deterioration vs. Drift

These are **fundamentally different phenomena**:

| Phenomenon | Definition | Detection |
|---|---|---|
| Feature drift (input shift) | Population characteristics have changed | CSI / PSI on features |
| Score drift (output shift) | Predicted PD distribution has changed | Score PSI |
| Performance deterioration | Model predictions no longer reflect outcomes | AUC, Brier, log loss on labelled outcomes |

A model can exhibit high PSI without performance deterioration (if the model generalises
across the shifted distribution), and can deteriorate without drift (if the relationship
between features and outcomes changes without distributional shift).

### Regulatory Alignment

SR 11-7 (Federal Reserve) and EBA Guidelines on Internal Models require:
- Ongoing performance monitoring on labelled outcomes
- Annual full model validation
- Documentation of material changes to data, model, or environment
- Escalation procedures for material performance deterioration
""")

# ── TAB 8: LIMITATIONS ───────────────────────────────────────────────────────
with tab_limits:
    st.subheader("Fundamental Limitations & Governance Warnings")

    st.markdown("""
### Critical Limitations of This Prototype
""")
    st.error("""
**1. Consumer Credit Dataset — Not Corporate Credit**

The underlying model is trained exclusively on **Taiwan consumer credit card** data (2005).
The corporate ECL module is a *structural demonstration* — it uses hypothetical financial
ratios and does NOT derive PDs from the UCI dataset. Any comparison to corporate credit
ratings is illustrative only.
""")
    st.warning("""
**2. Historical Data — No Current Macroeconomic Context**

Data is from April–September 2005. Economic conditions (interest rates, unemployment, credit
standards) differ materially from the present. The model is NOT calibrated for current conditions.
""")
    st.warning("""
**3. Geographic Specificity**

The dataset covers Taiwan's revolving credit market. Credit behaviour, regulatory frameworks,
and macroeconomic drivers differ significantly across jurisdictions. The model has NO
established validity outside its training context.
""")
    st.warning("""
**4. Model Stability — Out-of-Time Validation Not Performed**

The prototype uses a randomly sampled 20% hold-out for testing. A rigorous production
validation would require **out-of-time testing** (e.g., testing on a later calendar period)
to detect temporal decay in predictive power.
""")
    st.warning("""
**5. LGD and EAD Are Assumed Inputs — Not Modelled**

LGD (Loss Given Default) values are set to illustrative defaults (45% for unsecured retail).
EAD is approximated as a fraction of credit limit. In production IFRS 9 systems, LGD
and EAD require separate validated models.
""")
    st.warning("""
**6. Macroeconomic Scenario Weights Are Illustrative**

Scenario weights (50% Base, 30% Downturn, 20% Upturn) are placeholder values.
Production IFRS 9 implementation requires quantitative expert judgment, macroeconomic
model outputs, and senior management / audit committee approval.
""")
    st.info("""
**7. No Adverse Action Compliance**

Credit scoring models used in actual lending decisions must comply with adverse action
notice requirements (e.g., ECOA / Reg B in the US, GDPR Article 22 in Europe).
This prototype has no regulatory compliance infrastructure.
""")
    st.info("""
**8. Feature Importance ≠ Causal Effect**

Gradient boosting feature importances (gain, permutation) reflect predictive associations,
not causal relationships. Using feature importance rankings to justify credit decisions
without causal validation is methodologically unsound.
""")

# ── TAB 9: REFERENCES ────────────────────────────────────────────────────────
with tab_refs:
    st.subheader("References & Regulatory Framework")

    st.markdown("""
### Dataset & Modelling References

1. **Yeh, I-Cheng, and Che-hui Lien (2009).** "The comparisons of data mining techniques for the
   predictive accuracy of probability of default of credit card clients."
   *Expert Systems with Applications*, 36(2), 2473–2480.
   DOI: [10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020)

2. **Dua, D. and Graff, C. (2019).** UCI Machine Learning Repository.
   University of California, Irvine, School of Information and Computer Science.
   [https://archive.ics.uci.edu](https://archive.ics.uci.edu)

### Probability Calibration

3. **Niculescu-Mizil, A., and Caruana, R. (2005).** "Predicting Good Probabilities with
   Supervised Learning." *Proceedings of ICML 2005*.

4. **Zadrozny, B., and Elkan, C. (2002).** "Transforming classifier scores into accurate
   multiclass probability estimates." *Proceedings of KDD 2002*.

### IFRS 9 / ECL

5. **International Accounting Standards Board (2014).** *IFRS 9 Financial Instruments.*
   IFRS Foundation.

6. **Basel Committee on Banking Supervision (2006).** *International Convergence of Capital
   Measurement and Capital Standards (Basel II).* Bank for International Settlements.

7. **Basel Committee on Banking Supervision (2010).** *Basel III: A global regulatory
   framework for more resilient banks and banking systems.* BIS.

### Model Risk Management

8. **Board of Governors of the Federal Reserve System (2011).** *SR 11-7: Guidance on
   Model Risk Management.* Federal Reserve / OCC.

9. **European Banking Authority (2019).** *Guidelines on Internal Governance (GL 44).*
   EBA/GL/2017/11.

10. **BCBS (2013).** *Principles for Effective Risk Data Aggregation and Risk Reporting (BCBS 239).*
    Bank for International Settlements.

### Drift Detection

11. **Gama, J., et al. (2014).** "A survey on concept drift adaptation." *ACM Computing Surveys*, 46(4).

12. **Population Stability Index (PSI).** Industry standard from consumer credit bureau practice;
    no single canonical academic reference. Threshold conventions derived from Siddiqi (2006):
    *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.*
    John Wiley & Sons.

### Python Libraries

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | ≥1.4 | ML pipelines, calibration, cross-validation |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.26 | Numerical computation |
| scipy | ≥1.12 | Statistical tests (binomial, KS) |
| plotly | ≥5.20 | Interactive visualisations |
| streamlit | ≥1.34 | Dashboard framework |
| joblib | ≥1.3 | Model serialisation |
| ucimlrepo | ≥0.0.6 | Official UCI dataset API |
| sqlalchemy | ≥2.0 | Database abstraction layer |
| pytest | ≥8.0 | Test framework (129 passing tests) |
""")

    st.divider()
    st.markdown("""
### Platform Architecture Summary

```
CreditRisk Analytics — Module Overview
=======================================
src/
  config.py              — Central configuration (paths, hyperparameters, ECL params)
  data/
    ingestion.py         — UCI dataset download, caching, NEVER silently substitutes
    validation.py        — BCBS 239 data quality checks
    preprocessing.py     — Categorical cleaning, imputation, train/test split
  features/
    engineering.py       — Derived features (utilization, delinquency, trends)
  models/
    train.py             — Multi-model CV, champion selection, calibration, artifact save
    calibration.py       — IsotonicCalibrator wrapper + diagnostics
    evaluate.py          — EvaluationMetrics dataclass, bootstrap CI
  risk/
    segmentation.py      — CreditScorecard, MasterRatingScale, BorrowerRiskSegmenter
    ecl_analysis.py      — IFRS 9 ECL engine, scenario analysis, sensitivity
    expected_loss.py     — ECLCalculator (formula layer)
    pd_analysis.py       — Grade-level backtest, Basel traffic light
  monitoring/
    data_quality.py      — DataQualityMonitor
    drift.py             — PSI, CSI, DriftMonitor
    performance.py       — PerformanceMonitor
    prediction.py        — PredictionMonitor
    alerting.py          — AlertManager, configurable severity thresholds
    subsystem.py         — ModelMonitoringSubsystem orchestrator
    simulation.py        — Synthetic cohort simulation for monitoring demos
    visualizations.py    — Plotly monitoring charts
  reporting/
    charts.py            — Shared Plotly chart functions
  storage/
    database.py          — SQLAlchemy abstraction (SQLite / PostgreSQL)

dashboard/
  app.py                              — Entry point / system overview
  pages/
    1_🏛️_Executive_Overview.py       — KPIs, rating distribution, model summary
    2_📊_Portfolio_Analytics.py       — Demographic filters, default rates
    3_📈_Model_Performance.py         — ROC/PR curves, calibration, threshold explorer
    4_💰_ECL_&_Scenario_Analysis.py   — IFRS 9 staging, scenario stress testing
    5_🛡️_Monitoring_&_Stability.py    — PSI, CSI, performance decay, alerts
    6_🎯_Calibration_&_Segmentation.py — Reliability diagrams, decile analysis
    7_🔍_Individual_Risk_Assessment.py — Single-borrower interactive scoring
    8_📚_Methodology_&_Limitations.py  — This page

tests/                    — 129+ pytest unit tests
```
""")
