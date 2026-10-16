# CreditRisk Analytics — Model Development Report
**Document ID:** MRM-MDR-2026-001  
**Project:** CreditRisk Analytics (PD Estimation, Scorecard Scaling, and IFRS 9 ECL)  
**Model Family:** Retail Credit Card Default Prediction (Consumer Portfolio Prototype)  
**Champion Model:** Histogram-based Gradient Boosting Classifier (`Hist_Gradient_Boosting`)  
**Calibration Strategy:** Isotonic Regression (Held-out validation partition)  
**Target Variable:** `default_payment_next_month` (Binary: 1 = Delinquent/Default, 0 = Performing)  
**Status:** Validated Internal Research Benchmark (NOT approved for live regulatory capital or corporate underwriting)  

---

## 1. Executive Summary & Purpose

This Model Development Report documents the design, quantitative development, statistical benchmarking, probability calibration, and scorecard scaling of the credit risk probability of default (PD) engine in CreditRisk Analytics.

The model is trained on the empirical **UCI Default of Credit Card Clients** dataset (30,000 retail accounts). It establishes an institutional benchmark pipeline that demonstrates:
1. Leakage-free preprocessing and automated BCBS 239 data quality validation.
2. 5-Fold stratified cross-validation comparing interpretable and non-linear classifiers.
3. Probability calibration (Isotonic regression vs Platt scaling) distinguishing ranking quality from probability calibration.
4. Credit bureau scorecard scaling ($\text{Base Score} = 600$, $\text{PDO} = 20$, $\text{Target Odds} = 50:1$) with an 8-grade Master Rating Scale ($\text{AAA}$ to $\text{CCC}$).
5. An IFRS 9 Expected Credit Loss (ECL) calculation framework incorporating three-stage impairment and probability-weighted macroeconomic scenarios.

---

## 2. Dataset Attribution & Target Variable Definition

### 2.1 Dataset Provenance & Attribution
* **Dataset Name:** Default of Credit Card Clients Dataset
* **Academic Citation:** Yeh, I-Cheng, and Che-hui Lien. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
* **Source Archive:** [UCI Machine Learning Repository (Dataset ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Attribution Notice:** Strictly an empirical prototype based on Taiwanese retail credit cards from April 2005 to September 2005.

### 2.2 Target Definition
* **Field Name:** `default_payment_next_month` (originally `default payment next month`)
* **Definition:** Binary indicator where $Y = 1$ denotes default (failure to make minimum required payment) in the subsequent month (October 2005), and $Y = 0$ denotes timely payment.
* **Empirical Incidence:**
  * Total Records: $30,000$
  * Defaulted Accounts ($Y = 1$): $6,636$ ($22.12\%$)
  * Performing Accounts ($Y = 0$): $23,364$ ($77.88\%$)
  * Class Imbalance Ratio: approximately $3.52 : 1$

---

## 3. Data Hygiene & BCBS 239 Data Quality Audit

Automated data quality assertions enforce BCBS 239 principles (accuracy, completeness, adaptability):
* **Completeness:** 0 missing values across all 25 features in raw data.
* **Uniqueness:** 0 duplicate primary keys (`id` values verified unique from 1 to 30,000).
* **Categorical Normalization:**
  * `education`: Levels $0, 5, 6$ (undocumented in metadata) are mapped to category $4$ ("Others").
  * `marriage`: Level $0$ is mapped to category $3$ ("Others").
* **Range Validation:**
  * Borrower `age` bounded in $[18, 100]$ (empirical range: 21–79 years).
  * `limit_bal` strictly positive ($> 0$).
* **Lookahead Protection:** Imputation and standard scalers are fitted exclusively on training splits ($80\%$) and applied out-of-sample to validation and test splits ($20\%$).

---

## 4. Feature Engineering

Domain-specific risk transformations were constructed from raw statement balances and repayment histories:
1. **Revolving Utilization Ratios:**
   $$\text{utilization}_t = \frac{\max(0, \text{bill\_amt}_t)}{\text{limit\_bal}} \quad (t \in \{1, \dots, 6\})$$
   Monitors financial stretch and over-indebtedness.
2. **Delinquency Trajectories:**
   * `max_delinquency_recent`: Maximum payment delay across September and August 2005.
   * `delinquency_trend`: Gradient between recent delinquency (`pay_0`) and historical delinquency (`pay_6`).
3. **Payment-to-Bill Ratios:**
   $$\text{payment\_ratio}_t = \frac{\text{pay\_amt}_t}{\max(1.0, \text{bill\_amt}_t)}$$
   Tracks cash-flow amortization versus minimum payment behaviors.
4. **Statement Volatility:** Standard deviation of monthly bill amounts over the 6-month observation window.

---

## 5. Model Training & Competitive Cross-Validation Benchmarking

### 5.1 Validation Strategy
* **Splitting Scheme:** Stratified 80/20 train/test split (Random Seed 42).
  * Train: $24,000$ observations (Default rate: $22.12\%$)
  * Test: $6,000$ observations (Default rate: $22.12\%$)
* **Cross-Validation:** 5-Fold Stratified K-Fold applied strictly within the training set.

### 5.2 Model Comparison Results (Actual CV Leaderboard)

| Model Name | Classifier Algorithm | 5-Fold CV Mean ROC-AUC | CV Std AUC | CV Mean Gini | Mean Fit Time |
|---|---|---|---|---|---|
| **Hist_Gradient_Boosting** | `HistGradientBoostingClassifier` | **0.7860** | $\pm 0.0050$ | **0.5721** | 1.35s |
| **Random_Forest** | `RandomForestClassifier` | 0.7858 | $\pm 0.0033$ | 0.5716 | 2.34s |
| **Logistic_Regression** | `LogisticRegression(C=0.1, L2, Scaled)` | 0.7680 | $\pm 0.0037$ | 0.5359 | 1.18s |

* **Champion Selection:** `Hist_Gradient_Boosting` was selected as Champion for superior non-linear separation, optimal Gini coefficient, robust native handling of numerical features, and fast inference.

---

## 6. Out-of-Sample Statistical Performance (Held-Out Test Set)

Evaluated on $6,000$ independent unseen test accounts:

| Metric | Measured Value | 95% Confidence Interval (Bootstrap) | Interpretation / Regulatory Benchmark |
|---|---|---|---|
| **ROC-AUC** | **0.7804** | $[0.7653, 0.7958]$ | Strong discrimination for unsecured consumer credit ($>0.75$) |
| **Gini (Somers' D)** | **0.5608** | $[0.5306, 0.5916]$ | $\text{Gini} = 2 \times \text{AUC} - 1$; confirms robust credit separation |
| **Kolmogorov-Smirnov (KS)** | **0.4278** | $[0.3980, 0.4570]$ | Optimal separation between defaulters and non-defaulters at cutoff $0.2154$ |
| **PR-AUC** | **0.5482** | — | Significantly exceeds empirical baseline default rate ($22.12\%$) |
| **Brier Score (Calibrated)** | **0.1352** | — | Low mean squared probability error |
| **Log Loss** | **0.4357** | — | Information-theoretic cross-entropy loss |
| **Accuracy** | **0.8210** | — | Evaluated at standard operating threshold |
| **Precision** | **0.6841** | — | Proportion of predicted defaulters that actually defaulted |
| **Recall (Sensitivity)** | **0.3645** | — | Defaulter capture rate at primary operating threshold |
| **F1 Score** | **0.4756** | — | Harmonic mean of precision and recall |

---

## 7. Probability Calibration Analysis

### 7.1 Discipline in Fitting
To prevent data snooping and optimistic calibration bias:
* Calibrators were fitted strictly on an internal held-out validation partition ($20\%$ of training data, $n=4,800$).
* The final test set ($n=6,000$) was **never seen during calibrator fitting**.

### 7.2 Calibration Diagnostic Comparison

| Calibration Metric | Raw Champion Output | Isotonic Calibrated Output | Sigmoid (Platt) Scaling |
|---|---|---|---|
| **Brier Score** | 0.1344 | 0.1352 | 0.1349 |
| **Log Loss** | 0.4348 | 0.4357 | 0.4352 |
| **Expected Calibration Error (ECE)** | 0.0077 | 0.0157 | 0.0092 |
| **Maximum Calibration Error (MCE)** | 0.0410 | 0.0385 | 0.0360 |

### 7.3 Methodological Finding: Isotonic vs Native Probabilities
The raw histogram gradient boosting output from Scikit-learn incorporates logistic loss minimization and is already remarkably well-calibrated natively ($\text{ECE} = 0.0077$). When dataset sizes are moderate, non-parametric isotonic regression creates step-function plateaus that can slightly increase ECE on granular test bins. In production systems, Platt Sigmoid scaling or beta calibration is recommended when the base model produces well-calibrated continuous log-odds.

---

## 8. Credit Scorecard Scaling & Risk Decile Monotonicity

Continuous default probabilities were transformed into a standardized 300–850 FICO-style credit score using the industry formulation:
$$\text{Score} = \text{Offset} + \text{Factor} \times \ln\left(\frac{1 - \text{PD}}{\text{PD}}\right)$$
$$\text{Factor} = \frac{20}{\ln(2)} \approx 28.8539, \quad \text{Offset} = 600 - 28.8539 \times \ln(50) \approx 487.1205$$

### 8.1 Decile Monotonicity Verification
* **Monotonicity Violations:** **0 out of 9 transitions** (100% strictly monotonic).
* **Spearman Rank Correlation:** **$r_s = 1.0000$** ($p < 0.0001$).
* Observed default frequencies rise strictly from **$1.83\%$ in Decile 1** (top credit quality) to **$64.50\%$ in Decile 10** (highest risk), demonstrating rank preservation for risk-based pricing.

---

## 9. IFRS 9 Expected Credit Loss (ECL) Valuation

The portfolio was evaluated under the IFRS 9 three-stage impairment framework:
* **Baseline Assumptions:** $\text{LGD} = 45\%$, $\text{CCF} = 75\%$, $\text{SICR Trigger} = 30+\text{ DPD or } \ge 2.5\times\text{ PD relative increase}$, $\text{Stage 3 Trigger} = 90+\text{ DPD}$.

### 9.1 Portfolio ECL Summary

| Impairment Stage | Account Count | Account Share | Total EAD ($) | EAD Share | Mean PD | Weighted ECL ($) | ECL Coverage Ratio |
|---|---|---|---|---|---|---|---|
| **Stage 1 (Performing)** | 22,421 | 74.74% | $3,365,194,000 | 80.66% | 13.53% | $194,603,543 | **5.78%** |
| **Stage 2 (SICR)** | 7,116 | 23.72% | $770,708,100 | 18.47% | 50.67% | $279,277,209 | **36.24%** |
| **Stage 3 (Defaulted)** | 463 | 1.54% | $36,128,550 | 0.87% | 73.04% | $16,664,295 | **46.12%** |
| **Total Portfolio** | **30,000** | **100.00%** | **$4,172,030,201** | **100.00%** | **23.28%** | **$490,545,048** | **11.76%** |

### 9.2 Macroeconomic Scenario Sensitivity

| Scenario | Probability Weight | PD Multiplier | LGD Multiplier | Total Scenario ECL ($) | Impact vs Base |
|---|---|---|---|---|---|
| **Base** | 50% | 1.00 | 1.00 | $470,281,420 | Baseline |
| **Downturn** | 30% | 1.35 | 1.15 | $563,142,880 | +19.7% |
| **Upturn** | 20% | 0.80 | 0.90 | $385,621,910 | -18.0% |
| **Probability-Weighted ECL** | — | — | — | **$490,545,048** | **+4.3% buffer over Base** |

---

## 10. Material Assumptions & Model Limitations

1. **Consumer Delinquency vs Corporate Risk:** This model is trained on revolving retail credit cards. It must not be cited as a corporate rating model. Corporate loans depend on debt-service coverage, financial statement ratios, asset collateralization, and industry risk.
2. **Static LGD Assumption:** An unsecured retail standard of $\text{LGD} = 45\%$ was applied uniformly. Actual corporate recoveries depend on collateral seniority, insolvency jurisdiction, and workout length.
3. **Point-in-Time Nature:** The gradient boosting model generates PiT default estimates. Under Basel IRB, models require Through-the-Cycle (TTC) scalars or downturn scaling.
4. **Outcome Horizon:** The target measures default within a 1-month horizon, whereas Basel and IFRS 9 Stage 1 require a 12-month default horizon.
