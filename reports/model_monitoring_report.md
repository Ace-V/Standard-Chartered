# CreditRisk Analytics — Model Monitoring & Ongoing Surveillance Report
**Document ID:** MRM-MON-2026-001  
**Project:** CreditRisk Analytics Surveillance Subsystem  
**Scope:** Input Data Hygiene, Covariate Drift, Model Discrimination Decay, and Operational Alerting  
**Regulatory Guidance:** US Federal Reserve SR 11-7 / OCC 2011-12, PRA SS1/23, EBA Guidelines on PD Estimation  
**Surveillance Frequency:** Monthly Data Quality & Drift; Semi-Annual Performance Discrimination (cohort maturation)  
**Status:** Verified Operational Framework (Internal Prototype)  

---

## 1. Executive Summary & Surveillance Mandate

In banking model risk management, model validation does not conclude at deployment. Ongoing surveillance is mandatory to ensure that the credit risk model operates within its approved operational envelope and that shifts in economic conditions, underwriting policies, or borrower demographics do not invalidate provisions or capital reserves.

The CreditRisk Analytics monitoring subsystem provides four decoupled surveillance modules:
1. **Input Data Quality Monitor:** Detects missing value spikes, out-of-range attributes, unexpected category codes, and schema changes before scoring occurs.
2. **Feature & Population Drift Monitor (PSI & CSI):** Tracks shifts in borrower characteristic distributions (CSI) and overall scoring distributions (PSI).
3. **Model Performance & Discrimination Monitor:** Tracks out-of-time decay in ROC-AUC, Gini, KS, and Brier score as outcome labels mature.
4. **Alerting & Audit Trail Engine:** Generates severity-tiered alerts (`INFO`, `WARNING`, `CRITICAL`), enforces analyst acknowledgment workflows, and logs records to the database.

---

## 2. Input Data Quality Surveillance

### 2.1 Automated Quality Assertions

| Data Quality Check | Method & Rule | Warning Threshold | Critical Threshold | Action on Failure |
|---|---|---|---|---|
| **Missing Values** | Null rate per feature | $> 1.0\%$ | $> 5.0\%$ | Alert raised; critical fails pipeline run |
| **Out-of-Bounds Numbers** | Numerical bounds validation | $> 0.5\%$ records | $> 2.0\%$ records | Outliers quarantined; clipping applied |
| **Unexpected Categories** | Category set matching | Any unmapped code | $> 1.0\%$ novel codes | Mapped to fallback "Others"; flagged |
| **Duplicate Keys** | Primary key uniqueness check | $> 0$ duplicate IDs | $> 0$ duplicate IDs | Pipeline aborted; integrity exception |
| **Schema Drift** | Missing or unexpected columns | $\ge 1$ unexpected col | Missing required col | Pipeline aborted; schema mismatch |

### 2.2 Baseline Validation Results
During the baseline pipeline execution across $30,000$ accounts:
* **Missing Rate:** $0.00\%$ across all columns (Status: **GREEN**).
* **Duplicate IDs:** $0$ duplicates identified (Status: **GREEN**).
* **Unmapped Categories:** $0$ unhandled categories after automated standardizer mapping (Status: **GREEN**).

---

## 3. Population & Feature Stability Surveillance (PSI & CSI)

### 3.1 Mathematical Formulation
The Population Stability Index (PSI) measures the divergence between a baseline reference distribution (development population) and a monitoring cohort:

$$\text{PSI} = \sum_{b=1}^{B} \left(\text{Actual}_b - \text{Expected}_b\right) \times \ln\left(\frac{\text{Actual}_b}{\text{Expected}_b}\right)$$

Where:
* $B = 10$ quantile bins defined on the baseline development population.
* $\text{Expected}_b$: Fraction of baseline borrowers in bin $b$.
* $\text{Actual}_b$: Fraction of monitoring cohort borrowers in bin $b$.
* Small constant ($\epsilon = 10^{-4}$) is applied to prevent division by zero or undefined logarithms.

### 3.2 Regulatory Traffic Light Thresholds

| PSI Range | Traffic Light | Statistical Interpretation | Mandatory Governance Action |
|---|---|---|---|
| $\text{PSI} < 0.10$ | 🟢 **GREEN** | **Stable Population** | Model operating normally; continue standard surveillance. |
| $0.10 \le \text{PSI} < 0.25$ | 🟡 **AMBER** | **Moderate Shift** | Heightened surveillance; inspect CSI by feature; evaluate credit policy changes. |
| $\text{PSI} \ge 0.25$ | 🔴 **RED** | **Significant Drift** | Mandatory model recalibration or replacement; review underwriting score cutoffs. |

### 3.3 Characteristic Stability Index (CSI) by Feature
Characteristic Stability Index (CSI) applies the PSI formulation to individual continuous and categorical input features to isolate the root cause of population shift:

| Monitored Feature | Baseline vs OOT PSI | Traffic Light | Interpretation / Driver |
|---|---|---|---|
| **Calibrated PD (Overall)** | **0.0010** | 🟢 **GREEN** | Score distribution is perfectly stable across development and test splits. |
| `limit_bal` (Credit Limit) | 0.0024 | 🟢 **GREEN** | Credit line assignments remain consistent. |
| `utilization_1` (Recent Util) | 0.0031 | 🟢 **GREEN** | Borrower revolving usage patterns are stable. |
| `pay_0` (Recent Repayment) | 0.0018 | 🟢 **GREEN** | Recent delinquency incidence is consistent with baseline. |
| `age` (Borrower Age) | 0.0012 | 🟢 **GREEN** | Demographic distribution is identical. |

---

## 4. Methodological Distinction: Data Drift vs Model Deterioration

A critical responsibility of model validation is disentangling **covariate shift** from **concept drift**:

```
                         Distribution Shifts
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   Covariate Shift (Data Drift)                   Concept Drift (Model Decay)
  • Shifts in P(X)                               • Shifts in P(Y|X)
  • Borrower population characteristics change   • The fundamental relationship between
    (e.g., younger borrowers, higher limits)       features and default changes
  • Model discrimination (AUC) may hold          • Model discrimination (AUC, Gini) drops
  • Detected via PSI and CSI                     • Detected via out-of-time ROC-AUC & Brier
  • Addressed by score cut adjustment            • Mandatory model recalibration/retraining
```

### 4.1 False Alarm Prevention
* **High PSI with Stable AUC:** If the borrower profile shifts toward higher credit limits (elevating CSI for `limit_bal`), but the model continues to rank defaults accurately, the model does not require retraining. Instead, credit underwriting tiers or cutoffs may need minor adjustment.
* **Low PSI with Deteriorating AUC:** If macro deterioration causes borrowers with historically clean payment histories to default simultaneously (e.g., systemic economic shock), PSI will remain low while discrimination degrades rapidly. Tracking both dimensions is essential.

---

## 5. Performance Monitoring with Delayed Outcomes (Maturity Windows)

### 5.1 The Outcome Lag Challenge
In 12-month credit risk modelling:
* Borrowers scored in month $M$ cannot be confirmed as defaults or non-defaults until month $M + 12$.
* Attempting to compute ROC-AUC, Brier score, or KS statistic on unlabelled recent cohorts ($< 12$ months mature) produces biased or misleading metrics.

### 5.2 Two-Track Surveillance Strategy
1. **Recent Cohorts ($0 \le \text{Age} < 12$ months):**
   * Monitored exclusively using **Data Quality** and **Population Stability (PSI/CSI)**.
   * Model performance metrics are marked as `UNMATURED` with explicit disclaimers.
   * No false performance degradation alerts are triggered.
2. **Mature Cohorts ($\text{Age} \ge 12$ months):**
   * Realized default labels are joined with historical scores.
   * ROC-AUC, Gini, KS, and calibration curves are computed.
   * Basel binomial traffic light tests evaluate whether actual defaults exceed predicted bands.

---

## 6. Operational Alerting & Escalation Framework

### 6.1 Severity Tiers & SLA Requirements

| Severity Tier | Trigger Conditions | Operational SLA | Escalation Pathway |
|---|---|---|---|
| **INFO** | Minor shifts ($\text{PSI} \approx 0.05$); data format warnings | 5 Business Days | Model Risk Analyst review; logged in database audit trail |
| **WARNING** | Moderate drift ($0.10 \le \text{PSI} < 0.25$); missing data $>1\%$ | 48 Hours | Credit Risk Manager review; root-cause diagnosis |
| **CRITICAL** | Severe drift ($\text{PSI} \ge 0.25$); $\text{ROC-AUC} < 0.70$; missing data $>5\%$ | 24 Hours | Chief Risk Officer notification; model committee review |

### 6.2 Analyst Acknowledgment Workflow
All alerts are persisted to the `alert_records` database table:
* Alerts remain in state `acknowledged = False` until reviewed by an analyst.
* The system provides programmatic acknowledgment via `DatabaseManager.acknowledge_alert(alert_id)`.
* Every resolution requires an audit note detailing root cause and remedial actions.

---

## 7. Stress & Drift Simulation Scenarios

The monitoring subsystem includes an integrated simulation engine (`src.monitoring.simulation.SimulationEngine`) to test surveillance response across synthetic scenarios:
1. **Clean Baseline:** Healthy distribution; confirms all monitors produce GREEN status.
2. **Covariate Shift Scenario:** Injects $+25\%$ limit increases and demographic shifts; confirms PSI triggers AMBER/RED while logging specific CSI features.
3. **Concept Drift Scenario:** Artificially breaks the correlation between delinquency and default; verifies that mature cohort performance monitor flags ROC-AUC drop.
4. **Data Quality Defect Scenario:** Injects nulls and out-of-range values; verifies immediate quarantine and data quality alert generation.
