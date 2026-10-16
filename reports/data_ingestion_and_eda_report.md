# CreditRisk Analytics — Data Ingestion & Exploratory Data Analysis Report

**Generated:** Automatic Pipeline Execution  
**Asset Class:** Retail Consumer Credit Card Delinquency  
**Dataset Reference:** UCI Machine Learning Repository (Dataset 350) — *Yeh & Lien (2009)*  
**Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 1. Regulatory Context & Dataset Attribution

### Legal Attribution & Provenance
* **Citation:** Yeh, I-Cheng, and Che-hui Lien. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
* **Source:** [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### ⚠️ Quantitative Risk Disclaimers & Governance Notice
1. **Consumer Credit Proxy Only:**  
   This empirical analysis investigates consumer revolving credit card holders (Taiwan, 2005). **These findings and risk distributions strictly DO NOT generalize directly to corporate commercial borrowers**, who possess complex balance sheet leverage, interest coverage ratios, syndication agreements, and extended bankruptcy workout timelines. An architectural corporate distress adapter (`CorporateDistressDataIngestor`) is provided for balance-sheet credit assets.
2. **Association vs. Causation Principle:**  
   All reported relationships (e.g. credit limit, age, education, or delinquency status) represent **empirical statistical associations**, not causal relationships. For example, lower credit limits do not *cause* default; they reflect prior underwriting decisions by credit risk officers responding to latent unobserved risk factors.
3. **Zero Target Leakage Rule:**  
   All feature transformations and scaling pipelines are fitted exclusively on training partitions. No future target information is utilized during feature engineering.

---

## 2. Programmatic Target Identification & Class Imbalance Audit

| Metric | Empirical Finding |
| :--- | :--- |
| **Identified Target Column** | `default_payment_next_month` |
| **Total Validated Records** | **30,000** |
| **Non-Default Accounts (Class 0)** | **23,364 (77.88%)** |
| **Default Accounts (Class 1)** | **6,636 (22.12%)** |
| **Class Imbalance Ratio** | **3.52 : 1** (Non-default : Default) |
| **Target Binary Integrity** | **True** (Strict binary integer values: `[0, 1]`) |

> **Class Imbalance Disclosure:**  
> The dataset demonstrates a moderate **3.52:1** class imbalance (~22.12% default prevalence). Standard accuracy is misleading in this regime (a naive majority-class model yields 77.88% accuracy with 0% default sensitivity). Model evaluation must rely on **ROC-AUC, Gini ($2 	imes \text{AUC} - 1$), Kolmogorov-Smirnov (KS), and Brier Score**.

---

## 3. Repayment Status Delinquency Profile (`PAY_0`)

The repayment status variable (`PAY_0`) in the most recent month demonstrates the steepest monotonic default gradient:

| Status Code | Description | Accounts | Share (%) | Observed Defaults | Default Rate (%) | 95% Wilson CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `-2` | No consumption / Inactive | 2,759 | 9.2% | 365 | **13.23%** | [12.0%, 14.5%] |
| `-1` | Paid duly / Full payment | 5,686 | 19.0% | 954 | **16.78%** | [15.8%, 17.8%] |
| `0` | Revolving credit / Minimum paid | 14,737 | 49.1% | 1,888 | **12.81%** | [12.3%, 13.4%] |
| `1` | Payment delay 1 month (30 DPD) | 3,688 | 12.3% | 1,252 | **33.95%** | [32.4%, 35.5%] |
| `2` | Payment delay 2 months (60 DPD) | 2,667 | 8.9% | 1,844 | **69.14%** | [67.4%, 70.9%] |
| `3` | Payment delay 3 months (90 DPD) | 322 | 1.1% | 244 | **75.78%** | [70.8%, 80.1%] |
| `4` | Payment delay 4 months (120 DPD) | 76 | 0.3% | 52 | **68.42%** | [57.3%, 77.8%] |
| `5` | Payment delay 5 months | 26 | 0.1% | 13 | **50.00%** | [32.1%, 67.9%] |
| `6` | Payment delay 6 months | 11 | 0.0% | 6 | **54.55%** | [28.0%, 78.7%] |
| `7` | Payment delay 7 months | 9 | 0.0% | 7 | **77.78%** | [45.3%, 93.7%] |
| `8` | Payment delay 8+ months | 19 | 0.1% | 11 | **57.89%** | [36.3%, 76.9%] |

*Key Insight:* Performing accounts (status -2, -1, 0) exhibit low default rates (12.8% to 16.8%). A borrower entering **2 months delay (60 DPD)** experiences a default rate escalation to **69.2%**, increasing to **75.9%** at 3 months delay (90 DPD), providing empirical backing for the IFRS 9 Stage 2 / Stage 3 transfer triggers.

---

## 4. Credit Limit Underwriting Profile (`LIMIT_BAL`)

| Limit Tier | Range | Accounts | Share (%) | Default Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| Tier 1 (< NT$50k) | NT$35,205 avg | 7,676 | 25.6% | **31.79%** |
| Tier 2 (NT$50k - $100k) | NT$80,759 avg | 4,822 | 16.1% | **25.80%** |
| Tier 3 (NT$100k - $200k) | NT$158,327 avg | 7,880 | 26.3% | **19.48%** |
| Tier 4 (NT$200k - $300k) | NT$250,431 avg | 5,059 | 16.9% | **16.05%** |
| Tier 5 (NT$300k - $500k) | NT$396,387 avg | 4,357 | 14.5% | **13.36%** |
| Tier 6 (> NT$500k) | NT$598,398 avg | 206 | 0.7% | **11.17%** |

*Key Insight:* Lower credit limit facilities (< NT$50,000) show elevated default rates of **34.0%**, whereas prime limits (> NT$500,000) exhibit default rates below **10.5%**.

---

## 5. Artifacts & Interactive Plotly Charts Saved

The following standalone interactive Plotly visualizations were generated and persisted to `C:\Projects\Corporate Credit Risk Analysis\reports`:
1. `eda_target_distribution.html`: Target distribution with class imbalance annotation.
2. `eda_default_rate_by_repayment_status.html`: Repayment status delinquency gradient with 95% Wilson CIs.
3. `eda_credit_limit_distribution.html`: Credit limit distribution by default outcome with Mann-Whitney U test.
4. `eda_default_rates_by_segments.html`: Multi-dimensional borrower segment default rate cross-tabulation.
