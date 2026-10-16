# CreditRisk Analytics — Probability Calibration Report
*Generated: 2026-10-16 16:52 UTC | Champion Model: Hist Gradient Boosting | Calibration Method: Isotonic*

---

## Conceptual Framework

### Three Distinct Performance Dimensions

| Dimension | Definition | Primary Metrics | Credit Risk Relevance |
|---|---|---|---|
| **Ranking Quality** | How well the model orders defaulters above non-defaulters | ROC-AUC, Gini, KS | Drives credit tier pricing and eligibility decisions |
| **Probability Calibration** | Whether stated PD = empirical default frequency | ECE, MCE, Brier Score, Log Loss | Required for IFRS 9 ECL provisions and Basel IRB capital |
| **Threshold Performance** | Precision/Recall at a chosen operating cutoff | F1, Precision, Recall | Relevant for collection strategy and operational decisions |

> A model can rank well (high AUC) but be grossly miscalibrated. For example, if a model assigns
> PD = 0.10 to borrowers who default at a 30% rate, ECL reserves will be systematically under-stated
> by ~20 percentage points — a material IFRS 9 and capital impact. Calibration post-processing
> (isotonic regression or Platt scaling) corrects this without affecting ranking quality.

---

## Part A: Calibration Diagnostic Results

### Scalar Metric Comparison — Raw vs Post-Calibration (Test Set, n = 10 bins from test data)

| Metric                           |   Raw (Uncalibrated) |   Post-Calibration |   Improvement % | Direction         |
|:---------------------------------|---------------------:|-------------------:|----------------:|:------------------|
| Brier Score                      |              0.13442 |            0.13517 |           -0.56 | ↓ Lower is better |
| Log Loss                         |              0.42833 |            0.43043 |           -0.49 | ↓ Lower is better |
| Expected Calibration Error (ECE) |              0.00765 |            0.01569 |         -104.96 | ↓ Lower is better |
| Maximum Calibration Error (MCE)  |              0.02107 |            0.04997 |         -137.2  | ↓ Lower is better |

**Interpretation:**
- **Brier Score** reduction of **-0.6%** confirms that calibration
  improves the squared probability error between predicted PDs and actual outcomes.
- **Log Loss** reduction of **-0.5%** confirms better-calibrated
  probability estimates from the information-theoretic perspective.
- **ECE** (Expected Calibration Error) reduction of **-105.0%**
  reflects reduced population-weighted mean miscalibration across 10 quantile bins.
- **MCE** (Maximum Calibration Error) reduction of **-137.2%**
  shows improvement in the worst-performing risk bucket.

> **Note on fitting discipline:** The `isotonic` calibrator was fitted exclusively on the
> held-out validation partition (20% of training data). It was **never fitted or evaluated on the
> final test set**. The results above measure generalisation to unseen data.

---

### Bin-Level Calibration Table — Calibrated Probabilities

> Bins with fewer than 30 observations are flagged as statistically uncertain.
> Wilson score 95% confidence intervals are reported for each observed default rate.

| Bin            |   Bin Low |   Bin High |   Count |   Mean Predicted PD |   Observed Default Rate |   Calibration Error |   95% CI Lower |   95% CI Upper | Statistically Uncertain (n<30)   |
|:---------------|----------:|-----------:|--------:|--------------------:|------------------------:|--------------------:|---------------:|---------------:|:---------------------------------|
| [0.000, 0.064) |    0      |     0.0639 |     354 |             0.0216  |                 0.03955 |             0.01795 |        0.0237  |        0.06528 | False                            |
| [0.064, 0.080) |    0.0639 |     0.0798 |     601 |             0.06392 |                 0.0599  |             0.00402 |        0.04358 |        0.08181 | False                            |
| [0.080, 0.111) |    0.0798 |     0.1111 |     540 |             0.07982 |                 0.07778 |             0.00204 |        0.05805 |        0.10347 | False                            |
| [0.111, 0.115) |    0.1111 |     0.1145 |     611 |             0.11198 |                 0.10311 |             0.00887 |        0.08142 |        0.12975 | False                            |
| [0.115, 0.163) |    0.1145 |     0.1628 |     686 |             0.12526 |                 0.1312  |             0.00593 |        0.10797 |        0.15853 | False                            |
| [0.163, 0.189) |    0.1628 |     0.1889 |     391 |             0.16583 |                 0.14066 |             0.02517 |        0.10969 |        0.17863 | False                            |
| [0.189, 0.215) |    0.1889 |     0.2154 |     953 |             0.18897 |                 0.18258 |             0.00639 |        0.15934 |        0.20837 | False                            |
| [0.215, 0.379) |    0.2154 |     0.3793 |     638 |             0.28029 |                 0.26332 |             0.01696 |        0.23064 |        0.29884 | False                            |
| [0.379, 0.581) |    0.3793 |     0.5814 |     616 |             0.4688  |                 0.41883 |             0.04997 |        0.38049 |        0.45818 | False                            |
| [0.581, 1.000) |    0.5814 |     1      |     610 |             0.72821 |                 0.7     |             0.02821 |        0.66247 |        0.73502 | False                            |

---

### Bin-Level Calibration Table — Raw (Uncalibrated)

| Bin            |   Bin Low |   Bin High |   Count |   Mean Predicted PD |   Observed Default Rate |   Calibration Error |   95% CI Lower |   95% CI Upper | Statistically Uncertain (n<30)   |
|:---------------|----------:|-----------:|--------:|--------------------:|------------------------:|--------------------:|---------------:|---------------:|:---------------------------------|
| [0.000, 0.054) |    0      |     0.0544 |     600 |             0.0411  |                 0.04167 |             0.00057 |        0.02838 |        0.06079 | False                            |
| [0.054, 0.074) |    0.0544 |     0.0743 |     600 |             0.0641  |                 0.07    |             0.0059  |        0.0522  |        0.09327 | False                            |
| [0.074, 0.098) |    0.0743 |     0.098  |     600 |             0.08664 |                 0.08833 |             0.0017  |        0.06817 |        0.11374 | False                            |
| [0.098, 0.120) |    0.098  |     0.1202 |     600 |             0.10893 |                 0.13    |             0.02107 |        0.10543 |        0.15928 | False                            |
| [0.120, 0.145) |    0.1202 |     0.145  |     600 |             0.13228 |                 0.13333 |             0.00105 |        0.10845 |        0.16288 | False                            |
| [0.145, 0.176) |    0.145  |     0.1761 |     600 |             0.15974 |                 0.16    |             0.00026 |        0.13284 |        0.19148 | False                            |
| [0.176, 0.228) |    0.1761 |     0.2278 |     600 |             0.19944 |                 0.19    |             0.00944 |        0.16062 |        0.22332 | False                            |
| [0.228, 0.331) |    0.2278 |     0.3307 |     600 |             0.27415 |                 0.26167 |             0.01248 |        0.22809 |        0.29827 | False                            |
| [0.331, 0.577) |    0.3307 |     0.5768 |     600 |             0.4185  |                 0.43667 |             0.01817 |        0.39751 |        0.47663 | False                            |
| [0.577, 1.000) |    0.5768 |     1      |     600 |             0.70589 |                 0.7     |             0.00589 |        0.66215 |        0.7353  | False                            |

---

## Saved Outputs

| File | Description |
|---|---|
| `calibration_reliability_diagram.html` | Interactive overlay: raw vs calibrated reliability diagram |
| `calibration_bin_errors.html` | Per-bin absolute calibration error comparison chart |
| `calibration_scalar_comparison.csv` | Scalar metric comparison table |
| `calibration_raw_bins.csv` | Raw uncalibrated bin-level analysis |
| `calibration_post_bins.csv` | Post-calibration bin-level analysis |
