# CreditRisk Analytics — Borrower Risk Segmentation Report
*Generated: 2026-10-16 16:52 UTC*

> **DISCLOSURE:** All risk segment boundaries (Low Risk, Medium-Low, Medium-High, High, Very High)
> are **illustrative thresholds for demonstration purposes only** and are NOT validated bank
> underwriting policy, regulatory minimums, or supervisory-approved credit guidelines.
> Actual segment boundaries require IRB model validation, backtesting, and regulatory sign-off.

---

## Part B: Borrower Risk Segmentation Results

### Risk Segment Summary (Illustrative Bands)

| Segment (Illustrative)   |   PD Band Low |   PD Band High |   Borrower Count |   Portfolio Share % |   Mean Predicted PD |   Observed Default Rate |   Calibration Error |   95% CI Lower |   95% CI Upper | Statistically Uncertain (n<30)   |
|:-------------------------|--------------:|---------------:|-----------------:|--------------------:|--------------------:|------------------------:|--------------------:|---------------:|---------------:|:---------------------------------|
| Low Risk                 |          0    |           0.05 |             1811 |                6.04 |             0.02189 |                 0.01988 |             0.00201 |        0.01439 |        0.0274  | False                            |
| Medium-Low Risk          |          0.05 |           0.15 |            12165 |               40.55 |             0.09631 |                 0.07431 |             0.022   |        0.06978 |        0.07911 | False                            |
| Medium-High Risk         |          0.15 |           0.3  |             8725 |               29.08 |             0.19981 |                 0.19553 |             0.00428 |        0.18734 |        0.20399 | False                            |
| High Risk                |          0.3  |           0.6  |             4617 |               15.39 |             0.43687 |                 0.42387 |             0.013   |        0.40968 |        0.43818 | False                            |
| Very High Risk           |          0.6  |           1.01 |             2682 |                8.94 |             0.74792 |                 0.75802 |             0.01009 |        0.74145 |        0.77385 | False                            |

**Key observations:**
- Segments with "Statistically Uncertain" = True contain fewer than 30 borrowers. Observed default
  rates in these buckets carry wide confidence intervals and should be interpreted with caution.
- The 95% confidence intervals are computed using the Wilson score method, which is preferred over
  the normal approximation for small samples and extreme proportions.
- The calibration error per segment reflects |Mean Predicted PD − Observed Default Rate|.

---

## Risk Decile Analysis (D1 = Lowest PD Decile, D10 = Highest PD Decile)

|   Decile |   PD Low |   PD High |   Count |   Mean Predicted PD |   Observed Default Rate |   Calibration Error |   95% CI Lower |   95% CI Upper | Monotone (↑ from prev)   | Statistically Uncertain   |
|---------:|---------:|----------:|--------:|--------------------:|------------------------:|--------------------:|---------------:|---------------:|:-------------------------|:--------------------------|
|        1 |  0.0001  |   0.06392 |    4825 |             0.04814 |                 0.03212 |             0.01602 |        0.02751 |        0.03748 |                          | False                     |
|        2 |  0.07982 |   0.07982 |    2823 |             0.07982 |                 0.05668 |             0.02315 |        0.04873 |        0.06582 | True                     | False                     |
|        3 |  0.08812 |   0.11111 |    1730 |             0.1111  |                 0.07977 |             0.03133 |        0.06791 |        0.09349 | True                     | False                     |
|        4 |  0.11187 |   0.1145  |    2900 |             0.11397 |                 0.09517 |             0.01879 |        0.08502 |        0.10639 | True                     | False                     |
|        5 |  0.11857 |   0.16279 |    3360 |             0.14926 |                 0.12946 |             0.0198  |        0.11853 |        0.14124 | True                     | False                     |
|        6 |  0.1799  |   0.1889  |    5018 |             0.18861 |                 0.18533 |             0.00327 |        0.17482 |        0.19632 | True                     | False                     |
|        7 |  0.18895 |   0.21538 |     495 |             0.21471 |                 0.25455 |             0.03984 |        0.21816 |        0.29471 | True                     | False                     |
|        8 |  0.22222 |   0.37334 |    2849 |             0.29124 |                 0.29238 |             0.00114 |        0.27597 |        0.30936 | True                     | False                     |
|        9 |  0.37931 |   0.57895 |    3012 |             0.47457 |                 0.45385 |             0.02072 |        0.43614 |        0.47168 | True                     | False                     |
|       10 |  0.58045 |   0.9375  |    2988 |             0.73087 |                 0.74163 |             0.01077 |        0.72563 |        0.75701 | True                     | False                     |

### Monotonicity Assessment

| Diagnostic | Result | Interpretation |
|---|---|---|
| Monotonicity violations | **0/9** | Number of decile transitions where observed DR *decreases* |
| Spearman rank correlation | **1.0000** | ±1.0 = perfect rank ordering; 0 = no ordering |
| Spearman p-value | **0.000000** | Significance of rank correlation |
| Overall calibration error | **0.01848** | Mean |predicted − observed| across deciles |

✅ **Fully monotone** — Observed default rate increases consistently across all deciles.

**Spearman rank interpretation:** r = 1.0000 indicates
strong
positive monotone ordering between decile rank and observed
default frequency (p = 0.000000< 0.001 — highly significant).

---

## Saved Outputs

| File | Description |
|---|---|
| `segmentation_decile_chart.html` | Decile default rate bar chart with CIs and monotonicity flags |
| `segmentation_calibration_bubble.html` | Segment-level calibration bubble chart |
| `segmentation_segment_bar.html` | Segment bar chart: count, observed DR, predicted PD |
| `segmentation_risk_bands.csv` | Full segment-level table |
| `segmentation_deciles.csv` | Full decile table |
