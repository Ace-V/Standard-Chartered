# Standard Chartered SCMAC — Quantitative Risk & Analytics Interview Guide

**Target Unit:** Standard Chartered Model Assessment & Analytics Centre (SCMAC) / Model Risk Management (MRM)  
**Coverage:** 
- **Part I:** Corporate & Retail Credit Risk Analytics (PD, LGD, EAD, IFRS 9, Basel IRB, Scorecard Scaling, Model Surveillance)
- **Part II:** Time Series Forecasting, Hybrid Deep Learning & Stochastic Simulation (ARIMA, Prophet, LSTM, Monte Carlo, Market Risk)

---

# Part I: Credit Risk Modelling & Impairment Analytics (CCRA)

---

### [TOPIC 1: PROBABILITY OF DEFAULT & STATISTICAL DISCRIMINATION]

#### Question 1: Ranking Quality vs. Probability Calibration
**Question:**  
What is the theoretical distinction between ranking quality and probability calibration? Why can a machine learning credit model achieve a stellar ROC-AUC of 0.88 yet fail regulatory validation under IFRS 9?

**Technically Correct Answer:**  
* **Ranking Quality (Discrimination):** Evaluates how effectively the model orders defaulters relative to non-defaulters. Metrics such as **ROC-AUC**, **Gini (Somers' D)**, and the **Kolmogorov-Smirnov (KS) statistic** are rank-order statistics invariant to any strictly monotonic transformation of predicted probabilities. If model score $s_A > s_B$, borrower $A$ is ranked as riskier than $B$.
* **Probability Calibration:** Evaluates whether stated default probabilities match empirical default frequencies:
  $$\mathbb{E}[Y \mid \hat{p}(X) = p] = p$$
  Assessed via the **Brier score**, **Log Loss**, and **Expected Calibration Error (ECE)**.
* **Why High AUC Models Fail IFRS 9:**  
  A model can discriminate ranks with 0.88 AUC while predicting squashed probabilities (e.g., predicting $\hat{p} = 4\%$ for a segment defaulting at $20\%$). Under IFRS 9:
  $$\text{ECL} = \text{EAD} \times \text{PD} \times \text{LGD}$$
  Because credit loss provisions scale linearly with dollar exposure, a 16% underestimation in PD directly translates to balance sheet under-provisioning. In banking validation, high AUC guarantees tiering ability, but rigorous probability calibration is required before outputs can drive balance sheet provisioning or capital allocation.

---

#### Question 2: ROC-AUC, Gini Coefficient, and KS Statistic Formulation
**Question:**  
Define the mathematical relationships linking ROC-AUC, Gini (Somers' D), and the Kolmogorov-Smirnov (KS) statistic. What are the regulatory benchmark values expected in unsecured retail vs. corporate wholesale portfolios?

**Technically Correct Answer:**  
* **ROC-AUC & Gini Relationship:**  
  The Gini coefficient is the linear transformation of the area under the Receiver Operating Characteristic curve:
  $$\text{Gini} = 2 \times \text{ROC-AUC} - 1$$
  Gini reflects Somers' $D$ rank correlation between predicted scores and actual default indicators.
* **Kolmogorov-Smirnov (KS) Statistic:**  
  The maximum vertical divergence between cumulative distribution functions of defaulters ($F_{\text{Bad}}$) and non-defaulters ($F_{\text{Good}}$):
  $$\text{KS} = \max_{s} \left| F_{\text{Bad}}(s) - F_{\text{Good}}(s) \right|$$
  The score $s^*$ where this supremum occurs defines the optimal statistical separation cutoff.
* **Regulatory & Industry Benchmarks:**
  * **Retail Unsecured Revolvers (Credit Cards):** Typical $\text{ROC-AUC} \in [0.72, 0.80]$ ($\text{Gini} \in [0.44, 0.60]$), $\text{KS} \in [0.35, 0.45]$.
  * **Corporate / Large Wholesale Portfolios:** Low default portfolios (LDP) often exhibit $\text{ROC-AUC} \in [0.75, 0.88]$, but small sample sizes demand bootstrap confidence intervals to prove statistical stability.

---

#### Question 3: Non-parametric Isotonic Regression vs. Parametric Platt Scaling
**Question:**  
Compare Isotonic Regression against Platt Scaling for credit model probability calibration. Under what data conditions would you reject Isotonic Regression in favor of Platt scaling or Beta calibration?

**Technically Correct Answer:**  
* **Isotonic Regression:**  
  Fits a non-parametric, piece-wise constant monotonic step function by minimizing squared errors via the Pair Adjacent Violators Algorithm (PAVA):
  $$\min \sum_{i=1}^{N} \left( y_i - m(\hat{p}_i) \right)^2 \quad \text{subject to } m(\hat{p}_i) \le m(\hat{p}_j) \text{ whenever } \hat{p}_i \le \hat{p}_j$$
* **Platt Scaling:**  
  Fits a parametric sigmoid transformation to model log-odds via cross-entropy minimization:
  $$P(Y=1 \mid s) = \frac{1}{1 + \exp(A \cdot s + B)}$$
* **Why Reject Isotonic Regression:**  
  Isotonic regression is unconstrained between steps, leading to flat plateaus where diverse raw scores receive identical calibrated PDs. In thin or moderate validation samples ($n < 10,000$), it is susceptible to overfitting noise at the distribution tails. When the base model (such as gradient boosting with log loss) outputs smooth, monotonic continuous log-odds, **Platt scaling or Beta calibration** is preferred because it preserves score continuity and avoids step-function distortions.

---

### [TOPIC 2: SCORECARD ENGINEERING & RATING MECHANICS]

#### Question 4: Credit Scorecard Scaling Mathematics
**Question:**  
Derive the standard credit bureau scorecard formula transforming continuous default probability into a 300–850 credit score using Base Score, Points to Double Odds (PDO), and Target Odds.

**Technically Correct Answer:**  
A scorecard maps the log-odds of a borrower being "good" (non-defaulting) to a linear numerical scale:
$$\text{Odds} = \frac{P(\text{Good})}{P(\text{Bad})} = \frac{1 - \text{PD}}{\text{PD}}$$
$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

To determine $\text{Factor}$ and $\text{Offset}$:
1. If odds double from $\text{Odds}_0$ to $2 \times \text{Odds}_0$, the score increases by $\text{PDO}$:
   $$\text{Score} + \text{PDO} = \text{Offset} + \text{Factor} \times \ln(2 \times \text{Odds}_0) = \text{Score} + \text{Factor} \times \ln(2)$$
   $$\implies \text{Factor} = \frac{\text{PDO}}{\ln(2)}$$
2. At baseline $\text{BaseOdds}$, the score equals $\text{BaseScore}$:
   $$\text{BaseScore} = \text{Offset} + \text{Factor} \times \ln(\text{BaseOdds})$$
   $$\implies \text{Offset} = \text{BaseScore} - \text{Factor} \times \ln(\text{BaseOdds})$$

*For $\text{BaseScore} = 600$, $\text{PDO} = 20$, and $\text{BaseOdds} = 50:1$:*
$$\text{Factor} = \frac{20}{\ln(2)} \approx 28.8539$$
$$\text{Offset} = 600 - 28.8539 \times \ln(50) \approx 600 - 112.8795 = 487.1205$$
$$\text{Score} = 487.1205 + 28.8539 \times \ln\left(\frac{1 - \text{PD}}{\text{PD}}\right)$$

---

#### Question 5: Master Rating Scale Design & Decile Monotonicity
**Question:**  
Why is decile monotonicity mandatory in credit risk rating systems? How do you statistically validate whether a rating scale suffers from inversion?

**Technically Correct Answer:**  
* **Monotonicity Requirement:**  
  Borrowers grouped into ascending risk buckets (Decile 1 to 10 or AAA to CCC) must exhibit strictly non-decreasing observed default rates:
  $$\text{ODR}_{k} \le \text{ODR}_{k+1} \quad \forall k \in \{1, \dots, K-1\}$$
  A non-monotonic step (where a lower-risk band experiences a higher default rate than a higher-risk band) indicates rating inversion, undermining credit pricing, risk-weighted asset calculations, and underwriting delegation.
* **Statistical Validation:**
  1. **Monotonicity Inversion Count:** Count of adjacent pair violations:
     $$\text{Violations} = \sum_{k=1}^{K-1} \mathbb{I}\left(\text{ODR}_{k} > \text{ODR}_{k+1}\right)$$
  2. **Spearman Rank Correlation:** Rank correlation between assigned bucket rank and empirical default frequency:
     $$r_s = 1 - \frac{6 \sum d_i^2}{K(K^2 - 1)}$$
     A robust rating model requires $r_s \approx 1.000$ ($p < 0.01$) and zero inversions.

---

#### Question 6: Basel Binomial Traffic Light Backtesting Test
**Question:**  
Explain the mechanics of the Basel Binomial Traffic Light backtesting framework. How are Green, Amber, and Red zones determined for rating grade default rates?

**Technically Correct Answer:**  
Under the Basel Framework (BCBS guidelines on PD validation), realized defaults in each rating grade are modelled as a sequence of independent Bernoulli trials:
$$D_k \sim \text{Binomial}(N_k, \text{PD}_k)$$
Where $N_k$ is the number of active obligors in grade $k$, and $\text{PD}_k$ is the approved regulatory probability of default.

The validation test calculates the cumulative one-sided binomial probability of observing $K_k$ or more defaults:
$$p\text{-value} = P(X \ge K_k) = \sum_{x = K_k}^{N_k} \binom{N_k}{x} \text{PD}_k^x (1 - \text{PD}_k)^{N_k - x}$$

* **Zone Thresholds:**
  * 🟢 **Green Zone ($p \ge 0.05$):** Realized defaults do not exceed statistical expectations. The model is conservative or well-calibrated.
  * 🟡 **Amber Zone ($0.001 \le p < 0.05$):** Statistical anomaly at the 95% level; indicates potential underestimation. Triggers heightened supervisory monitoring.
  * 🔴 **Red Zone ($p < 0.001$):** Rejects the hypothesis that true PD is $\le \text{PD}_k$ at the 99.9% confidence level. Mandatory supervisory capital add-on and model recalibration.

---

### [TOPIC 3: IFRS 9 / CECL IMPAIRMENT & CAPITAL GOVERNANCE]

#### Question 7: Three-Stage Impairment & The SICR "Cliff Effect"
**Question:**  
Explain the staging criteria under IFRS 9. How does Significant Increase in Credit Risk (SICR) trigger a provision "cliff effect," and what quantitative rules govern Stage 1 to Stage 2 migration?

**Technically Correct Answer:**  
* **IFRS 9 Staging Classification:**
  * **Stage 1 (Performing):** Credit risk has not increased significantly since origination. Provision = **12-month Expected Credit Loss ($\text{ECL}_{12\text{m}}$)**.
  * **Stage 2 (Underperforming - SICR):** Significant Increase in Credit Risk since origination, but no objective credit impairment. Provision = **Lifetime Expected Credit Loss ($\text{ECL}_{\text{Lifetime}}$)**.
  * **Stage 3 (Credit-Impaired / Defaulted):** Objective evidence of impairment ($90+$ DPD or insolvency). Provision = **Lifetime ECL calculated on net carrying amount**.
* **The "Cliff Effect":**  
  Because revolving and term loans typically have multi-year contractual or behavioural horizons, migrating an account from Stage 1 (12-month horizon) to Stage 2 (lifetime horizon) instantly scales required reserves by $300\%\text{ to }600\%$. This sudden spike in credit impairment hits bank P&L and tier-1 capital reserves.
* **Quantitative Migration Rules:**
  1. **Relative PD Deterioration:** Ratio of current lifetime PD to origination lifetime PD:
     $$\frac{\text{PD}_{\text{current}}}{\text{PD}_{\text{origination}}} \ge \theta_{\text{SICR}} \quad (\text{e.g., } 2.5\times\text{ to }3.0\times)$$
  2. **Regulatory Backstop:** $30+$ Days Past Due (DPD) mandates Stage 2 classification regardless of statistical score.

---

#### Question 8: Forward-Looking Macroeconomic Scenario Weighting
**Question:**  
Why does IFRS 9 prohibit using a single baseline economic forecast for provisioning? How is probability-weighted scenario expectation computed?

**Technically Correct Answer:**  
* **Non-Linear Loss Dynamics:**  
  Credit losses are convex non-linear functions of macroeconomic variables (GDP contraction, unemployment spikes cause exponential default increases). Using only a baseline ("most likely") economic forecast underestimates expected losses due to Jensen's Inequality:
  $$\mathbb{E}[\text{ECL}(X)] \ne \text{ECL}(\mathbb{E}[X])$$
* **Mathematical Weighting:**  
  Banks must incorporate multiple forward-looking probability-weighted macroeconomic scenarios:
  $$\text{ECL}_{\text{Final}} = \sum_{s=1}^{S} w_s \cdot \text{ECL}_s(\text{PD}_s, \text{LGD}_s, \text{EAD}_s) \quad \text{where } \sum_{s=1}^{S} w_s = 1.0$$
  *Typical portfolio configuration:*
  * **Baseline ($w_{\text{Base}} = 0.50$):** Central bank consensus trajectory.
  * **Downturn ($w_{\text{Down}} = 0.30$):** Severe stress scenario ($1.35\times\text{PD}, 1.15\times\text{LGD}$).
  * **Upturn ($w_{\text{Up}} = 0.20$):** Favourable economic scenario ($0.80\times\text{PD}, 0.90\times\text{LGD}$).

---

#### Question 9: Point-in-Time (PiT) vs. Through-the-Cycle (TTC) PDs
**Question:**  
Contrast Point-in-Time (PiT) and Through-the-Cycle (TTC) default probabilities. Which is mandated for IFRS 9 and which for Basel IRB capital, and how do you convert between them?

**Technically Correct Answer:**  
* **Point-in-Time (PiT) PD:**  
  Reflects the probability of default under current economic conditions and recent borrower trends. Highly procyclical: rises during recessions and falls during booms. **Mandated for IFRS 9 accounting provisions**.
* **Through-the-Cycle (TTC) PD:**  
  Reflects average long-run default likelihood across an entire economic cycle, removing cyclical macroeconomic fluctuations. **Mandated for Basel II/III Internal Ratings-Based (IRB) capital calculations** to prevent procyclical bank capital requirements.
* **Conversion Methodology (Vasicek Single Factor Model):**  
  Converting between TTC and PiT utilizes the Merton-Vasicek structural distribution:
  $$\text{PD}_{\text{PiT}}(Z) = \Phi\left( \frac{\Phi^{-1}(\text{PD}_{\text{TTC}}) - \sqrt{\rho} \cdot Z}{\sqrt{1 - \rho}} \right)$$
  Where $\Phi$ is the standard normal cumulative distribution, $\rho$ is asset correlation, and $Z$ represents the standardized systematic macroeconomic factor.

---

#### Question 10: Exposure at Default (EAD) & Credit Conversion Factor (CCF)
**Question:**  
How is Exposure at Default (EAD) calculated for revolving credit lines? What is the role of Credit Conversion Factors (CCF)?

**Technically Correct Answer:**  
For revolving credit facilities (e.g., credit cards, corporate overdrafts), total exposure consists of the current drawn balance plus anticipated drawdown on undrawn limits prior to default:
$$\text{EAD} = \text{Drawn Balance} + \text{CCF} \times \max\left(0, \text{Limit} - \text{Drawn Balance}\right)$$

* **Credit Conversion Factor (CCF):**  
  The percentage of undrawn commitment expected to be drawn down prior to default event:
  $$\text{CCF} = \frac{\text{Drawdown at Default} - \text{Drawn Balance}_{t}}{\text{Undrawn Commitment}_{t}}$$
  * Basel Foundation IRB prescribes a standard retail CCF of **75%** for revolving lines.
  * In Advanced IRB (A-IRB), banks estimate empirical CCF models factoring in historical utilization acceleration and credit limit management actions.

---

### [TOPIC 4: MODEL RISK SURVEILLANCE & DRIFT MONITORING]

#### Question 11: Population Stability Index (PSI) & Characteristic Stability Index (CSI)
**Question:**  
State the mathematical formulation of the Population Stability Index (PSI). Why does PSI use Kullback-Leibler symmetric divergence, and what do the standard banking thresholds signify?

**Technically Correct Answer:**  
* **Mathematical Formulation:**  
  Given $B$ quantile bins determined on the baseline development reference set:
  $$\text{PSI} = \sum_{b=1}^{B} \left( A_b - E_b \right) \times \ln\left( \frac{A_b}{E_b} \right)$$
  Where $E_b$ is the expected proportion of baseline accounts in bin $b$, and $A_b$ is the actual proportion of accounts in the monitoring cohort.
* **Connection to Information Theory:**  
  PSI is the symmetric form of Kullback-Leibler (KL) divergence (Jeffreys divergence):
  $$\text{PSI} = D_{\text{KL}}(A \parallel E) + D_{\text{KL}}(E \parallel A)$$
  The weighting $(A_b - E_b)$ ensures that bins with equal shares contribute zero drift, while large discrepancies contribute heavily to the total index.
* **Banking Traffic Light Standards:**
  * $\text{PSI} < 0.10$: 🟢 **Green (Stable Population):** No material distribution shift. Standard ongoing surveillance.
  * $0.10 \le \text{PSI} < 0.25$: 🟡 **Amber (Moderate Drift):** Notable population shift. Model Risk team inspects Characteristic Stability Indices (CSI) on individual features.
  * $\text{PSI} \ge 0.25$: 🔴 **Red (Severe Drift):** Substantial structural population shift. Mandates model recalibration or redevelopment.

---

#### Question 12: Covariate Shift (Data Drift) vs. Concept Drift (Performance Decay)
**Question:**  
How do you distinguish covariate shift from concept drift in credit risk surveillance? What operational decisions follow each?

**Technically Correct Answer:**  
* **Covariate Shift (Data Drift):**  
  A change in the marginal distribution of inputs $P(X)$, while the conditional probability of default $P(Y \mid X)$ remains unchanged:
  $$P_{\text{baseline}}(X) \ne P_{\text{monitor}}(X) \quad \text{but} \quad P(Y \mid X) \text{ is constant}$$
  *Example:* Marketing campaigns attract younger borrowers, shifting the age distribution (high CSI on `age`), but the credit model still correctly ranks defaulters among young borrowers.
  *Decision:* Model discrimination holds. Do not retrain the model; adjust underwriting approval cutoffs or policy tiers.
* **Concept Drift (Model Degradation):**  
  A breakdown in the relationship between borrower characteristics and default outcomes:
  $$P_{\text{baseline}}(Y \mid X) \ne P_{\text{monitor}}(Y \mid X)$$
  *Example:* A macroeconomic shock or regulatory change causes historically prime borrowers to default unexpectedly. ROC-AUC and Gini drop sharply on mature cohorts.
  *Decision:* Model logic is invalidated. Mandates full model redevelopment or recalibration.

---

#### Question 13: Delayed Performance Outcomes in Matured vs. Unmatured Credit Cohorts
**Question:**  
Credit risk defaults require 12 months to observe. How must a model monitoring subsystem handle recent cohorts without generating false performance decay alarms?

**Technically Correct Answer:**  
Calculating ROC-AUC, Brier score, or confusion matrices on recent origination cohorts ($< 12$ months old) introduces severe truncation bias (defaulters who default in month 8 or 10 have not yet defaulted).

* **The Two-Track Surveillance Protocol:**
  1. **Unmatured Track ($0 \le \text{Cohort Age} < 12$ Months):**  
     Monitored exclusively using **non-outcome metrics**:
     * Input Data Quality (null spikes, range boundary breaches, categorical changes).
     * Population Stability (PSI on scores, CSI on input variables).
     * Performance metrics are explicitly suppressed and marked as `UNMATURED`.
  2. **Matured Track ($\text{Cohort Age} \ge 12$ Months):**  
     Historical predictions are joined with realized 12-month default outcomes:
     * Full statistical evaluation: ROC-AUC, Gini, KS, PR-AUC, and Brier calibration curves.
     * Basel Binomial backtests evaluate whether observed defaults in each rating grade match predicted bands.

---

### [TOPIC 5: MODEL RISK GOVERNANCE & ENGINEERING INTEGRITY]

#### Question 14: BCBS 239 Data Quality Principles & Target Leakage Prevention
**Question:**  
How does CreditRisk Analytics enforce BCBS 239 data quality standards? What engineering controls eliminate target leakage?

**Technically Correct Answer:**  
* **BCBS 239 Principles Applied:**
  * **Completeness & Accuracy:** Automated assertions verify zero missing values and unique primary keys before data ingestion completes.
  * **Categorical Normalization:** Unmapped or anomalous codes are mapped to standard categories under documented rules.
  * **Adaptability:** Abstract ingestor base classes decouple data ingestion from model algorithms.
* **Zero Target Leakage Controls:**
  * The target column `default_payment_next_month` is strictly excluded from all feature engineering routines.
  * Preprocessors (mean/median imputers, robust scalers) are **fitted exclusively on training splits** ($X_{\text{train}}$) and applied as fixed transforms on validation and test sets ($X_{\text{test}}$), preventing lookahead bias.

---

#### Question 15: US Federal Reserve SR 11-7 & PRA SS1/23 Model Risk Standards
**Question:**  
What are the primary tenets of SR 11-7 and PRA SS1/23 regarding independent model validation and conceptual soundness?

**Technically Correct Answer:**  
* **Core Tenets of SR 11-7 / OCC 2011-12 & PRA SS1/23:**
  1. **Conceptual Soundness:** The model must be theoretically justified. Algorithms, variables, and mathematical forms must align with established financial economics.
  2. **Independent Validation:** Model development must be evaluated by an independent validation unit (such as SCMAC) with organizational independence and authority to challenge assumptions.
  3. **Ongoing Monitoring & Outcomes Analysis:** Continuous tracking of model performance against predetermined thresholds (PSI, Gini decay, Basel binomial tests).
  4. **Documented Limitations & Overlays:** Every model has boundaries where it degrades. Assumptions must be explicitly documented, and qualitative management overlays must be transparent and auditable.

---

# Part II: Time Series Forecasting, Hybrid Deep Learning, & Stochastic Simulation

---

### [TOPIC 6: CLASSICAL & ADDITIVE TIME SERIES METHODS]

#### Question 16: Stationarity, Dickey-Fuller Testing, and ARIMA $(p,d,q)$ Formulation
**Question:**  
Define weak stationarity. How does the Augmented Dickey-Fuller (ADF) test determine the integration order $d$ in an $\text{ARIMA}(p,d,q)$ model?

**Technically Correct Answer:**  
* **Weak (Covariance) Stationarity:**  
  A discrete-time stochastic process $\{X_t\}$ is covariance stationary if:
  1. Mean is constant over time: $\mathbb{E}[X_t] = \mu \quad \forall t$.
  2. Variance is finite and constant: $\text{Var}(X_t) = \sigma^2 < \infty \quad \forall t$.
  3. Autocovariance depends only on lag $k$: $\text{Cov}(X_t, X_{t+k}) = \gamma(k) \quad \forall t, k$.
* **Augmented Dickey-Fuller (ADF) Test:**  
  Tests for the presence of a unit root ($\rho = 1$):
  $$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{k} \delta_i \Delta X_{t-i} + \epsilon_t$$
  * Null Hypothesis ($H_0$): $\gamma = 0$ (Unit root present; series is non-stationary).
  * Alternative ($H_1$): $\gamma < 0$ (Series is stationary).  
  If $p > 0.05$, the series is differenced: $Y_t = \Delta X_t = X_t - X_{t-1}$ (setting $d=1$) and re-tested until stationarity is attained.
* **$\text{ARIMA}(p,d,q)$ Formulation:**  
  $$\left(1 - \sum_{i=1}^{p} \phi_i L^i\right) (1 - L)^d X_t = c + \left(1 + \sum_{j=1}^{q} \theta_j L^j\right) \epsilon_t$$
  Where $L$ is the lag operator ($L^k X_t = X_{t-k}$).

---

#### Question 17: Facebook Prophet Architecture & Decomposable Time Series Mechanics
**Question:**  
Explain the structural decomposition used in Facebook Prophet. How does it handle trend change points and calendar seasonality compared to classical ARIMA?

**Technically Correct Answer:**  
Prophet frames time series forecasting as a generalized additive curve-fitting regression:
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

1. **Trend Model $g(t)$:**  
   Models non-periodic linear or logistic saturation growth. Change points are placed automatically at dates $s_j$:
   $$g(t) = (k + \boldsymbol{a}(t)^T \boldsymbol{\delta}) t + (m + \boldsymbol{a}(t)^T \boldsymbol{\gamma})$$
   Where $\boldsymbol{\delta}$ represents rate adjustments with a Laplace prior ($\delta_j \sim \text{Laplace}(0, \tau)$) acting as an L1 regularizer.
2. **Seasonality $s(t)$:**  
   Modelled using standard Fourier series:
   $$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right)$$
   (e.g., $P = 365.25$ for yearly seasonality, $P = 7$ for weekly).
3. **Holiday & Event Effects $h(t)$:** Incorporates dummy variables for market closures and policy shifts.
* **Advantage over ARIMA:** Does not require evenly spaced data, handles missing dates naturally without imputation, and accommodates multiple overlapping seasonalities without differencing.

---

### [TOPIC 7: DEEP LEARNING & HYBRID ARCHITECTURES]

#### Question 18: Hybrid ARIMA + LSTM: Residual Decomposition Rationale
**Question:**  
What is the mathematical justification for combining ARIMA with a Long Short-Term Memory (LSTM) network in a hybrid architecture? Why not use LSTM alone?

**Technically Correct Answer:**  
Real-world financial asset prices contain both linear autocorrelated structures and complex non-linear residual dynamics. A pure linear model (ARIMA) cannot capture non-linear volatility clustering, while a pure deep neural network (LSTM) can struggle to converge on simple linear trends when sample sizes are finite.

* **Additive Hybrid Decomposition:**
  1. The target series is decomposed:
     $$y_t = L_t + N_t$$
     Where $L_t$ is the linear component and $N_t$ is the non-linear residual.
  2. **Step 1 (Linear Trend Fitting):** An $\text{ARIMA}(p,d,q)$ model is fitted to estimate $\hat{L}_t$:
     $$e_t = y_t - \hat{L}_t$$
  3. **Step 2 (Non-linear Residual Learning):** An LSTM network is trained on sliding lookback windows of the residuals:
     $$\hat{e}_t = f_{\text{LSTM}}(e_{t-1}, e_{t-2}, \dots, e_{t-k})$$
  4. **Step 3 (Final Combined Forecast):**
     $$\hat{y}_t = \hat{L}_t + \hat{e}_t$$
  By combining both paradigms, ARIMA isolates the macroscopic trend while LSTM models residual non-linearities, yielding lower MAE and RMSE than either model independently.

---

#### Question 19: Long Short-Term Memory (LSTM) Internal Gating Mechanics
**Question:**  
Write down the mathematical gating equations for an LSTM cell. How do the forget, input, and output gates prevent the vanishing gradient problem?

**Technically Correct Answer:**  
At time step $t$, given input $x_t$ and prior hidden state $h_{t-1}$:

1. **Forget Gate ($f_t$):** Determines what proportion of prior memory to discard:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$) & Candidate Memory ($\tilde{C}_t$):** Determines new information to store:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$):** Linear combination of preserved memory and new input:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t$) & Hidden State ($h_t$):**
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

* **Prevention of Vanishing Gradients:**  
  In standard RNNs, backpropagating through time involves repeated matrix multiplications $\prod W^T$, which decay exponentially to zero. In LSTMs, the cell state gradient $\frac{\partial C_t}{\partial C_{t-1}} = f_t + \dots$ acts as an additive "constant error carousel" whenever the forget gate $f_t \approx 1$, allowing gradients to flow backward through hundreds of time steps without vanishing.

---

#### Question 20: Forecasting Raw Prices vs. Log Returns in Financial Econometrics
**Question:**  
Why do quantitative risk analysts strongly discourage forecasting raw stock prices directly with neural networks? What statistical problems arise, and why are log returns preferred?

**Technically Correct Answer:**  
* **Non-Stationarity & Spurious Correlation:**  
  Asset prices $S_t$ exhibit a unit root ($I(1)$ process) with time-varying mean and variance (random walk). Feeding non-stationary prices directly into neural networks produces models that appear accurate on MSE simply by outputting $\hat{S}_{t+1} \approx S_t$ (persistence forecast), while failing to predict price direction.
* **Why Log Returns Are Preferred:**
  $$\text{Return } r_t = \ln\left(\frac{S_t}{S_{t-1}}\right) = \ln(S_t) - \ln(S_{t-1})$$
  1. **Stationarity:** Differencing log prices removes unit roots, yielding a mean-stationary series suitable for statistical learning.
  2. **Additivity:** Multi-period log returns sum linearly:
     $$r_{t, t+k} = \sum_{j=1}^{k} r_{t+j}$$
  3. **Scale Invariance:** Returns are unit-free and comparable across assets, whereas raw price levels depend on stock split histories and share counts.

---

### [TOPIC 8: STOCHASTIC SIMULATION & MODEL EVALUATION]

#### Question 21: Geometric Brownian Motion (GBM) & Monte Carlo Simulation
**Question:**  
Derive the analytical solution of Geometric Brownian Motion (GBM) used in `MonteCarlo.py`. How does Monte Carlo simulation generate forward risk distributions?

**Technically Correct Answer:**  
Geometric Brownian Motion models asset prices as a continuous-time stochastic differential equation (SDE):
$$\frac{dS_t}{S_t} = \mu \, dt + \sigma \, dW_t$$
Where $\mu$ is drift, $\sigma$ is volatility, and $W_t$ is standard Brownian motion ($dW_t = \epsilon \sqrt{dt}, \, \epsilon \sim \mathcal{N}(0, 1)$).

Applying **Itô's Lemma** to $f(S_t) = \ln(S_t)$:
$$df = \left( \frac{\partial f}{\partial S} \mu S_t + \frac{\partial f}{\partial t} + \frac{1}{2} \frac{\partial^2 f}{\partial S^2} \sigma^2 S_t^2 \right) dt + \frac{\partial f}{\partial S} \sigma S_t \, dW_t$$
$$\implies d(\ln S_t) = \left( \mu - \frac{1}{2}\sigma^2 \right) dt + \sigma \, dW_t$$

Integrating from $0$ to $t$:
$$S_t = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right) t + \sigma W_t \right)$$

* **Discrete Monte Carlo Step:**
  $$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right) \Delta t + \sigma \sqrt{\Delta t} \, Z \right) \quad \text{where } Z \sim \mathcal{N}(0, 1)$$
  Simulating $M = 10,000$ independent sample paths generates the full terminal probability distribution $P(S_T)$, allowing direct extraction of Value at Risk ($\text{VaR}_\alpha$) and Expected Shortfall ($\text{ES}_\alpha$).

---

#### Question 22: Evaluation Metrics: MAE vs. RMSE vs. MAPE in Financial Forecasting
**Question:**  
Compare Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) for evaluating financial price predictions. Which metric should govern model selection under asymmetric risk?

**Technically Correct Answer:**  
$$\text{MAE} = \frac{1}{N} \sum_{t=1}^{N} |y_t - \hat{y}_t|, \quad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2}, \quad \text{MAPE} = \frac{100\%}{N} \sum_{t=1}^{N} \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

* **Comparison:**
  * **MAE:** Measures median absolute deviation. Robust to outliers and treats errors linearly.
  * **RMSE:** Squares errors prior to averaging. Heavily penalizes large forecast misses.
  * **MAPE:** Expresses errors in percentage terms, but divides by zero when $y_t \to 0$ and penalizes over-forecasts more heavily than under-forecasts.
* **Governance Under Asymmetric Financial Risk:**  
  In risk management and trading desk forecasting, **large deviations carry catastrophic non-linear costs** (e.g., margin calls or illiquid inventory liquidation). Therefore, **RMSE** is preferred over MAE as the primary loss metric because squaring errors penalizes tail deviations. For multi-asset comparison, normalized root mean squared error (NRMSE) or directional symmetry (sign accuracy) is used alongside RMSE.
