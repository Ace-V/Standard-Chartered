"""
CreditRisk Analytics — Expected Credit Loss (ECL) Analysis Engine
================================================================

This module implements an institutional Expected Credit Loss (ECL) calculation
and scenario stress-testing engine based on the fundamental credit risk formula:

    ECL = PD × LGD × EAD

--------------------------------------------------------------------------------
CORE DEFINITIONS & UNITS:
--------------------------------------------------------------------------------
1. PD (Probability of Default):
   - Definition: The statistical likelihood that a counterparty defaults on its debt
     obligations over a specified observation horizon (typically 12 months for 1-period
     models).
   - Range / Unit: Dimensionless probability in the closed interval [0.0, 1.0].
   - Note: If percentage values are provided (e.g. 3.5%), they must be divided by 100.

2. LGD (Loss Given Default):
   - Definition: The economic proportion of exposure that is lost when default occurs,
     after accounting for recovery, collateral liquidation, and legal workout costs.
   - Range / Unit: Dimensionless fraction in the closed interval [0.0, 1.0].
   - Formula: LGD = 1.0 - Recovery Rate.

3. EAD (Exposure at Default):
   - Definition: The gross monetary credit exposure at the time of counterparty default,
     consisting of drawn balance plus credit conversion factor (CCF) applied to undrawn
     committed credit lines.
   - Range / Unit: Currency amount in non-negative real numbers [0.0, +inf).

4. ECL (Expected Credit Loss):
   - Definition: The expected monetary loss arising from counterparty credit default:
     ECL = PD × LGD × EAD.
   - Range / Unit: Currency amount in non-negative real numbers [0.0, EAD].

--------------------------------------------------------------------------------
CRITICAL REGULATORY & METHODOLOGICAL DISTINCTIONS:
--------------------------------------------------------------------------------
1. Consumer Credit Card Data vs. Wholesale Corporate PD:
   - The primary empirical dataset in this repository (UCI Default of Credit Card Clients)
     reflects retail consumer revolving credit balances and payment delays in Taiwan.
   - Under no circumstances should predicted default probabilities from consumer credit
     card models be claimed as validated corporate Probability of Default (PD) estimates.
   - Corporate wholesale credit analysis requires company balance-sheet fundamentals
     (Interest Coverage, Debt/EBITDA, Cash Flow/Debt, Liquidity), audited financial
     statements, macroeconomic industry cyclicality, qualitative management assessment,
     and agency rating alignments.
   - To demonstrate the ECL engine independently and rigorously, this module provides
     clearly labelled hypothetical corporate borrower facilities.

2. Simple One-Period Formula vs. Full Institutional IFRS 9 / US CECL Standards:
   The simple formula `ECL = PD × LGD × EAD` serves as a core risk calculation,
   but does NOT constitute a complete regulatory IFRS 9 or CECL impairment framework:
   - Discounting: Simple ECL omits time value of money. Under IFRS 9 paragraph 5.5.17,
     cash shortfalls must be discounted back to the reporting date using the loan's
     Effective Interest Rate (EIR):
         ECL = sum_{t=1}^T [ Marginal_PD_t * LGD_t * EAD_t / (1 + EIR)^t ]
   - Forecast Horizons & Staging: The simple formula applies a static 1-year horizon
     (analogous to Stage 1 performing loans). It does not dynamically model Significant
     Increase in Credit Risk (SICR) migration or multi-year cumulative lifetime default
     probabilities required for Stage 2 (underperforming) and Stage 3 (defaulted).
   - Macroeconomic Conditioning: Linear multipliers (e.g., 1.45x for adverse) provide
     an illustrative proxy for stress testing. Full institutional IFRS 9 models use
     econometric satellite models (e.g. Vasicek credit cycle indexes, Vector
     Autoregression) conditioning marginal PDs and LGDs on macroeconomic variables
     (GDP growth, unemployment, commercial property indices, base interest rates).
   - Exposure Dynamics: Static EAD assumes exposure remains unchanged until default,
     ignoring contractual amortization schedules, voluntary prepayments, and dynamic
     CCF drawdown behavior during financial distress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Institutional Color Palette for Visualizations
# ---------------------------------------------------------------------------
CHART_COLORS = {
    "primary": "#1E3A8A",       # Deep Corporate Navy
    "secondary": "#0284C7",     # Slate Blue
    "accent": "#0D9488",        # Emerald Teal
    "danger": "#DC2626",        # Red (Adverse)
    "warning": "#F59E0B",       # Amber (Base)
    "success": "#10B981",       # Green (Favourable)
    "neutral": "#64748B",       # Slate Grey
    "card_bg": "#F8FAFC",       # Off White
}


# ===========================================================================
# 1. Input Validation & Core Formula
# ===========================================================================

def validate_ecl_inputs(
    pd: float,
    lgd: float,
    ead: float,
    allow_zero_exposure: bool = True,
) -> None:
    """
    Validate credit risk parameters for range, type, and financial consistency.

    Parameters
    ----------
    pd : float
        Probability of Default. Must be a finite float in [0.0, 1.0].
    lgd : float
        Loss Given Default. Must be a finite float in [0.0, 1.0].
    ead : float
        Exposure at Default. Must be a finite non-negative float [0.0, +inf).
    allow_zero_exposure : bool, default=True
        Whether EAD=0.0 is permitted (e.g., paid-off or cancelled facilities).

    Raises
    ------
    TypeError
        If any input is non-numeric or boolean.
    ValueError
        If any input violates required mathematical ranges.
    """
    # Strict type checks (reject booleans because bool is a subclass of int in Python)
    for name, val in [("PD", pd), ("LGD", lgd), ("EAD", ead)]:
        if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
            raise TypeError(f"{name} must be a numeric float or int, got {type(val).__name__}.")
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"{name} must be finite (not NaN or Inf), got {val}.")

    # Numerical range checks
    if not (0.0 <= pd <= 1.0):
        raise ValueError(
            f"PD (Probability of Default) must be between 0.0 and 1.0 (got {pd}). "
            "If entering percentage points (e.g., 5%), divide by 100 first."
        )

    if not (0.0 <= lgd <= 1.0):
        raise ValueError(
            f"LGD (Loss Given Default) must be between 0.0 and 1.0 (got {lgd}). "
            "If entering percentage points (e.g., 45%), divide by 100 first."
        )

    if ead < 0.0:
        raise ValueError(
            f"EAD (Exposure at Default) must be non-negative (got {ead}). "
            "Financial exposures cannot be negative."
        )

    if not allow_zero_exposure and ead == 0.0:
        raise ValueError("EAD (Exposure at Default) must be strictly greater than 0.0.")


def calculate_expected_loss(
    pd: float,
    lgd: float,
    ead: float,
) -> float:
    """
    Calculate Expected Credit Loss (ECL) using the fundamental credit formula:

        ECL = PD × LGD × EAD

    Parameters
    ----------
    pd : float
        Probability of Default in [0.0, 1.0].
    lgd : float
        Loss Given Default in [0.0, 1.0].
    ead : float
        Exposure at Default in [0.0, +inf).

    Returns
    -------
    float
        Expected monetary credit loss in currency units.
        Guaranteed to be within [0.0, EAD].

    Examples
    --------
    >>> calculate_expected_loss(pd=0.02, lgd=0.45, ead=10000000.0)
    90000.0
    >>> calculate_expected_loss(pd=0.0, lgd=0.45, ead=10000000.0)
    0.0
    >>> calculate_expected_loss(pd=0.05, lgd=0.0, ead=10000000.0)
    0.0
    >>> calculate_expected_loss(pd=0.05, lgd=0.45, ead=0.0)
    0.0
    """
    validate_ecl_inputs(pd=pd, lgd=lgd, ead=ead)
    return float(pd * lgd * ead)


# ===========================================================================
# 2. Data Structures for Hypothetical Borrowers & Scenarios
# ===========================================================================

@dataclass
class Scenario:
    """
    Definition of a macroeconomic forward-looking scenario.

    Parameters
    ----------
    name : str
        Descriptive label (e.g., "Base", "Favourable", "Adverse").
    weight : float
        Probability weighting for this scenario in [0.0, 1.0].
    pd_multiplier : float
        Stress or relief factor applied to baseline PD (>= 0.0).
    lgd_multiplier : float
        Stress or relief factor applied to baseline LGD (>= 0.0).
    description : str
        Narrative macroeconomic assumptions underpinning the scenario.
    """
    name: str
    weight: float
    pd_multiplier: float
    lgd_multiplier: float
    description: str = ""

    def __post_init__(self):
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Scenario weight must be in [0.0, 1.0], got {self.weight}.")
        if self.pd_multiplier < 0.0:
            raise ValueError(f"pd_multiplier must be >= 0.0, got {self.pd_multiplier}.")
        if self.lgd_multiplier < 0.0:
            raise ValueError(f"lgd_multiplier must be >= 0.0, got {self.lgd_multiplier}.")


@dataclass
class HypotheticalBorrower:
    """
    A single hypothetical corporate credit facility.

    Parameters
    ----------
    borrower_id : str
        Unique facility or borrower reference code.
    borrower_name : str
        Fictional corporate borrower name.
    industry : str
        Economic sector / industry category.
    rating_grade : str
        Internal credit rating grade (e.g., AAA, AA, A, BBB, BB, B, CCC).
    pd : float
        Baseline 12-month Probability of Default in [0.0, 1.0].
    lgd : float
        Baseline Loss Given Default in [0.0, 1.0].
    ead : float
        Gross Exposure at Default in currency units (>= 0.0).
    facility_type : str
        Facility structure (e.g., "Term Loan", "Revolving Credit", "Project Finance").
    is_hypothetical : bool
        Explicit marker confirming this is synthetic illustrative data.
    """
    borrower_id: str
    borrower_name: str
    industry: str
    rating_grade: str
    pd: float
    lgd: float
    ead: float
    facility_type: str = "Senior Secured Term Loan"
    is_hypothetical: bool = True

    def __post_init__(self):
        validate_ecl_inputs(pd=self.pd, lgd=self.lgd, ead=self.ead)

    @property
    def base_ecl(self) -> float:
        """Baseline Expected Credit Loss."""
        return calculate_expected_loss(self.pd, self.lgd, self.ead)


@dataclass
class BorrowerScenarioResult:
    """Consolidated scenario and weighted ECL valuation for an individual borrower."""
    borrower_id: str
    borrower_name: str
    industry: str
    rating_grade: str
    facility_type: str
    ead: float
    base_pd: float
    base_lgd: float
    ecl_base: float
    ecl_favourable: float
    ecl_adverse: float
    ecl_weighted: float
    loss_rate_pct: float     # (Weighted ECL / EAD) * 100
    is_hypothetical: bool = True


@dataclass
class CorporatePortfolioECLResult:
    """
    Aggregated multi-borrower portfolio ECL metrics across macroeconomic scenarios.
    """
    borrowers: List[BorrowerScenarioResult]
    total_ead: float
    total_ecl_base: float
    total_ecl_favourable: float
    total_ecl_adverse: float
    total_ecl_weighted: float
    portfolio_weighted_pd: float
    portfolio_weighted_lgd: float
    portfolio_loss_rate_pct: float
    scenarios: Dict[str, Scenario]
    is_hypothetical: bool = True

    def to_dataframe(self) -> pd.DataFrame:
        """Return facility-level results as a Pandas DataFrame."""
        records = []
        for b in self.borrowers:
            records.append({
                "Facility ID": b.borrower_id,
                "Borrower Name": b.borrower_name,
                "Industry": b.industry,
                "Rating Grade": b.rating_grade,
                "Facility Type": b.facility_type,
                "EAD ($)": round(b.ead, 2),
                "Base PD": round(b.base_pd, 4),
                "Base LGD": round(b.base_lgd, 4),
                "ECL Base ($)": round(b.ecl_base, 2),
                "ECL Favourable ($)": round(b.ecl_favourable, 2),
                "ECL Adverse ($)": round(b.ecl_adverse, 2),
                "ECL Weighted ($)": round(b.ecl_weighted, 2),
                "Loss Rate (%)": round(b.loss_rate_pct, 4),
                "Hypothetical": b.is_hypothetical,
            })
        return pd.DataFrame(records)

    def summary_by_industry(self) -> pd.DataFrame:
        """Aggregate EAD and Expected Loss grouped by industry sector."""
        df = self.to_dataframe()
        grouped = df.groupby("Industry").agg(
            Facilities=("Facility ID", "count"),
            Total_EAD=("EAD ($)", "sum"),
            Total_Weighted_ECL=("ECL Weighted ($)", "sum"),
        ).reset_index()
        grouped["EAD_Share_Pct"] = (grouped["Total_EAD"] / self.total_ead * 100).round(2)
        grouped["Loss_Rate_Pct"] = np.where(
            grouped["Total_EAD"] > 0,
            (grouped["Total_Weighted_ECL"] / grouped["Total_EAD"] * 100).round(4),
            0.0,
        )
        return grouped.sort_values(by="Total_Weighted_ECL", ascending=False)

    def summary_by_rating(self) -> pd.DataFrame:
        """Aggregate EAD and Expected Loss grouped by rating grade."""
        df = self.to_dataframe()
        grouped = df.groupby("Rating Grade").agg(
            Facilities=("Facility ID", "count"),
            Total_EAD=("EAD ($)", "sum"),
            Total_Weighted_ECL=("ECL Weighted ($)", "sum"),
        ).reset_index()
        grouped["EAD_Share_Pct"] = (grouped["Total_EAD"] / self.total_ead * 100).round(2)
        grouped["Loss_Rate_Pct"] = np.where(
            grouped["Total_EAD"] > 0,
            (grouped["Total_Weighted_ECL"] / grouped["Total_EAD"] * 100).round(4),
            0.0,
        )
        return grouped.sort_values(by="Total_Weighted_ECL", ascending=False)

    def scenario_summary_table(self) -> pd.DataFrame:
        """Compare aggregate portfolio loss figures across scenarios."""
        return pd.DataFrame([
            {
                "Scenario": "Favourable (Upturn)",
                "Weight (%)": self.scenarios["Favourable"].weight * 100,
                "Total ECL ($)": round(self.total_ecl_favourable, 2),
                "Portfolio Loss Rate (%)": round(
                    (self.total_ecl_favourable / self.total_ead * 100) if self.total_ead > 0 else 0.0, 4
                ),
            },
            {
                "Scenario": "Base (Consensus)",
                "Weight (%)": self.scenarios["Base"].weight * 100,
                "Total ECL ($)": round(self.total_ecl_base, 2),
                "Portfolio Loss Rate (%)": round(
                    (self.total_ecl_base / self.total_ead * 100) if self.total_ead > 0 else 0.0, 4
                ),
            },
            {
                "Scenario": "Adverse (Downturn)",
                "Weight (%)": self.scenarios["Adverse"].weight * 100,
                "Total ECL ($)": round(self.total_ecl_adverse, 2),
                "Portfolio Loss Rate (%)": round(
                    (self.total_ecl_adverse / self.total_ead * 100) if self.total_ead > 0 else 0.0, 4
                ),
            },
            {
                "Scenario": "Probability-Weighted Total",
                "Weight (%)": 100.0,
                "Total ECL ($)": round(self.total_ecl_weighted, 2),
                "Portfolio Loss Rate (%)": round(self.portfolio_loss_rate_pct, 4),
            },
        ])


@dataclass
class SensitivityAnalysisResult:
    """
    Two-dimensional sensitivity analysis results for PD and LGD variations.
    """
    base_pd: float
    base_lgd: float
    ead: float
    base_ecl: float
    pd_values: np.ndarray
    lgd_values: np.ndarray
    ecl_matrix: np.ndarray
    marginal_pd_sensitivity: float    # dECL / dPD = LGD * EAD ($ per 100% PD shift)
    marginal_lgd_sensitivity: float   # dECL / dLGD = PD * EAD ($ per 100% LGD shift)
    shock_100bps_pd_impact: float     # Dollar ECL change for +100 bps (+0.01) PD increase
    shock_100bps_lgd_impact: float    # Dollar ECL change for +100 bps (+0.01) LGD increase

    def to_dataframe(self) -> pd.DataFrame:
        """Convert sensitivity grid into a clean tabular DataFrame."""
        rows = []
        for i, pd_val in enumerate(self.pd_values):
            for j, lgd_val in enumerate(self.lgd_values):
                ecl_val = self.ecl_matrix[i, j]
                diff_dollar = ecl_val - self.base_ecl
                diff_pct = (diff_dollar / self.base_ecl * 100) if self.base_ecl > 0 else 0.0
                rows.append({
                    "PD": round(pd_val, 4),
                    "PD_Pct": f"{pd_val * 100:.2f}%",
                    "LGD": round(lgd_val, 4),
                    "LGD_Pct": f"{lgd_val * 100:.2f}%",
                    "ECL ($)": round(ecl_val, 2),
                    "Delta to Base ($)": round(diff_dollar, 2),
                    "Delta to Base (%)": round(diff_pct, 2),
                })
        return pd.DataFrame(rows)


# ===========================================================================
# 3. Macroeconomic Scenarios & Portfolio Calculations
# ===========================================================================

def get_default_scenarios() -> Dict[str, Scenario]:
    """
    Return standard 3-scenario macroeconomic stress testing definitions.
    Weights sum to exactly 1.00.
    """
    return {
        "Base": Scenario(
            name="Base",
            weight=0.50,
            pd_multiplier=1.00,
            lgd_multiplier=1.00,
            description="Baseline macroeconomic consensus; stable inflation, moderate growth, historical default rates.",
        ),
        "Favourable": Scenario(
            name="Favourable",
            weight=0.25,
            pd_multiplier=0.75,
            lgd_multiplier=0.85,
            description="Economic expansion; robust corporate cash flows, lower default frequency, improved collateral values.",
        ),
        "Adverse": Scenario(
            name="Adverse",
            weight=0.25,
            pd_multiplier=1.45,
            lgd_multiplier=1.20,
            description="Recessionary downturn; tighter credit conditions, elevated corporate distress, impaired recovery rates.",
        ),
    }


def validate_scenarios(scenarios: Dict[str, Scenario]) -> None:
    """Validate scenario set consistency and weight normalization."""
    required = {"Base", "Favourable", "Adverse"}
    if not required.issubset(scenarios.keys()):
        raise ValueError(f"Scenarios dictionary must contain at least {required}, got {set(scenarios.keys())}.")

    total_weight = sum(sc.weight for sc in scenarios.values())
    if not np.isclose(total_weight, 1.0, atol=1e-4):
        raise ValueError(
            f"Scenario weights must sum to 1.0 (current sum: {total_weight:.4f}). "
            "Re-balance weights across Base, Favourable, and Adverse."
        )


def calculate_portfolio_ecl(
    borrowers: List[HypotheticalBorrower],
    scenarios: Optional[Dict[str, Scenario]] = None,
) -> CorporatePortfolioECLResult:
    """
    Calculate Expected Credit Loss across a portfolio of corporate borrowers
    under multiple macroeconomic scenarios and calculate probability-weighted loss.

    Parameters
    ----------
    borrowers : List[HypotheticalBorrower]
        Collection of corporate facilities to evaluate.
    scenarios : Optional[Dict[str, Scenario]], default=None
        Custom scenario dictionary. If None, uses get_default_scenarios().

    Returns
    -------
    CorporatePortfolioECLResult
        Aggregated portfolio metrics, facility-level breakdowns, and scenario tables.
    """
    if not borrowers:
        raise ValueError("Cannot calculate portfolio ECL: borrowers list is empty.")

    scenarios = scenarios or get_default_scenarios()
    validate_scenarios(scenarios)

    borrower_results: List[BorrowerScenarioResult] = []

    total_ead = 0.0
    total_ecl_base = 0.0
    total_ecl_favourable = 0.0
    total_ecl_adverse = 0.0
    total_ecl_weighted = 0.0
    sum_pd_times_ead = 0.0
    sum_lgd_times_ead = 0.0

    for b in borrowers:
        # Base scenario
        ecl_base = calculate_expected_loss(b.pd, b.lgd, b.ead)

        # Favourable scenario (clipped to max 1.0)
        fav_pd = min(1.0, b.pd * scenarios["Favourable"].pd_multiplier)
        fav_lgd = min(1.0, b.lgd * scenarios["Favourable"].lgd_multiplier)
        ecl_fav = calculate_expected_loss(fav_pd, fav_lgd, b.ead)

        # Adverse scenario (clipped to max 1.0)
        adv_pd = min(1.0, b.pd * scenarios["Adverse"].pd_multiplier)
        adv_lgd = min(1.0, b.lgd * scenarios["Adverse"].lgd_multiplier)
        ecl_adv = calculate_expected_loss(adv_pd, adv_lgd, b.ead)

        # Probability-weighted ECL
        ecl_weighted = (
            scenarios["Base"].weight * ecl_base
            + scenarios["Favourable"].weight * ecl_fav
            + scenarios["Adverse"].weight * ecl_adv
        )

        loss_rate_pct = (ecl_weighted / b.ead * 100) if b.ead > 0 else 0.0

        borrower_results.append(BorrowerScenarioResult(
            borrower_id=b.borrower_id,
            borrower_name=b.borrower_name,
            industry=b.industry,
            rating_grade=b.rating_grade,
            facility_type=b.facility_type,
            ead=b.ead,
            base_pd=b.pd,
            base_lgd=b.lgd,
            ecl_base=ecl_base,
            ecl_favourable=ecl_fav,
            ecl_adverse=ecl_adv,
            ecl_weighted=ecl_weighted,
            loss_rate_pct=loss_rate_pct,
            is_hypothetical=b.is_hypothetical,
        ))

        total_ead += b.ead
        total_ecl_base += ecl_base
        total_ecl_favourable += ecl_fav
        total_ecl_adverse += ecl_adv
        total_ecl_weighted += ecl_weighted
        sum_pd_times_ead += b.pd * b.ead
        sum_lgd_times_ead += b.lgd * b.ead

    portfolio_weighted_pd = (sum_pd_times_ead / total_ead) if total_ead > 0 else 0.0
    portfolio_weighted_lgd = (sum_lgd_times_ead / total_ead) if total_ead > 0 else 0.0
    portfolio_loss_rate = (total_ecl_weighted / total_ead * 100) if total_ead > 0 else 0.0

    return CorporatePortfolioECLResult(
        borrowers=borrower_results,
        total_ead=total_ead,
        total_ecl_base=total_ecl_base,
        total_ecl_favourable=total_ecl_favourable,
        total_ecl_adverse=total_ecl_adverse,
        total_ecl_weighted=total_ecl_weighted,
        portfolio_weighted_pd=portfolio_weighted_pd,
        portfolio_weighted_lgd=portfolio_weighted_lgd,
        portfolio_loss_rate_pct=portfolio_loss_rate,
        scenarios=scenarios,
        is_hypothetical=True,
    )


# ===========================================================================
# 4. Clearly Labelled Hypothetical Corporate Portfolio
# ===========================================================================

def get_hypothetical_corporate_portfolio() -> List[HypotheticalBorrower]:
    """
    Generate a representative portfolio of 10 hypothetical corporate borrowers
    spanning multiple industries, facility types, and internal credit ratings.

    CRITICAL DISCLOSURE:
    This dataset is entirely synthetic and hypothetical. It is designed to
    demonstrate corporate wholesale ECL calculation independently of retail
    credit-card dataset models.

    Returns
    -------
    List[HypotheticalBorrower]
        10 validated corporate credit facilities. Total EAD: $175,600,000.
    """
    return [
        HypotheticalBorrower(
            borrower_id="CORP-001",
            borrower_name="Apex Industrial Machinery Ltd",
            industry="Manufacturing",
            rating_grade="BBB",
            pd=0.035,
            lgd=0.40,
            ead=12_500_000.0,
            facility_type="Senior Secured Term Loan",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-002",
            borrower_name="Helios Renewable Energy Corp",
            industry="Utilities / Energy",
            rating_grade="A",
            pd=0.012,
            lgd=0.35,
            ead=28_000_000.0,
            facility_type="Project Finance Term Loan",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-003",
            borrower_name="Vanguard Global Logistics Inc",
            industry="Transportation",
            rating_grade="BB",
            pd=0.075,
            lgd=0.45,
            ead=8_200_000.0,
            facility_type="Equipment Lease / Term Facility",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-004",
            borrower_name="Meridian BioTherapeutics PLC",
            industry="Healthcare",
            rating_grade="AA",
            pd=0.005,
            lgd=0.30,
            ead=15_000_000.0,
            facility_type="Senior Unsecured Revolving Credit",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-005",
            borrower_name="Stratosphere Cloud SaaS Systems",
            industry="Technology",
            rating_grade="BBB",
            pd=0.028,
            lgd=0.40,
            ead=10_000_000.0,
            facility_type="Senior Term Loan",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-006",
            borrower_name="Beacon Prime Commercial REIT",
            industry="Commercial Real Estate",
            rating_grade="BB",
            pd=0.065,
            lgd=0.35,
            ead=22_000_000.0,
            facility_type="Commercial Mortgage Facility",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-007",
            borrower_name="Titan Mining & Refining PLC",
            industry="Basic Materials",
            rating_grade="B",
            pd=0.140,
            lgd=0.55,
            ead=18_500_000.0,
            facility_type="Senior Subordinated Debt",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-008",
            borrower_name="Solaria Consumer Brands International",
            industry="Consumer Goods",
            rating_grade="B",
            pd=0.110,
            lgd=0.50,
            ead=6_400_000.0,
            facility_type="Unsecured Working Capital Line",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-009",
            borrower_name="Pinnacle Aerospace & Defense Corp",
            industry="Industrials",
            rating_grade="AAA",
            pd=0.001,
            lgd=0.25,
            ead=35_000_000.0,
            facility_type="Syndicated Term Facility",
        ),
        HypotheticalBorrower(
            borrower_id="CORP-010",
            borrower_name="Quantum Foundry Semiconductor NV",
            industry="Technology",
            rating_grade="A",
            pd=0.015,
            lgd=0.35,
            ead=20_000_000.0,
            facility_type="Capex Facility",
        ),
    ]


# ===========================================================================
# 5. Sensitivity Analysis for PD and LGD
# ===========================================================================

def perform_sensitivity_analysis(
    pd: float,
    lgd: float,
    ead: float,
    pd_steps: int = 9,
    lgd_steps: int = 9,
    pd_multiplier_range: Tuple[float, float] = (0.5, 1.5),
    lgd_multiplier_range: Tuple[float, float] = (0.5, 1.5),
) -> SensitivityAnalysisResult:
    """
    Perform a 2D grid sensitivity analysis evaluating Expected Credit Loss
    across a spectrum of PD and LGD assumptions.

    Also derives first-order marginal risk sensitivities:
        d(ECL) / d(PD)  = LGD × EAD   (Dollar loss per 100% change in PD)
        d(ECL) / d(LGD) = PD × EAD    (Dollar loss per 100% change in LGD)

    Parameters
    ----------
    pd : float
        Base Probability of Default in [0.0, 1.0].
    lgd : float
        Base Loss Given Default in [0.0, 1.0].
    ead : float
        Gross Exposure at Default in currency units (>= 0.0).
    pd_steps : int, default=9
        Number of grid evaluation points for PD.
    lgd_steps : int, default=9
        Number of grid evaluation points for LGD.
    pd_multiplier_range : Tuple[float, float], default=(0.5, 1.5)
        Range of multipliers around baseline PD.
    lgd_multiplier_range : Tuple[float, float], default=(0.5, 1.5)
        Range of multipliers around baseline LGD.

    Returns
    -------
    SensitivityAnalysisResult
        Contains the 2D evaluation matrix, marginal derivatives, and 100 bps shock metrics.
    """
    validate_ecl_inputs(pd, lgd, ead)

    base_ecl = calculate_expected_loss(pd, lgd, ead)

    # Generate grid boundaries, safely clamped to [0.0001, 1.0]
    min_pd_mult, max_pd_mult = pd_multiplier_range
    min_lgd_mult, max_lgd_mult = lgd_multiplier_range

    min_pd = max(0.0001, min(1.0, pd * min_pd_mult))
    max_pd = max(0.0001, min(1.0, pd * max_pd_mult))
    min_lgd = max(0.05, min(1.0, lgd * min_lgd_mult))
    max_lgd = max(0.05, min(1.0, lgd * max_lgd_mult))

    pd_vals = np.linspace(min_pd, max_pd, pd_steps)
    lgd_vals = np.linspace(min_lgd, max_lgd, lgd_steps)

    # 2D Meshgrid computation: ECL = PD_i * LGD_j * EAD
    # Shape: (len(pd_vals), len(lgd_vals))
    ecl_matrix = np.outer(pd_vals, lgd_vals) * ead

    # Analytical marginal sensitivities (partial derivatives)
    d_ecl_d_pd = lgd * ead
    d_ecl_d_lgd = pd * ead

    # 100 bps (+1.00% = +0.01) shock impacts
    shock_pd_100bps = 0.01 * lgd * ead
    shock_lgd_100bps = 0.01 * pd * ead

    return SensitivityAnalysisResult(
        base_pd=pd,
        base_lgd=lgd,
        ead=ead,
        base_ecl=base_ecl,
        pd_values=pd_vals,
        lgd_values=lgd_vals,
        ecl_matrix=ecl_matrix,
        marginal_pd_sensitivity=d_ecl_d_pd,
        marginal_lgd_sensitivity=d_ecl_d_lgd,
        shock_100bps_pd_impact=shock_pd_100bps,
        shock_100bps_lgd_impact=shock_lgd_100bps,
    )


# ===========================================================================
# 6. Interactive Plotly Visualizations
# ===========================================================================

def plot_scenario_losses(portfolio_result: CorporatePortfolioECLResult) -> go.Figure:
    """
    Produce an institutional bar chart comparing portfolio Expected Credit Loss
    across Favourable, Base, Adverse, and Probability-Weighted scenarios.

    Parameters
    ----------
    portfolio_result : CorporatePortfolioECLResult
        Portfolio calculation output.

    Returns
    -------
    go.Figure
        Plotly interactive bar chart with currency annotations and loss rates.
    """
    scenarios = [
        ("Favourable (Upturn)", portfolio_result.total_ecl_favourable, CHART_COLORS["success"], "0.25 wt"),
        ("Base (Consensus)", portfolio_result.total_ecl_base, CHART_COLORS["primary"], "0.50 wt"),
        ("Adverse (Downturn)", portfolio_result.total_ecl_adverse, CHART_COLORS["danger"], "0.25 wt"),
        ("Probability-Weighted", portfolio_result.total_ecl_weighted, CHART_COLORS["secondary"], "Blended Total"),
    ]

    names = [s[0] for s in scenarios]
    losses = [s[1] for s in scenarios]
    colors = [s[2] for s in scenarios]
    subtitles = [s[3] for s in scenarios]
    loss_rates = [
        (loss / portfolio_result.total_ead * 100) if portfolio_result.total_ead > 0 else 0.0
        for loss in losses
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=losses,
        marker_color=colors,
        text=[f"${loss:,.0f}<br>({rate:.2f}% EAD)" for loss, rate in zip(losses, loss_rates)],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Expected Credit Loss: $%{y:,.2f}<br>"
            "Portfolio Loss Rate: %{customdata:.3f}%<br>"
            "<extra></extra>"
        ),
        customdata=loss_rates,
    ))

    fig.update_layout(
        title={
            "text": "<b>Portfolio Expected Credit Loss (ECL) Across Macroeconomic Scenarios</b><br>"
                    "<sup>Hypothetical Corporate Loan Portfolio (EAD: $"
                    f"{portfolio_result.total_ead:,.0f})</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 16, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Macroeconomic Scenario",
        yaxis_title="Expected Credit Loss ($ USD)",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
        yaxis={"tickprefix": "$", "tickformat": ",.0f", "showgrid": True, "gridcolor": "#E2E8F0"},
        xaxis={"showgrid": False},
        showlegend=False,
    )

    return fig


def plot_ecl_sensitivity_heatmap(sensitivity: SensitivityAnalysisResult) -> go.Figure:
    """
    Produce an interactive 2D heatmap showing Expected Credit Loss ($)
    as a joint function of Probability of Default (PD) and Loss Given Default (LGD).

    Parameters
    ----------
    sensitivity : SensitivityAnalysisResult
        Output of perform_sensitivity_analysis.

    Returns
    -------
    go.Figure
        Plotly heatmap with contour lines and baseline indicator.
    """
    x_labels = [f"{lgd * 100:.1f}%" for lgd in sensitivity.lgd_values]
    y_labels = [f"{pd * 100:.2f}%" for pd in sensitivity.pd_values]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=sensitivity.ecl_matrix,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        colorbar={
            "title": "ECL ($)",
            "tickprefix": "$",
            "tickformat": ",.0f",
        },
        hovertemplate=(
            "<b>PD:</b> %{y}<br>"
            "<b>LGD:</b> %{x}<br>"
            "<b>Expected Credit Loss:</b> $%{z:,.2f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title={
            "text": "<b>ECL Sensitivity Heatmap: Joint Variation of PD and LGD</b><br>"
                    f"<sup>Facility EAD: ${sensitivity.ead:,.0f} | Base PD: {sensitivity.base_pd:.2%} | "
                    f"Base LGD: {sensitivity.base_lgd:.1%} | Base ECL: ${sensitivity.base_ecl:,.0f}</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Loss Given Default (LGD)",
        yaxis_title="Probability of Default (PD)",
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
    )

    return fig


def plot_portfolio_ecl_by_industry(portfolio_result: CorporatePortfolioECLResult) -> go.Figure:
    """
    Produce a grouped bar chart displaying Total Exposure (EAD) vs Weighted Expected Loss (ECL)
    by industry sector to evaluate credit concentration.

    Parameters
    ----------
    portfolio_result : CorporatePortfolioECLResult
        Portfolio calculation output.

    Returns
    -------
    go.Figure
        Dual-variable grouped bar chart.
    """
    summary_df = portfolio_result.summary_by_industry()

    fig = go.Figure()

    # Total EAD Bar
    fig.add_trace(go.Bar(
        x=summary_df["Industry"],
        y=summary_df["Total_EAD"],
        name="Exposure at Default (EAD)",
        marker_color=CHART_COLORS["primary"],
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Total EAD: $%{y:,.0f}<extra></extra>",
    ))

    # Weighted ECL Bar (secondary axis)
    fig.add_trace(go.Bar(
        x=summary_df["Industry"],
        y=summary_df["Total_Weighted_ECL"],
        name="Weighted Expected Loss (ECL)",
        marker_color=CHART_COLORS["danger"],
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Weighted ECL: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title={
            "text": "<b>Corporate Credit Risk by Industry: EAD vs. Expected Credit Loss</b><br>"
                    "<sup>Evaluating sector concentration and risk-adjusted loss distribution</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Industry Sector",
        yaxis={"title": "Total EAD ($ USD)", "tickprefix": "$", "tickformat": ",.0f", "showgrid": True},
        yaxis2={
            "title": "Weighted ECL ($ USD)",
            "tickprefix": "$",
            "tickformat": ",.0f",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        barmode="group",
        template="plotly_white",
        margin={"l": 70, "r": 70, "t": 70, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
    )

    return fig


def plot_borrower_ecl_waterfall(portfolio_result: CorporatePortfolioECLResult) -> go.Figure:
    """
    Produce a ranked horizontal bar chart showing individual borrower contributions
    to total portfolio expected loss.

    Parameters
    ----------
    portfolio_result : CorporatePortfolioECLResult
        Portfolio calculation output.

    Returns
    -------
    go.Figure
        Horizontal bar chart sorted by loss contribution.
    """
    df = portfolio_result.to_dataframe().sort_values(by="ECL Weighted ($)", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["Borrower Name"],
        x=df["ECL Weighted ($)"],
        orientation="h",
        marker_color=CHART_COLORS["secondary"],
        text=[f"${x:,.0f} ({r})" for x, r in zip(df["ECL Weighted ($)"], df["Rating Grade"])],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Weighted ECL: $%{x:,.2f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title={
            "text": "<b>Individual Borrower Contribution to Portfolio Expected Loss</b><br>"
                    "<sup>Ranked by probability-weighted ECL ($ USD) with internal rating grade</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Weighted Expected Credit Loss ($ USD)",
        yaxis_title="",
        xaxis={"tickprefix": "$", "tickformat": ",.0f", "showgrid": True, "gridcolor": "#E2E8F0"},
        template="plotly_white",
        margin={"l": 200, "r": 80, "t": 70, "b": 50},
        font={"family": "Inter, Arial, sans-serif", "size": 11, "color": "#1E293B"},
        showlegend=False,
    )

    return fig


# ===========================================================================
# 7. Documentation & Accounting Disclosures
# ===========================================================================

def get_ecl_methodology_and_limitations() -> Dict[str, Any]:
    """
    Return comprehensive technical explanations and regulatory limitations
    concerning the ECL calculation engine.
    """
    return {
        "formula": "ECL = PD × LGD × EAD",
        "inputs": {
            "PD": {
                "name": "Probability of Default",
                "definition": "The probability that a borrower defaults over a 12-month period.",
                "valid_range": "[0.0, 1.0]",
                "unit": "Dimensionless probability",
                "notes": "Must be derived from corporate balance-sheet fundamentals or agency ratings.",
            },
            "LGD": {
                "name": "Loss Given Default",
                "definition": "Percentage of exposure lost upon default after workout and collateral recovery.",
                "valid_range": "[0.0, 1.0]",
                "unit": "Dimensionless fraction (1.0 - Recovery Rate)",
                "notes": "Varies by facility seniority, collateral quality, and jurisdiction.",
            },
            "EAD": {
                "name": "Exposure at Default",
                "definition": "Gross credit exposure expected at default (drawn debt + CCF × undrawn).",
                "valid_range": "[0.0, +inf)",
                "unit": "Currency units ($ USD)",
                "notes": "Includes drawn balances and credit conversion factors on committed facilities.",
            },
        },
        "critical_distinctions": [
            {
                "topic": "Consumer Credit vs Corporate Wholesale PD",
                "detail": (
                    "The initial benchmark dataset (UCI Credit Card Clients) contains consumer retail revolving "
                    "credit data. These predicted probabilities must NOT be used as corporate wholesale PDs. "
                    "Corporate default analysis relies on corporate financial statements, interest coverage, "
                    "leverage metrics, industry dynamics, and agency ratings. All corporate demonstrations "
                    "in this module strictly use clearly labelled hypothetical facilities."
                ),
            },
            {
                "topic": "Simple ECL vs. Institutional IFRS 9 / CECL Standards",
                "detail": (
                    "The simple 1-period ECL formula provides a foundational risk calculation but does not "
                    "constitute a complete accounting standard compliance implementation. Key omissions include:"
                ),
                "limitations": [
                    "Discounting: Omission of Effective Interest Rate (EIR) discounting back to the reporting date.",
                    "Staging & Horizons: Fixed 1-year horizon (Stage 1) omitting SICR triggers and multi-year lifetime ECL.",
                    "Macroeconomic Conditioning: Linear multipliers approximate econometric satellite models (VAR/Vasicek).",
                    "Exposure Dynamics: Static EAD assumes fixed exposure, omitting contractual amortization and prepayment curves.",
                    "Correlation & Diversification: Portfolio loss is the sum of expectations; unexpected loss (UL) requires copula models.",
                ],
            },
        ],
    }
