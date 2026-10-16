"""
CreditRisk Analytics — Expected Credit Loss (ECL) Calculation Engine
=====================================================================

Implements institutional IFRS 9 / CECL three-stage impairment framework:
- Exposure at Default (EAD) with Credit Conversion Factor (CCF)
- Significant Increase in Credit Risk (SICR) staging rules
- 12-Month ECL (Stage 1) vs. Lifetime ECL (Stage 2 & Stage 3)
- Multi-scenario probability-weighted macroeconomic projections (Base, Downturn, Upturn)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ECLConfig, ecl_config, logger


@dataclass
class LoanECLRecord:
    """Individual loan facility ECL valuation output."""
    account_id: Any
    current_balance: float
    credit_limit: float
    ead: float
    stage: int
    pd_12m: float
    pd_lifetime: float
    lgd: float
    ecl_base: float
    ecl_downturn: float
    ecl_upturn: float
    ecl_weighted: float


@dataclass
class ECLPortfolioResult:
    """Consolidated portfolio ECL impairment metrics and stage distribution."""
    total_accounts: int
    total_credit_limit: float
    total_balance: float
    total_ead: float
    total_ecl_weighted: float
    total_ecl_base: float
    total_ecl_downturn: float
    total_ecl_upturn: float
    ecl_coverage_ratio_pct: float
    stage_breakdown: Dict[str, Dict[str, Any]]
    scenario_weights: Dict[str, float]
    detailed_df: pd.DataFrame = field(repr=False)

    def summary_table(self) -> pd.DataFrame:
        """Tabular summary of stage migration and reserve requirements."""
        rows = []
        for stage_name, metrics in self.stage_breakdown.items():
            rows.append({
                "Stage": stage_name,
                "Accounts": metrics["account_count"],
                "Account_Share_Pct": round(metrics["account_share_pct"], 2),
                "Total_EAD": round(metrics["total_ead"], 2),
                "EAD_Share_Pct": round(metrics["ead_share_pct"], 2),
                "Average_PD": round(metrics["avg_pd"], 4),
                "Weighted_ECL": round(metrics["total_ecl_weighted"], 2),
                "Coverage_Ratio_Pct": round(metrics["coverage_ratio_pct"], 2),
            })
        return pd.DataFrame(rows)


class ECLCalculator:
    """
    Computes facility and portfolio level Expected Credit Loss under IFRS 9 standards.
    """

    def __init__(self, config: Optional[ECLConfig] = None):
        self.config = config or ecl_config

    def calculate_ead(self, balance: pd.Series | np.ndarray, limit: pd.Series | np.ndarray) -> np.ndarray:
        """
        Calculate Exposure at Default:
        EAD = Drawn Balance + CCF * max(0, Limit - Drawn Balance)
        """
        bal = np.asarray(balance, dtype=float)
        lim = np.asarray(limit, dtype=float)

        undrawn = np.maximum(0.0, lim - bal)
        ead = bal + (self.config.ccf * undrawn)
        # Ensure EAD is at least the drawn balance
        return np.maximum(bal, ead)

    def determine_staging(
        self,
        pd_current: np.ndarray,
        pd_origination: Optional[np.ndarray] = None,
        delinquency_status: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Assign loans to IFRS 9 Staging:
        Stage 1: Performing (12m ECL)
        Stage 2: Underperforming / SICR (Relative PD increase >= threshold OR 30+ DPD)
        Stage 3: Credit-Impaired / Default (Delinquency status >= 3 or 90+ DPD)
        """
        n = len(pd_current)
        stages = np.ones(n, dtype=int)  # Default: Stage 1

        # Check SICR relative PD shift (if origination PD provided, else proxy using portfolio average)
        if pd_origination is None:
            baseline_pd = np.median(pd_current)
            pd_ratio = pd_current / max(baseline_pd, 1e-4)
        else:
            pd_ratio = pd_current / np.maximum(pd_origination, 1e-4)

        # Stage 2: SICR trigger
        is_sicr = pd_ratio >= self.config.sicr_pd_multiplier
        if delinquency_status is not None:
            # Payment delay of 1 or 2 months corresponds to 30-60 DPD
            is_30_dpd = delinquency_status >= 1
            is_sicr = is_sicr | is_30_dpd

        stages[is_sicr] = 2

        # Stage 3: Default trigger
        if delinquency_status is not None:
            # Payment delay >= 3 months corresponds to 90+ DPD (Basel Default definition)
            is_default = delinquency_status >= 3
            stages[is_default] = 3

        return stages

    def compute_portfolio_ecl(
        self,
        df: pd.DataFrame,
        pd_col: str = "calibrated_pd",
        balance_col: str = "bill_amt1",
        limit_col: str = "limit_bal",
        delinquency_col: Optional[str] = "pay_0",
        id_col: Optional[str] = "id",
        custom_lgd: Optional[float] = None,
    ) -> ECLPortfolioResult:
        """
        Calculate facility-level EAD, Staging, and Multi-scenario weighted ECL for an entire portfolio.
        """
        data = df.copy()
        n_accounts = len(data)

        if n_accounts == 0:
            raise ValueError("Input DataFrame is empty.")

        pd_12m = np.asarray(data[pd_col], dtype=float)
        # Compute multi-year lifetime PD using survival approximation
        t_years = self.config.lifetime_horizon_years
        pd_lifetime = 1.0 - ((1.0 - pd_12m) ** t_years)

        # EAD calculation
        balance = np.maximum(0.0, data[balance_col].values if balance_col in data.columns else np.zeros(n_accounts))
        limit = np.maximum(1000.0, data[limit_col].values if limit_col in data.columns else np.ones(n_accounts) * 10000.0)
        ead = self.calculate_ead(balance, limit)

        # Delinquency status for staging
        delinq = data[delinquency_col].values if (delinquency_col and delinquency_col in data.columns) else None
        stages = self.determine_staging(pd_12m, delinquency_status=delinq)

        # LGD baseline
        base_lgd = custom_lgd if custom_lgd is not None else self.config.baseline_lgd

        # Calculate scenario-specific ECLs
        scenarios = self.config.scenarios
        scenario_ecl_arrays: Dict[str, np.ndarray] = {}

        for sc_name, sc_params in scenarios.items():
            pd_mult = sc_params["pd_multiplier"]
            lgd_mult = sc_params["lgd_multiplier"]

            sc_pd_12m = np.clip(pd_12m * pd_mult, 0.0, 1.0)
            sc_pd_lifetime = np.clip(pd_lifetime * pd_mult, 0.0, 1.0)
            sc_lgd = np.clip(base_lgd * lgd_mult, 0.0, 1.0)

            # Apply IFRS 9 Staging logic:
            # Stage 1: 12-Month ECL
            # Stage 2: Lifetime ECL
            # Stage 3: Full default (PD=1.0) Lifetime ECL
            facility_ecl = np.zeros(n_accounts)
            facility_ecl[stages == 1] = ead[stages == 1] * sc_pd_12m[stages == 1] * sc_lgd
            facility_ecl[stages == 2] = ead[stages == 2] * sc_pd_lifetime[stages == 2] * sc_lgd
            facility_ecl[stages == 3] = ead[stages == 3] * 1.0 * sc_lgd

            scenario_ecl_arrays[sc_name] = facility_ecl

        # Weighted ECL
        weighted_ecl = np.zeros(n_accounts)
        weights_map = {}
        for sc_name, sc_params in scenarios.items():
            w = sc_params["weight"]
            weights_map[sc_name] = w
            weighted_ecl += w * scenario_ecl_arrays[sc_name]

        # Attach computed risk attributes to output DataFrame
        data["ead"] = np.round(ead, 2)
        data["ifrs9_stage"] = stages
        data["pd_12m"] = np.round(pd_12m, 4)
        data["pd_lifetime"] = np.round(pd_lifetime, 4)
        data["ecl_base"] = np.round(scenario_ecl_arrays.get("Base", weighted_ecl), 2)
        data["ecl_downturn"] = np.round(scenario_ecl_arrays.get("Downturn", weighted_ecl), 2)
        data["ecl_upturn"] = np.round(scenario_ecl_arrays.get("Upturn", weighted_ecl), 2)
        data["ecl_weighted"] = np.round(weighted_ecl, 2)

        # Portfolio aggregate KPIs
        total_limit = float(np.sum(limit))
        total_bal = float(np.sum(balance))
        total_ead = float(np.sum(ead))
        tot_weighted_ecl = float(np.sum(weighted_ecl))
        tot_base_ecl = float(np.sum(scenario_ecl_arrays.get("Base", 0.0)))
        tot_down_ecl = float(np.sum(scenario_ecl_arrays.get("Downturn", 0.0)))
        tot_up_ecl = float(np.sum(scenario_ecl_arrays.get("Upturn", 0.0)))

        coverage_ratio = (tot_weighted_ecl / total_ead) * 100 if total_ead > 0 else 0.0

        # Stage distribution breakdown
        stage_breakdown = {}
        stage_names = {1: "Stage 1 (Performing)", 2: "Stage 2 (SICR)", 3: "Stage 3 (Impaired)"}

        for st_num in [1, 2, 3]:
            st_mask = stages == st_num
            st_count = int(np.sum(st_mask))
            st_ead = float(np.sum(ead[st_mask]))
            st_ecl = float(np.sum(weighted_ecl[st_mask]))
            st_avg_pd = float(np.mean(pd_12m[st_mask])) if st_count > 0 else 0.0

            stage_breakdown[stage_names[st_num]] = {
                "stage_number": st_num,
                "account_count": st_count,
                "account_share_pct": (st_count / n_accounts) * 100,
                "total_ead": st_ead,
                "ead_share_pct": (st_ead / total_ead) * 100 if total_ead > 0 else 0.0,
                "avg_pd": st_avg_pd,
                "total_ecl_weighted": st_ecl,
                "coverage_ratio_pct": (st_ecl / st_ead) * 100 if st_ead > 0 else 0.0,
            }

        return ECLPortfolioResult(
            total_accounts=n_accounts,
            total_credit_limit=total_limit,
            total_balance=total_bal,
            total_ead=total_ead,
            total_ecl_weighted=tot_weighted_ecl,
            total_ecl_base=tot_base_ecl,
            total_ecl_downturn=tot_down_ecl,
            total_ecl_upturn=tot_up_ecl,
            ecl_coverage_ratio_pct=coverage_ratio,
            stage_breakdown=stage_breakdown,
            scenario_weights=weights_map,
            detailed_df=data,
        )
