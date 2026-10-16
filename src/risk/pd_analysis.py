"""
CreditRisk Analytics — Probability of Default (PD) Backtesting & Analytics
===========================================================================

Implements Basel Committee on Banking Supervision (BCBS) backtesting protocols
for Probability of Default (PD) validation, including the exact Binomial Traffic Light Test,
marginal vs. cumulative PD projections, and decile discrimination analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import logger


def backtest_pd_by_grade(
    df: pd.DataFrame,
    grade_col: str = "rating_grade",
    pd_col: str = "calibrated_pd",
    target_col: str = "default_payment_next_month",
) -> pd.DataFrame:
    """
    Executes the regulatory Basel Binomial Traffic Light test per rating grade.

    Tests whether observed defaults significantly exceed expected defaults:
        H0: Realized default rate <= Model-predicted PD
        p-value = P(X >= K | N, PD_predicted)
    Traffic Light Zones:
        Green: p-value >= 0.05 (Valid / Conservative calibration)
        Amber: 0.001 <= p-value < 0.05 (Warning / Elevated defaults)
        Red:   p-value < 0.001 (Critical failure: Significant underestimation)
    """
    required_cols = [grade_col, pd_col, target_col]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in DataFrame for PD backtesting.")

    records = []
    unique_grades = sorted(df[grade_col].unique())

    for grade in unique_grades:
        sub = df[df[grade_col] == grade]
        n_accounts = len(sub)
        if n_accounts == 0:
            continue

        n_defaults = int(sub[target_col].sum())
        obs_dr = float(n_defaults / n_accounts)
        pred_pd = float(sub[pd_col].mean())

        # Exact Binomial test: Probability of observing >= n_defaults under pred_pd
        # Note: if pred_pd <= 0, handle boundary safely
        safe_pd = min(max(pred_pd, 1e-5), 0.9999)
        if n_defaults == 0:
            p_val = 1.0
        else:
            # 1 - binom.cdf(k - 1, n, p) gives P(X >= k)
            p_val = float(1.0 - stats.binom.cdf(n_defaults - 1, n_accounts, safe_pd))

        if p_val >= 0.05:
            traffic_light = "GREEN"
            interpretation = "Model PD is statistically conservative or well-calibrated."
        elif p_val >= 0.001:
            traffic_light = "AMBER"
            interpretation = "Moderate default excess; model warrants heightened supervisory monitoring."
        else:
            traffic_light = "RED"
            interpretation = "Significant default excess; model under-predicts risk; recalibration required."

        records.append({
            "Grade": grade,
            "Accounts": n_accounts,
            "Observed_Defaults": n_defaults,
            "Observed_Default_Rate": round(obs_dr, 4),
            "Predicted_PD": round(pred_pd, 4),
            "PD_Spread": round(obs_dr - pred_pd, 4),
            "Binomial_P_Value": round(p_val, 4),
            "Traffic_Light": traffic_light,
            "Interpretation": interpretation,
        })

    return pd.DataFrame(records)


class PDAnalyzer:
    """
    Quantitative analysis tools for term-structure projections and decile lift analysis.
    """

    @staticmethod
    def project_cumulative_pd(
        pd_12m: float | np.ndarray,
        horizons_years: List[int] = [1, 2, 3, 5],
    ) -> Dict[int, float | np.ndarray]:
        """
        Projects multi-year cumulative PD assuming constant hazard rates:
        PD(t) = 1 - (1 - PD_12m)^t
        """
        p = np.asarray(pd_12m, dtype=float)
        projections = {}
        for h in horizons_years:
            cum_pd = 1.0 - ((1.0 - p) ** h)
            projections[h] = cum_pd
        return projections

    @staticmethod
    def decile_lift_analysis(
        df: pd.DataFrame,
        pd_col: str = "calibrated_pd",
        target_col: str = "default_payment_next_month",
        n_deciles: int = 10,
    ) -> pd.DataFrame:
        """
        Calculates decile lift table, cumulative response, and capture rates.
        """
        data = df[[pd_col, target_col]].copy()
        # Sort descending by predicted PD (worst credit first)
        data = data.sort_values(by=pd_col, ascending=False).reset_index(drop=True)

        data["decile"] = pd.qcut(data.index, q=n_deciles, labels=[f"D{i}" for i in range(1, n_deciles + 1)])

        total_defaults = data[target_col].sum()
        total_accounts = len(data)
        baseline_rate = total_defaults / total_accounts if total_accounts > 0 else 0.0

        summary = data.groupby("decile", observed=False).agg(
            accounts=(target_col, "count"),
            defaults=(target_col, "sum"),
            mean_predicted_pd=(pd_col, "mean"),
        )

        summary["observed_default_rate"] = summary["defaults"] / summary["accounts"]
        summary["lift"] = summary["observed_default_rate"] / baseline_rate if baseline_rate > 0 else 1.0
        summary["cum_defaults"] = summary["defaults"].cumsum()
        summary["cum_defaults_captured_pct"] = (summary["cum_defaults"] / total_defaults) * 100 if total_defaults > 0 else 0.0
        summary["cum_accounts"] = summary["accounts"].cumsum()
        summary["cum_accounts_pct"] = (summary["cum_accounts"] / total_accounts) * 100

        return summary.reset_index().round(4)
