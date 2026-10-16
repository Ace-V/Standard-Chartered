"""
CreditRisk Analytics — Monitoring Subsystem: Simulation Engine
==============================================================

Provides realistic synthetic monitoring scenarios to demonstrate model surveillance
without fabricating real-world historical metrics.

CRITICAL IMPLEMENTATION RULES:
1. Never claim simulated monitoring cohorts represent real unverified production history.
2. Every simulation run is prominently tagged with '[SIMULATION: <Scenario Name>]' and metadata.
3. Separates feature drift (covariate shift) from model performance decay.
4. Correctly demonstrates behaviour when outcome labels are unavailable (e.g. recent loan intake).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulatedCohort:
    """A simulated monitoring cohort with explicit synthetic metadata."""
    name: str
    description: str
    data: pd.DataFrame
    labels: Optional[pd.Series]
    labels_available: bool
    is_simulation: bool = True
    disclaimer: str = (
        "PROVABLY SYNTHETIC: This dataset partition is a synthetic simulation generated to "
        "test and demonstrate credit risk monitoring algorithms. It does not represent actual bank history."
    )


def simulate_monitoring_cohort(
    base_df: pd.DataFrame,
    scenario_type: str = "stable",
    target_col: str = "default_payment_next_month",
    seed: int = 42,
) -> SimulatedCohort:
    """
    Generate an explicitly labelled simulated monitoring cohort.

    Scenarios:
    1. 'stable': Identical distribution to baseline; tests false positive resistance (PSI < 0.10).
    2. 'covariate_shift': Macroeconomic credit stress; higher utilization and delinquency (PSI >= 0.25).
    3. 'data_quality_defect': Ingestion pipeline bug; introduces null values and negative ages.
    4. 'unlabelled_recent': Recent loan origination cohort (3-month vintage); labels are unavailable (None).
    5. 'small_sample': Insufficient sample size (n=20) to test graceful degradation.
    """
    rng = np.random.RandomState(seed)
    df = base_df.copy()
    n = len(df)

    if scenario_type == "stable":
        # Random sample with replacement to simulate stable production intake
        idx = rng.choice(n, size=min(n, 3000), replace=True)
        sim_df = df.iloc[idx].copy().reset_index(drop=True)
        y = sim_df[target_col] if target_col in sim_df.columns else None
        return SimulatedCohort(
            name="[SIMULATION: Stable Production]",
            description="Simulated ongoing intake with identical risk characteristics (clean baseline).",
            data=sim_df,
            labels=y,
            labels_available=y is not None,
        )

    elif scenario_type == "covariate_shift":
        # Simulate macroeconomic stress: inflation, elevated utilization, credit limit contractions
        sim_df = df.sample(n=min(n, 3000), random_state=seed).copy().reset_index(drop=True)
        if "limit_bal" in sim_df.columns:
            sim_df["limit_bal"] = (sim_df["limit_bal"] * rng.uniform(0.60, 0.85, size=len(sim_df))).round(-2)
        if "bill_amt1" in sim_df.columns:
            sim_df["bill_amt1"] = (sim_df["bill_amt1"] * rng.uniform(1.20, 1.60, size=len(sim_df))).round(2)
        if "pay_0" in sim_df.columns:
            # Shift payment delay upwards by 1-2 steps
            sim_df["pay_0"] = np.clip(sim_df["pay_0"] + rng.choice([0, 1, 2], size=len(sim_df), p=[0.4, 0.4, 0.2]), -2, 8)

        # Ground truth labels are present (e.g. mature 12m cohort under stress)
        y = sim_df[target_col] if target_col in sim_df.columns else None
        return SimulatedCohort(
            name="[SIMULATION: Macroeconomic Covariate Shift]",
            description="Simulated credit stress: balance inflation, limit contraction, and increased 30+ DPD delays.",
            data=sim_df,
            labels=y,
            labels_available=y is not None,
        )

    elif scenario_type == "data_quality_defect":
        # Simulate upstream pipeline corruption
        sim_df = df.sample(n=min(n, 2000), random_state=seed).copy().reset_index(drop=True)
        # Inject 8% missing values into limit_bal
        mask_miss = rng.rand(len(sim_df)) < 0.08
        if "limit_bal" in sim_df.columns:
            sim_df.loc[mask_miss, "limit_bal"] = np.nan
        # Inject out-of-range negative age
        if "age" in sim_df.columns:
            mask_age = rng.rand(len(sim_df)) < 0.03
            sim_df.loc[mask_age, "age"] = -99
        # Inject unexpected category in EDUCATION
        if "education" in sim_df.columns:
            mask_edu = rng.rand(len(sim_df)) < 0.04
            sim_df.loc[mask_edu, "education"] = 999

        y = sim_df[target_col] if target_col in sim_df.columns else None
        return SimulatedCohort(
            name="[SIMULATION: Upstream Data Pipeline Defect]",
            description="Simulated pipeline failure injecting 8% missing limits, negative ages, and corrupt category codes.",
            data=sim_df,
            labels=y,
            labels_available=y is not None,
        )

    elif scenario_type == "unlabelled_recent":
        # Simulate fresh loan intake from the last quarter (labels not yet mature!)
        sim_df = df.sample(n=min(n, 2500), random_state=seed).copy().reset_index(drop=True)
        # Labels are completely withheld / None
        return SimulatedCohort(
            name="[SIMULATION: Recent Unlabelled Originations]",
            description="Recent 3-month loan intake where 12-month default maturity has not yet elapsed (labels = None).",
            data=sim_df,
            labels=None,
            labels_available=False,
        )

    elif scenario_type == "small_sample":
        # Simulate a small pilot portfolio of 20 accounts
        sim_df = df.sample(n=20, random_state=seed).copy().reset_index(drop=True)
        y = sim_df[target_col] if target_col in sim_df.columns else None
        return SimulatedCohort(
            name="[SIMULATION: Small Sample Pilot (n=20)]",
            description="Small account sample to verify statistical uncertainty warnings and graceful degradation.",
            data=sim_df,
            labels=y,
            labels_available=y is not None,
        )

    else:
        raise ValueError(f"Unknown simulation scenario_type: '{scenario_type}'.")
