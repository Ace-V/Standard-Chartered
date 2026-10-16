"""
Tests for Risk Segmentation, Credit Scorecard, and IFRS 9 ECL Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.expected_loss import ECLCalculator
from src.risk.segmentation import CreditScorecard, MasterRatingScale


def test_credit_scorecard_scaling():
    scorecard = CreditScorecard()

    # Good credit (very low PD, e.g. 0.01) must have high score
    score_low_pd = scorecard.pd_to_score(0.01)
    # Bad credit (high PD, e.g. 0.50) must have low score
    score_high_pd = scorecard.pd_to_score(0.50)

    assert score_low_pd > score_high_pd
    assert 300 <= score_low_pd <= 850
    assert 300 <= score_high_pd <= 850


def test_master_rating_scale_assignment():
    scale = MasterRatingScale()

    assert scale.assign_grade(0.002) == "AAA"
    assert scale.assign_grade(0.010) == "AA"
    assert scale.assign_grade(0.025) == "A"
    assert scale.assign_grade(0.050) == "BBB"
    assert scale.assign_grade(0.100) == "BB"
    assert scale.assign_grade(0.200) == "B"
    assert scale.assign_grade(0.400) == "CCC"


def test_ecl_ead_calculation():
    calculator = ECLCalculator()

    # Balance 20,000, Limit 100,000 -> Undrawn 80,000. CCF = 0.75 -> EAD = 20,000 + 0.75 * 80,000 = 80,000
    ead = calculator.calculate_ead(balance=np.array([20000.0]), limit=np.array([100000.0]))
    assert np.isclose(ead[0], 80000.0)

    # Balance > Limit -> EAD is at least drawn balance
    ead_over = calculator.calculate_ead(balance=np.array([120000.0]), limit=np.array([100000.0]))
    assert np.isclose(ead_over[0], 120000.0)


def test_ecl_staging_rules():
    calculator = ECLCalculator()

    # Baseline PD = 0.02, Current PD = 0.02 (no SICR, no delay) -> Stage 1
    stages = calculator.determine_staging(
        pd_current=np.array([0.02]),
        pd_origination=np.array([0.02]),
        delinquency_status=np.array([0]),
    )
    assert stages[0] == 1

    # Significant increase in PD (from 0.02 to 0.08 -> 4x increase > 2.5x threshold) -> Stage 2
    stages_sicr = calculator.determine_staging(
        pd_current=np.array([0.08]),
        pd_origination=np.array([0.02]),
        delinquency_status=np.array([0]),
    )
    assert stages_sicr[0] == 2

    # Default trigger: delinquency status >= 3 -> Stage 3
    stages_default = calculator.determine_staging(
        pd_current=np.array([0.02]),
        pd_origination=np.array([0.02]),
        delinquency_status=np.array([3]),
    )
    assert stages_default[0] == 3
