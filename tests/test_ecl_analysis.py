"""
CreditRisk Analytics — Unit & Integration Tests for ECL Analysis Engine
========================================================================

Tests coverage:
1. Core Formula & Input Validation:
   - ECL = PD × LGD × EAD numerical accuracy
   - Permissible ranges: PD in [0, 1], LGD in [0, 1], EAD >= 0
   - Strict validation: negative values, >1.0 bounds, non-numeric types, booleans, NaN/Inf
2. Edge Cases:
   - Zero exposure: EAD = 0 -> ECL = 0
   - Zero default probability: PD = 0 -> ECL = 0
   - Full recovery: LGD = 0 -> ECL = 0
   - Total loss: PD = 1.0 and LGD = 1.0 -> ECL = EAD
3. Hypothetical Corporate Portfolio:
   - Clear synthetic labelling (is_hypothetical=True)
   - Multi-borrower representation across diverse industries and credit ratings
4. Macroeconomic Scenario Modeling:
   - Base, Favourable, Adverse definitions
   - Weight normalization: sum(weights) == 1.0
   - Stress monotonicity: Favourable ECL < Base ECL < Adverse ECL
   - Probability-weighted ECL consistency
5. Sensitivity Analysis:
   - 2D grid evaluation
   - Analytical marginal sensitivities: d(ECL)/d(PD) = LGD * EAD, d(ECL)/d(LGD) = PD * EAD
   - 100 bps (+1%) shock impact calculations
6. Plotly Visualizations:
   - Non-empty traces, correct layouts, valid figure objects
7. Methodological Distinctions & Regulatory Limitations:
   - Distinction between consumer credit-card data and wholesale corporate PDs
   - Accounting limitations: discounting (EIR), horizons/staging, macro satellite models, dynamic EAD
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.risk.ecl_analysis import (
    BorrowerScenarioResult,
    CorporatePortfolioECLResult,
    HypotheticalBorrower,
    Scenario,
    SensitivityAnalysisResult,
    calculate_expected_loss,
    calculate_portfolio_ecl,
    get_default_scenarios,
    get_ecl_methodology_and_limitations,
    get_hypothetical_corporate_portfolio,
    perform_sensitivity_analysis,
    plot_borrower_ecl_waterfall,
    plot_ecl_sensitivity_heatmap,
    plot_portfolio_ecl_by_industry,
    plot_scenario_losses,
    validate_ecl_inputs,
    validate_scenarios,
)


# ===========================================================================
# 1. Core Formula & Mathematical Accuracy Tests
# ===========================================================================

class TestCoreECLFormula:
    """Verify numerical correctness of ECL = PD × LGD × EAD."""

    def test_standard_corporate_loan_calculation(self):
        # Facility: EAD = $10M, PD = 2.5%, LGD = 40%
        # Expected Loss = 0.025 * 0.40 * 10,000,000 = $100,000
        ecl = calculate_expected_loss(pd=0.025, lgd=0.40, ead=10_000_000.0)
        assert np.isclose(ecl, 100_000.0)

    def test_high_yield_stressed_facility(self):
        # High-yield: EAD = $50M, PD = 12.0%, LGD = 60%
        # Expected Loss = 0.12 * 0.60 * 50,000,000 = $3,600,000
        ecl = calculate_expected_loss(pd=0.12, lgd=0.60, ead=50_000_000.0)
        assert np.isclose(ecl, 3_600_000.0)

    def test_hypothetical_borrower_property(self):
        borrower = HypotheticalBorrower(
            borrower_id="TEST-001",
            borrower_name="Test Corp",
            industry="Technology",
            rating_grade="A",
            pd=0.015,
            lgd=0.35,
            ead=20_000_000.0,
        )
        assert np.isclose(borrower.base_ecl, 0.015 * 0.35 * 20_000_000.0)
        assert borrower.is_hypothetical is True


# ===========================================================================
# 2. Edge Cases: Zero Exposure, Boundary Probabilities, Total Loss
# ===========================================================================

class TestECLEdgeCases:
    """Verify robust handling of zero exposure and boundary states."""

    def test_zero_exposure_produces_zero_loss(self):
        # Paid-off or unutilised zero-limit facility
        ecl = calculate_expected_loss(pd=0.05, lgd=0.45, ead=0.0)
        assert ecl == 0.0

    def test_zero_pd_risk_free_facility(self):
        # Sovereign or fully cash-collateralised zero default risk
        ecl = calculate_expected_loss(pd=0.0, lgd=0.45, ead=15_000_000.0)
        assert ecl == 0.0

    def test_zero_lgd_full_recovery(self):
        # 100% over-collateralised with prime liquid securities
        ecl = calculate_expected_loss(pd=0.10, lgd=0.0, ead=15_000_000.0)
        assert ecl == 0.0

    def test_boundary_full_loss_scenario(self):
        # Complete default with zero recovery: PD = 1.0, LGD = 1.0 -> ECL = EAD
        ead = 8_500_000.0
        ecl = calculate_expected_loss(pd=1.0, lgd=1.0, ead=ead)
        assert np.isclose(ecl, ead)

    def test_disallowing_zero_exposure_when_specified(self):
        with pytest.raises(ValueError, match="strictly greater than 0.0"):
            validate_ecl_inputs(pd=0.02, lgd=0.40, ead=0.0, allow_zero_exposure=False)


# ===========================================================================
# 3. Input Validation & Exception Handling
# ===========================================================================

class TestECLInputValidation:
    """Verify strict validation and informative error messages."""

    @pytest.mark.parametrize("invalid_pd", [-0.01, -1.0, 1.0001, 1.5, 50.0])
    def test_pd_out_of_bounds_raises_value_error(self, invalid_pd):
        with pytest.raises(ValueError, match="PD.*between 0.0 and 1.0"):
            validate_ecl_inputs(pd=invalid_pd, lgd=0.40, ead=1_000_000.0)

    @pytest.mark.parametrize("invalid_lgd", [-0.05, -1.0, 1.01, 2.0, 45.0])
    def test_lgd_out_of_bounds_raises_value_error(self, invalid_lgd):
        with pytest.raises(ValueError, match="LGD.*between 0.0 and 1.0"):
            validate_ecl_inputs(pd=0.02, lgd=invalid_lgd, ead=1_000_000.0)

    @pytest.mark.parametrize("invalid_ead", [-1.0, -100_000.0, -0.0001])
    def test_negative_ead_raises_value_error(self, invalid_ead):
        with pytest.raises(ValueError, match="EAD.*must be non-negative"):
            validate_ecl_inputs(pd=0.02, lgd=0.40, ead=invalid_ead)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_nan_and_inf_raise_value_error(self, bad_val):
        with pytest.raises(ValueError, match="must be finite"):
            validate_ecl_inputs(pd=bad_val, lgd=0.40, ead=1_000_000.0)
        with pytest.raises(ValueError, match="must be finite"):
            validate_ecl_inputs(pd=0.02, lgd=bad_val, ead=1_000_000.0)
        with pytest.raises(ValueError, match="must be finite"):
            validate_ecl_inputs(pd=0.02, lgd=0.40, ead=bad_val)

    def test_non_numeric_and_boolean_raise_type_error(self):
        # Reject booleans (True/False evaluate to 1/0 in Python if not guarded)
        with pytest.raises(TypeError, match="must be a numeric float or int"):
            validate_ecl_inputs(pd=True, lgd=0.40, ead=1_000_000.0)
        with pytest.raises(TypeError, match="must be a numeric float or int"):
            validate_ecl_inputs(pd=0.02, lgd="0.40", ead=1_000_000.0)
        with pytest.raises(TypeError, match="must be a numeric float or int"):
            validate_ecl_inputs(pd=0.02, lgd=0.40, ead=None)


# ===========================================================================
# 4. Hypothetical Corporate Portfolio Tests
# ===========================================================================

class TestHypotheticalCorporatePortfolio:
    """Verify structure, diversity, and explicit labelling of synthetic loans."""

    def test_portfolio_generation_completeness(self):
        portfolio = get_hypothetical_corporate_portfolio()
        assert len(portfolio) == 10

        total_ead = sum(b.ead for b in portfolio)
        assert np.isclose(total_ead, 175_600_000.0)

    def test_all_borrowers_clearly_labelled_hypothetical(self):
        portfolio = get_hypothetical_corporate_portfolio()
        for b in portfolio:
            assert b.is_hypothetical is True
            assert 0.0 <= b.pd <= 1.0
            assert 0.0 <= b.lgd <= 1.0
            assert b.ead > 0.0
            assert len(b.industry) > 0
            assert len(b.rating_grade) > 0

    def test_portfolio_spans_multiple_industries_and_ratings(self):
        portfolio = get_hypothetical_corporate_portfolio()
        industries = {b.industry for b in portfolio}
        ratings = {b.rating_grade for b in portfolio}

        assert len(industries) >= 6
        assert {"AAA", "AA", "A", "BBB", "BB", "B"}.issubset(ratings)


# ===========================================================================
# 5. Macroeconomic Scenario Engine Tests
# ===========================================================================

class TestMacroeconomicScenarios:
    """Verify Base, Favourable, and Adverse multi-scenario stress engine."""

    def test_default_scenarios_weights_sum_to_one(self):
        scenarios = get_default_scenarios()
        total_w = sum(s.weight for s in scenarios.values())
        assert np.isclose(total_w, 1.0)
        assert "Base" in scenarios
        assert "Favourable" in scenarios
        assert "Adverse" in scenarios

    def test_scenario_weight_validation_rejects_unbalanced_weights(self):
        scenarios = get_default_scenarios()
        scenarios["Base"] = Scenario("Base", weight=0.80, pd_multiplier=1.0, lgd_multiplier=1.0)
        # Sum is now 0.80 + 0.25 + 0.25 = 1.30 != 1.0
        with pytest.raises(ValueError, match="Scenario weights must sum to 1.0"):
            validate_scenarios(scenarios)

    def test_portfolio_ecl_stress_monotonicity(self):
        """Under stress, Favourable ECL < Base ECL < Adverse ECL."""
        portfolio = get_hypothetical_corporate_portfolio()
        result = calculate_portfolio_ecl(portfolio)

        assert result.total_ecl_favourable < result.total_ecl_base
        assert result.total_ecl_base < result.total_ecl_adverse

        # Weighted loss must fall strictly between favourable and adverse
        assert result.total_ecl_favourable < result.total_ecl_weighted < result.total_ecl_adverse

    def test_empty_portfolio_raises_value_error(self):
        with pytest.raises(ValueError, match="borrowers list is empty"):
            calculate_portfolio_ecl([])

    def test_portfolio_result_dataframes_and_aggregations(self):
        portfolio = get_hypothetical_corporate_portfolio()
        result = calculate_portfolio_ecl(portfolio)

        # Facility DataFrame
        df_facilities = result.to_dataframe()
        assert len(df_facilities) == 10
        assert "ECL Weighted ($)" in df_facilities.columns
        assert "Loss Rate (%)" in df_facilities.columns

        # Industry summary
        df_ind = result.summary_by_industry()
        assert np.isclose(df_ind["Total_EAD"].sum(), result.total_ead)
        assert np.isclose(df_ind["Total_Weighted_ECL"].sum(), result.total_ecl_weighted)

        # Rating summary
        df_rat = result.summary_by_rating()
        assert np.isclose(df_rat["Total_EAD"].sum(), result.total_ead)
        assert np.isclose(df_rat["Total_Weighted_ECL"].sum(), result.total_ecl_weighted)

        # Scenario comparison table
        df_sc = result.scenario_summary_table()
        assert len(df_sc) == 4
        assert "Portfolio Loss Rate (%)" in df_sc.columns


# ===========================================================================
# 6. Sensitivity Analysis Engine Tests
# ===========================================================================

class TestSensitivityAnalysis:
    """Verify 2D sensitivity grid and first-order marginal derivatives."""

    def test_sensitivity_grid_dimensions_and_base_point(self):
        pd_base = 0.04
        lgd_base = 0.45
        ead = 10_000_000.0

        sens = perform_sensitivity_analysis(pd=pd_base, lgd=lgd_base, ead=ead, pd_steps=7, lgd_steps=9)

        assert sens.ecl_matrix.shape == (7, 9)
        assert np.isclose(sens.base_ecl, pd_base * lgd_base * ead)

    def test_analytical_marginal_sensitivities(self):
        # d(ECL)/d(PD) = LGD * EAD
        # d(ECL)/d(LGD) = PD * EAD
        pd_val = 0.05
        lgd_val = 0.40
        ead = 20_000_000.0

        sens = perform_sensitivity_analysis(pd_val, lgd_val, ead)

        expected_d_pd = lgd_val * ead
        expected_d_lgd = pd_val * ead
        assert np.isclose(sens.marginal_pd_sensitivity, expected_d_pd)
        assert np.isclose(sens.marginal_lgd_sensitivity, expected_d_lgd)

        # 100 bps (+1.00%) shock impact
        assert np.isclose(sens.shock_100bps_pd_impact, 0.01 * expected_d_pd)
        assert np.isclose(sens.shock_100bps_lgd_impact, 0.01 * expected_d_lgd)

    def test_sensitivity_to_dataframe(self):
        sens = perform_sensitivity_analysis(pd=0.03, lgd=0.35, ead=5_000_000.0, pd_steps=5, lgd_steps=5)
        df_sens = sens.to_dataframe()
        assert len(df_sens) == 25
        assert "Delta to Base ($)" in df_sens.columns
        assert "Delta to Base (%)" in df_sens.columns


# ===========================================================================
# 7. Plotly Visual Analytics Tests
# ===========================================================================

class TestPlotlyVisualizations:
    """Verify Plotly charts return valid, populated interactive figures."""

    @pytest.fixture
    def portfolio_result(self) -> CorporatePortfolioECLResult:
        portfolio = get_hypothetical_corporate_portfolio()
        return calculate_portfolio_ecl(portfolio)

    def test_plot_scenario_losses(self, portfolio_result):
        fig = plot_scenario_losses(portfolio_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 4  # 4 scenarios

    def test_plot_ecl_sensitivity_heatmap(self):
        sens = perform_sensitivity_analysis(pd=0.03, lgd=0.40, ead=10_000_000.0)
        fig = plot_ecl_sensitivity_heatmap(sens)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "heatmap"

    def test_plot_portfolio_ecl_by_industry(self, portfolio_result):
        fig = plot_portfolio_ecl_by_industry(portfolio_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # EAD and ECL traces

    def test_plot_borrower_ecl_waterfall(self, portfolio_result):
        fig = plot_borrower_ecl_waterfall(portfolio_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].y) == 10  # 10 borrowers


# ===========================================================================
# 8. Regulatory Limitations & Methodology Documentation
# ===========================================================================

class TestMethodologyAndLimitations:
    """Verify complete documentation of assumptions, distinctions, and accounting limitations."""

    def test_documentation_dictionary_contains_required_topics(self):
        docs = get_ecl_methodology_and_limitations()

        assert docs["formula"] == "ECL = PD × LGD × EAD"
        assert "PD" in docs["inputs"]
        assert "LGD" in docs["inputs"]
        assert "EAD" in docs["inputs"]

        # Verify critical distinction between consumer and corporate wholesale PD
        topics = [item["topic"] for item in docs["critical_distinctions"]]
        assert any("Consumer Credit" in t for t in topics)
        assert any("Institutional IFRS 9" in t or "Simple ECL" in t for t in topics)

        # Verify the 5 key accounting limitations are documented
        limitations = []
        for item in docs["critical_distinctions"]:
            if "limitations" in item:
                limitations.extend(item["limitations"])

        assert any("Discounting" in lim for lim in limitations)
        assert any("Staging" in lim or "Horizons" in lim for lim in limitations)
        assert any("Macroeconomic Conditioning" in lim for lim in limitations)
        assert any("Exposure Dynamics" in lim for lim in limitations)
