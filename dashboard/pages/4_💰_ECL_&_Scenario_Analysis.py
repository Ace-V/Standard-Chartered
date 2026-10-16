"""
CreditRisk Analytics — IFRS 9 Expected Credit Loss (ECL) & Scenario Analysis
=============================================================================

Implements institutional IFRS 9 three-stage impairment calculations, Master Rating
Scale segmentation, Basel Binomial backtesting, and interactive macroeconomic stress testing.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ECLConfig, MODELS_DIR
from src.data.ingestion import load_raw_dataset
from src.reporting.charts import (
    plot_ecl_waterfall,
    plot_rating_distribution,
    plot_scenario_comparison,
)
from src.risk.ecl_analysis import (
    Scenario,
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
)
from src.risk.expected_loss import ECLCalculator
from src.risk.pd_analysis import backtest_pd_by_grade
from src.risk.segmentation import MasterRatingScale, RiskSegmenter
from src.storage.database import get_db_manager

st.set_page_config(page_title="ECL & Scenario Analysis | CreditRisk", page_icon="💰", layout="wide")

st.title("💰 IFRS 9 Expected Credit Loss (ECL) & Stress Testing")
st.caption("Stage 1 (12-Month) vs. Stage 2/3 (Lifetime) provisions with forward-looking macroeconomic weighting.")

artifact_path = MODELS_DIR / "pd_model_best.joblib"
if not artifact_path.exists():
    st.warning("Please visit '3_📈_PD_Model_Performance' first to train and serialize the champion PD model.")
    st.stop()

@st.cache_data
def get_scored_portfolio():
    df_raw = load_raw_dataset(source="uci")
    model_art = joblib.load(artifact_path)
    # Predict calibrated default probabilities
    probs = model_art.predict_proba(df_raw, calibrate=True)
    df_raw["calibrated_pd"] = probs
    # Segment portfolio
    segmenter = RiskSegmenter()
    df_segmented = segmenter.segment(df_raw, pd_col="calibrated_pd")
    return df_segmented

df_scored = get_scored_portfolio()

# Top Navigation Tabs
tab_ecl, tab_corp, tab_rating_scale, tab_basel_test = st.tabs([
    "Retail IFRS 9 Staging (Empirical Dataset)",
    "🏢 Corporate ECL (PD × LGD × EAD)",
    "Master Rating Scale Distribution",
    "Basel Binomial Traffic Light Backtest",
])

with tab_ecl:
    st.subheader("Interactive ECL Valuation & Parameter Calibration")

    with st.expander("⚙️ Adjust Macroeconomic Stress & Regulatory Impairment Parameters", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            custom_lgd = st.slider("Loss Given Default (LGD)", min_value=0.20, max_value=0.80, value=0.45, step=0.05, help="Standard Basel Foundation retail unsecured LGD = 45%")
            custom_ccf = st.slider("Credit Conversion Factor (CCF)", min_value=0.30, max_value=1.00, value=0.75, step=0.05, help="Drawdown factor for undrawn facility limits")
        with col_p2:
            sicr_mult = st.slider("SICR Relative PD Multiplier", min_value=1.5, max_value=4.0, value=2.5, step=0.25, help="Relative PD jump triggering Stage 2 transfer")
            lifetime_years = st.slider("Lifetime Horizon (Years)", min_value=1, max_value=5, value=3, step=1)
        with col_p3:
            w_base = st.slider("Base Scenario Weight", min_value=0.10, max_value=0.80, value=0.50, step=0.05)
            w_down = st.slider("Downturn Stress Weight", min_value=0.10, max_value=0.60, value=0.30, step=0.05)
            w_up = max(0.0, round(1.0 - w_base - w_down, 2))
            st.write(f"**Upturn Scenario Weight:** {w_up:.2f} (auto-balanced to 1.00)")

    # Execute ECL Calculation
    ecl_cfg = ECLConfig(
        baseline_lgd=custom_lgd,
        ccf=custom_ccf,
        sicr_pd_multiplier=sicr_mult,
        lifetime_horizon_years=lifetime_years,
        scenarios={
            "Base": {"weight": w_base, "pd_multiplier": 1.0, "lgd_multiplier": 1.0},
            "Downturn": {"weight": w_down, "pd_multiplier": 1.35, "lgd_multiplier": 1.15},
            "Upturn": {"weight": w_up, "pd_multiplier": 0.80, "lgd_multiplier": 0.90},
        },
    )
    calculator = ECLCalculator(config=ecl_cfg)
    ecl_result = calculator.compute_portfolio_ecl(df_scored, custom_lgd=custom_lgd)

    # Key Impairment Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Exposure at Default (EAD)", f"${ecl_result.total_ead:,.0f}")
    with c2:
        st.metric("Total Weighted ECL Reserve", f"${ecl_result.total_ecl_weighted:,.0f}", delta=f"{ecl_result.ecl_coverage_ratio_pct:.2f}% Coverage")
    with c3:
        st.metric("Base Scenario ECL", f"${ecl_result.total_ecl_base:,.0f}")
    with c4:
        st.metric("Downturn Stress ECL", f"${ecl_result.total_ecl_downturn:,.0f}", delta=f"+${ecl_result.total_ecl_downturn - ecl_result.total_ecl_base:,.0f} Impact", delta_color="inverse")

    st.markdown("---")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        fig_waterfall = plot_ecl_waterfall(ecl_result)
        st.plotly_chart(fig_waterfall, use_container_width=True)
    with col_w2:
        fig_scenarios = plot_scenario_comparison(ecl_result)
        st.plotly_chart(fig_scenarios, use_container_width=True)

    # Stage Migration Breakdown Table
    st.subheader("IFRS 9 Impairment Stage Distribution")
    st.dataframe(ecl_result.summary_table(), use_container_width=True)

    # Database Persistence
    if st.button("💾 Persist Valuation Run to Database", type="secondary"):
        db = get_db_manager()
        run_id = f"RUN_{uuid.uuid4().hex[:8].upper()}"
        b = ecl_result.stage_breakdown
        st1_ecl = b.get("Stage 1 (Performing)", {}).get("total_ecl_weighted", 0.0)
        st2_ecl = b.get("Stage 2 (SICR)", {}).get("total_ecl_weighted", 0.0)
        st3_ecl = b.get("Stage 3 (Impaired)", {}).get("total_ecl_weighted", 0.0)

        db.save_ecl_run(
            run_id=run_id,
            total_accounts=ecl_result.total_accounts,
            total_ead=ecl_result.total_ead,
            total_ecl_weighted=ecl_result.total_ecl_weighted,
            total_ecl_base=ecl_result.total_ecl_base,
            total_ecl_downturn=ecl_result.total_ecl_downturn,
            total_ecl_upturn=ecl_result.total_ecl_upturn,
            coverage_ratio_pct=ecl_result.ecl_coverage_ratio_pct,
            stage_1_ecl=st1_ecl,
            stage_2_ecl=st2_ecl,
            stage_3_ecl=st3_ecl,
            parameters={"lgd": custom_lgd, "ccf": custom_ccf, "sicr_mult": sicr_mult},
        )
        st.success(f"Successfully recorded ECL run '{run_id}' into database!")

with tab_corp:
    st.subheader("🏢 Wholesale Corporate Expected Credit Loss (ECL = PD × LGD × EAD)")
    st.caption("Hypothetical corporate loan facilities, multi-scenario stress weighting, and 2D PD/LGD sensitivity.")

    # Critical Methodology and Regulatory Distinction Banner
    st.info(
        "**⚖️ Critical Methodological & Regulatory Distinctions:**\n\n"
        "1. **Consumer Data vs. Wholesale Corporate PD:** The benchmark dataset in this platform is consumer retail credit-card data (Taiwan). "
        "Its predicted default probabilities reflect consumer revolving credit behavior and must **NOT** be claimed as validated corporate PDs. "
        "Corporate wholesale credit evaluation requires balance-sheet metrics (leverage, interest coverage, cash-flow coverage), audited financials, "
        "and agency rating agency alignment. To demonstrate the ECL engine independently and soundly, this tab evaluates **clearly labelled hypothetical corporate credit facilities**.\n\n"
        "2. **Simple One-Period Formula vs. Institutional IFRS 9 / US CECL:** The simple `ECL = PD × LGD × EAD` formula is a core credit risk calculation, "
        "but does **NOT** constitute a complete IFRS 9 accounting standard implementation. Limitations include: **(a) Discounting:** Omits Effective Interest Rate (EIR) discounting over multi-year cash flow shortfalls; "
        "**(b) Horizons & Staging:** Assumes a fixed 1-year horizon without dynamic SICR (Significant Increase in Credit Risk) staging to lifetime ECL; "
        "**(c) Macro Scenarios:** Linear multipliers approximate complex econometric satellite transition models; "
        "**(d) Exposure Dynamics:** Static EAD omits contractual amortization, prepayment curves, and dynamic CCF facility drawdown volatility under distress."
    )

    # Section 1: Interactive Single-Facility Calculator
    st.markdown("### 1. Interactive Single-Facility ECL Calculator")
    st.markdown("Calculate Expected Credit Loss for an individual facility with instant parameter validation.")

    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        calc_pd = st.number_input(
            "Probability of Default (PD)",
            min_value=0.0,
            max_value=1.0,
            value=0.035,
            step=0.005,
            format="%.4f",
            help="12-Month default probability [0.0 to 1.0]. E.g., 0.035 = 3.50%.",
        )
    with c_in2:
        calc_lgd = st.number_input(
            "Loss Given Default (LGD)",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            format="%.2f",
            help="Economic loss fraction upon default [0.0 to 1.0]. Senior secured typically ~35-45%.",
        )
    with c_in3:
        calc_ead = st.number_input(
            "Exposure at Default (EAD in $ USD)",
            min_value=0.0,
            max_value=500_000_000.0,
            value=12_500_000.0,
            step=500_000.0,
            format="%.0f",
            help="Gross exposure expected at default (drawn + CCF * undrawn) in $ USD.",
        )

    try:
        single_ecl = calculate_expected_loss(calc_pd, calc_lgd, calc_ead)
        sens_single = perform_sensitivity_analysis(calc_pd, calc_lgd, calc_ead)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Credit Loss (ECL)", f"${single_ecl:,.2f}", f"{(single_ecl / calc_ead * 100):.2f}% of EAD" if calc_ead > 0 else "0.00%")
        m2.metric("Gross Exposure (EAD)", f"${calc_ead:,.0f}")
        m3.metric("+100 bps PD Shock Impact", f"+${sens_single.shock_100bps_pd_impact:,.2f}", "Delta per 1% PD increase")
        m4.metric("+100 bps LGD Shock Impact", f"+${sens_single.shock_100bps_lgd_impact:,.2f}", "Delta per 1% LGD increase")
    except Exception as e:
        st.error(f"Input validation error: {e}")

    st.divider()

    # Section 2: Portfolio Aggregation & Multi-Scenario Stress Testing
    st.markdown("### 2. Hypothetical Corporate Loan Portfolio & Macroeconomic Scenarios")
    st.markdown("Aggregate Expected Credit Loss across 10 diverse corporate facilities spanning investment grade to high yield.")

    corp_portfolio = get_hypothetical_corporate_portfolio()

    with st.expander("⚙️ Customize Macroeconomic Scenario Parameters", expanded=False):
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        with sc_col1:
            st.markdown("**Base Scenario (Consensus)**")
            base_w = st.slider("Base Weight", 0.10, 0.80, 0.50, 0.05, key="sc_base_w")
            base_pd_m = st.number_input("Base PD Multiplier", 0.5, 2.0, 1.0, 0.05, key="sc_base_pdm")
            base_lgd_m = st.number_input("Base LGD Multiplier", 0.5, 2.0, 1.0, 0.05, key="sc_base_lgdm")
        with sc_col2:
            st.markdown("**Favourable Scenario (Upturn)**")
            fav_w = st.slider("Favourable Weight", 0.05, 0.50, 0.25, 0.05, key="sc_fav_w")
            fav_pd_m = st.number_input("Fav PD Multiplier", 0.1, 1.0, 0.75, 0.05, key="sc_fav_pdm")
            fav_lgd_m = st.number_input("Fav LGD Multiplier", 0.1, 1.0, 0.85, 0.05, key="sc_fav_lgdm")
        with sc_col3:
            st.markdown("**Adverse Scenario (Downturn)**")
            adv_w = max(0.0, round(1.0 - base_w - fav_w, 2))
            st.markdown(f"**Adverse Weight:** `{adv_w:.2f}` *(auto-balanced to 1.00)*")
            adv_pd_m = st.number_input("Adv PD Multiplier", 1.0, 3.0, 1.45, 0.05, key="sc_adv_pdm")
            adv_lgd_m = st.number_input("Adv LGD Multiplier", 1.0, 2.0, 1.20, 0.05, key="sc_adv_lgdm")

    custom_scenarios = {
        "Base": Scenario("Base", base_w, base_pd_m, base_lgd_m, "Baseline economic consensus"),
        "Favourable": Scenario("Favourable", fav_w, fav_pd_m, fav_lgd_m, "Expansionary macroeconomic conditions"),
        "Adverse": Scenario("Adverse", adv_w, adv_pd_m, adv_lgd_m, "Recessionary downturn stress"),
    }

    corp_result = calculate_portfolio_ecl(corp_portfolio, scenarios=custom_scenarios)

    # Portfolio KPI Summary
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    pk1.metric("Total Corporate EAD", f"${corp_result.total_ead:,.0f}")
    pk2.metric("Weighted Expected Loss", f"${corp_result.total_ecl_weighted:,.0f}", f"{corp_result.portfolio_loss_rate_pct:.3f}% of EAD")
    pk3.metric("Base Consensus ECL", f"${corp_result.total_ecl_base:,.0f}")
    pk4.metric("Adverse Downturn ECL", f"${corp_result.total_ecl_adverse:,.0f}", f"+${(corp_result.total_ecl_adverse - corp_result.total_ecl_base):,.0f} stress")
    pk5.metric("Portfolio Weighted PD", f"{corp_result.portfolio_weighted_pd:.2%}", f"Avg LGD: {corp_result.portfolio_weighted_lgd:.1%}")

    # Visualizations Row 1: Scenario Comparison & Industry Concentration
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.plotly_chart(plot_scenario_losses(corp_result), use_container_width=True)
    with row1_c2:
        st.plotly_chart(plot_portfolio_ecl_by_industry(corp_result), use_container_width=True)

    # Visualizations Row 2: Sensitivity Heatmap & Borrower Waterfall
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.plotly_chart(plot_ecl_sensitivity_heatmap(sens_single), use_container_width=True)
    with row2_c2:
        st.plotly_chart(plot_borrower_ecl_waterfall(corp_result), use_container_width=True)

    # Facility-Level Table
    st.markdown("#### Facility-Level Corporate Credit Risk Ledger")
    st.dataframe(corp_result.to_dataframe(), use_container_width=True)

    # Industry and Rating Grade Aggregations
    col_ind, col_rat = st.columns(2)
    with col_ind:
        st.markdown("#### Expected Loss by Industry Sector")
        st.dataframe(corp_result.summary_by_industry(), use_container_width=True)
    with col_rat:
        st.markdown("#### Expected Loss by Internal Rating Grade")
        st.dataframe(corp_result.summary_by_rating(), use_container_width=True)

    st.divider()

with tab_rating_scale:
    st.subheader("Master Rating Scale Segmentation (AAA to CCC)")
    scale = MasterRatingScale()
    rating_summary = scale.summarize_portfolio(df_scored, pd_col="calibrated_pd", target_col="default_payment_next_month")
    st.dataframe(rating_summary, use_container_width=True)

    fig_rating = plot_rating_distribution(rating_summary)
    st.plotly_chart(fig_rating, use_container_width=True)

with tab_basel_test:
    st.subheader("Basel Committee (BCBS) Binomial Traffic Light Test")
    st.caption("Statistical hypothesis test of predicted PD vs realized default rate across rating grades.")

    backtest_results = backtest_pd_by_grade(df_scored, grade_col="rating_grade", pd_col="calibrated_pd", target_col="default_payment_next_month")

    def highlight_traffic_light(val):
        colors = {
            "GREEN": "background-color: #DCFCE7; color: #166534; font-weight: bold",
            "AMBER": "background-color: #FEF3C7; color: #92400E; font-weight: bold",
            "RED": "background-color: #FEE2E2; color: #991B1B; font-weight: bold",
        }
        return colors.get(val, "")

    styled_backtest = backtest_results.style.map(highlight_traffic_light, subset=["Traffic_Light"])
    st.dataframe(styled_backtest, use_container_width=True)
