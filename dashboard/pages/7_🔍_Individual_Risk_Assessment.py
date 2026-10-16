"""
CreditRisk Analytics -- Page 7: Individual Borrower Risk Assessment
==================================================================

Interactive single-borrower scoring tool. The user enters credit card account
characteristics via sidebar widgets; the page runs the pre-trained champion PD
model end-to-end and returns:

  * Calibrated Probability of Default (PD)
  * Credit bureau-style score (300-850)
  * Master Rating Scale grade (AAA -> CCC/Default)
  * IFRS 9 / CECL stage classification (Stage 1, 2, or 3)
  * Individual Expected Credit Loss (ECL) under Base / Downturn / Upturn scenarios

IMPORTANT DISCLAIMERS
---------------------
1. This is a *prototype* demonstration tool. Output probabilities are derived
   from a consumer credit-card delinquency model trained on the UCI Default of
   Credit Card Clients dataset and are NOT validated underwriting decisions.
2. Predicted PD values must NOT be used as lending decisions, regulatory capital
   inputs, or any purpose beyond academic/demonstration use.
3. IFRS 9 stage thresholds and ECL parameters are illustrative and do not
   constitute an audited impairment model.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ECLConfig,
    MODELS_DIR,
    RATING_SCALE,
    UCI_DATASET_ATTRIBUTION,
    ecl_config,
    scorecard_config,
)
from src.risk.segmentation import CreditScorecard, MasterRatingScale

st.set_page_config(
    page_title="Individual Risk Assessment | CreditRisk",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Page header & disclaimer
# ---------------------------------------------------------------------------
st.title("🔍 Individual Borrower Risk Assessment")
st.caption(
    "Enter account characteristics to obtain a calibrated Probability of Default, "
    "bureau credit score, and IFRS 9 staging. For demonstration purposes only."
)

st.warning(
    f"**⚖️ GOVERNANCE NOTICE:**  {UCI_DATASET_ATTRIBUTION.disclaimer}\n\n"
    "All outputs are **prototype estimates** derived from a consumer credit-card "
    "model. They are NOT lending decisions, regulatory capital inputs, or validated "
    "creditworthiness assessments."
)

# ---------------------------------------------------------------------------
# Load model artifact
# ---------------------------------------------------------------------------
artifact_path = MODELS_DIR / "pd_model_best.joblib"
if not artifact_path.exists():
    st.error(
        f"**Champion Model Artifact Not Found:** `{artifact_path}`\n\n"
        "Run `python run_pipeline.py` to train and serialise the PD model before "
        "using this page."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(artifact_path)


model_art = load_model()

scorecard = CreditScorecard(config=scorecard_config)
rating_scale = MasterRatingScale()

# ---------------------------------------------------------------------------
# Sidebar -- Borrower Input Form
# ---------------------------------------------------------------------------
st.sidebar.header("🏦 Borrower Account Inputs")
st.sidebar.caption("Adjust parameters and press **Score Borrower** to assess risk.")

with st.sidebar:
    st.markdown("### 💳 Credit Facility")
    limit_bal = st.number_input(
        "Credit Limit (NT$)",
        min_value=10_000,
        max_value=1_000_000,
        value=150_000,
        step=10_000,
        help="Total approved revolving credit limit in New Taiwan Dollars.",
    )
    st.markdown("### 👤 Demographic Profile")
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox(
        "Education Level",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x],
    )
    marriage = st.selectbox(
        "Marital Status",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x],
    )
    age = st.slider("Age", min_value=21, max_value=79, value=35, step=1)

    st.markdown("### 📅 Repayment Status (Most Recent → 6 Months Ago)")
    st.caption(
        "−2 = No consumption; −1 = Duly paid; 0 = Revolving credit used; "
        "1–8 = Months of payment delay"
    )
    pay_status_opts = list(range(-2, 9))

    def fmt_pay(v: int) -> str:
        if v == -2:
            return "−2  No consumption"
        if v == -1:
            return "−1  Duly paid"
        if v == 0:
            return "  0  Revolving credit used"
        return f"+{v}  Month(s) delay"

    pay_0 = st.selectbox("Repayment Status (Month −1)", options=pay_status_opts, index=2, format_func=fmt_pay)
    pay_2 = st.selectbox("Repayment Status (Month −2)", options=pay_status_opts, index=2, format_func=fmt_pay)
    pay_3 = st.selectbox("Repayment Status (Month −3)", options=pay_status_opts, index=2, format_func=fmt_pay)
    pay_4 = st.selectbox("Repayment Status (Month −4)", options=pay_status_opts, index=2, format_func=fmt_pay)
    pay_5 = st.selectbox("Repayment Status (Month −5)", options=pay_status_opts, index=2, format_func=fmt_pay)
    pay_6 = st.selectbox("Repayment Status (Month −6)", options=pay_status_opts, index=2, format_func=fmt_pay)

    st.markdown("### 💰 Bill Statement Amounts (NT$)")
    bill_amt1 = st.number_input("Bill Amount (Month −1)", min_value=-100_000, max_value=1_000_000, value=35_000, step=1_000)
    bill_amt2 = st.number_input("Bill Amount (Month −2)", min_value=-100_000, max_value=1_000_000, value=32_000, step=1_000)
    bill_amt3 = st.number_input("Bill Amount (Month −3)", min_value=-100_000, max_value=1_000_000, value=30_000, step=1_000)
    bill_amt4 = st.number_input("Bill Amount (Month −4)", min_value=-100_000, max_value=1_000_000, value=28_000, step=1_000)
    bill_amt5 = st.number_input("Bill Amount (Month −5)", min_value=-100_000, max_value=1_000_000, value=25_000, step=1_000)
    bill_amt6 = st.number_input("Bill Amount (Month −6)", min_value=-100_000, max_value=1_000_000, value=22_000, step=1_000)

    st.markdown("### 💳 Payment Amounts Made (NT$)")
    pay_amt1 = st.number_input("Payment Made (Month −1)", min_value=0, max_value=1_000_000, value=3_000, step=500)
    pay_amt2 = st.number_input("Payment Made (Month −2)", min_value=0, max_value=1_000_000, value=2_500, step=500)
    pay_amt3 = st.number_input("Payment Made (Month −3)", min_value=0, max_value=1_000_000, value=2_000, step=500)
    pay_amt4 = st.number_input("Payment Made (Month −4)", min_value=0, max_value=1_000_000, value=2_000, step=500)
    pay_amt5 = st.number_input("Payment Made (Month −5)", min_value=0, max_value=1_000_000, value=1_500, step=500)
    pay_amt6 = st.number_input("Payment Made (Month −6)", min_value=0, max_value=1_000_000, value=1_500, step=500)

    st.markdown("### 🏦 ECL Inputs (Illustrative)")
    lgd_pct = st.slider(
        "Loss Given Default — LGD (%)",
        min_value=0.0, max_value=100.0, value=45.0, step=1.0,
        help="Estimated % of exposure lost if the borrower defaults. 45% is a regulatory Basel floor for unsecured retail.",
    )
    ead = st.number_input(
        "Exposure at Default — EAD (NT$)",
        min_value=0,
        max_value=1_000_000,
        value=120_000,
        step=5_000,
        help="Outstanding balance expected at point of default. Defaults to 80% of credit limit as a conservative estimate.",
    )

    score_btn = st.button("🔎 Score Borrower", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Assemble input row
# ---------------------------------------------------------------------------
input_record = pd.DataFrame([{
    "limit_bal": float(limit_bal),
    "sex": int(sex),
    "education": int(education),
    "marriage": int(marriage),
    "age": int(age),
    "pay_0": int(pay_0),
    "pay_2": int(pay_2),
    "pay_3": int(pay_3),
    "pay_4": int(pay_4),
    "pay_5": int(pay_5),
    "pay_6": int(pay_6),
    "bill_amt1": float(bill_amt1),
    "bill_amt2": float(bill_amt2),
    "bill_amt3": float(bill_amt3),
    "bill_amt4": float(bill_amt4),
    "bill_amt5": float(bill_amt5),
    "bill_amt6": float(bill_amt6),
    "pay_amt1": float(pay_amt1),
    "pay_amt2": float(pay_amt2),
    "pay_amt3": float(pay_amt3),
    "pay_amt4": float(pay_amt4),
    "pay_amt5": float(pay_amt5),
    "pay_amt6": float(pay_amt6),
}])


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
def score_borrower(row: pd.DataFrame, model) -> dict:
    """Run end-to-end inference and return a structured result dict."""
    proba_arr = model.predict_proba(row, calibrate=True)
    pd_val = float(proba_arr[0])
    credit_score_val = scorecard.pd_to_score(pd_val)

    # Determine rating grade
    grade = "CCC"
    grade_desc = "Substandard / Impaired / Default Warning"
    for entry in RATING_SCALE:
        if entry["pd_min"] <= pd_val < entry["pd_max"]:
            grade = entry["grade"]
            grade_desc = entry["description"]
            break

    # IFRS 9 Stage classification (illustrative thresholds)
    sicr_pd_threshold = 0.10  # Absolute PD > 10% as illustrative SICR
    max_delay = max(pay_0, pay_2, pay_3, pay_4, pay_5, pay_6)

    if max_delay >= 3 or pd_val >= 0.30:
        stage = 3
        stage_label = "Stage 3 — Credit-Impaired (90+ DPD / PD ≥ 30%)"
        ecl_horizon = "Lifetime"
        ecl_horizon_years = ecl_config.lifetime_horizon_years
    elif pd_val >= sicr_pd_threshold or max_delay >= 1:
        stage = 2
        stage_label = "Stage 2 — Significant Increase in Credit Risk (SICR)"
        ecl_horizon = "Lifetime"
        ecl_horizon_years = ecl_config.lifetime_horizon_years
    else:
        stage = 1
        stage_label = "Stage 1 — Performing (12-Month ECL)"
        ecl_horizon = "12-Month"
        ecl_horizon_years = 1

    # ECL under three scenarios (illustrative)
    lgd = lgd_pct / 100.0
    ead_val = float(ead)

    results_by_scenario = {}
    for scen_name, scen in ecl_config.scenarios.items():
        pd_adj = min(pd_val * scen["pd_multiplier"], 1.0)
        lgd_adj = min(lgd * scen["lgd_multiplier"], 1.0)
        if ecl_horizon_years == 1:
            ecl_amount = pd_adj * lgd_adj * ead_val
        else:
            # Simplified multi-period: sum of annual ECL contributions
            ecl_amount = pd_adj * lgd_adj * ead_val * ecl_horizon_years
        results_by_scenario[scen_name] = {
            "PD_adj": pd_adj,
            "LGD_adj": lgd_adj,
            "ECL": ecl_amount,
        }

    weighted_ecl = sum(
        ecl_config.scenarios[s]["weight"] * results_by_scenario[s]["ECL"]
        for s in ecl_config.scenarios
    )

    return {
        "pd": pd_val,
        "credit_score": float(credit_score_val),
        "grade": grade,
        "grade_desc": grade_desc,
        "stage": stage,
        "stage_label": stage_label,
        "ecl_horizon": ecl_horizon,
        "scenarios": results_by_scenario,
        "weighted_ecl": weighted_ecl,
        "ead": ead_val,
        "lgd": lgd,
    }


# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
if score_btn:
    with st.spinner("Scoring borrower through the champion PD model pipeline..."):
        try:
            result = score_borrower(input_record, model_art)
        except Exception as exc:
            st.error(f"**Scoring Failed:** {exc}")
            st.stop()

    pd_val = result["pd"]
    credit_score_out = result["credit_score"]
    grade = result["grade"]
    stage = result["stage"]
    weighted_ecl = result["weighted_ecl"]

    # Grade / Stage colour mapping
    grade_colors = {
        "AAA": ("#10B981", "#DCFCE7"),
        "AA":  ("#059669", "#D1FAE5"),
        "A":   ("#0284C7", "#E0F2FE"),
        "BBB": ("#0369A1", "#DBEAFE"),
        "BB":  ("#D97706", "#FEF3C7"),
        "B":   ("#EA580C", "#FFEDD5"),
        "CCC": ("#DC2626", "#FEE2E2"),
    }
    stage_colors = {
        1: ("#10B981", "#DCFCE7"),
        2: ("#F59E0B", "#FEF3C7"),
        3: ("#EF4444", "#FEE2E2"),
    }
    g_text, g_bg = grade_colors.get(grade, ("#64748B", "#F1F5F9"))
    s_text, s_bg = stage_colors[stage]

    st.divider()
    st.subheader("📊 Borrower Risk Assessment Results")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Calibrated PD", f"{pd_val*100:.2f}%",
                  help="Calibrated Probability of Default from the isotonic-calibrated champion model.")
    with k2:
        delta_dir = "Above" if credit_score_out >= 600 else "Below"
        st.metric("Bureau Credit Score", f"{credit_score_out:.0f} / 850",
                  delta=f"{'▲' if credit_score_out >= 600 else '▼'} {delta_dir} Base 600")
    with k3:
        short_desc = result["grade_desc"][:30] + ("…" if len(result["grade_desc"]) > 30 else "")
        st.metric("Rating Grade", grade, delta=short_desc)
    with k4:
        st.metric("IFRS 9 Stage", f"Stage {stage}", delta=result["ecl_horizon"] + " ECL")
    with k5:
        st.metric("Weighted ECL (Illustrative)", f"NT$ {weighted_ecl:,.0f}",
                  help="Probability-weighted ECL across Base (50%), Downturn (30%), and Upturn (20%) scenarios.")

    col_grade, col_stage = st.columns(2)
    with col_grade:
        st.markdown(
            f"""<div style="background:{g_bg};border:1px solid {g_text};border-radius:8px;
            padding:14px 20px;text-align:center;margin-top:8px;">
            <span style="font-size:2rem;font-weight:700;color:{g_text};">{grade}</span><br/>
            <span style="color:{g_text};font-size:0.9rem;">{result['grade_desc']}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_stage:
        st.markdown(
            f"""<div style="background:{s_bg};border:1px solid {s_text};border-radius:8px;
            padding:14px 20px;text-align:center;margin-top:8px;">
            <span style="font-size:2rem;font-weight:700;color:{s_text};">Stage {stage}</span><br/>
            <span style="color:{s_text};font-size:0.9rem;">{result['stage_label']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("📈 ECL Scenario Analysis (Illustrative)")

    col_scenario, col_gauge = st.columns([1.3, 0.7])

    with col_scenario:
        scen_rows = []
        for scen_name, scen_data in result["scenarios"].items():
            weight = ecl_config.scenarios[scen_name]["weight"]
            scen_rows.append({
                "Scenario": scen_name,
                "Weight": f"{weight*100:.0f}%",
                "Adj. PD": f"{scen_data['PD_adj']*100:.2f}%",
                "Adj. LGD": f"{scen_data['LGD_adj']*100:.1f}%",
                "EAD (NT$)": f"{result['ead']:,.0f}",
                "ECL (NT$)": f"{scen_data['ECL']:,.0f}",
                "Wtd. Contribution (NT$)": f"{weight * scen_data['ECL']:,.0f}",
            })
        scen_df = pd.DataFrame(scen_rows)

        def color_scenario(row):
            palette = {"Base": "#DBEAFE", "Downturn": "#FEE2E2", "Upturn": "#DCFCE7"}
            c = palette.get(row["Scenario"], "")
            return [f"background-color: {c}"] * len(row)

        st.dataframe(scen_df.style.apply(color_scenario, axis=1), use_container_width=True, hide_index=True)
        st.info(
            f"**Weighted ECL = NT$ {weighted_ecl:,.0f}** "
            f"({result['ecl_horizon']} ECL horizon, illustrative IFRS 9 Stage {stage})\n\n"
            "ECL parameters (LGD, horizon) are illustrative and NOT an audited impairment estimate."
        )

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickformat": ".0f", "nticks": 6},
                "bar": {"color": g_text},
                "steps": [
                    {"range": [0.0,  3.5],  "color": "#D1FAE5"},
                    {"range": [3.5,  7.5],  "color": "#DBEAFE"},
                    {"range": [7.5,  15.0], "color": "#FEF3C7"},
                    {"range": [15.0, 30.0], "color": "#FFEDD5"},
                    {"range": [30.0, 100.0],"color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#DC2626", "width": 3},
                    "thickness": 0.85,
                    "value": 30,
                },
            },
            title={"text": f"PD Gauge — {grade} Grade", "font": {"size": 14}},
        ))
        fig_gauge.update_layout(
            height=280,
            margin={"l": 20, "r": 20, "t": 50, "b": 10},
            template="plotly_white",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    with st.expander("📋 Review Submitted Account Characteristics", expanded=False):
        col_demo, col_pay, col_bill = st.columns(3)
        with col_demo:
            st.markdown("**Demographic & Facility**")
            st.write(f"- Credit Limit: NT${limit_bal:,}")
            st.write(f"- Sex: {'Male' if sex == 1 else 'Female'}")
            edu_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
            mar_map = {1: "Married", 2: "Single", 3: "Others"}
            st.write(f"- Education: {edu_map[education]}")
            st.write(f"- Marital Status: {mar_map[marriage]}")
            st.write(f"- Age: {age}")
        with col_pay:
            st.markdown("**Repayment Status (6 months)**")
            for lbl, val in zip(
                ["Month −1", "Month −2", "Month −3", "Month −4", "Month −5", "Month −6"],
                [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6],
            ):
                st.write(f"- {lbl}: {fmt_pay(val)}")
        with col_bill:
            st.markdown("**Bill & Payment Amounts (NT$)**")
            for idx, (b, p) in enumerate(zip(
                [bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6],
                [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6],
            )):
                st.write(f"- Month −{idx+1}: Bill {b:,} | Paid {p:,}")

else:
    st.info(
        "👈 **Configure borrower account characteristics in the sidebar and press "
        "'Score Borrower' to obtain a risk assessment.**\n\n"
        "The scoring engine will run the pre-trained champion PD model end-to-end "
        "and display a calibrated PD, bureau credit score, rating grade, IFRS 9 stage, "
        "and illustrative ECL estimate."
    )

    st.subheader("📖 Master Rating Scale Reference")
    ref_rows = []
    for entry in RATING_SCALE:
        ref_rows.append({
            "Grade": entry["grade"],
            "PD Lower (%)": f"{entry['pd_min']*100:.2f}%",
            "PD Upper (%)": f"{entry['pd_max']*100:.2f}%",
            "Description": entry["description"],
        })
    ref_df = pd.DataFrame(ref_rows)
    grade_bgs = {
        "AAA": "#DCFCE7", "AA": "#D1FAE5", "A": "#DBEAFE", "BBB": "#EDE9FE",
        "BB": "#FEF3C7", "B": "#FFEDD5", "CCC": "#FEE2E2",
    }

    def hl_grade(row):
        bg = grade_bgs.get(row["Grade"], "#F8FAFC")
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(ref_df.style.apply(hl_grade, axis=1), use_container_width=True, hide_index=True)
    st.caption(
        "⚠️ Rating grade boundaries are illustrative thresholds derived from conventional "
        "credit agency PD ranges. They are NOT regulatory-approved grade cutoffs."
    )
