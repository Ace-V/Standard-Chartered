"""
CreditRisk Analytics — Visual Analytics & Institutional Plotly Charts
====================================================================

Generates interactive, publication-quality Plotly visualizations for credit risk:
- ROC Curve with Gini and AUC annotations
- Kolmogorov-Smirnov (KS) Cumulative Distribution separation plot
- Calibration reliability curves with Expected Calibration Error (ECE)
- Credit bureau score distributions by default outcome
- Master Rating Scale distribution and PD term structure
- IFRS 9 Staging impairment waterfall
- Macroeconomic scenario stress comparison
- Population Stability Index (PSI) decile distribution shifts
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from src.models.calibration import compute_calibration_curve
from src.monitoring.drift import PSIResult
from src.risk.expected_loss import ECLPortfolioResult

# Institutional Color Palette
COLORS = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0284C7",    # Slate Blue
    "accent": "#0D9488",       # Emerald Teal
    "danger": "#E11D48",       # Rose / Red
    "warning": "#F59E0B",      # Amber
    "neutral": "#64748B",      # Slate Grey
    "light": "#F8FAFC",        # Off white
    "dark": "#0F172A",         # Navy Charcoal
}


def _apply_theme(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Applies a clean, modern banking terminal theme to Plotly figures."""
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0.05, "xanchor": "left", "font": {"size": 16, "family": "Inter, Arial, sans-serif"}},
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        hovermode="x unified",
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E2E8F0", zeroline=False)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float, gini: float) -> go.Figure:
    """Plot Receiver Operating Characteristic curve with regulatory Gini metric."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Model (AUC = {roc_auc:.3f} | Gini = {gini:.3f})",
        line={"color": COLORS["primary"], "width": 3},
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
    ))

    return _apply_theme(fig, "Receiver Operating Characteristic (ROC)", "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity)")


def plot_ks_curve(y_true: np.ndarray, y_prob: np.ndarray, ks_stat: float, ks_threshold: float) -> go.Figure:
    """Plot Kolmogorov-Smirnov cumulative distributions for defaults and non-defaults."""
    desc_idx = np.argsort(y_prob)[::-1]
    y_true_s = y_true[desc_idx]
    y_prob_s = y_prob[desc_idx]

    n_def = np.sum(y_true_s == 1)
    n_non = np.sum(y_true_s == 0)

    cum_def = np.cumsum(y_true_s == 1) / max(n_def, 1)
    cum_non = np.cumsum(y_true_s == 0) / max(n_non, 1)

    # Subsample for smooth rendering
    step = max(1, len(y_prob_s) // 500)
    idx_sub = np.arange(0, len(y_prob_s), step)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_prob_s[idx_sub], y=cum_def[idx_sub],
        mode="lines", name="Cumulative Defaults",
        line={"color": COLORS["danger"], "width": 2.5},
    ))
    fig.add_trace(go.Scatter(
        x=y_prob_s[idx_sub], y=cum_non[idx_sub],
        mode="lines", name="Cumulative Non-Defaults",
        line={"color": COLORS["accent"], "width": 2.5},
    ))

    # Add vertical line at maximum separation
    fig.add_vline(
        x=ks_threshold,
        line_width=2,
        line_dash="dot",
        line_color=COLORS["dark"],
        annotation_text=f"Max KS = {ks_stat:.3f} (Cutoff: {ks_threshold:.3f})",
        annotation_position="top right",
    )

    return _apply_theme(fig, "Kolmogorov-Smirnov (KS) Separation Analysis", "Probability Threshold (Descending Risk)", "Cumulative Proportion")


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> go.Figure:
    """Plot calibration curve comparing predicted default probability to empirical rate."""
    prob_true, prob_pred, counts, ece, mce = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
    ))
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode="lines+markers",
        name=f"Model Calibration (ECE = {ece:.3f} | MCE = {mce:.3f})",
        line={"color": COLORS["primary"], "width": 3},
        marker={"size": 8, "color": COLORS["primary"]},
    ))

    return _apply_theme(fig, "Probability Reliability & Calibration Curve", "Mean Predicted Probability (PD)", "Observed Default Frequency")


def plot_score_distribution(df: pd.DataFrame, score_col: str = "credit_score", target_col: str = "default_payment_next_month") -> go.Figure:
    """Overlaid distribution of credit scores separated by true default status."""
    non_defaults = df[df[target_col] == 0][score_col].dropna()
    defaults = df[df[target_col] == 1][score_col].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=non_defaults,
        name="Non-Defaults (Performing)",
        opacity=0.6,
        marker_color=COLORS["accent"],
        nbinsx=35,
    ))
    fig.add_trace(go.Histogram(
        x=defaults,
        name="Defaults (Non-Performing)",
        opacity=0.7,
        marker_color=COLORS["danger"],
        nbinsx=35,
    ))
    fig.update_layout(barmode="overlay")
    return _apply_theme(fig, "Credit Score Distribution by Default Status", "Credit Bureau Score (300 - 850)", "Account Count")


def plot_rating_distribution(summary_df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart displaying account volume and average PD across Master Rating Grades."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=summary_df["Grade"],
        y=summary_df["Account_Count"],
        name="Account Count",
        marker_color=COLORS["primary"],
        yaxis="y",
    ))

    avg_pd = summary_df.get("Average_Predicted_PD", summary_df.get("average_predicted_pd", pd.Series(dtype=float)))
    if not avg_pd.empty:
        fig.add_trace(go.Scatter(
            x=summary_df["Grade"],
            y=avg_pd * 100,
            name="Average Predicted PD (%)",
            mode="lines+markers",
            line={"color": COLORS["danger"], "width": 3},
            marker={"size": 9},
            yaxis="y2",
        ))

    fig.update_layout(
        yaxis={"title": "Number of Accounts", "showgrid": True},
        yaxis2={"title": "Average PD (%)", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "right", "x": 1.0},
    )
    return _apply_theme(fig, "Master Rating Scale Distribution & PD Profile", "Rating Grade (AAA to CCC)", "Accounts")


def plot_ecl_waterfall(ecl_result: ECLPortfolioResult) -> go.Figure:
    """Breakdown of portfolio Expected Credit Loss by IFRS 9 Staging."""
    stages = ["Stage 1 (12m ECL)", "Stage 2 (Lifetime SICR)", "Stage 3 (Defaulted)", "Total Weighted ECL"]
    b = ecl_result.stage_breakdown
    st1 = b.get("Stage 1 (Performing)", {}).get("total_ecl_weighted", 0.0)
    st2 = b.get("Stage 2 (SICR)", {}).get("total_ecl_weighted", 0.0)
    st3 = b.get("Stage 3 (Impaired)", {}).get("total_ecl_weighted", 0.0)
    total = ecl_result.total_ecl_weighted

    fig = go.Figure(go.Waterfall(
        name="ECL Staging",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=stages,
        textposition="outside",
        text=[f"${st1:,.0f}", f"${st2:,.0f}", f"${st3:,.0f}", f"${total:,.0f}"],
        y=[st1, st2, st3, total],
        connector={"line": {"color": COLORS["neutral"]}},
        decreasing={"marker": {"color": COLORS["accent"]}},
        increasing={"marker": {"color": COLORS["danger"]}},
        totals={"marker": {"color": COLORS["primary"]}},
    ))

    return _apply_theme(fig, "IFRS 9 Expected Credit Loss (ECL) Impairment by Stage", "Impairment Stage", "ECL Reserve ($)")


def plot_scenario_comparison(ecl_result: ECLPortfolioResult) -> go.Figure:
    """Compares Expected Credit Loss reserve requirements across economic scenarios."""
    scenarios = ["Base (50% Wt)", "Downturn (30% Wt)", "Upturn (20% Wt)", "Probability-Weighted"]
    values = [
        ecl_result.total_ecl_base,
        ecl_result.total_ecl_downturn,
        ecl_result.total_ecl_upturn,
        ecl_result.total_ecl_weighted,
    ]
    colors = [COLORS["secondary"], COLORS["danger"], COLORS["accent"], COLORS["primary"]]

    fig = go.Figure(go.Bar(
        x=scenarios,
        y=values,
        text=[f"${v:,.0f}" for v in values],
        textposition="auto",
        marker_color=colors,
    ))

    return _apply_theme(fig, "Macroeconomic Stress Scenario ECL Reserves", "Macroeconomic Scenario", "Total ECL Provision ($)")


def plot_psi_bins(psi_result: PSIResult) -> go.Figure:
    """Grouped bar chart comparing baseline vs monitoring population proportions across deciles."""
    df_bins = psi_result.bin_details
    if df_bins.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_bins["bin"],
        y=df_bins["expected_pct"],
        name="Baseline / Development %",
        marker_color=COLORS["secondary"],
    ))
    fig.add_trace(go.Bar(
        x=df_bins["bin"],
        y=df_bins["actual_pct"],
        name="Monitoring / Out-of-Time %",
        marker_color=COLORS["warning"] if psi_result.traffic_light == "AMBER" else (
            COLORS["danger"] if psi_result.traffic_light == "RED" else COLORS["accent"]
        ),
    ))

    fig.update_layout(barmode="group")
    return _apply_theme(fig, f"Population Drift (PSI = {psi_result.psi_value:.4f} | {psi_result.traffic_light})", "Risk Score Decile Bins", "Population Share (%)")


# ---------------------------------------------------------------------------
# Part A: Calibration comparison charts
# ---------------------------------------------------------------------------

def plot_calibration_comparison(
    raw_prob_true: np.ndarray,
    raw_prob_pred: np.ndarray,
    cal_prob_true: np.ndarray,
    cal_prob_pred: np.ndarray,
    raw_ece: float,
    cal_ece: float,
) -> go.Figure:
    """
    Overlay reliability diagram comparing raw (uncalibrated) vs calibrated probabilities
    against the perfect-calibration 45° line.
    """
    fig = go.Figure()

    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
        hoverinfo="skip",
    ))

    # Raw uncalibrated model
    fig.add_trace(go.Scatter(
        x=raw_prob_pred, y=raw_prob_true,
        mode="lines+markers",
        name=f"Raw / Uncalibrated (ECE = {raw_ece:.4f})",
        line={"color": COLORS["warning"], "width": 2.5, "dash": "dot"},
        marker={"size": 8, "symbol": "circle-open", "color": COLORS["warning"]},
    ))

    # Calibrated model
    fig.add_trace(go.Scatter(
        x=cal_prob_pred, y=cal_prob_true,
        mode="lines+markers",
        name=f"Post-Calibration (ECE = {cal_ece:.4f})",
        line={"color": COLORS["primary"], "width": 3},
        marker={"size": 9, "color": COLORS["primary"]},
    ))

    return _apply_theme(
        fig,
        "Reliability Diagram — Raw vs Post-Calibration Comparison",
        "Mean Predicted Default Probability",
        "Observed Default Frequency (Empirical)",
    )


def plot_calibration_bin_errors(
    raw_bin_table: pd.DataFrame,
    cal_bin_table: pd.DataFrame,
) -> go.Figure:
    """
    Grouped bar chart showing absolute calibration error per probability bin
    for raw vs calibrated models. Bins with n<30 are annotated as uncertain.
    """
    fig = go.Figure()

    bin_labels = raw_bin_table["Bin"].tolist()

    fig.add_trace(go.Bar(
        x=bin_labels,
        y=raw_bin_table["Calibration Error"].tolist(),
        name="Raw / Uncalibrated",
        marker_color=COLORS["warning"],
        opacity=0.85,
    ))

    if not cal_bin_table.empty:
        # Align to same bins
        cal_merged = raw_bin_table[["Bin"]].merge(
            cal_bin_table[["Bin", "Calibration Error", "Statistically Uncertain (n<30)"]],
            on="Bin", how="left",
        )
        fig.add_trace(go.Bar(
            x=bin_labels,
            y=cal_merged["Calibration Error"].fillna(0).tolist(),
            name="Post-Calibration",
            marker_color=COLORS["primary"],
            opacity=0.85,
        ))
        # Annotate uncertain bins
        uncertain_mask = cal_merged["Statistically Uncertain (n<30)"].fillna(True)
        for i, (label, uncertain) in enumerate(zip(bin_labels, uncertain_mask)):
            if uncertain:
                fig.add_annotation(
                    x=label, y=cal_merged["Calibration Error"].fillna(0).iloc[i] + 0.005,
                    text="⚠ n<30", showarrow=False,
                    font={"size": 9, "color": COLORS["warning"]},
                )

    fig.update_layout(barmode="group")
    return _apply_theme(
        fig,
        "Absolute Calibration Error per Probability Bin",
        "Probability Bin",
        "Absolute Calibration Error |Predicted − Observed|",
    )


# ---------------------------------------------------------------------------
# Part B: Risk segmentation charts
# ---------------------------------------------------------------------------

def plot_decile_default_rates(decile_table: pd.DataFrame) -> go.Figure:
    """
    Bar chart of observed default rates across predicted-probability deciles,
    with 95% Wilson score CI error bars and monotonicity violation annotations.
    The chart tests whether observed default frequency increases across risk deciles.
    """
    fig = go.Figure()

    decile_labels = [f"D{int(d)}" for d in decile_table["Decile"]]
    obs_drs = decile_table["Observed Default Rate"].tolist()
    mean_pds = decile_table["Mean Predicted PD"].tolist()
    ci_lo = decile_table["95% CI Lower"].tolist()
    ci_hi = decile_table["95% CI Upper"].tolist()
    monotone_flags = decile_table["Monotone (↑ from prev)"].tolist()
    counts = decile_table["Count"].tolist()

    # Error bar half-widths
    err_minus = [max(0.0, obs - lo) for obs, lo in zip(obs_drs, ci_lo)]
    err_plus = [max(0.0, hi - obs) for obs, hi in zip(obs_drs, ci_hi)]

    # Bar colors: grey for violations, teal for monotone
    bar_colors = []
    for i, flag in enumerate(monotone_flags):
        if flag is False:
            bar_colors.append(COLORS["danger"])  # monotonicity violation
        else:
            bar_colors.append(COLORS["accent"])

    fig.add_trace(go.Bar(
        x=decile_labels,
        y=obs_drs,
        name="Observed Default Rate",
        marker_color=bar_colors,
        error_y={"type": "data", "symmetric": False, "array": err_plus, "arrayminus": err_minus,
                 "color": COLORS["neutral"], "thickness": 2, "width": 5},
        text=[f"{dr:.1%}<br>n={cnt:,}" for dr, cnt in zip(obs_drs, counts)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Observed DR: %{y:.2%}<br>95% CI: [%{customdata[0]:.2%}, %{customdata[1]:.2%}]<extra></extra>",
        customdata=list(zip(ci_lo, ci_hi)),
    ))

    # Overlay mean predicted PD as line
    fig.add_trace(go.Scatter(
        x=decile_labels,
        y=mean_pds,
        mode="lines+markers",
        name="Mean Predicted PD",
        line={"color": COLORS["primary"], "width": 2.5, "dash": "dot"},
        marker={"size": 7, "color": COLORS["primary"], "symbol": "diamond"},
    ))

    # Annotate monotonicity violations
    for i, flag in enumerate(monotone_flags):
        if flag is False:
            fig.add_annotation(
                x=decile_labels[i],
                y=obs_drs[i] + max(err_plus[i], 0.02) + 0.015,
                text="⚠ Non-monotone",
                showarrow=False,
                font={"size": 9, "color": COLORS["danger"]},
            )

    return _apply_theme(
        fig,
        "Risk Decile Analysis — Observed Default Rate vs Predicted PD (D1=Lowest Risk, D10=Highest)",
        "Predicted Probability Decile",
        "Observed Default Rate",
    )


def plot_segment_calibration(segment_table: pd.DataFrame) -> go.Figure:
    """
    Bubble chart displaying mean predicted PD vs observed default rate per risk segment.
    Bubble size encodes borrower count. Segments on or near the diagonal are well calibrated.
    Points below the diagonal indicate over-estimation of risk (conservative PDs).
    Points above the diagonal indicate under-estimation (optimistic PDs).
    """
    SEGMENT_COLORS = {
        "Low Risk": COLORS["accent"],
        "Medium-Low Risk": COLORS["secondary"],
        "Medium-High Risk": COLORS["warning"],
        "High Risk": COLORS["danger"],
        "Very High Risk": "#7C3AED",
    }

    fig = go.Figure()

    # Perfect calibration diagonal
    max_val = max(
        segment_table["Mean Predicted PD"].max(),
        segment_table["Observed Default Rate"].max(),
        0.1,
    ) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        name="Perfect Calibration",
        line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
        hoverinfo="skip",
    ))

    for _, row in segment_table.iterrows():
        seg = row.get("Segment (Illustrative)", "Unknown")
        count = int(row["Borrower Count"])
        if count == 0:
            continue
        bubble_size = max(10, min(70, count / 150))
        color = SEGMENT_COLORS.get(seg, COLORS["neutral"])
        uncertain = row.get("Statistically Uncertain (n<30)", False)
        marker_sym = "circle-open" if uncertain else "circle"

        fig.add_trace(go.Scatter(
            x=[row["Mean Predicted PD"]],
            y=[row["Observed Default Rate"]],
            mode="markers+text",
            name=seg,
            marker={
                "size": bubble_size,
                "color": color,
                "symbol": marker_sym,
                "line": {"width": 2, "color": color},
            },
            text=[seg],
            textposition="top center",
            hovertemplate=(
                f"<b>{seg}</b><br>"
                f"Mean Predicted PD: {{x:.2%}}<br>"
                f"Observed Default Rate: {{y:.2%}}<br>"
                f"Borrower Count: {count:,}<br>"
                f"Cal. Error: {row['Calibration Error']:.4f}<br>"
                f"{'⚠ Statistically Uncertain' if uncertain else ''}"
                "<extra></extra>"
            ),
        ))

    return _apply_theme(
        fig,
        "Segment Calibration — Mean Predicted PD vs Observed Default Rate<br>"
        "<sup>⚠ Illustrative segment boundaries — not bank policy. See disclosure.</sup>",
        "Mean Predicted Default Probability",
        "Observed Default Frequency",
    )


def plot_segment_bar(segment_table: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: borrower count and observed default rate per risk segment,
    with 95% Wilson CI bands on the observed default rate axis.
    """
    segs = segment_table[segment_table["Borrower Count"] > 0]["Segment (Illustrative)"].tolist()
    counts = segment_table[segment_table["Borrower Count"] > 0]["Borrower Count"].tolist()
    obs_drs = segment_table[segment_table["Borrower Count"] > 0]["Observed Default Rate"].tolist()
    ci_lo = segment_table[segment_table["Borrower Count"] > 0]["95% CI Lower"].tolist()
    ci_hi = segment_table[segment_table["Borrower Count"] > 0]["95% CI Upper"].tolist()
    mean_pds = segment_table[segment_table["Borrower Count"] > 0]["Mean Predicted PD"].tolist()

    err_minus = [max(0.0, obs - lo) for obs, lo in zip(obs_drs, ci_lo)]
    err_plus = [max(0.0, hi - obs) for obs, hi in zip(obs_drs, ci_hi)]

    fig = go.Figure()

    # Borrower counts (left axis)
    fig.add_trace(go.Bar(
        x=segs, y=counts,
        name="Borrower Count",
        marker_color=COLORS["primary"],
        opacity=0.7,
        yaxis="y",
    ))

    # Observed default rate (right axis) with CI
    fig.add_trace(go.Scatter(
        x=segs, y=obs_drs,
        name="Observed Default Rate",
        mode="lines+markers",
        line={"color": COLORS["danger"], "width": 2.5},
        marker={"size": 9},
        error_y={"type": "data", "symmetric": False, "array": err_plus, "arrayminus": err_minus,
                 "color": COLORS["danger"], "thickness": 1.5, "width": 4},
        yaxis="y2",
    ))

    # Mean predicted PD (right axis) overlay
    fig.add_trace(go.Scatter(
        x=segs, y=mean_pds,
        name="Mean Predicted PD",
        mode="lines+markers",
        line={"color": COLORS["warning"], "width": 2, "dash": "dot"},
        marker={"size": 7, "symbol": "diamond"},
        yaxis="y2",
    ))

    fig.update_layout(
        yaxis={"title": "Borrower Count", "showgrid": True},
        yaxis2={"title": "Default Rate / Predicted PD", "overlaying": "y", "side": "right",
                "showgrid": False, "tickformat": ".1%"},
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return _apply_theme(
        fig,
        "Borrower Risk Segments — Count, Observed Default Rate & Predicted PD<br>"
        "<sup>⚠ Illustrative segment boundaries — not validated bank policy</sup>",
        "Risk Segment",
        "Borrower Count",
    )
