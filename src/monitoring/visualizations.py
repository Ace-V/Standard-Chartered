"""
CreditRisk Analytics — Monitoring Visual Analytics & Plotly Reports
====================================================================

Generates publication-quality Plotly charts for production model surveillance:
1. Feature CSI ranking with supervisory threshold reference lines (0.10 and 0.25)
2. Predicted probability density comparison (Baseline vs. Monitoring)
3. Risk rating segment migration bar chart
4. Data quality missingness & range error diagnostics
5. Monitoring alert breakdown by category and severity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.monitoring.alerting import Alert, AlertManager, AlertSeverity
from src.monitoring.data_quality import DataQualityReport


COLORS = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0284C7",    # Slate Blue
    "accent": "#0D9488",       # Emerald Teal
    "danger": "#DC2626",       # Red
    "warning": "#F59E0B",      # Amber
    "success": "#10B981",      # Green
    "neutral": "#64748B",      # Slate Grey
}


def plot_feature_csi_ranking(csi_df: pd.DataFrame) -> go.Figure:
    """
    Produce a ranked horizontal bar chart of feature Characteristic Stability Index (CSI)
    values with supervisory reference lines at 0.10 (Warning) and 0.25 (Critical).
    """
    df = csi_df.copy().sort_values(by="CSI_Value", ascending=True)

    colors = []
    for val in df["CSI_Value"]:
        if val >= 0.25:
            colors.append(COLORS["danger"])
        elif val >= 0.10:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["success"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["Feature"],
        x=df["CSI_Value"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in df["CSI_Value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>CSI: %{x:.4f}<extra></extra>",
    ))

    # Supervisory threshold lines
    fig.add_vline(x=0.10, line_dash="dash", line_color=COLORS["warning"], annotation_text="Warning (0.10)", annotation_position="top")
    fig.add_vline(x=0.25, line_dash="dash", line_color=COLORS["danger"], annotation_text="Critical (0.25)", annotation_position="top")

    fig.update_layout(
        title={
            "text": "<b>Characteristic Stability Index (CSI) — Feature Drift Ranking</b><br>"
                    "<sup>Supervisory thresholds: Green < 0.10 <= Amber < 0.25 <= Red</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Characteristic Stability Index (CSI)",
        yaxis_title="",
        template="plotly_white",
        margin={"l": 150, "r": 60, "t": 70, "b": 50},
        font={"family": "Inter, Arial, sans-serif", "size": 11, "color": "#1E293B"},
        showlegend=False,
    )
    return fig


def plot_prediction_distribution_shift(
    baseline_probs: np.ndarray,
    monitoring_probs: np.ndarray,
    n_bins: int = 40,
) -> go.Figure:
    """
    Overlay histogram comparing baseline and monitoring predicted probability distributions.
    """
    b_p = np.asarray(baseline_probs, dtype=float)
    m_p = np.asarray(monitoring_probs, dtype=float)
    b_p = b_p[np.isfinite(b_p)]
    m_p = m_p[np.isfinite(m_p)]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=b_p,
        histnorm="probability density",
        name="Baseline (Training / In-Time)",
        marker_color=COLORS["primary"],
        opacity=0.6,
        nbinsx=n_bins,
    ))

    fig.add_trace(go.Histogram(
        x=m_p,
        histnorm="probability density",
        name="Monitoring (Recent / Out-of-Time)",
        marker_color=COLORS["warning"],
        opacity=0.6,
        nbinsx=n_bins,
    ))

    fig.update_layout(
        title={
            "text": "<b>Predicted Probability Distribution Comparison</b><br>"
                    "<sup>Detecting systemic score shifts and macro-driven covariate changes</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Predicted Default Probability (PD)",
        yaxis_title="Probability Density",
        barmode="overlay",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
    )
    return fig


def plot_risk_segment_migration(migration_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart displaying portfolio percentage share in each Master Rating Scale grade.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=migration_df["Rating Grade"],
        y=migration_df["Baseline Share (%)"],
        name="Baseline Portfolio (%)",
        marker_color=COLORS["primary"],
        opacity=0.85,
    ))

    fig.add_trace(go.Bar(
        x=migration_df["Rating Grade"],
        y=migration_df["Monitoring Share (%)"],
        name="Monitoring Portfolio (%)",
        marker_color=COLORS["secondary"],
        opacity=0.85,
    ))

    fig.update_layout(
        title={
            "text": "<b>Risk Segment Migration Across Master Rating Scale</b><br>"
                    "<sup>Tracking portfolio risk migration from prime (AAA) to impaired (CCC)</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Master Rating Grade",
        yaxis_title="Portfolio Share (%)",
        barmode="group",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
    )
    return fig


def plot_data_quality_issues(dq_report: DataQualityReport) -> go.Figure:
    """
    Bar chart visualizing missingness and invalid range rates across flagged columns.
    """
    df_missing = dq_report.missingness_summary()

    fig = go.Figure()

    if not df_missing.empty:
        fig.add_trace(go.Bar(
            x=df_missing["Column"],
            y=df_missing["Missing (%)"],
            name="Missing Rate (%)",
            marker_color=COLORS["danger"],
            text=[f"{v:.1f}%" for v in df_missing["Missing (%)"]],
            textposition="outside",
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["warning"], annotation_text="Warning (1%)")
        fig.add_hline(y=5.0, line_dash="dash", line_color=COLORS["danger"], annotation_text="Critical (5%)")

    fig.update_layout(
        title={
            "text": "<b>Data Quality Audit — Missing-Value Rates per Feature</b><br>"
                    f"<sup>Batch Size: {dq_report.total_records:,} rows | Overall Quality: {dq_report.overall_status}</sup>",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 15, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Feature Name",
        yaxis_title="Missingness Percentage (%)",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#1E293B"},
        showlegend=False,
    )
    return fig
