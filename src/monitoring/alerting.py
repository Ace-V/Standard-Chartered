"""
CreditRisk Analytics — Monitoring Subsystem: Alerting Engine (Module E)
=======================================================================

Manages monitoring threshold evaluations, severity states, human-readable explanations,
and alert lifecycle history across Data Quality, Feature Drift, Prediction Drift,
and Model Performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    DATA_QUALITY = "DATA_QUALITY"
    FEATURE_DRIFT = "FEATURE_DRIFT"
    PREDICTION_DRIFT = "PREDICTION_DRIFT"
    MODEL_PERFORMANCE = "MODEL_PERFORMANCE"


@dataclass
class Alert:
    """Individual monitoring alert record."""
    alert_id: str
    timestamp: str
    category: AlertCategory
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    message: str
    recommended_action: str
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Alert ID": self.alert_id,
            "Timestamp": self.timestamp,
            "Category": self.category.value,
            "Severity": self.severity.value,
            "Metric": self.metric_name,
            "Current Value": self.current_value,
            "Warning Thresh": self.threshold_warning,
            "Critical Thresh": self.threshold_critical,
            "Message": self.message,
            "Recommended Action": self.recommended_action,
            "Acknowledged": self.acknowledged,
        }


@dataclass
class AlertRule:
    """Configurable threshold rule specification for a metric."""
    metric_name: str
    category: AlertCategory
    warning_threshold: float
    critical_threshold: float
    comparator: str  # "gte" (greater than or equal) or "lte" (less than or equal)
    description: str
    action_guidance: str


def get_default_alert_rules() -> Dict[str, AlertRule]:
    """Return standard banking supervisory alert rules."""
    return {
        "psi_score": AlertRule(
            metric_name="psi_score",
            category=AlertCategory.FEATURE_DRIFT,
            warning_threshold=0.10,
            critical_threshold=0.25,
            comparator="gte",
            description="Population Stability Index for Credit Score / PD output",
            action_guidance="Review recent origination underwriting criteria and credit risk segmentation.",
        ),
        "csi_feature": AlertRule(
            metric_name="csi_feature",
            category=AlertCategory.FEATURE_DRIFT,
            warning_threshold=0.10,
            critical_threshold=0.25,
            comparator="gte",
            description="Characteristic Stability Index for input risk factors",
            action_guidance="Investigate upstream data feeds or macro shifts in borrower balances/delinquency.",
        ),
        "missing_rate": AlertRule(
            metric_name="missing_rate",
            category=AlertCategory.DATA_QUALITY,
            warning_threshold=1.0,
            critical_threshold=5.0,
            comparator="gte",
            description="Per-feature missing value percentage",
            action_guidance="Inspect data ingestion pipeline and upstream core banking extracts.",
        ),
        "gini_decay_pct": AlertRule(
            metric_name="gini_decay_pct",
            category=AlertCategory.MODEL_PERFORMANCE,
            warning_threshold=15.0,
            critical_threshold=25.0,
            comparator="gte",
            description="Percentage erosion in Gini coefficient (Somers' D)",
            action_guidance="Perform formal model recalibration or initiate model redevelopment under SR 11-7.",
        ),
        "mean_pd_shift_pct": AlertRule(
            metric_name="mean_pd_shift_pct",
            category=AlertCategory.PREDICTION_DRIFT,
            warning_threshold=15.0,
            critical_threshold=30.0,
            comparator="gte",
            description="Percentage shift in average predicted probability of default",
            action_guidance="Assess credit portfolio stress or underwriting policy loosening.",
        ),
    }


class AlertManager:
    """
    Evaluates monitoring metrics against configurable rules and maintains an alert history ledger.
    """

    def __init__(self, rules: Optional[Dict[str, AlertRule]] = None):
        self.rules = rules or get_default_alert_rules()
        self._history: List[Alert] = []

    def evaluate_metric(
        self,
        rule_name: str,
        current_value: float,
        metric_display_name: Optional[str] = None,
        context_details: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Evaluate a scalar metric against its defined rule and record an alert if triggered.
        """
        if rule_name not in self.rules:
            return None

        rule = self.rules[rule_name]
        val = float(current_value)
        disp_name = metric_display_name or rule.metric_name

        severity: Optional[AlertSeverity] = None

        if rule.comparator == "gte":
            if val >= rule.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif val >= rule.warning_threshold:
                severity = AlertSeverity.WARNING
        elif rule.comparator == "lte":
            if val <= rule.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif val <= rule.warning_threshold:
                severity = AlertSeverity.WARNING

        if severity is None:
            return None

        msg = (
            f"[{severity.value}] {disp_name} = {val:.4f} crossed {severity.value.lower()} threshold "
            f"({rule.critical_threshold if severity == AlertSeverity.CRITICAL else rule.warning_threshold:.4f}). "
            f"{rule.description}. {context_details or ''}".strip()
        )

        alert = Alert(
            alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            category=rule.category,
            severity=severity,
            metric_name=disp_name,
            current_value=round(val, 4),
            threshold_warning=rule.warning_threshold,
            threshold_critical=rule.critical_threshold,
            message=msg,
            recommended_action=rule.action_guidance,
            acknowledged=False,
        )

        self._history.append(alert)
        return alert

    def record_custom_alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        metric_name: str,
        current_value: float,
        threshold_warning: float,
        threshold_critical: float,
        message: str,
        recommended_action: str,
    ) -> Alert:
        """Manually construct and record an alert."""
        alert = Alert(
            alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            category=category,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            message=message,
            recommended_action=recommended_action,
            acknowledged=False,
        )
        self._history.append(alert)
        return alert

    def get_active_alerts(self, unacknowledged_only: bool = True) -> List[Alert]:
        """Return all active alerts."""
        if unacknowledged_only:
            return [a for a in self._history if not a.acknowledged]
        return list(self._history)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for a in self._history:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    def history_dataframe(self) -> pd.DataFrame:
        """Export all recorded alerts as a Pandas DataFrame."""
        if not self._history:
            return pd.DataFrame(columns=[
                "Alert ID", "Timestamp", "Category", "Severity", "Metric",
                "Current Value", "Warning Thresh", "Critical Thresh", "Message",
                "Recommended Action", "Acknowledged"
            ])
        return pd.DataFrame([a.to_dict() for a in self._history])
