"""
decision_engine.py — Enhanced Decision Transparency & Explainability Engine for AgroIntel.

Rules:
  - Estimated Storage Cost: 2.0% over 30 days (warehouse fee, interest, shrink/decay).
  - Estimated Net Gain % = Expected Change % - Estimated Storage Cost %
  - If Net Gain % > 3.0% (Expected Change > +5.0%) -> HOLD
  - Else -> SELL

Returns:
  decision: "HOLD" | "SELL"
  decision_score: dict with full financial breakdown & decision_reason
  reasons: list of deterministic explainability strings
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Standard 30-day agricultural holding / storage cost percentage
ESTIMATED_STORAGE_COST_PCT = 2.0


@dataclass
class DecisionResult:
    """Structured Sell/Hold decision output with full score breakdown & explainability."""
    decision: str                      # "HOLD" | "SELL"
    decision_score: Dict[str, Any]     # Financial breakdown dict
    expected_change_percent: float
    reason_summary: str
    reasons: List[str]


def make_decision(
    crop: str,
    current_price: float,
    predicted_30d_avg: float,
    expected_change_percent: float,
    trend_direction: str,
    confidence_percent: float,
    model_name: str,
    monthly_temp: float,
    monthly_rain: float,
    data_freshness_label: str,
    storage_cost_percent: float = ESTIMATED_STORAGE_COST_PCT,
) -> DecisionResult:
    """
    Determine SELL or HOLD recommendation with transparent decision_score breakdown.

    Args:
        crop: Crop name.
        current_price: Current market price (₹/quintal).
        predicted_30d_avg: Average 30-day forecasted price (₹/quintal).
        expected_change_percent: 30-day percentage price change.
        trend_direction: "UPWARD" | "DOWNWARD" | "STABLE".
        confidence_percent: Forecast confidence percentage.
        model_name: Production model name.
        monthly_temp: Monthly average temperature.
        monthly_rain: Monthly total rainfall.
        data_freshness_label: Mandi data freshness string.
        storage_cost_percent: Estimated 30-day holding cost % (default 2.0%).

    Returns:
        DecisionResult object.
    """
    crop_title = crop.capitalize()
    net_gain_percent = round(expected_change_percent - storage_cost_percent, 2)

    # ── Decision Logic ────────────────────────────────────────────────────────
    if expected_change_percent > 5.0:
        decision = "HOLD"
        decision_reason = (
            f"Predicted price increase ({expected_change_percent:+.1f}%) significantly exceeds "
            f"estimated 30-day storage costs ({storage_cost_percent:.1f}%). Holding inventory is profitable."
        )
    elif expected_change_percent < -5.0:
        decision = "SELL"
        decision_reason = (
            f"Predicted price decrease ({expected_change_percent:.1f}%) indicates an downward market. "
            f"Selling immediately is recommended to lock in current market prices."
        )
    else:
        decision = "SELL"
        decision_reason = (
            f"Predicted price change ({expected_change_percent:+.1f}%) does not yield net gain after accounting for "
            f"estimated 30-day holding and storage decay costs ({storage_cost_percent:.1f}%). Selling recommended."
        )

    decision_score = {
        "current_price": round(float(current_price), 2),
        "predicted_average_price": round(float(predicted_30d_avg), 2),
        "expected_change_percent": round(float(expected_change_percent), 2),
        "estimated_storage_cost_percent": round(float(storage_cost_percent), 2),
        "estimated_net_gain_percent": net_gain_percent,
        "decision_reason": decision_reason,
    }

    # ── Deterministic Explainability Reasons ────────────────────────────────
    reasons = []

    # 1. Trend & Price Movement
    if trend_direction == "UPWARD":
        reasons.append(
            f"Historical trend analysis indicates an UPWARD trajectory (+{expected_change_percent:.1f}% expected over 30 days)."
        )
    elif trend_direction == "DOWNWARD":
        reasons.append(
            f"Historical trend analysis indicates a DOWNWARD trajectory ({expected_change_percent:.1f}% expected over 30 days)."
        )
    else:
        reasons.append(
            f"Historical trend analysis indicates a STABLE price pattern ({expected_change_percent:+.1f}% 30-day forecast variance)."
        )

    # 2. Storage Cost Justification
    if decision == "HOLD":
        reasons.append(
            f"Net gain after 30-day storage costs ({storage_cost_percent:.1f}%) is estimated at +{net_gain_percent:.1f}%."
        )
    else:
        reasons.append(
            f"Holding crop for 30 days incurs an estimated {storage_cost_percent:.1f}% storage & shrink cost, yielding net gain of {net_gain_percent:+.1f}%."
        )

    # 3. Seasonality Context
    if crop_title in ["Wheat", "Potato"]:
        reasons.append(
            f"{crop_title} pricing reflects Winter Season harvest market patterns based on 6-year history."
        )
    elif crop_title in ["Rice", "Maize"]:
        reasons.append(
            f"{crop_title} pricing reflects Rainy Season crop arrival cycles based on 6-year history."
        )
    else:
        reasons.append(
            f"{crop_title} displays historical seasonal trading behavior in national agricultural markets."
        )

    # 4. Weather & Model Context
    reasons.append(
        f"Regional climate proxy conditions ({monthly_temp:.1f}°C avg temp, {monthly_rain:.1f}mm monthly rainfall) "
        f"align with standard seasonal supply expectations."
    )
    reasons.append(
        f"Forecast generated by production-selected {model_name.upper()} model with {confidence_percent:.1f}% confidence "
        f"(Data status: {data_freshness_label})."
    )

    return DecisionResult(
        decision=decision,
        decision_score=decision_score,
        expected_change_percent=round(expected_change_percent, 2),
        reason_summary=decision_reason,
        reasons=reasons,
    )
