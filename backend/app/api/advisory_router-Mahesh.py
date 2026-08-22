"""
advisory_router.py — Combined Agricultural Advisory Router for AgroIntel.

Endpoints:
  POST /api/advisory — Integrated Crop Recommendation & Price Forecast Advisory Engine
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.constants import PRICE_PREDICTION_CROPS
from app.ml.inference import predict_price
from app.services.phase6_integration_service import AgroIntelPhase6Engine
from app.services.location_normalizer import normalize_location

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Combined Advisory"])

_engine = AgroIntelPhase6Engine()


class AdvisoryRequest(BaseModel):
    state: str = Field(..., description="Target state", example="Maharashtra")
    district: str = Field(..., description="Target district", example="Ahilya Nagar")
    season: Optional[str] = Field("Kharif", description="Season", example="Kharif")
    crop: Optional[str] = Field(None, description="Optional target crop for price prediction", example="wheat")
    n: Optional[float] = Field(None, description="Nitrogen content (kg/ha)")
    p: Optional[float] = Field(None, description="Phosphorus content (kg/ha)")
    k: Optional[float] = Field(None, description="Potassium content (kg/ha)")
    ph: Optional[float] = Field(None, description="Soil pH")
    lat: Optional[float] = Field(None, description="Latitude")
    lon: Optional[float] = Field(None, description="Longitude")


@router.post("/advisory", summary="Get Integrated Crop Recommendation & Price Forecast Advisory")
def get_combined_advisory(request: AdvisoryRequest):
    """
    Generate combined agricultural advisory:
      1. Run Multi-Stage Crop Recommendation for target region & soil.
      2. Predict 30-day Price Forecast & Sell/Hold Decision for target/top crop.
      3. Output unified advisory response with combined explainability reasons.
    """
    t_start = time.perf_counter()

    # 1. Location Normalization
    loc = normalize_location(request.district, request.state)
    canon_state = loc["state"]
    canon_district = loc["canonical_name"]

    # 2. Step 1: Crop Recommendation Pipeline
    rec_result = _engine.evaluate_recommendation(
        state=canon_state,
        district=canon_district,
        season=request.season or "Kharif",
        soil_ph=request.ph,
        previous_crop=request.crop,
        lat=request.lat,
        lon=request.lon
    )

    recs = rec_result.get("recommendations", [])
    top_crop_name = recs[0]["crop"] if recs else "Rice"

    # 3. Step 2: Target Crop Selection for Price Prediction
    # Use user-selected crop if provided, otherwise top recommended crop
    target_crop = request.crop or top_crop_name
    target_crop_lower = target_crop.lower().strip()

    # 4. Step 3: Authoritative State-Aware Price Forecast
    # CRITICAL: Each crop uses ONLY its own validated model. No cross-crop substitution.
    if target_crop_lower in PRICE_PREDICTION_CROPS:
        price_result = predict_price(
            crop=target_crop_lower,
            state=canon_state,
            horizon_days=30
        )
        curr_p = price_result.get("current_price")
        pred_p = price_result.get("predicted_price")
        # Use forecast decision only if the forecast actually succeeded
        if price_result.get("available", False) and pred_p is not None:
            decision_action = price_result.get("recommendation")  # SELL / HOLD / WAIT
            decision_reason = price_result.get("recommendation_reason", "")
        else:
            decision_action = None
            decision_reason = "A reliable 30-day price forecast is currently unavailable for this crop."
    else:
        # Non-benchmark crop (e.g. Sugarcane, Cotton) — no price forecasting model exists
        # Provide current mandi price if available, but NO fake decision
        m_vec = rec_result.get("market", {})
        curr_p = m_vec.get("current_price")
        pred_p = None
        decision_action = None
        decision_reason = "A reliable 30-day price forecast is currently unavailable for this crop."
        price_result = {
            "available": False,
            "crop": target_crop,
            "state": canon_state,
            "current_price": curr_p,
            "predicted_price": None,
            "recommendation": None,
            "recommendation_reason": decision_reason
        }

    # 5. Combined Summary (Farmer-Oriented)
    rec_top1 = recs[0] if recs else {}
    expl = rec_top1.get("nlp_explanation", {})
    season_friendly_map = {"kharif": "Rainy Season", "rabi": "Winter Season", "zaid": "Summer Season", "summer": "Summer Season", "whole year": "Whole Year"}
    friendly_season_label = season_friendly_map.get(str(request.season).lower(), request.season)
    why_text = expl.get("why_recommended") or f"{top_crop_name} is recommended for the {friendly_season_label} in {canon_district}."

    if pred_p and curr_p:
        change_pct = round(((pred_p - curr_p) / curr_p) * 100.0, 1)
        price_summary = f"30-day price outlook for {target_crop.title()} indicates {decision_action} (forecast ₹{pred_p:,.0f}/qtl, {change_pct:+.1f}%)."
    elif curr_p:
        price_summary = f"Latest mandi price for {target_crop.title()} was ₹{curr_p:,.0f}/qtl."
    else:
        price_summary = f"Market data monitored for regional mandis in {canon_state}."

    combined_summary = f"Recommended crop for {canon_district}, {canon_state} ({friendly_season_label}): {top_crop_name}. {price_summary}"

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)

    # Build price prediction block — forecast_available tells frontend what to display
    price_prediction_block = {
        "crop": target_crop,
        "forecast_available": price_result.get("available", False),
        "current_price": curr_p,
        "predicted_30d_avg": pred_p,  # null when no forecast
        "decision": decision_action,  # null when no forecast — NEVER default HOLD
        "decision_reason": decision_reason,
        "observation_date": price_result.get("observation_date"),
        "forecast_series": price_result.get("predictions", []),
        "date_labels": price_result.get("date_labels", []),
        "farmer_message": decision_reason if decision_action is None else None,
    }

    return {
        "state": canon_state,
        "district": canon_district,
        "season": rec_result.get("season", request.season),
        "target_price_crop": target_crop,
        "combined_summary": combined_summary,
        "crop_recommendations": recs,
        "price_prediction": price_prediction_block,
        "response_time_ms": latency_ms
    }
