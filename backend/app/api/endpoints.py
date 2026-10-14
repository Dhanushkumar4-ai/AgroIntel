"""
endpoints.py — API routes for AgroIntel AI
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from app.ml.inference import predict_price, predict_crop, get_crop_info, clear_model_cache
from app.ml.train import train_all_crops, train_price_prediction_model, CROPS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "app" / "data" / "experimental"

class CropRequest(BaseModel):
    location: str
    season: str = "Auto"  # Auto | Kharif | Rabi | Zaid
    lat: float = None     # Optional GPS latitude  (enables satellite soil detection)
    lon: float = None     # Optional GPS longitude (enables satellite soil detection)

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_CROPS = list(CROPS.keys())

# ── Models are already training in background ─────────────────────────────────
_training_status = {"status": "idle", "message": "No training run yet."}


def _run_training_task():
    """Background task that trains all crop models."""
    global _training_status
    _training_status = {"status": "running", "message": "Training in progress…"}
    try:
        results = train_all_crops()
        clear_model_cache()
        best_models = {c: r.get("best_model", "?") for c, r in results.items() if r.get("status") == "success"}
        _training_status = {
            "status": "success",
            "message": "Training complete.",
            "results": results,
            "best_models": best_models,
        }
        logger.info("Background training complete: %s", best_models)
    except Exception as e:
        _training_status = {"status": "error", "message": str(e)}
        logger.error("Training failed: %s", e)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """Trigger retraining of all models in the background."""
    if _training_status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    background_tasks.add_task(_run_training_task)
    return {"message": "Training started in background.", "crops": SUPPORTED_CROPS,
            "note": "Includes COVID-19 (2020) and Russia-Ukraine war (2022) data."}


@router.get("/train/status")
def training_status():
    """Get the status of the last training run."""
    return _training_status


@router.post("/train/{crop}")
def train_single_crop(crop: str):
    """Train models for a single crop synchronously."""
    crop = crop.lower()
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"Unsupported crop '{crop}'. Choose: {SUPPORTED_CROPS}")
    try:
        result = train_price_prediction_model(crop)
        clear_model_cache()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/crop")
def predict_crop_endpoint(request: CropRequest):
    """Predict the best crop to grow based on location, weather, satellite soil, and season."""
    try:
        result = predict_crop(
            request.location,
            request.season,
            lat=request.lat,
            lon=request.lon,
        )
        return result
    except Exception as e:
        logger.error("Crop prediction error for %s: %s", request.location, e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict")
def predict(
    crop: str = Query(default="wheat", description="Crop name"),
    state: str = Query(default="All", description="State name for accurate local price"),
    horizon_days: int = Query(default=30, ge=1, le=90, description="Forecast horizon in days"),
):
    """Get price prediction for a crop using the best available model."""
    crop_clean = crop.lower()
    if crop_clean not in ["rice", "wheat", "maize", "onion", "potato"]:
        return {
            "available": False,
            "crop": crop,
            "forecast": {
                "available": False,
                "status": "FORECAST_UNAVAILABLE",
                "reason": "Price prediction is currently unavailable for this crop because a validated forecasting model is not available."
            },
            "advisory": {
                "decision": "INSUFFICIENT_DATA",
                "reason": "No validated forecasting model available."
            },
            "message": "Price prediction is currently unavailable for this crop because a validated forecasting model is not available."
        }
    try:
        result = predict_price(crop_clean, state, horizon_days)

        # ── Build clean farmer-facing response ────────────────────────────────
        # Strip ALL technical internal fields before returning to frontend.
        # Farmers must never see: model names, MAE, RMSE, model_level,
        # forecast_scope, best_model, model_comparison, price_cached_time, etc.

        farmer_response = {
            # Core availability
            "available":             result.get("available", False),
            "crop":                  result.get("crop", crop_clean),
            "state":                 result.get("state", state),

            # Price information
            "current_price":         result.get("current_price"),
            "predicted_price":       result.get("predicted_price"),
            "predictions":           result.get("predictions", []),
            "date_labels":           result.get("date_labels", []),
            "horizon_days":          result.get("horizon_days", horizon_days),
            "percent_change":        result.get("percent_change"),

            # Observation metadata & State-Aware Intelligence
            "observation_date":      result.get("observation_date") or result.get("current_price_date"),
            "market_name":           result.get("market_name"),
            "data_age_days":         result.get("data_age_days"),
            "price_data_source":     result.get("price_data_source", "Agmarknet 2.0 (Official Live Mandi)"),
            "price_type":            result.get("price_type"),
            "period_start":          result.get("period_start"),
            "period_end":            result.get("period_end"),
            "unit":                  result.get("unit", "₹/quintal"),
            "min_price":             result.get("min_price"),
            "max_price":             result.get("max_price"),
            "msp_price":             result.get("msp_price"),
            "msp_season":            result.get("msp_season"),
            "variety":               result.get("variety", "All Reporting Varieties (State-wide Pre-Aggregated)"),
            "market_scope":          result.get("market_scope") or (f"State-wide Scope ({state})" if state and state.lower() != "all" else "National APMC Mandi Benchmark"),
            "contributing_records_count": result.get("contributing_records_count", 1),
            "contributing_records":  result.get("contributing_records", []),
            "aggregation_method":    result.get("aggregation_method", "Official Arrival-Weighted State Modal Average (wt_avg_price)"),
            "farm_gate_modal_price": result.get("farm_gate_modal_price"),
            "farm_gate_range":       result.get("farm_gate_range"),
            "state_composite_index": result.get("state_composite_index"),
            "reconciliation_notice": result.get("reconciliation_notice"),
            "district_breakdown":    result.get("district_breakdown", []),
            "price_transparency":    result.get("price_transparency", {}),
            "interval_type":         result.get("interval_type", "Estimated Error Range (Holdout RMSE Basis)"),
            "interval_calibrated":   result.get("interval_calibrated", False),
            "forecast_scope":        result.get("forecast_scope") or (f"State-Specific Forecast ({state})" if state and state.lower() != "all" else "National APMC Mandi Benchmark"),
            "model_level":           result.get("model_level", "STATE_AWARE"),
            "best_model":            result.get("best_model"),
            "best_model_label":      "State-Aware XGBoost Model" if "state" in str(result.get("best_model", "")).lower() else "Agricultural Time-Series Model",
            "reference_benchmark":   result.get("reference_benchmark"),

            # Advisory
            "recommendation":        result.get("recommendation", "WAIT"),
            "recommendation_reason": result.get("recommendation_reason", ""),
            "advisory":              result.get("advisory") or {"decision": result.get("recommendation", "WAIT"), "reason": result.get("recommendation_reason", "")},
            "black_swan_warning":    result.get("black_swan_warning"),
            "storage_cost_pct":      result.get("storage_cost_pct"),
            "net_expected_change":   result.get("net_expected_change"),
            "decision_score":        result.get("decision_score"),
            "historical_series":     result.get("historical_series", []),
            "forecast_series":       result.get("forecast_series", []),
            "model_comparison":      result.get("model_comparison", {}),

            # Nested objects for frontend compatibility
            "market": {
                "current_price":              result.get("current_price"),
                "observation_date":           result.get("observation_date") or result.get("current_price_date"),
                "period_start":               result.get("period_start"),
                "period_end":                 result.get("period_end"),
                "price_type":                 result.get("price_type"),
                "unit":                       result.get("unit", "₹/quintal"),
                "min_price":                  result.get("min_price"),
                "max_price":                  result.get("max_price"),
                "msp_price":                  result.get("msp_price"),
                "msp_season":                 result.get("msp_season"),
                "variety":                    result.get("variety", "All Reporting Varieties (State-wide Pre-Aggregated)"),
                "market_scope":               result.get("market_scope") or (f"State-wide Scope ({state})" if state and state.lower() != "all" else "National APMC Mandi Benchmark"),
                "contributing_records_count": result.get("contributing_records_count", 1),
                "contributing_records":       result.get("contributing_records", []),
                "aggregation_method":         result.get("aggregation_method", "Official Arrival-Weighted State Modal Average (wt_avg_price)"),
                "farm_gate_modal_price":      result.get("farm_gate_modal_price"),
                "farm_gate_range":            result.get("farm_gate_range"),
                "state_composite_index":      result.get("state_composite_index"),
                "reconciliation_notice":      result.get("reconciliation_notice"),
                "district_breakdown":         result.get("district_breakdown", []),
                "market_name":                result.get("market_name"),
                "data_age_days":              result.get("data_age_days"),
                "price_data_source":          result.get("price_data_source", "Agmarknet 2.0 (Official Live Mandi)"),
                "price_transparency":         result.get("price_transparency", {}),
                "price_label":                "Latest Available Mandi Price",
            },
            "forecast": {
                "available":           result.get("available", False),
                "predicted_price":     result.get("predicted_price"),
                "prediction_start":    result.get("prediction_start"),
                "forecast_series":     result.get("forecast_series", []),
                "historical_series":   result.get("historical_series", []),
                "raw_predictions":     result.get("predictions", []),
                "date_labels":         result.get("date_labels", []),
                "forecast_scope":      result.get("forecast_scope") or (f"State-Specific Forecast ({state})" if state and state.lower() != "all" else "National APMC Mandi Benchmark"),
                "model_comparison":    result.get("model_comparison", {}),
                "reference_benchmark": result.get("reference_benchmark"),
                "interval_type":       result.get("interval_type", "Estimated Error Range (Holdout RMSE Basis)"),
                "interval_calibrated": result.get("interval_calibrated", False),
            },
            "advisory": {
                "decision":            result.get("recommendation", "WAIT"),
                "reason":              result.get("recommendation_reason", ""),
                "storage_cost_pct":    result.get("storage_cost_pct"),
                "net_expected_change": result.get("net_expected_change"),
                "decision_score":      result.get("decision_score"),
                "news_alignment":      result.get("news_intelligence", {}).get("forecast_alignment", "NEUTRAL"),
                "market_signal":       result.get("news_intelligence", {}).get("market_signal", "NEUTRAL"),
                "news_explanation":    result.get("news_intelligence", {}).get("advisory_explanation", ""),
            },
            "news_intelligence":       result.get("news_intelligence", {}),
        }
        return farmer_response

    except Exception as e:
        logger.error("Prediction error for %s in %s: %s", crop, state, e)
        return {
            "available": False,
            "crop": crop,
            "forecast": {
                "available": False,
                "status": "FORECAST_UNAVAILABLE",
                "reason": str(e)
            },
            "advisory": {
                "decision": "INSUFFICIENT_DATA",
                "reason":   str(e)
            },
            "message": "Price prediction is currently unavailable for this crop because a validated forecasting model is not available."
        }




@router.get("/crops")
def list_crops():
    """List all supported crops and their model status."""
    crops_info = []
    for crop in SUPPORTED_CROPS:
        try:
            info = get_crop_info(crop)
        except Exception:
            info = {"crop": crop, "ready": False}
        crops_info.append(info)
    return {"crops": crops_info, "supported": SUPPORTED_CROPS}


@router.get("/crops/{crop}")
def crop_info(crop: str):
    """Get model status and metrics for a specific crop."""
    crop = crop.lower()
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"Unknown crop: {crop}")
    try:
        return get_crop_info(crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agricultural News Intelligence Endpoints ─────────────────────────────────

@router.get("/news", summary="Agricultural News & Intelligence Feed")
def get_agricultural_news(
    crop: Optional[str] = Query(None, description="Filter by crop name (e.g. wheat, rice, onion)"),
    state: Optional[str] = Query(None, description="Filter by state name (e.g. Maharashtra, Punjab)"),
    market_effect: Optional[str] = Query(None, description="Filter by market effect: BULLISH, BEARISH, MIXED, UNCERTAIN"),
    verification_status: Optional[str] = Query(None, description="Filter by verification: VERIFIED, CORROBORATED, SINGLE_SOURCE, UNVERIFIED"),
    limit: int = Query(default=20, ge=1, le=100, description="Max articles to return"),
):
    """
    Returns verified, scored agricultural news articles.
    Supports filtering by crop, state, market effect, and verification status.
    """
    events_path = EXP_DIR / "news_events.json"
    if not events_path.exists():
        events_path = EXP_DIR / "current_intelligence.json"
    if not events_path.exists():
        return {
            "total_articles": 0,
            "filtered_articles": 0,
            "articles": [],
            "message": "No news articles found. Run POST /api/news/refresh to ingest.",
        }

    with open(events_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        articles = []

    from app.services.news_intelligence_service import calculate_article_relevance_score

    filtered = []
    for art in articles:
        # Filter by market effect
        if market_effect and art.get("market_effect", "").upper() != market_effect.upper():
            continue
        # Filter by verification status
        if verification_status and art.get("verification_status", "").upper() != verification_status.upper():
            continue

        # If crop or state specified, re-score article relevance for that context
        if crop or state:
            art_copy = dict(art)
            rel_score = calculate_article_relevance_score(art, target_crop=crop, target_state=state)
            art_copy["relevance_score"] = rel_score
            # Include if relevant
            if rel_score["composite_score"] >= 25.0:
                filtered.append(art_copy)
        else:
            filtered.append(art)

    # Sort by composite relevance score descending, then recency
    filtered.sort(
        key=lambda x: (
            x.get("relevance_score", {}).get("composite_score", 50.0),
            -float(x.get("age_days", 999))
        ),
        reverse=True
    )

    return {
        "total_articles": len(articles),
        "filtered_count": len(filtered),
        "crop_filter": crop or "All",
        "state_filter": state or "All",
        "market_effect_filter": market_effect or "All",
        "verification_filter": verification_status or "All",
        "articles": filtered[:limit],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/news/refresh", summary="Manual Trigger for RSS News Ingestion")
def refresh_agricultural_news():
    """
    Manually triggers daily RSS news ingestion, duplicate pruning,
    market impact classification, and cross-source verification.
    """
    from app.jobs.fetch_news import NewsIngestionJob
    from app.jobs.process_news import NewsProcessingJob

    t_start = datetime.now(timezone.utc)
    ingestion_job = NewsIngestionJob()
    ingest_res = ingestion_job.run()

    proc_job = NewsProcessingJob()
    proc_res = proc_job.run()

    return {
        "status": "SUCCESS",
        "message": "Daily agricultural news feeds refreshed and verified successfully.",
        "ingestion_summary": ingest_res,
        "processing_summary": proc_res,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round((datetime.now(timezone.utc) - t_start).total_seconds(), 2),
    }


@router.get("/news/status", summary="News Feed Health & Freshness Status")
def news_pipeline_status():
    """
    Returns news pipeline health, feed statuses, total articles, and data freshness.
    """
    ingest_audit_path = EXP_DIR / "news_ingestion_audit.json"
    verif_audit_path = EXP_DIR / "news_cross_verification_report.json"
    source_audit_path = EXP_DIR / "news_source_runtime_audit.json"

    ingest_info = {}
    verif_info = {}
    sources_info = []

    if ingest_audit_path.exists():
        with open(ingest_audit_path, "r", encoding="utf-8") as f:
            ingest_info = json.load(f)
    if verif_audit_path.exists():
        with open(verif_audit_path, "r", encoding="utf-8") as f:
            verif_info = json.load(f)
    if source_audit_path.exists():
        with open(source_audit_path, "r", encoding="utf-8") as f:
            sources_info = json.load(f)

    # Compute overall freshness status
    last_fetch = ingest_info.get("last_successful_fetch_time")
    freshness = "FRESH"
    if last_fetch:
        try:
            lf_dt = datetime.fromisoformat(last_fetch)
            hrs = (datetime.now(timezone.utc) - lf_dt).total_seconds() / 3600.0
            if hrs > 36.0:
                freshness = "STALE"
            elif hrs > 24.0:
                freshness = "DUE_FOR_REFRESH"
        except Exception:
            pass

    return {
        "freshness_status": freshness,
        "last_successful_fetch_time": last_fetch,
        "total_articles_stored": ingest_info.get("total_stored_articles", 0),
        "total_sources_audited": ingest_info.get("total_sources_audited", len(sources_info)),
        "sources_active": ingest_info.get("sources_active", sum(1 for s in sources_info if s.get("status") == "SUCCESS")),
        "sources_failed": ingest_info.get("sources_failed", sum(1 for s in sources_info if s.get("status") != "SUCCESS")),
        "verified_events_count": verif_info.get("verified_events_count", 0),
        "verified_clusters_count": verif_info.get("verified_clusters_count", 0),
        "source_breakdown": sources_info[:15],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

