"""
price_router.py — Price Prediction & Mandi Market API Router for AgroIntel v4.0.

Endpoints:
  GET  /api/predict/price  — Multi-horizon price forecast, decision score & graph series
  GET  /api/market/latest  — Fetch latest mandi market price from Agmarknet API / cache
  POST /api/train          — Trigger asynchronous retraining of all price models
  POST /api/train/{crop}   — Retrain price prediction models for a single crop synchronously
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from app.core.constants import PRICE_PREDICTION_CROPS
from app.ml.price_predictor import predict_crop_price
from app.ml.price_trainer import train_all_crops, train_price_models_for_crop
from app.services.mandi_service import get_latest_price

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Price Prediction & Market"])

_training_status = {"status": "idle", "message": "No active training run."}


def _background_training_task():
    global _training_status
    _training_status = {"status": "running", "message": "Price model training in progress..."}
    try:
        results = train_all_crops()
        _training_status = {
            "status": "success",
            "message": "Model training complete.",
            "results": results,
        }
        logger.info("Background training task completed successfully.")
    except Exception as e:
        _training_status = {"status": "error", "message": str(e)}
        logger.error(f"Background training failed: {e}")


@router.get("/predict/price", summary="Predict Crop Price Forecast & Decision")
def get_price_prediction(
    crop: str = Query(..., description="Target crop (wheat, rice, maize, potato, onion)"),
    state: Optional[str] = Query(None, description="Optional Indian state for mandi lookup"),
    horizon_days: int = Query(30, description="Forecast horizon in days (7, 15, 30, 60, 90)"),
    lat: float = Query(21.1458, description="Optional latitude for weather lookup"),
    lon: float = Query(79.0882, description="Optional longitude for weather lookup"),
):
    """
    Generate unified price prediction, graph visualization series, trend statistics,
    confidence breakdown, decision score, and explainability reasons for a crop.
    """
    crop_clean = crop.lower().strip()
    if crop_clean not in PRICE_PREDICTION_CROPS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported crop '{crop}'. Supported crops: {PRICE_PREDICTION_CROPS}"
        )

    if horizon_days not in [7, 15, 30, 60, 90]:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid horizon_days '{horizon_days}'. Must be one of [7, 15, 30, 60, 90]."
        )

    try:
        return predict_crop_price(
            crop=crop_clean,
            state=state,
            horizon_days=horizon_days,
            lat=lat,
            lon=lon,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Price prediction error for crop='{crop}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction failure: {str(e)}")


@router.get("/market/latest", summary="Fetch Latest Mandi Market Price")
def get_latest_mandi_price(
    crop: str = Query(..., description="Target crop name"),
    state: Optional[str] = Query(None, description="Optional Indian state"),
):
    """Fetch real-time or cached Mandi market price from Government Agmarknet API."""
    res = get_latest_price(crop.lower().strip(), state)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"No mandi price data found for crop='{crop}', state='{state}'"
        )

    return {
        "crop": res.crop,
        "state": state or "National",
        "market": res.market,
        "modal_price": res.modal_price,
        "min_price": res.min_price,
        "max_price": res.max_price,
        "msp_price": res.msp_price,
        "msp_season": res.msp_season,
        "arrival_date": res.arrival_date,
        "observation_date": res.observation_date,
        "source": res.source,
        "variety": res.variety,
        "market_scope": res.market_scope,
        "contributing_records_count": res.contributing_records_count,
        "contributing_records": res.contributing_records,
        "aggregation_method": res.aggregation_method,
        "freshness_label": res.freshness_label,
        "data_age_days": res.data_age_days,
        "price_status": res.price_status,
        "status_label": res.status_label,
        "price_type": getattr(res, "price_type", "Modal Price"),
        "period_start": getattr(res, "period_start", res.arrival_date),
        "period_end": getattr(res, "period_end", res.arrival_date),
        "unit": getattr(res, "unit", "₹/quintal"),
        "price_transparency": res.to_transparency_dict(),
    }


@router.post("/train", summary="Trigger Retraining of All Price Models")
def trigger_training(background_tasks: BackgroundTasks):
    """Trigger background retraining of Prophet, ARIMA, XGBoost, and LSTM models across all crops."""
    if _training_status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    background_tasks.add_task(_background_training_task)
    return {
        "message": "Price model training started in background.",
        "supported_crops": PRICE_PREDICTION_CROPS,
    }


@router.post("/train/{crop}", summary="Synchronously Retrain Models for a Single Crop")
def train_single_crop(crop: str):
    """Retrain models for a single crop synchronously."""
    crop_clean = crop.lower().strip()
    if crop_clean not in PRICE_PREDICTION_CROPS:
        raise HTTPException(status_code=422, detail=f"Unsupported crop '{crop}'. Choose: {PRICE_PREDICTION_CROPS}")

    try:
        res = train_price_models_for_crop(crop_clean)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed for {crop}: {str(e)}")
