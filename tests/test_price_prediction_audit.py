"""
test_price_prediction_audit.py — Comprehensive Audit & Verification Test Suite for AgroIntel Price Prediction.

Covers all 9 audit & improvement phases:
  Phase 1: Trace Current Mandi Price Source (Agmarknet 2.0 Live vs Cache vs Fallback)
  Phase 2: Current Price Correctness & Provenance (Crop, State, Market, ₹/q, Observation Date)
  Phase 3: Google Search Price Discrepancies & Market Distinctions
  Phase 4: 2024-2026 Dataset Coverage & Deduplication Audit
  Phase 5: Retraining & Validation of All 4 Models (XGBoost, ARIMA, Prophet, PyTorch LSTM)
  Phase 6: State-Specific Forecast Verification (Genuine State Differentiation, Encoder Isolation)
  Phase 7: Forecast Graph Data Integrity (Historical Observed Series vs ML Trajectory & Confidence Bounds)
  Phase 8: Sell/Hold Advisory Economic Audit (Storage Costs, Perishability, Explainability)
  Phase 9: Edge Cases, Unsupported Crops, and Robust Failure Handling
"""

import os
import json
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent / "backend"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "app" / "data"


# ==============================================================================
# PHASE 1 & 2: CURRENT MANDI PRICE SOURCE & PROVENANCE AUDIT
# ==============================================================================

def test_mandi_service_source_and_cache_detection():
    """Verify Agmarknet 2.0 source detection and caching policy."""
    from app.services.mandi_service import get_latest_price, MandiPriceResult

    res = get_latest_price("wheat", "Maharashtra")
    assert res is not None, "get_latest_price should return a MandiPriceResult"
    assert isinstance(res, MandiPriceResult)
    assert res.crop.lower() == "wheat"
    assert res.modal_price > 0, f"Expected positive modal price, got {res.modal_price}"
    assert "Agmarknet" in res.source or "data.gov.in" in res.source or "cache" in res.source
    assert res.arrival_date is not None
    assert isinstance(res.data_age_days, int)
    assert res.freshness_label in ("Fresh", "Recent", "Historical")


def test_current_price_provenance_and_metadata():
    """Verify that predict_price returns complete provenance without fabrications."""
    from app.ml.inference import predict_price

    res = predict_price("wheat", "Maharashtra", horizon_days=7)
    assert res["available"] is True
    assert res["current_price"] > 0
    assert res["price_data_source"] is not None
    assert res["observation_date"] is not None
    # Validate date format YYYY-MM-DD
    datetime.strptime(res["observation_date"], "%Y-%m-%d")
    assert res["market_name"] is not None
    assert res["data_age_days"] is not None


def test_statutory_fallback_has_zero_random_noise():
    """Verify that MSP fallback does NOT inject pseudo-random noise."""
    from app.data.ingestion import DataIngestion

    # Test an obscure state where live API and state-tail might not have records
    res1 = DataIngestion.fetch_live_market_data("wheat", "Goa")
    res2 = DataIngestion.fetch_live_market_data("wheat", "Goa")
    assert res1["current_price"] == res2["current_price"], "Fallback prices must be deterministic and repeatable"


# ==============================================================================
# PHASE 3: PRICE DISCREPANCY AUDIT & AGGREGATION LEVEL
# ==============================================================================

def test_wholesale_quintal_unit_and_state_basis():
    """Verify prices represent wholesale APMC modal prices in Rs/quintal."""
    from app.ml.inference import predict_price

    res_wheat_mh = predict_price("wheat", "Maharashtra", horizon_days=7)
    res_wheat_pb = predict_price("wheat", "Punjab", horizon_days=7)

    # Wholesale wheat price in India is realistically between 2,200 and 4,500 Rs/quintal
    assert 2000.0 <= res_wheat_mh["current_price"] <= 5000.0, f"Unexpected wheat price in MH: {res_wheat_mh['current_price']}"
    assert 2000.0 <= res_wheat_pb["current_price"] <= 4500.0, f"Unexpected wheat price in PB: {res_wheat_pb['current_price']}"


# ==============================================================================
# PHASE 4: 2024-2026 DATASET COVERAGE AUDIT
# ==============================================================================

def test_historical_datasets_2024_2026_coverage():
    """Verify real_historical_prices.csv and real_historical_prices_state.csv contain 2024-2026 records."""
    csv_crop = DATA_DIR / "real_historical_prices.csv"
    csv_state = DATA_DIR / "real_historical_prices_state.csv"

    assert csv_crop.exists(), "real_historical_prices.csv must exist"
    assert csv_state.exists(), "real_historical_prices_state.csv must exist"

    df_crop = pd.read_csv(csv_crop)
    df_crop["ds"] = pd.to_datetime(df_crop["ds"])
    assert df_crop["ds"].dt.year.max() == 2026, "Dataset must extend through 2026"
    assert 2024 in df_crop["ds"].dt.year.values
    assert 2025 in df_crop["ds"].dt.year.values
    assert 2026 in df_crop["ds"].dt.year.values

    # Check zero duplicates per (ds, crop)
    dups_crop = df_crop.duplicated(subset=["ds", "crop"]).sum()
    assert dups_crop == 0, f"Found {dups_crop} duplicates in real_historical_prices.csv"

    df_state = pd.read_csv(csv_state)
    df_state["ds"] = pd.to_datetime(df_state["ds"])
    assert df_state["ds"].dt.year.max() == 2026, "State dataset must extend through 2026"
    assert df_state["state"].nunique() >= 25, "State dataset must cover major Indian states"


# ==============================================================================
# PHASE 5: FOUR FORECASTING MODELS RETRAINING & ARTIFACT AUDIT
# ==============================================================================

@pytest.mark.parametrize("crop", ["wheat", "rice", "maize", "potato", "onion"])
def test_all_four_model_artifacts_exist(crop):
    """Verify that all 4 models + State XGBoost artifacts exist on disk."""
    xgboost_path = MODELS_DIR / f"xgboost_{crop}.pkl"
    arima_path = MODELS_DIR / f"arima_{crop}.pkl"
    prophet_path = MODELS_DIR / f"prophet_{crop}.pkl"
    lstm_pt_path = MODELS_DIR / f"lstm_{crop}.pt"
    lstm_scaler_path = MODELS_DIR / f"lstm_scaler_{crop}.pkl"
    state_xgb_path = MODELS_DIR / f"xgboost_state_{crop}.pkl"

    assert xgboost_path.exists(), f"XGBoost model missing for {crop}"
    assert arima_path.exists(), f"ARIMA model missing for {crop}"
    assert prophet_path.exists(), f"Prophet model missing for {crop}"
    assert lstm_pt_path.exists(), f"PyTorch LSTM model missing for {crop}"
    assert lstm_scaler_path.exists(), f"LSTM scaler missing for {crop}"
    assert state_xgb_path.exists(), f"State-aware XGBoost model missing for {crop}"


def test_model_registry_has_verified_lstm_metrics():
    """Verify model_registry.json contains genuine, evaluated LSTM metrics (< 999999.0)."""
    reg_path = MODELS_DIR / "model_registry.json"
    assert reg_path.exists()
    with open(reg_path, "r") as f:
        reg = json.load(f)

    for crop in ["wheat", "rice", "maize", "potato", "onion"]:
        crop_info = reg["registry"].get(crop, {})
        assert crop_info.get("lstm_availability") is True, f"LSTM not marked available for {crop}"
        lstm_mae = crop_info.get("all_models_mae", {}).get("lstm")
        assert lstm_mae is not None and lstm_mae < 500.0, f"Invalid LSTM MAE for {crop}: {lstm_mae}"


# ==============================================================================
# PHASE 6: STATE-SPECIFIC FORECAST VERIFICATION
# ==============================================================================

def test_state_specific_forecast_differentiation():
    """Verify that forecasts differ meaningfully across states and do not leak across states."""
    from app.ml.inference import predict_price, clear_model_cache
    clear_model_cache()

    res_mh = predict_price("wheat", "Maharashtra", horizon_days=15)
    res_pb = predict_price("wheat", "Punjab", horizon_days=15)
    res_ka = predict_price("wheat", "Karnataka", horizon_days=15)

    assert res_mh["forecast_scope"] != "National APMC Mandi Benchmark"
    assert res_pb["forecast_scope"] != "National APMC Mandi Benchmark"

    # Prices should reflect regional economic logistics (Punjab production surplus vs Karnataka consumer deficit)
    assert res_mh["predictions"] != res_pb["predictions"], "State forecasts must differ between MH and PB"
    assert res_pb["predictions"] != res_ka["predictions"], "State forecasts must differ between PB and KA"


# ==============================================================================
# PHASE 7: GRAPH DATA INTEGRITY & UNANCHORED TRAJECTORY AUDIT
# ==============================================================================

def test_graph_data_historical_and_forecast_separation():
    """Verify historical series and forecast series are clean, separated, and unanchored."""
    from app.ml.inference import predict_price

    res = predict_price("wheat", "Maharashtra", horizon_days=30)
    hist_series = res["historical_series"]
    fcast_series = res["forecast_series"]

    assert len(hist_series) >= 10, "Historical series should have at least 10 observations"
    assert len(fcast_series) == 30, "Forecast series should have exactly 30 observations"

    # Verify chronological ordering
    last_hist_date = hist_series[-1]["date"]
    first_fcast_date = fcast_series[0]["date"]
    assert last_hist_date < first_fcast_date, f"Historical date {last_hist_date} must precede forecast start {first_fcast_date}"

    # Verify uncertainty bounds
    for pt in fcast_series:
        assert pt["lower"] <= pt["price"] <= pt["upper"], f"Invalid confidence bound: lower={pt['lower']}, price={pt['price']}, upper={pt['upper']}"


@pytest.mark.parametrize("horizon", [7, 15, 30, 60, 90])
def test_all_forecast_horizons_supported(horizon):
    """Verify horizons 7, 15, 30, 60, 90 return exact counts."""
    from app.ml.inference import predict_price

    res = predict_price("wheat", "Maharashtra", horizon_days=horizon)
    assert len(res["predictions"]) == horizon
    assert len(res["date_labels"]) == horizon
    assert len(res["forecast_series"]) == horizon


# ==============================================================================
# PHASE 8: SELL / HOLD ADVISORY ECONOMIC AUDIT
# ==============================================================================

def test_sell_hold_storage_cost_deduction():
    """Verify storage holding costs are factored into sell/hold decision."""
    from app.ml.inference import predict_price

    res_wheat = predict_price("wheat", "Maharashtra", horizon_days=30)
    res_onion = predict_price("onion", "Maharashtra", horizon_days=30)

    # Wheat holding rate is ~2.0% for 30d, Onion is ~4.0% for 30d (higher perishability)
    assert res_wheat["storage_cost_pct"] == 2.0
    assert res_onion["storage_cost_pct"] == 4.0

    # Net change must equal percent_change - storage_cost_pct
    expected_net = round(res_wheat["percent_change"] - res_wheat["storage_cost_pct"], 2)
    assert res_wheat["net_expected_change"] == expected_net

    assert res_wheat["recommendation"] in ("HOLD", "SELL", "WAIT")
    assert len(res_wheat["recommendation_reason"]) > 20


# ==============================================================================
# PHASE 9: UNSUPPORTED CROPS & EDGE CASES
# ==============================================================================

def test_unsupported_crop_returns_honest_unavailable():
    """Verify unsupported crops return honest unavailable status rather than inventing prices."""
    from app.api.endpoints import predict

    res = predict(crop="dragonfruit", state="Karnataka", horizon_days=30)
    assert res["available"] is False
    assert "unavailable" in res["message"].lower()
    assert res["forecast"]["available"] is False


# ==============================================================================
# REMEDIATION AUDIT: TESTS FOR FIXES 1 THROUGH 8
# ==============================================================================

def test_remediated_price_semantics_and_no_synthetic_bounds():
    """Fix 1: Mandi price has explicit price_type, period dates, unit, and NO synthetic +/-5% bounds."""
    from app.services.mandi_service import get_latest_price
    from app.api.endpoints import predict

    res_mandi = get_latest_price("wheat", "Maharashtra")
    assert res_mandi is not None
    assert hasattr(res_mandi, "price_type")
    assert hasattr(res_mandi, "period_start")
    assert hasattr(res_mandi, "period_end")
    assert res_mandi.unit == "₹/quintal"
    # Never fabricate min/max as modal +/- 5%
    if res_mandi.min_price is not None and res_mandi.max_price is not None:
        assert not (round(res_mandi.min_price, 2) == round(res_mandi.modal_price * 0.95, 2) and
                    round(res_mandi.max_price, 2) == round(res_mandi.modal_price * 1.05, 2)), \
            "Fabricated +/-5% bounds detected!"

    api_res = predict(crop="wheat", state="Maharashtra", horizon_days=30)
    assert api_res["price_type"] is not None
    assert api_res["unit"] == "₹/quintal"
    assert "market" in api_res
    assert api_res["market"]["unit"] == "₹/quintal"


def test_state_aware_xgboost_metrics_populated():
    """Fix 3: State-aware XGBoost returns genuine non-null MAE & RMSE in model_comparison."""
    from app.api.endpoints import predict

    res = predict(crop="wheat", state="Maharashtra", horizon_days=30)
    comparison = res.get("model_comparison", {})
    assert "xgboost_state" in comparison, "xgboost_state missing from model_comparison"
    xgb_state = comparison["xgboost_state"]
    assert xgb_state["mae"] is not None and xgb_state["mae"] > 0, f"MAE must be non-null positive, got {xgb_state['mae']}"
    assert xgb_state["rmse"] is not None and xgb_state["rmse"] > 0, f"RMSE must be non-null positive, got {xgb_state['rmse']}"
    assert xgb_state["sample_count"] is not None and xgb_state["sample_count"] > 1000
    assert "State-Level Test Population" in xgb_state["evaluation_scope"]


def test_reference_benchmark_transparency():
    """Fix 5: Reference benchmark is present in model_comparison and benchmark superiority is transparently surfaced."""
    from app.api.endpoints import predict

    res_wheat = predict(crop="wheat", state="Maharashtra", horizon_days=30)
    comp_wheat = res_wheat.get("model_comparison", {})
    assert "reference_benchmark" in comp_wheat
    assert "Naive 7-Day Moving Average" in comp_wheat["reference_benchmark"]["label"]
    assert comp_wheat["reference_benchmark"]["sample_count"] == 60

    # For Rice, Naive MA-7 MAE 31.71 outperforms all models (Prophet 44.07, XGBoost 56.81, LSTM 66.05)
    res_rice = predict(crop="rice", state="Karnataka", horizon_days=30)
    comp_rice = res_rice.get("model_comparison", {})
    ref_rice = comp_rice.get("reference_benchmark", {})
    assert ref_rice.get("benchmark_outperformed") is True, "Rice reference benchmark must transparently show benchmark_outperformed=True"


def test_uncertainty_interval_labeling_and_uncalibrated_status():
    """Fix 6: Uncertainty intervals are labeled Estimated Error Range with uncalibrated status."""
    from app.api.endpoints import predict

    res = predict(crop="wheat", state="Maharashtra", horizon_days=30)
    assert res["interval_type"] == "Estimated Error Range (Holdout RMSE Basis)"
    assert res["interval_calibrated"] is False
    forecast_series = res["forecast_series"]
    assert len(forecast_series) == 30
    for pt in forecast_series:
        assert pt["interval_type"] == "Estimated Error Range (Holdout RMSE Basis)"
        assert pt["lower"] <= pt["price"] <= pt["upper"]


def test_dataset_provenance_manifest_integrity():
    """Fix 7: dataset_provenance_manifest.json exists and accurately details datasets and zero duplicates."""
    manifest_path = DATA_DIR / "dataset_provenance_manifest.json"
    assert manifest_path.exists(), "dataset_provenance_manifest.json must exist"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["manifest_version"] == "1.0.0"
    assert "national" in manifest["datasets"]
    assert "state" in manifest["datasets"]
    assert manifest["datasets"]["national"]["duplicate_rows"] == 0
    assert manifest["datasets"]["state"]["duplicate_rows"] == 0

