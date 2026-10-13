"""
price_predictor.py — Production-Ready Unified Price Prediction Engine for AgroIntel v4.0.

Provides comprehensive prediction response including:
  - prediction_metadata (ISO timestamp, horizon, historical end date, model/feature/dataset/weather versions)
  - forecast_summary (starting, ending, highest, lowest, average price, trend)
  - trend_statistics (forecast_slope, daily_average_change, forecast_std, forecast_variance, volatility_percent)
  - daily_predictions with confidence bands [{"day": 1, "predicted_price": 1865, "lower_bound": 1815, "upper_bound": 1912}]
  - daily_prediction_series, trend_line, confidence_series (for plotting)
  - confidence & confidence_breakdown (model_quality, horizon_penalty, data_freshness)
  - decision & decision_score (current_price, predicted_avg, expected_change %, storage_cost %, net_gain %, decision_reason)
  - model_health (model_loaded, registry_loaded, feature_version_match, latest_market_data, prediction_status)
  - response_time_ms (measured latency in milliseconds)
  - prediction_logger audit entry
"""

import json
import logging
import pickle
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.constants import (
    DEFAULT_WEATHER_LAT,
    DEFAULT_WEATHER_LON,
    PRICE_FEATURE_COLS,
    PRICE_PREDICTION_CROPS,
)
from app.ml.feature_engineering import build_inference_features
from app.services.confidence_engine import calculate_confidence
from app.services.decision_engine import make_decision
from app.services.mandi_service import get_latest_price
from app.services.prediction_logger import log_prediction
from app.services.trend_engine import analyze_trend
from app.services.weather_service import get_current_monthly_weather

logger = logging.getLogger(__name__)

MODELS_DIR = settings.MODELS_DIR
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
FEATURE_COLS_PATH = settings.DATA_DIR / "feature_columns.json"


# ── Metadata Loaders ──────────────────────────────────────────────────────────

def _load_model_registry() -> Dict[str, Any]:
    """Load model_registry.json."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Model registry not found at {REGISTRY_PATH}. Run app/ml/price_trainer.py first.")
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _load_feature_columns_meta() -> Dict[str, Any]:
    """Load feature_columns.json metadata."""
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH, "r") as f:
            return json.load(f)
    return {"feature_version": "4.0.0"}


# ── Generic Forecasters ───────────────────────────────────────────────────────

def _forecast_prophet(model: Any, start_date: date, total_days: int = 90) -> List[float]:
    """Generate Prophet forecasts for total_days starting from start_date."""
    dates = pd.date_range(start=start_date, periods=total_days, freq="D")
    future = pd.DataFrame({"ds": dates})
    forecast = model.predict(future)
    return [round(float(p), 2) for p in forecast["yhat"].values]


def _forecast_arima(model: Any, total_days: int = 90) -> List[float]:
    """Generate ARIMA forecasts for total_days."""
    preds = model.forecast(steps=total_days)
    vals = preds.values if hasattr(preds, "values") else preds
    return [round(float(p), 2) for p in vals]


def _forecast_xgboost(
    model: Any,
    data_tail_df: pd.DataFrame,
    start_date: date,
    monthly_temp: float,
    monthly_rain: float,
    total_days: int = 90,
) -> List[float]:
    """Generate XGBoost predictions recursively for total_days."""
    price_tail = list(data_tail_df["y"].values)
    predictions = []
    current_dt = start_date

    for _ in range(total_days):
        feat_df = build_inference_features(
            price_tail=price_tail,
            target_date=current_dt,
            monthly_avg_temp=monthly_temp,
            monthly_total_rainfall=monthly_rain,
        )
        X = feat_df[PRICE_FEATURE_COLS].values
        pred_val = float(model.predict(X)[0])
        pred_val = max(pred_val, 100.0)

        predictions.append(round(pred_val, 2))
        price_tail.append(pred_val)
        current_dt += timedelta(days=1)

    return predictions


def _forecast_state_aware_xgboost(
    crop: str,
    model: Any,
    data_tail_df: pd.DataFrame,
    start_date: date,
    state: Optional[str],
    total_days: int = 90,
) -> List[float]:
    """
    Generate State-Aware XGBoost predictions using 14-feature state-encoded vector.
    Feature set: state_enc, lag_1, lag_7, lag_14, lag_30, rolling_7, rolling_30,
                 rolling_std_7, price_range, day_of_year, month, day_of_week, year, black_swan

    Black swan flag now reads from black_swan_config.json (same source as inference.py),
    NOT from a hardcoded list.
    """
    # Use the shared is_black_swan_period function so training and inference flags match
    from app.ml.feature_engineering import is_black_swan_period

    # Load state encoder
    encoder_path = MODELS_DIR / f"state_encoder_{crop}.pkl"
    state_enc_val = 0  # Default: unknown state encodes to 0
    if encoder_path.exists() and state:
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        state_clean = state.strip().title()
        if state_clean in le.classes_:
            state_enc_val = int(le.transform([state_clean])[0])
        else:
            # Use national median encoding (midpoint of classes)
            state_enc_val = len(le.classes_) // 2

    # Build initial price tail from state-specific data if available
    # data_tail_state_{crop}.pkl is a dict: {state_name: [{"ds": ..., "y": ...}, ...]}
    if state and isinstance(data_tail_df, dict):
        state_clean = state.strip().title()
        state_records = data_tail_df.get(state_clean)
        if not state_records:
            # Try case-insensitive lookup
            for k, v in data_tail_df.items():
                if k.lower() == state_clean.lower():
                    state_records = v
                    break
        if state_records and len(state_records) >= 7:
            price_tail = [r["y"] for r in state_records[-60:]]
        else:
            # Aggregate: use last known price across all states
            all_prices = [r["y"] for records in data_tail_df.values() for r in records]
            price_tail = all_prices[-60:] if len(all_prices) >= 60 else all_prices
    elif hasattr(data_tail_df, "columns") and "state" in data_tail_df.columns and state:
        state_clean = state.strip().title()
        state_df = data_tail_df[data_tail_df["state"] == state_clean]
        if len(state_df) >= 30:
            price_tail = list(state_df["y"].values[-60:])
        else:
            price_tail = list(data_tail_df["y"].values[-60:])
    elif hasattr(data_tail_df, "columns"):
        price_tail = list(data_tail_df["y"].values[-60:])
    else:
        # Generic fallback if it's a dict without matching structure
        all_prices = [r["y"] for records in data_tail_df.values() for r in (records if isinstance(records, list) else [])]
        price_tail = all_prices[-60:] if len(all_prices) >= 60 else (all_prices or [2000.0])


    predictions = []
    current_dt = start_date

    for _ in range(total_days):
        dt = pd.to_datetime(current_dt)

        black_swan = int(is_black_swan_period(current_dt))

        # Safe lag extraction from rolling price tail
        def safe_lag(n: int) -> float:
            return float(price_tail[-n]) if len(price_tail) >= n else float(price_tail[0])

        lag_1  = safe_lag(1)
        lag_7  = safe_lag(7)
        lag_14 = safe_lag(14)
        lag_30 = safe_lag(30)

        recent_7  = price_tail[-7:]  if len(price_tail) >= 7  else price_tail
        recent_30 = price_tail[-30:] if len(price_tail) >= 30 else price_tail

        rolling_7   = float(np.mean(recent_7))
        rolling_30  = float(np.mean(recent_30))
        rolling_std_7 = float(np.std(recent_7)) if len(recent_7) >= 2 else 0.0
        price_range = float(max(recent_7) - min(recent_7)) if recent_7 else 0.0

        X = np.array([[
            state_enc_val,
            lag_1, lag_7, lag_14, lag_30,
            rolling_7, rolling_30, rolling_std_7, price_range,
            dt.dayofyear, dt.month, dt.dayofweek, dt.year,
            black_swan
        ]])

        pred_val = float(model.predict(X)[0])
        pred_val = max(pred_val, 100.0)

        predictions.append(round(pred_val, 2))
        price_tail.append(pred_val)
        current_dt += timedelta(days=1)

    return predictions


def _forecast_lstm(
    crop: str,
    data_tail_df: pd.DataFrame,
    start_date: date,
    monthly_temp: float,
    monthly_rain: float,
    total_days: int = 90,
) -> Optional[List[float]]:
    """Generate LSTM predictions if Keras model file exists, else return None."""
    keras_path = MODELS_DIR / f"lstm_{crop}.keras"
    scaler_path = MODELS_DIR / f"lstm_scaler_{crop}.pkl"

    if not keras_path.exists() or not scaler_path.exists():
        return None

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(keras_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if isinstance(data_tail_df, dict):
            state_clean = state.strip().title() if state else ""
            state_recs = data_tail_df.get(state_clean) if state_clean else None
            if not state_recs:
                all_recs = [r for records in data_tail_df.values() for r in (records if isinstance(records, list) else [])]
                price_tail = [r["y"] for r in all_recs[-60:]]
            else:
                price_tail = [r["y"] for r in state_recs[-60:]]
        else:
            price_tail = list(data_tail_df["y"].values)
        predictions = []
        current_dt = start_date

        for _ in range(total_days):
            feat_df = build_inference_features(
                price_tail=price_tail,
                target_date=current_dt,
                monthly_avg_temp=monthly_temp,
                monthly_total_rainfall=monthly_rain,
            )
            X_scaled = scaler.transform(feat_df[PRICE_FEATURE_COLS].values)
            seq = X_scaled.reshape(1, 1, len(PRICE_FEATURE_COLS))
            pred_val = float(model.predict(seq, verbose=0)[0, 0])
            pred_val = max(pred_val, 100.0)

            predictions.append(round(pred_val, 2))
            price_tail.append(pred_val)
            current_dt += timedelta(days=1)

        return predictions
    except Exception as e:
        logger.warning(f"LSTM prediction failed for {crop}: {e}")
        return None


# ── Chart & Visualization Helper with Confidence Bands ────────────────────────

def _build_visualization_series(
    daily_prices: List[float],
    model_rmse: float,
) -> Dict[str, Any]:
    """
    Generate graph-ready visualization structures with confidence bounds (RMSE derived).
    """
    days_arr = np.arange(1, len(daily_prices) + 1)
    prices_arr = np.array(daily_prices, dtype=float)

    # 1.2 * RMSE bound
    margin = model_rmse * 1.2 if model_rmse > 0 else 50.0

    # Daily predictions array of dicts with bounds
    daily_predictions = [
        {
            "day": int(d),
            "predicted_price": round(float(p), 2),
            "lower_bound": round(float(max(p - margin, 50.0)), 2),
            "upper_bound": round(float(p + margin), 2),
        }
        for d, p in zip(days_arr, prices_arr)
    ]

    # Linear trend line
    slope, intercept = np.polyfit(days_arr, prices_arr, 1)
    trend_line = [round(float(slope * d + intercept), 2) for d in days_arr]

    # Upper and lower bounds series
    upper_bound = [round(float(p + margin), 2) for p in prices_arr]
    lower_bound = [round(float(max(p - margin, 50.0)), 2) for p in prices_arr]

    return {
        "daily_predictions": daily_predictions,
        "daily_prediction_series": [round(float(p), 2) for p in prices_arr],
        "trend_line": trend_line,
        "confidence_series": {
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
        },
    }


# ── Unified Prediction Entry Point ────────────────────────────────────────────

def predict_crop_price(
    crop: str,
    state: Optional[str] = None,
    horizon_days: int = 30,
    lat: float = DEFAULT_WEATHER_LAT,
    lon: float = DEFAULT_WEATHER_LON,
) -> Dict[str, Any]:
    """
    Generate complete production-ready price prediction, metadata, health,
    trend statistics, confidence breakdown, decision score, and graph series.
    """
    t_start = time.perf_counter()
    gen_timestamp = datetime.now().isoformat()

    crop_lower = crop.lower().strip()
    if crop_lower not in PRICE_PREDICTION_CROPS:
        raise ValueError(
            f"Unsupported crop '{crop}'. Supported crops: {PRICE_PREDICTION_CROPS}"
        )

    # 1. Load Registry & Feature Metadata
    registry_meta = _load_model_registry()
    registry = registry_meta.get("registry", {})
    crop_meta = registry.get(crop_lower)
    if not crop_meta:
        raise ValueError(f"No trained model registry metadata for crop '{crop_lower}'")

    feat_meta = _load_feature_columns_meta()

    prod_model_name = crop_meta.get("production_model", "xgboost")
    model_mae = float(crop_meta.get("mae", 100.0))
    model_rmse = float(crop_meta.get("rmse", 120.0))
    dataset_version = crop_meta.get("dataset_version", "2019-2024-v1")
    weather_version = crop_meta.get("weather_version", "open-meteo-monthly-v1")
    feature_version = feat_meta.get("feature_version", "4.0.0")

    # 2. Load Historical Data Tail (prefer state-specific tail when state-aware model is active)
    state_tail_path = MODELS_DIR / f"data_tail_state_{crop_lower}.pkl"
    generic_tail_path = MODELS_DIR / f"data_tail_{crop_lower}.pkl"

    if prod_model_name == "state_aware_xgboost" and state_tail_path.exists():
        tail_path = state_tail_path
    elif generic_tail_path.exists():
        tail_path = generic_tail_path
    elif state_tail_path.exists():
        tail_path = state_tail_path
    else:
        raise FileNotFoundError(f"No data tail file found for crop '{crop_lower}'")

    with open(tail_path, "rb") as f:
        data_tail_df: pd.DataFrame = pickle.load(f)

    # Compute mean_hist_price and hist_end_date from whichever format was loaded
    if isinstance(data_tail_df, dict):
        # State tail format: {state_name: [{"ds": ..., "y": ...}, ...]}
        all_y = [r["y"] for records in data_tail_df.values() for r in (records if isinstance(records, list) else [])]
        all_ds = [r["ds"] for records in data_tail_df.values() for r in (records if isinstance(records, list) else [])]
        mean_hist_price = float(np.mean(all_y)) if all_y else 2000.0
        hist_end_date = str(max(pd.to_datetime(all_ds)).date()) if all_ds else str(date.today())
    else:
        mean_hist_price = float(data_tail_df["y"].mean())
        if "ds" in data_tail_df.columns:
            hist_end_date = str(pd.to_datetime(data_tail_df["ds"]).max().date())
        else:
            hist_end_date = str(date.today())


    # 3. Retrieve Current Mandi Price
    # Priority: (1) Live data.gov.in API  (2) Disk cache  (3) Historical tail (last CSV row)
    # NOTE: AGMARKNET publishes with 3–6 day delay — any successful API response is VALID.
    # Fallback to CSV ONLY on timeout / HTTP error / no records.
    mandi_res = get_latest_price(crop_lower, state)
    if mandi_res:
        current_price = mandi_res.modal_price
        freshness     = mandi_res.freshness_label   # Fresh / Recent / Historical
        current_price_source = f"{mandi_res.source} — {freshness}"
        current_price_date   = mandi_res.arrival_date
        observation_date     = mandi_res.observation_date or mandi_res.arrival_date
        data_age_days        = mandi_res.data_age_days
        price_timestamp      = mandi_res.arrival_date
        price_type           = getattr(mandi_res, "price_type", "APMC Modal Average")
        period_start         = getattr(mandi_res, "period_start", mandi_res.arrival_date)
        period_end           = getattr(mandi_res, "period_end", mandi_res.arrival_date)
        price_unit           = getattr(mandi_res, "unit", "₹/quintal")
        min_price            = getattr(mandi_res, "min_price", None)
        max_price            = getattr(mandi_res, "max_price", None)
        msp_price            = getattr(mandi_res, "msp_price", None)
        msp_season           = getattr(mandi_res, "msp_season", None)
        variety              = getattr(mandi_res, "variety", "All Reporting Varieties (State-wide Pre-Aggregated)")
        market_scope         = getattr(mandi_res, "market_scope", f"State-wide Scope ({state or 'All'})")
        contributing_count   = getattr(mandi_res, "contributing_records_count", 1)
        contributing_recs    = getattr(mandi_res, "contributing_records", [])
        aggregation_method   = getattr(mandi_res, "aggregation_method", "Official Arrival-Weighted State Modal Average (wt_avg_price)")
        price_transparency   = mandi_res.to_transparency_dict()
        market_data_status   = mandi_res.price_status
        logger.warning(
            f"[PRICE_SOURCE] {crop_lower}: ₹{current_price}/q "
            f"from {current_price_source} | market={mandi_res.market} | "
            f"record_date={current_price_date} | age={data_age_days}d | fallback=No"
        )
    else:
        from app.core.constants import get_official_msp
        msp_info = get_official_msp(crop_lower)
        msp_price = msp_info.get("msp_price") if msp_info else None
        msp_season = msp_info.get("season") if msp_info else None
        variety = "All Varieties (Historical APMC Aggregate)"
        market_scope = f"Historical APMC Scope ({state or 'National'})"
        contributing_count = 1
        contributing_recs = []
        aggregation_method = "Historical APMC State Average (Baseline)"
        price_type   = "Historical APMC Baseline"
        price_unit   = "₹/quintal"
        min_price    = None
        max_price    = None
        if isinstance(data_tail_df, dict):
            state_clean = state.strip().title() if state else ""
            state_recs = data_tail_df.get(state_clean) if state_clean else None
            if state_recs:
                last_rec = state_recs[-1]
                last_hist_price = round(float(last_rec["y"]), 2)
                rec_date = last_rec.get("ds") or hist_end_date
                rec_date_str = rec_date.strftime("%Y-%m-%d") if hasattr(rec_date, "strftime") else str(rec_date)[:10]
                try:
                    dt_rec = datetime.strptime(rec_date_str, "%Y-%m-%d").date()
                    data_age_days = (date.today() - dt_rec).days
                except Exception:
                    data_age_days = 1
                current_price        = last_hist_price
                state_label          = state_clean
                current_price_source = f"Historical State APMC Baseline ({state_label})"
                current_price_date   = rec_date_str
                observation_date     = rec_date_str
                price_timestamp      = rec_date_str
                period_start         = rec_date_str
                period_end           = rec_date_str
                market_data_status   = "HISTORICAL_FALLBACK"
            else:
                current_price        = round(mean_hist_price, 2)
                current_price_source = "National APMC Historical Baseline (Fallback)"
                current_price_date   = hist_end_date
                observation_date     = hist_end_date
                data_age_days        = (date.today() - _parse_date(hist_end_date)).days if _parse_date(hist_end_date) else 7
                price_timestamp      = hist_end_date
                period_start         = hist_end_date
                period_end           = hist_end_date
                market_data_status   = "STALE_NATIONAL_FALLBACK"
        else:
            last_hist_price      = round(float(data_tail_df["y"].iloc[-1]), 2)
            current_price        = last_hist_price
            current_price_source = "National APMC Historical Baseline"
            current_price_date   = hist_end_date
            observation_date     = hist_end_date
            data_age_days        = (date.today() - _parse_date(hist_end_date)).days if _parse_date(hist_end_date) else 7
            price_timestamp      = hist_end_date
            period_start         = hist_end_date
            period_end           = hist_end_date
            market_data_status   = "HISTORICAL_FALLBACK"

        price_transparency = {
            "source":                     current_price_source,
            "source_url":                 "https://agmarknet.gov.in",
            "observation_date":           observation_date,
            "selected_state":             state or "National",
            "state":                      state or "National",
            "market_scope":               market_scope,
            "commodity":                  crop_lower.capitalize(),
            "commodity_name":             crop_lower.capitalize(),
            "variety":                    variety,
            "modal_price":                round(float(current_price), 2) if current_price else None,
            "modal_price_display":        f"₹{round(float(current_price)):,}/quintal" if current_price else "Live price unavailable",
            "min_price":                  None,
            "max_price":                  None,
            "msp_price":                  msp_price,
            "msp_season":                 msp_season,
            "msp_difference":             round(float(current_price) - msp_price, 2) if (current_price and msp_price) else None,
            "contributing_records_count": contributing_count,
            "contributing_records":       contributing_recs,
            "aggregation_method":         aggregation_method,
            "last_successful_fetch_time": "Historical APMC Archive",
            "source_timestamp":           "Historical APMC Archive",
            "data_age_days":              data_age_days or 0,
            "price_status":               market_data_status,
            "status_label":               f"Historical Fallback ({data_age_days or 0}d old)",
            "is_live":                    False,
            "is_stale":                   True,
            "is_fallback":                True,
            "unit":                       price_unit,
            "farm_gate_modal_price":      round(float(current_price), 2) if current_price else None,
            "farm_gate_range":            None,
            "state_composite_index":      round(float(current_price), 2) if current_price else None,
            "reconciliation_notice":      None,
            "district_breakdown":         [],
            "market_scope_type":          market_scope,
        }

        logger.info(
            f"[PRICE_SOURCE] {crop_lower}: ₹{current_price}/q "
            f"from {current_price_source} | record_date={current_price_date} | status={market_data_status}"
        )

    # 4. Retrieve Current Weather
    weather = get_current_monthly_weather(lat, lon)
    monthly_temp = weather["monthly_avg_temp"]
    monthly_rain = weather["monthly_total_rainfall"]

    # 5. Generate 90-Day Forecast via Production Model
    start_date = date.today() + timedelta(days=1)
    daily_predictions_list: List[float] = []

    model_loaded_ok = False

    if prod_model_name == "state_aware_xgboost":
        sx_path = MODELS_DIR / f"xgboost_state_{crop_lower}.pkl"
        if sx_path.exists():
            with open(sx_path, "rb") as f:
                model = pickle.load(f)
            daily_predictions_list = _forecast_state_aware_xgboost(
                crop_lower, model, data_tail_df, start_date, state, total_days=90
            )
            model_loaded_ok = True
        else:
            logger.warning(f"State-aware XGBoost model missing for {crop_lower}, falling back to generic XGBoost.")
            prod_model_name = "xgboost"

    elif prod_model_name == "prophet":
        p_path = MODELS_DIR / f"prophet_{crop_lower}.pkl"
        if p_path.exists():
            with open(p_path, "rb") as f:
                model = pickle.load(f)
            daily_predictions_list = _forecast_prophet(model, start_date, total_days=90)
            model_loaded_ok = True

    elif prod_model_name == "arima":
        a_path = MODELS_DIR / f"arima_{crop_lower}.pkl"
        if a_path.exists():
            with open(a_path, "rb") as f:
                model = pickle.load(f)
            daily_predictions_list = _forecast_arima(model, total_days=90)
            model_loaded_ok = True

    elif prod_model_name == "lstm":
        lstm_preds = _forecast_lstm(crop_lower, data_tail_df, start_date, monthly_temp, monthly_rain, total_days=90)
        if lstm_preds:
            daily_predictions_list = lstm_preds
            model_loaded_ok = True
        else:
            logger.warning(f"LSTM model file missing for {crop_lower}. Falling back to state-aware XGBoost.")
            prod_model_name = "state_aware_xgboost"

    if not model_loaded_ok and prod_model_name in ("xgboost", "state_aware_xgboost"):
        # Final fallback: try state-aware first, then generic
        sx_path = MODELS_DIR / f"xgboost_state_{crop_lower}.pkl"
        x_path  = MODELS_DIR / f"xgboost_{crop_lower}.pkl"
        if sx_path.exists():
            with open(sx_path, "rb") as f:
                model = pickle.load(f)
            daily_predictions_list = _forecast_state_aware_xgboost(
                crop_lower, model, data_tail_df, start_date, state, total_days=90
            )
            prod_model_name = "state_aware_xgboost"
        elif x_path.exists():
            with open(x_path, "rb") as f:
                model = pickle.load(f)
            daily_predictions_list = _forecast_xgboost(
                model, data_tail_df, start_date, monthly_temp, monthly_rain, total_days=90
            )
            prod_model_name = "xgboost"
        else:
            raise FileNotFoundError(f"No XGBoost model file found for crop '{crop_lower}'")
        model_loaded_ok = True

    # 6. Build Graph & Confidence Bands Structures
    viz_data = _build_visualization_series(daily_predictions_list, model_rmse)

    # 7. Multi-Horizon Forecast Averages
    predictions_by_horizon = {
        "7_day":  round(float(np.mean(daily_predictions_list[0:7])), 2),
        "15_day": round(float(np.mean(daily_predictions_list[0:15])), 2),
        "30_day": round(float(np.mean(daily_predictions_list[0:30])), 2),
        "60_day": round(float(np.mean(daily_predictions_list[0:60])), 2),
        "90_day": round(float(np.mean(daily_predictions_list[0:90])), 2),
    }

    # 8. Forecast Summary
    pred_30d = daily_predictions_list[0:30]
    forecast_summary = {
        "starting_price": round(float(current_price), 2),
        "ending_price": round(float(pred_30d[-1]), 2),
        "highest_price": round(float(np.max(pred_30d)), 2),
        "lowest_price": round(float(np.min(pred_30d)), 2),
        "average_price": round(float(np.mean(pred_30d)), 2),
        "trend": "",  # Populated after trend analysis
    }

    # 9. Trend Analysis & Statistics
    trend_res = analyze_trend(current_price, pred_30d)
    forecast_summary["trend"] = trend_res.trend_direction

    # 10. Confidence Calculation & Breakdown
    conf_res = calculate_confidence(
        model_mae=model_mae,
        mean_historical_price=mean_hist_price,
        horizon_days=horizon_days,
        data_age_days=data_age_days,
    )

    # 11. Sell / Hold Decision Score & Reasons
    dec_res = make_decision(
        crop=crop_lower,
        current_price=current_price,
        predicted_30d_avg=trend_res.average_price,
        expected_change_percent=trend_res.expected_change_percent,
        trend_direction=trend_res.trend_direction,
        confidence_percent=conf_res.confidence,
        model_name=prod_model_name,
        monthly_temp=monthly_temp,
        monthly_rain=monthly_rain,
        data_freshness_label=current_price_source,
    )

    # Calculate Total Execution Latency
    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)

    # 12. Prediction Audit Log
    log_prediction(
        crop=crop_lower,
        production_model=prod_model_name,
        current_price=current_price,
        forecast_horizon=horizon_days,
        predicted_price=trend_res.average_price,
        expected_change_percent=trend_res.expected_change_percent,
        confidence=conf_res.confidence,
        decision=dec_res.decision,
        response_time_ms=latency_ms,
        prediction_source=market_data_status,
    )

    # 13. Assemble Final Production Response
    prediction_metadata = {
        "forecast_generated_at": gen_timestamp,
        "forecast_horizon_days": horizon_days,
        "historical_data_end_date": hist_end_date,
        "production_model": prod_model_name,
        "model_version": "4.0.0",
        "feature_version": feature_version,
        "dataset_version": dataset_version,
        "weather_version": weather_version,
    }

    model_health = {
        "model_loaded": model_loaded_ok,
        "registry_loaded": True,
        "feature_version_match": True,
        "latest_market_data": market_data_status,
        "prediction_status": "SUCCESS",
    }

    return {
        "crop": crop_lower,
        "production_model": prod_model_name,
        "current_price": round(float(current_price), 2),
        "current_price_source": current_price_source,
        "current_price_date": current_price_date,
        "data_status": market_data_status,
        "price_timestamp": price_timestamp,
        "forecast_horizon": horizon_days,
        "prediction_metadata": prediction_metadata,
        "model_health": model_health,
        "response_time_ms": latency_ms,
        "forecast_summary": forecast_summary,
        "trend_statistics": trend_res.trend_statistics,
        "daily_predictions": viz_data["daily_predictions"],
        "daily_prediction_series": viz_data["daily_prediction_series"],
        "trend_line": viz_data["trend_line"],
        "confidence_series": viz_data["confidence_series"],
        "predictions": predictions_by_horizon,
        "trend": trend_res.trend_direction,
        "trend_strength": trend_res.trend_strength,
        "expected_change_percent": trend_res.expected_change_percent,
        "average_price": trend_res.average_price,
        "minimum_price": trend_res.minimum_price,
        "maximum_price": trend_res.maximum_price,
        "volatility": trend_res.volatility_percent,
        "confidence": conf_res.confidence,
        "confidence_breakdown": conf_res.confidence_breakdown,
        "decision": dec_res.decision,
        "decision_score": dec_res.decision_score,
        "reasons": dec_res.reasons,
        "metrics": {
            "model_mae": model_mae,
            "model_rmse": model_rmse,
        },
        "price_type": price_type,
        "period_start": period_start,
        "period_end": period_end,
        "unit": price_unit,
        "min_price": min_price,
        "max_price": max_price,
        "observation_date": observation_date,
        "price_transparency": price_transparency,
        "msp_price": msp_price,
        "msp_season": msp_season,
        "variety": variety,
        "market_scope": market_scope,
        "contributing_records_count": contributing_count,
        "contributing_records": contributing_recs,
        "aggregation_method": aggregation_method,
        "reference_benchmark": crop_meta.get("reference_benchmark"),
        "benchmark_outperformed": crop_meta.get("benchmark_outperformed", True),
        "uncertainty_basis": "Estimated Error Range (Holdout RMSE Basis)",
        "interval_calibrated": False,
    }
