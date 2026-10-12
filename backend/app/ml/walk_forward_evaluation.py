"""
walk_forward_evaluation.py — Comprehensive Multi-Horizon Walk-Forward Evaluation

Evaluates 1-, 7-, 15-, 30-, 60-, and 90-day forecasts on untouched chronological
rolling origins using only data available at each forecast origin.

Models evaluated on identical timestamps and targets:
  - XGBoost
  - ARIMA
  - Prophet
  - PyTorch LSTM
  - Reference Benchmark (Naive 7-Day Moving Average)
  - State-Aware XGBoost (evaluated on state-level APMC records)
"""

import os
import json
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.ml.inference import (
    _load_models,
    _predict_xgboost,
    _predict_prophet,
    _predict_arima,
    _predict_lstm,
    _predict_state_aware_xgboost,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_JSON = DATA_DIR / "experimental" / "walk_forward_multi_horizon_metrics.json"

CROPS = ["wheat", "rice", "maize", "potato", "onion"]
HORIZONS = [1, 7, 15, 30, 60, 90]


def run_national_walk_forward():
    logger.info("=== Running National Multi-Horizon Walk-Forward Evaluation ===")
    df = pd.read_csv(DATA_DIR / "real_historical_prices.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    results = {}

    for crop in CROPS:
        logger.info(f"Evaluating Crop: {crop.upper()}")
        bundle = _load_models(crop)
        crop_df = df[df["crop"] == crop].sort_values("ds").reset_index(drop=True)

        # Generate rolling origins across 2026 test partition
        # Earliest origin: 2026-01-07 (allowing past history), Latest: up to 2026-10-03
        # Spaced every 7 days (weekly rolling origins)
        all_dates = crop_df["ds"].tolist()
        min_date = pd.to_datetime("2026-01-07")
        max_date = pd.to_datetime("2026-10-03")

        eligible_origins = [d for d in all_dates if min_date <= d <= max_date and d.dayofweek == 0]  # Every Monday
        if not eligible_origins:
            eligible_origins = all_dates[-25:]

        logger.info(f"  {crop}: {len(eligible_origins)} weekly rolling origins selected.")

        # Data structures to accumulate predictions and actuals per horizon
        # {model: {horizon: {'actuals': [], 'preds': []}}}
        model_names = ["xgboost", "prophet", "arima", "lstm", "reference_benchmark"]
        accum = {m: {h: {"actuals": [], "preds": []} for h in HORIZONS} for m in model_names}

        for origin_dt in eligible_origins:
            hist_df = crop_df[crop_df["ds"] <= origin_dt]
            future_df = crop_df[crop_df["ds"] > origin_dt]

            if len(hist_df) < 60:
                continue

            recent_prices = hist_df["y"].tail(60).tolist()
            target_date = origin_dt.date() + timedelta(days=1)

            max_h_avail = len(future_df)
            if max_h_avail == 0:
                continue

            # Run 90-day predictions from all models
            h_eval = min(90, max_h_avail)
            p_xgb = _predict_xgboost(bundle["xgboost"], recent_prices, target_date, h_eval)
            p_prophet = _predict_prophet(bundle["prophet"], target_date, h_eval)
            p_arima = _predict_arima(bundle["arima"], h_eval)
            p_lstm = _predict_lstm(bundle["lstm"], recent_prices, target_date, h_eval) if bundle.get("lstm") else [recent_prices[-1]] * h_eval
            p_bench = [float(np.mean(recent_prices[-7:]))] * h_eval

            actuals_list = future_df["y"].head(h_eval).tolist()

            for h in HORIZONS:
                if h <= len(actuals_list):
                    idx = h - 1
                    act_val = actuals_list[idx]
                    accum["xgboost"][h]["actuals"].append(act_val)
                    accum["xgboost"][h]["preds"].append(p_xgb[idx])

                    accum["prophet"][h]["actuals"].append(act_val)
                    accum["prophet"][h]["preds"].append(p_prophet[idx])

                    accum["arima"][h]["actuals"].append(act_val)
                    accum["arima"][h]["preds"].append(p_arima[idx])

                    accum["lstm"][h]["actuals"].append(act_val)
                    accum["lstm"][h]["preds"].append(p_lstm[idx])

                    accum["reference_benchmark"][h]["actuals"].append(act_val)
                    accum["reference_benchmark"][h]["preds"].append(p_bench[idx])

        # Compute metrics per horizon
        crop_metrics = {}
        for h in HORIZONS:
            crop_metrics[f"{h}_day"] = {}
            for m in model_names:
                act = np.array(accum[m][h]["actuals"])
                pred = np.array(accum[m][h]["preds"])
                if len(act) > 0:
                    mae = float(mean_absolute_error(act, pred))
                    rmse = float(np.sqrt(mean_squared_error(act, pred)))
                    mape = float(np.mean(np.abs((act - pred) / np.maximum(act, 1))) * 100)
                    medae = float(np.median(np.abs(act - pred)))
                    crop_metrics[f"{h}_day"][m] = {
                        "sample_count": len(act),
                        "mae": round(mae, 2),
                        "rmse": round(rmse, 2),
                        "mape": round(mape, 2),
                        "medae": round(medae, 2),
                    }
                else:
                    crop_metrics[f"{h}_day"][m] = {"sample_count": 0, "status": "insufficient_data"}

        results[crop] = crop_metrics

    return results


def run_state_walk_forward():
    logger.info("=== Running State-Aware Multi-Horizon Walk-Forward Evaluation ===")
    df = pd.read_csv(DATA_DIR / "real_historical_prices_state.csv")
    df["ds"] = pd.to_datetime(df["ds"])

    # Key representative states across agricultural zones
    TEST_STATES = ["Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka", "Madhya Pradesh"]
    state_results = {}

    for crop in CROPS:
        state_results[crop] = {}
        bundle = _load_models(crop)
        for st in TEST_STATES:
            st_df = df[(df["crop"] == crop) & (df["state"] == st)].sort_values("ds").reset_index(drop=True)
            if len(st_df) < 100:
                continue

            all_dates = st_df["ds"].tolist()
            min_date = pd.to_datetime("2026-01-07")
            max_date = pd.to_datetime("2026-10-03")
            eligible_origins = [d for d in all_dates if min_date <= d <= max_date and d.dayofweek == 0]

            if not eligible_origins:
                eligible_origins = all_dates[-15:]

            accum_state = {h: {"actuals": [], "preds": []} for h in HORIZONS}
            accum_bench = {h: {"actuals": [], "preds": []} for h in HORIZONS}

            for origin_dt in eligible_origins:
                hist_df = st_df[st_df["ds"] <= origin_dt]
                future_df = st_df[st_df["ds"] > origin_dt]

                if len(hist_df) < 30 or len(future_df) == 0:
                    continue

                recent_prices = hist_df["y"].tail(60).tolist()
                target_date = origin_dt.date() + timedelta(days=1)
                h_eval = min(90, len(future_df))

                live_curr = recent_prices[-1]
                p_state, _, _, _ = _predict_state_aware_xgboost(
                    bundle, crop, st, live_curr, target_date, h_eval
                )
                if not p_state:
                    continue

                p_bench = [float(np.mean(recent_prices[-7:]))] * h_eval
                actuals = future_df["y"].head(h_eval).tolist()

                for h in HORIZONS:
                    if h <= len(actuals):
                        idx = h - 1
                        act_val = actuals[idx]
                        accum_state[h]["actuals"].append(act_val)
                        accum_state[h]["preds"].append(p_state[idx])
                        accum_bench[h]["actuals"].append(act_val)
                        accum_bench[h]["preds"].append(p_bench[idx])

            st_metrics = {}
            for h in HORIZONS:
                act = np.array(accum_state[h]["actuals"])
                pred = np.array(accum_state[h]["preds"])
                b_pred = np.array(accum_bench[h]["preds"])
                if len(act) > 0:
                    mae = float(mean_absolute_error(act, pred))
                    rmse = float(np.sqrt(mean_squared_error(act, pred)))
                    b_mae = float(mean_absolute_error(act, b_pred))
                    b_rmse = float(np.sqrt(mean_squared_error(act, b_pred)))
                    st_metrics[f"{h}_day"] = {
                        "sample_count": len(act),
                        "state_xgboost": {
                            "mae": round(mae, 2),
                            "rmse": round(rmse, 2),
                            "mape": round(float(np.mean(np.abs((act - pred) / np.maximum(act, 1))) * 100), 2)
                        },
                        "reference_benchmark": {
                            "mae": round(b_mae, 2),
                            "rmse": round(b_rmse, 2),
                            "mape": round(float(np.mean(np.abs((act - b_pred) / np.maximum(act, 1))) * 100), 2)
                        }
                    }
                else:
                    st_metrics[f"{h}_day"] = {"sample_count": 0, "status": "insufficient_data"}

            state_results[crop][st] = st_metrics

    return state_results


def main():
    national_eval = run_national_walk_forward()
    state_eval = run_state_walk_forward()

    final_payload = {
        "generated_at": datetime.now().isoformat(),
        "evaluation_protocol": "Chronological Walk-Forward Rolling Origins (Origins spaced weekly across 2026)",
        "untouched_test_partition": "2026-01-07 to 2026-10-04",
        "horizons_evaluated": HORIZONS,
        "national_evaluation": national_eval,
        "state_aware_evaluation": state_eval,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    logger.info(f"Saved complete walk-forward evaluation to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
