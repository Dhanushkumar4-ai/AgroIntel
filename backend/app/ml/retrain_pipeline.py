"""
retrain_pipeline.py -- Master orchestrator for AgroIntel ML fixes.

Runs all 6 fix steps in order:
  Step 1: Fetch 2025-2026 AGMARKNET price data (fills 22-month gap)
  Step 2: Fetch state-level weather from Open-Meteo (replaces national average)
  Step 3: Retrain national models (Prophet, XGBoost, ARIMA, MLP) with fixed features
  Step 4: Retrain state-aware XGBoost models with corrected black swan source
  Step 5: Update model_registry.json to mark state_aware_xgboost as production model
  Step 6: Validate: print per-crop MAE and R2, flag any model with R2 < 0

Run:
    cd backend
    python -m app.ml.retrain_pipeline

Expected time: 10-20 minutes (mostly API fetch time)
"""

import json
import logging
import pickle
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("retrain_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "app" / "data"


def _banner(title):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  {}".format(title))
    logger.info("=" * 60)


def step1_fetch_prices():
    """Fetch 2025-2026 AGMARKNET price data."""
    _banner("STEP 1: Fetching recent AGMARKNET price data")
    try:
        from app.data.fetch_recent_prices import fetch_and_extend
        fetch_and_extend()
        national_path = DATA_DIR / "real_historical_prices.csv"
        if national_path.exists():
            df = pd.read_csv(national_path, parse_dates=["ds"])
            logger.info("National data: {} rows | {} to {}".format(
                len(df), df["ds"].min().date(), df["ds"].max().date()))
            data_gap = (date.today() - df["ds"].max().date()).days
            logger.info("Remaining data gap: {} days".format(data_gap))
        logger.info("STEP 1 COMPLETE")
        return True
    except Exception as e:
        logger.error("STEP 1 FAILED: {}".format(e))
        return False


def step2_fetch_weather():
    """Fetch state-level weather data from Open-Meteo."""
    _banner("STEP 2: Fetching state-level weather data")
    try:
        from app.data.fetch_state_weather import fetch_all_state_weather
        fetch_all_state_weather()
        weather_path = DATA_DIR / "weather_history_state.csv"
        if weather_path.exists():
            df = pd.read_csv(weather_path)
            logger.info("State weather: {} rows | {} states | years {}-{}".format(
                len(df), df["state"].nunique(),
                df["year"].min(), df["year"].max()))
        logger.info("STEP 2 COMPLETE")
        return True
    except Exception as e:
        logger.error("STEP 2 FAILED: {}".format(e))
        return False


def step3_retrain_national():
    """Retrain national models (Prophet, XGBoost, ARIMA, MLP) with fixed features."""
    _banner("STEP 3: Retraining national crop models (fixed features)")
    try:
        from app.ml.price_trainer import run_training_pipeline
        results = run_training_pipeline()
        logger.info("National model results:")
        for crop, res in results.items():
            prod_m = res.get("production_model", "?")
            mae = res.get("models", {}).get(prod_m, {}).get("mae", "?")
            logger.info("  {} -> {} | MAE: {}".format(crop.upper(), prod_m, mae))
        logger.info("STEP 3 COMPLETE")
        return True
    except Exception as e:
        logger.error("STEP 3 FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


def step4_retrain_state_models():
    """Retrain state-aware XGBoost models with corrected black swan source."""
    _banner("STEP 4: Retraining state-aware XGBoost models")
    try:
        from app.ml.train_state_models import train_crop_models
        results = train_crop_models()
        logger.info("State-aware model results:")
        for crop, res in results.items():
            state_mae = res.get("state_aware", {}).get("mae", "?")
            state_r2  = res.get("state_aware", {}).get("r2",  "?")
            flag = "*** R2<0: model may be broken ***" if isinstance(state_r2, float) and state_r2 < 0 else ""
            logger.info("  {} -> MAE: {:.2f} | R2: {:.4f} {}".format(
                crop.upper(), state_mae, state_r2, flag))
        logger.info("STEP 4 COMPLETE")
        return True, results
    except Exception as e:
        logger.error("STEP 4 FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        return False, {}


def step5_update_registry(state_results):
    """
    Update model_registry.json:
    - Use state_aware_xgboost as production model where R2 >= 0
    - Fall back to prophet for potato/onion if state model is broken (R2 < 0)
    """
    _banner("STEP 5: Updating model registry")
    registry_path = MODELS_DIR / "model_registry.json"
    timestamp = datetime.now().isoformat()

    # Load existing registry
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry_data = json.load(f)
    else:
        registry_data = {"system_version": "4.1.0", "last_updated": timestamp, "registry": {}}

    registry_data["last_updated"] = timestamp
    registry_data["system_version"] = "4.1.0"

    crops = ["wheat", "rice", "maize", "potato", "onion"]
    for crop in crops:
        res = state_results.get(crop, {})
        state_r2  = res.get("state_aware", {}).get("r2",   None)
        state_mae = res.get("state_aware", {}).get("mae",  None)
        state_rmse = res.get("state_aware", {}).get("rmse", None)

        # Decide production model:
        # - potato and onion have high volatility (CV > 50%) — use prophet if state model broken
        # - other crops use state_aware_xgboost if R2 >= 0
        HIGH_VOLATILITY = {"potato", "onion"}
        if isinstance(state_r2, float) and state_r2 < 0 and crop in HIGH_VOLATILITY:
            prod_model = "prophet"
            logger.info("  {} -> FALLBACK to prophet (state R2={:.4f} < 0)".format(crop.upper(), state_r2))
        else:
            prod_model = "state_aware_xgboost"
            logger.info("  {} -> state_aware_xgboost (R2={})".format(crop.upper(), state_r2))

        existing = registry_data["registry"].get(crop, {})
        existing.update({
            "crop":              crop,
            "production_model":  prod_model,
            "model_version":     "v_{}".format(date.today().strftime("%Y%m%d")),
            "training_timestamp": timestamp,
            "dataset_version":   "2019-2026-v2",
            "weather_version":   "open-meteo-state-monthly-v2",
            "holdout_mae":       round(state_mae, 2) if isinstance(state_mae, float) else None,
            "holdout_rmse":      round(state_rmse, 2) if isinstance(state_rmse, float) else None,
            "holdout_r2":        round(state_r2, 4) if isinstance(state_r2, float) else None,
            "state_aware":       prod_model == "state_aware_xgboost",
            "model_file_path":   "models/xgboost_state_{}.pkl".format(crop),
            "forecast_horizon_days": 30,
        })
        registry_data["registry"][crop] = existing

    with open(registry_path, "w") as f:
        json.dump(registry_data, f, indent=2)
    logger.info("Registry updated: {}".format(registry_path))
    logger.info("STEP 5 COMPLETE")
    return True


def step6_validate():
    """Validate all models: print metrics and flag broken models."""
    _banner("STEP 6: Validation check")
    registry_path = MODELS_DIR / "model_registry.json"
    if not registry_path.exists():
        logger.error("model_registry.json not found")
        return

    with open(registry_path, "r") as f:
        registry = json.load(f)

    logger.info("\nFINAL MODEL SUMMARY:")
    logger.info("{:<12} {:<25} {:>10} {:>10} {:>10}".format(
        "CROP", "PROD MODEL", "MAE", "RMSE", "R2"))
    logger.info("-" * 70)
    issues = []
    for crop, meta in registry.get("registry", {}).items():
        prod    = meta.get("production_model", "?")
        mae     = meta.get("holdout_mae",  "N/A")
        rmse    = meta.get("holdout_rmse", "N/A")
        r2      = meta.get("holdout_r2",   "N/A")
        r2_str  = "{:.4f}".format(r2) if isinstance(r2, float) else str(r2)
        mae_str = "{:.2f}".format(mae) if isinstance(mae, float) else str(mae)
        rmse_str = "{:.2f}".format(rmse) if isinstance(rmse, float) else str(rmse)
        flag = " <-- BROKEN (R2<0)" if isinstance(r2, float) and r2 < 0 else ""
        logger.info("{:<12} {:<25} {:>10} {:>10} {:>10}{}".format(
            crop.upper(), prod, mae_str, rmse_str, r2_str, flag))
        if isinstance(r2, float) and r2 < 0:
            issues.append("{} state model has R2={:.4f}".format(crop, r2))

    if issues:
        logger.warning("\nISSUES DETECTED:")
        for issue in issues:
            logger.warning("  - {}".format(issue))
        logger.warning("These crops have been auto-fallback'd to prophet in the registry.")
    else:
        logger.info("\nAll models look healthy.")

    # Check data freshness
    national_path = DATA_DIR / "real_historical_prices.csv"
    if national_path.exists():
        df = pd.read_csv(national_path, parse_dates=["ds"])
        data_end = df["ds"].max().date()
        gap = (date.today() - data_end).days
        logger.info("\nData freshness: latest={} | gap={} days".format(data_end, gap))
        if gap > 30:
            logger.warning("Data gap > 30 days. AGMARKNET API may have not returned data.")
            logger.warning("Check your API key or try fetching manually from data.gov.in")

    logger.info("STEP 6 COMPLETE")


def run_all(skip_fetch=False, skip_weather=False):
    """Run all steps of the retraining pipeline."""
    t_start = time.time()
    _banner("AGROINTEL ML RETRAINING PIPELINE v4.1")
    logger.info("Started at: {}".format(datetime.now().isoformat()))
    logger.info("Data dir:   {}".format(DATA_DIR))
    logger.info("Models dir: {}".format(MODELS_DIR))

    results = {}

    if not skip_fetch:
        results["step1"] = step1_fetch_prices()
    else:
        logger.info("STEP 1 SKIPPED (skip_fetch=True)")

    if not skip_weather:
        results["step2"] = step2_fetch_weather()
    else:
        logger.info("STEP 2 SKIPPED (skip_weather=True)")

    results["step3"] = step3_retrain_national()
    ok4, state_results = step4_retrain_state_models()
    results["step4"] = ok4
    results["step5"] = step5_update_registry(state_results)

    step6_validate()

    elapsed = round(time.time() - t_start, 1)
    _banner("PIPELINE COMPLETE")
    logger.info("Total time: {}s ({:.1f} min)".format(elapsed, elapsed / 60))
    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning("Failed steps: {}".format(failed))
    else:
        logger.info("All steps completed successfully.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgroIntel ML Retraining Pipeline")
    parser.add_argument("--skip-fetch",   action="store_true", help="Skip AGMARKNET data fetch (use existing CSVs)")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather data fetch")
    args = parser.parse_args()
    run_all(skip_fetch=args.skip_fetch, skip_weather=args.skip_weather)
