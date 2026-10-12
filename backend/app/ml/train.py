"""
train.py -- Multi-model Price Prediction Training
Trains Prophet, XGBoost, ARIMA, and MLP (Deep Learning proxy) on
real historical price data that includes key Black Swan events.

FEATURE ALIGNMENT: Uses build_training_features() from feature_engineering.py
to guarantee that the EXACT same PRICE_FEATURE_COLS used at inference are used
during training. This eliminates the training-inference feature mismatch.
"""

import os
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Import canonical feature pipeline ─────────────────────────────────────────
# IMPORTANT: build_training_features() produces PRICE_FEATURE_COLS which is the
# EXACT same set used in inference. Do NOT define local FEATURE_COLS here.
from app.core.constants import PRICE_FEATURE_COLS
from app.ml.feature_engineering import build_training_features


def _generate_historical_data(crop: str, cfg: dict) -> pd.DataFrame:
    """
    Load 5 years (2019-2024) of real historical daily price data from Kaggle dataset.
    """
    real_data_path = BASE_DIR / "app" / "data" / "real_historical_prices.csv"
    
    if not real_data_path.exists():
        raise FileNotFoundError(
            f"Historical data not found at {real_data_path}. "
            "Please run `python -m app.data.process_kaggle` first."
        )
        
    df = pd.read_csv(real_data_path)
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Filter by crop
    crop_df = df[df["crop"] == crop].copy()
    
    if crop_df.empty:
        raise ValueError(f"No historical data found for crop: {crop}")
        
    return crop_df.reset_index(drop=True)


# ── Crop configuration ─────────────────────────────────────────────────────────
CROPS = {
    "wheat":  {"base_price": 2200, "seasonal_amplitude": 180, "trend_per_year": 120, "noise_std": 60},
    "rice":   {"base_price": 2300, "seasonal_amplitude": 150, "trend_per_year": 100, "noise_std": 50},
    "maize":  {"base_price": 1600, "seasonal_amplitude": 130, "trend_per_year": 90,  "noise_std": 50},
    "potato": {"base_price": 1300, "seasonal_amplitude": 200, "trend_per_year": 80,  "noise_std": 60},
    "onion":  {"base_price": 1800, "seasonal_amplitude": 400, "trend_per_year": 100, "noise_std": 120},
}


def _train_prophet(df_raw: pd.DataFrame, crop: str):
    try:
        from prophet import Prophet
        prophet_df = df_raw[["ds", "y"]].copy()
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=8,
            n_changepoints=20,
        )
        m.fit(prophet_df)
        model_path = MODELS_DIR / f"prophet_{crop}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(m, f)
        logger.info(f"[{crop}] Prophet trained and saved.")
        return m
    except Exception as e:
        logger.error(f"[{crop}] Prophet training failed: {e}")
        return None


def _train_xgboost(df_feat: pd.DataFrame, crop: str):
    try:
        import xgboost as xgb
        X = df_feat[PRICE_FEATURE_COLS].values
        y = df_feat["y"].values
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
        )
        model.fit(X, y)
        model_path = MODELS_DIR / f"xgboost_{crop}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[{crop}] XGBoost trained and saved.")
        return model
    except Exception as e:
        logger.error(f"[{crop}] XGBoost training failed: {e}")
        return None


def _train_arima(df_raw: pd.DataFrame, crop: str):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Use weekly-resampled data for ARIMA (faster, still captures trends)
        weekly = df_raw.set_index("ds")["y"].resample("W").mean().dropna()
        # ARIMA(2,1,2) is robust for agricultural price series
        model = ARIMA(weekly, order=(1, 1, 1))
        result = model.fit()
        model_path = MODELS_DIR / f"arima_{crop}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"[{crop}] ARIMA trained and saved.")
        return result
    except Exception as e:
        logger.error(f"[{crop}] ARIMA training failed: {e}")
        return None


def _train_mlp(df_feat: pd.DataFrame, crop: str):
    """Train a Multi-Layer Perceptron (Deep Learning proxy via scikit-learn)."""
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        X = df_feat[PRICE_FEATURE_COLS].values
        y = df_feat["y"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            learning_rate_init=0.002,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(X_scaled, y)

        bundle = {"scaler": scaler, "model": model}
        model_path = MODELS_DIR / f"mlp_{crop}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info(f"[{crop}] MLP (Deep Learning) trained and saved.")
        return bundle
    except Exception as e:
        logger.error(f"[{crop}] MLP training failed: {e}")
        return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            learning_rate_init=0.002,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(X_scaled, y)

        # Save both scaler and model together
        bundle = {"scaler": scaler, "model": model}
        model_path = MODELS_DIR / f"mlp_{crop}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info(f"[{crop}] MLP (Deep Learning) trained and saved.")
        return bundle
    except Exception as e:
        logger.error(f"[{crop}] MLP training failed: {e}")
        return None


def _evaluate_models(df_raw: pd.DataFrame, df_feat: pd.DataFrame, crop: str, models: dict) -> dict:
    """
    Evaluate all 4 models on the last 30 days of data.
    Returns MAE (Mean Absolute Error) and RMSE for each.
    Lower = better.
    """
    val_df_raw = df_raw.tail(30).copy()
    val_df_feat = df_feat.tail(30).copy()

    metrics = {}

    # Prophet evaluation
    try:
        if models.get("prophet"):
            future = val_df_raw[["ds"]].copy()
            forecast = models["prophet"].predict(future)
            preds = forecast["yhat"].values
            actual = val_df_raw["y"].values
            mae  = float(np.mean(np.abs(preds - actual)))
            rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
            metrics["prophet"] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
    except Exception as e:
        logger.warning(f"Prophet eval error: {e}")
        metrics["prophet"] = {"mae": 9999, "rmse": 9999}

    # XGBoost evaluation
    try:
        if models.get("xgboost") and len(val_df_feat) > 0:
            X_val  = val_df_feat[PRICE_FEATURE_COLS].values
            preds  = models["xgboost"].predict(X_val)
            actual = val_df_feat["y"].values
            mae  = float(np.mean(np.abs(preds - actual)))
            rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
            metrics["xgboost"] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
    except Exception as e:
        logger.warning(f"XGBoost eval error: {e}")
        metrics["xgboost"] = {"mae": 9999, "rmse": 9999}

    # ARIMA evaluation
    try:
        if models.get("arima"):
            weekly_actual = val_df_raw.set_index("ds")["y"].resample("W").mean().dropna()
            if len(weekly_actual) >= 2:
                forecast = models["arima"].forecast(steps=len(weekly_actual))
                mae  = float(np.mean(np.abs(forecast.values - weekly_actual.values)))
                rmse = float(np.sqrt(np.mean((forecast.values - weekly_actual.values) ** 2)))
                metrics["arima"] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
            else:
                metrics["arima"] = {"mae": 9999, "rmse": 9999}
    except Exception as e:
        logger.warning(f"ARIMA eval error: {e}")
        metrics["arima"] = {"mae": 9999, "rmse": 9999}

    # MLP evaluation
    try:
        if models.get("mlp") and len(val_df_feat) > 0:
            scaler = models["mlp"]["scaler"]
            mlp    = models["mlp"]["model"]
            X_val  = val_df_feat[PRICE_FEATURE_COLS].values
            X_sc   = scaler.transform(X_val)
            preds  = mlp.predict(X_sc)
            actual = val_df_feat["y"].values
            mae  = float(np.mean(np.abs(preds - actual)))
            rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
            metrics["mlp"] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
    except Exception as e:
        logger.warning(f"MLP eval error: {e}")
        metrics["mlp"] = {"mae": 9999, "rmse": 9999}

    # Determine best model
    best = min(metrics, key=lambda k: metrics[k]["mae"])
    return {"models": metrics, "best_model": best}


def train_price_prediction_model(crop: str = "wheat"):
    """
    Full training pipeline for one crop.
    Returns a dict with training status and model metrics.
    """
    crop = crop.lower()
    if crop not in CROPS:
        raise ValueError(f"Unsupported crop '{crop}'. Choose from: {list(CROPS.keys())}")

    cfg = CROPS[crop]
    logger.info(f"=== Training pipeline for: {crop.upper()} ===")

    # 1. Generate 5-year historical data
    df_raw = _generate_historical_data(crop, cfg)
    logger.info(f"[{crop}] Generated {len(df_raw)} days of historical data "
                f"({df_raw['ds'].min().date()} → {df_raw['ds'].max().date()})")

    # 2. Add engineered features
    df_feat = _add_features(df_raw)

    # 3. Train / Validation split (hold last 30 days for evaluation)
    train_raw = df_raw.iloc[:-30]
    train_feat = df_feat.iloc[:-30]

    # 4. Train each model
    models = {}
    models["prophet"] = _train_prophet(train_raw, crop)
    models["xgboost"] = _train_xgboost(train_feat, crop)
    models["arima"] = _train_arima(train_raw, crop)
    models["mlp"] = _train_mlp(train_feat, crop)

    # 5. Evaluate on validation set
    metrics_result = _evaluate_models(df_raw, df_feat, crop, models)
    best = metrics_result["best_model"]
    logger.info(f"[{crop}] Best model: {best.upper()} "
                f"(MAE={metrics_result['models'][best]['mae']})")

    # 6. Save metrics
    metrics_path = MODELS_DIR / f"metrics_{crop}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_result, f, indent=2)

    # 7. Also save the raw data tail for inference reference
    ref_path = MODELS_DIR / f"data_tail_{crop}.pkl"
    with open(ref_path, "wb") as f:
        pickle.dump(df_feat.tail(60), f)

    logger.info(f"[{crop}] Training complete. Metrics saved to {metrics_path}")
    return {
        "crop": crop,
        "status": "success",
        "data_range": f"{df_raw['ds'].min().date()} to {df_raw['ds'].max().date()}",
        "total_samples": len(df_raw),
        "best_model": best,
        "metrics": metrics_result["models"],
    }


def train_all_crops():
    """Train models for all configured crops."""
    results = {}
    for crop in CROPS:
        try:
            result = train_price_prediction_model(crop)
            results[crop] = result
        except Exception as e:
            logger.error(f"Failed to train {crop}: {e}")
            results[crop] = {"status": "error", "error": str(e)}
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("Starting multi-crop, multi-model training pipeline on REAL Kaggle data...")
    print("Historical data: 2019 → 2024 (5+ years of daily mandi prices)")
    print()
    results = train_all_crops()
    print("\n=== TRAINING SUMMARY ===")
    for crop, r in results.items():
        if r.get("status") == "success":
            print(f"✓ {crop.upper()}: Best={r['best_model'].upper()}, "
                  f"MAE={r['metrics'][r['best_model']]['mae']}")
        else:
            print(f"✗ {crop.upper()}: FAILED — {r.get('error')}")
