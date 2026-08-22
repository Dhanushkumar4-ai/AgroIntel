"""
train_lstm.py — Standalone Isolated LSTM Trainer for AgroIntel v4.0.

This script is designed to be executed in a dedicated Python 3.11 environment
where TensorFlow/Keras is fully supported and compatible.

Workflow:
  1. Loads historical prices (real_historical_prices.csv) + weather history
  2. Performs exact 11-feature engineering (feature_engineering.py)
  3. Splits chronologically (last 60 days validation)
  4. Trains LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1)
  5. Saves models/lstm_{crop}.keras & models/lstm_scaler_{crop}.pkl
  6. Updates metrics_{crop}.json & model_registry.json with LSTM results

Usage (in Python 3.11 environment):
  python -m app.ml.train_lstm
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.constants import PRICE_FEATURE_COLS, PRICE_PREDICTION_CROPS
from app.ml.feature_engineering import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = settings.MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_single_crop_lstm(crop: str, df_crop: pd.DataFrame) -> Dict[str, Any]:
    """Train LSTM model for a single crop and save artifacts."""
    logger.info(f"=== Standalone LSTM Trainer for {crop.upper()} ===")

    df_feat = add_features(df_crop)
    val_size = 60
    train_df = df_feat.iloc[:-val_size].copy()
    val_df = df_feat.iloc[-val_size:].copy()

    start_time = time.time()
    try:
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        logger.info(f"[{crop.upper()}] TensorFlow version: {tf.__version__}")

        scaler = StandardScaler()
        X_train_raw = train_df[PRICE_FEATURE_COLS].values
        X_train_scaled = scaler.fit_transform(X_train_raw)
        y_train = train_df["y"].values

        seq_len = 30
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_train_scaled)):
            X_seq.append(X_train_scaled[i - seq_len : i])
            y_seq.append(y_train[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, X_train_scaled.shape[1])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(X_seq, y_seq, epochs=25, batch_size=32, verbose=1)

        # Validation prediction
        full_X = pd.concat([train_df.tail(seq_len), val_df], ignore_index=True)[PRICE_FEATURE_COLS].values
        full_X_scaled = scaler.transform(full_X)

        val_preds = []
        for i in range(len(val_df)):
            seq = full_X_scaled[i : i + seq_len].reshape(1, seq_len, X_train_scaled.shape[1])
            pred_val = float(model.predict(seq, verbose=0)[0, 0])
            val_preds.append(pred_val)

        actuals = val_df["y"].values
        mae = float(np.mean(np.abs(actuals - val_preds)))
        rmse = float(np.sqrt(np.mean((actuals - val_preds) ** 2)))
        elapsed = round(time.time() - start_time, 2)

        logger.info(f"[{crop.upper()}] LSTM Training Complete in {elapsed}s — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Save model and scaler
        keras_path = MODELS_DIR / f"lstm_{crop}.keras"
        scaler_path = MODELS_DIR / f"lstm_scaler_{crop}.pkl"

        model.save(keras_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        res = {
            "status": "success",
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "training_time_sec": elapsed,
            "hyperparameters": {
                "sequence_length": 30,
                "architecture": "LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1)",
                "epochs": 25,
                "batch_size": 32,
                "optimizer": "adam",
            },
            "model_file": str(keras_path),
            "scaler_file": str(scaler_path),
        }

        # Update per-crop metrics JSON if exists
        metrics_path = MODELS_DIR / f"metrics_{crop}.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            metrics_data["models"]["lstm"] = res
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2)

        return res

    except Exception as e:
        logger.error(f"[{crop.upper()}] LSTM training failed: {e}")
        return {"status": "failed", "error": str(e), "mae": 999999.0, "rmse": 999999.0}


def run_all_lstm_training():
    """Run isolated LSTM training across all crops."""
    raw_path = settings.DATA_DIR / "real_historical_prices.csv"
    df_raw = pd.read_csv(raw_path, parse_dates=["ds"])

    results = {}
    for crop in PRICE_PREDICTION_CROPS:
        df_crop = df_raw[df_raw["crop"] == crop].copy()
        res = train_single_crop_lstm(crop, df_crop)
        results[crop] = res

    # Update model registry
    reg_path = MODELS_DIR / "model_registry.json"
    if reg_path.exists():
        with open(reg_path, "r") as f:
            reg = json.load(f)
        for crop, res in results.items():
            if crop in reg.get("registry", {}):
                reg["registry"][crop]["lstm_availability"] = (res.get("status") == "success")
                if "all_models_mae" not in reg["registry"][crop]:
                    reg["registry"][crop]["all_models_mae"] = {}
                reg["registry"][crop]["all_models_mae"]["lstm"] = res.get("mae")
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)

    logger.info("Standalone LSTM training finished.")
    return results


if __name__ == "__main__":
    run_all_lstm_training()
