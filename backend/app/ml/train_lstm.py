"""
train_lstm.py — Standalone PyTorch LSTM Sequence Model Trainer for AgroIntel v4.1.

Replaces deprecated/incompatible TensorFlow runtime with native PyTorch (torch 2.7+).
Preserves the sequence architecture:
  Input (11 features) -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16, ReLU) -> Dense(1)

Validation Strategy:
  - Strict Chronological Holdout (Last 60 days of 2026 as validation set).
  - Feature and Target scalers fit strictly on training set (zero target leakage).
  - Metrics computed: MAE, RMSE, MAPE, R2.
  - Model artifacts saved:
      models/lstm_{crop}.pt
      models/lstm_scaler_{crop}.pkl
  - Updates metrics_{crop}.json and model_registry.json.
"""

import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn

from app.core.config import settings
from app.core.constants import PRICE_FEATURE_COLS, PRICE_PREDICTION_CROPS
from app.ml.feature_engineering import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = settings.MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class AgroLSTM(nn.Module):
    """Deep sequence LSTM for agricultural price series forecasting."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        # Take the last time-step representation
        last_step = out[:, -1, :]
        h = self.relu(self.fc1(last_step))
        return self.fc2(h)


def train_single_crop_lstm(
    crop: str,
    df_crop: pd.DataFrame,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.003,
) -> Dict[str, Any]:
    """Train PyTorch LSTM model for a single crop and persist artifacts."""
    logger.info(f"=== Training PyTorch LSTM Sequence Model for [{crop.upper()}] ===")
    start_time = time.time()

    # 1. Feature Engineering
    df_feat = add_features(df_crop)
    val_size = 60
    train_df = df_feat.iloc[:-val_size].copy()
    val_df = df_feat.iloc[-val_size:].copy()

    # 2. Fit Scalers on Train data only (Leakage Prevention)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_raw = train_df[PRICE_FEATURE_COLS].values
    X_train_scaled = x_scaler.fit_transform(X_train_raw)

    y_train_raw = train_df["y"].values.reshape(-1, 1)
    y_train_scaled = y_scaler.fit_transform(y_train_raw)

    # 3. Create 30-day sliding sequences
    seq_len = 30
    if len(X_train_scaled) <= seq_len:
        raise ValueError(f"Insufficient training rows ({len(X_train_scaled)}) for seq_len={seq_len}")

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_train_scaled)):
        X_seq.append(X_train_scaled[i - seq_len : i])
        y_seq.append(y_train_scaled[i])

    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)

    # 4. Initialize Model, Loss, Optimizer
    torch.manual_seed(42)
    model = AgroLSTM(input_dim=len(PRICE_FEATURE_COLS))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    # 6. Chronological Validation Evaluation
    model.eval()
    full_X = pd.concat([train_df.tail(seq_len), val_df], ignore_index=True)[PRICE_FEATURE_COLS].values
    full_X_scaled = x_scaler.transform(full_X)

    val_preds = []
    with torch.no_grad():
        for i in range(len(val_df)):
            seq = torch.tensor(full_X_scaled[i : i + seq_len], dtype=torch.float32).unsqueeze(0)
            scaled_p = model(seq).numpy()
            real_p = float(y_scaler.inverse_transform(scaled_p)[0, 0])
            val_preds.append(real_p)

    val_actuals = val_df["y"].values
    mae = float(mean_absolute_error(val_actuals, val_preds))
    rmse = float(np.sqrt(mean_squared_error(val_actuals, val_preds)))
    mape = float(np.mean(np.abs((val_actuals - val_preds) / np.maximum(val_actuals, 1.0))) * 100)
    r2 = float(r2_score(val_actuals, val_preds))
    elapsed = round(time.time() - start_time, 2)

    logger.info(
        f"[{crop.upper()}] PyTorch LSTM Training Complete in {elapsed}s -> "
        f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}"
    )

    # 7. Save Model Weights and Scalers
    pt_path = MODELS_DIR / f"lstm_{crop}.pt"
    scaler_path = MODELS_DIR / f"lstm_scaler_{crop}.pkl"

    torch.save(model.state_dict(), pt_path)
    with open(scaler_path, "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)

    res = {
        "status": "success",
        "framework": "PyTorch 2.7+",
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "training_time_sec": elapsed,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "hyperparameters": {
            "sequence_length": seq_len,
            "architecture": "LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1)",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
        },
        "model_file": str(pt_path.name),
        "scaler_file": str(scaler_path.name),
    }

    # 8. Update per-crop metrics JSON
    metrics_path = MODELS_DIR / f"metrics_{crop}.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        metrics_data.setdefault("models", {})["lstm"] = res
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

    return res


def run_all_lstm_training() -> Dict[str, Any]:
    """Train PyTorch LSTM models for all 5 crops and update model registry."""
    raw_path = settings.DATA_DIR / "real_historical_prices.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    df_raw = pd.read_csv(raw_path, parse_dates=["ds"])
    logger.info(f"Loaded {len(df_raw)} records across {df_raw['crop'].nunique()} crops for PyTorch LSTM training.")

    results = {}
    for crop in PRICE_PREDICTION_CROPS:
        df_crop = df_raw[df_raw["crop"] == crop].copy()
        if len(df_crop) == 0:
            continue
        res = train_single_crop_lstm(crop, df_crop)
        results[crop] = res

    # Update global model_registry.json
    reg_path = MODELS_DIR / "model_registry.json"
    if reg_path.exists():
        with open(reg_path, "r") as f:
            reg = json.load(f)
        for crop, res in results.items():
            if crop in reg.get("registry", {}):
                reg["registry"][crop]["lstm_availability"] = True
                reg["registry"][crop]["lstm_framework"] = "PyTorch 2.7"
                if "all_models_mae" not in reg["registry"][crop]:
                    reg["registry"][crop]["all_models_mae"] = {}
                reg["registry"][crop]["all_models_mae"]["lstm"] = res.get("mae")
                reg["registry"][crop]["all_models_rmse"] = reg["registry"][crop].get("all_models_rmse", {})
                reg["registry"][crop]["all_models_rmse"]["lstm"] = res.get("rmse")
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)

    logger.info("All PyTorch LSTM models successfully trained and registered.")
    return results


if __name__ == "__main__":
    run_all_lstm_training()
