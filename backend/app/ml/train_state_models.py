"""
train_state_models.py — Train State-Aware ML Price Models for AgroIntel

Reads real_historical_prices_state.csv.
Trains state-aware XGBoost models per crop + dedicated state models where sufficient data exists.
Evaluates state-aware model vs baseline crop-only model using chronological train-test split (2019-2023 train, 2024 test).
"""

import os
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR / "app" / "data"

# Load black swan events from config (same source as inference) for feature consistency
from app.ml.feature_engineering import _load_black_swan_config as _bsc_loader, is_black_swan_period
def _get_bs_events():
    """Return events as dicts with datetime 'start'/'end' keys."""
    raw = _bsc_loader()
    events = []
    for e in raw:
        try:
            events.append({
                "name":  e.get("label", "event"),
                "start": e["start_date"],
                "end":   e["end_date"],
            })
        except KeyError:
            pass
    return events
BLACK_SWAN_EVENTS = _get_bs_events()

CROPS = ["rice", "wheat", "maize", "onion", "potato"]


def load_state_dataset() -> pd.DataFrame:
    path = DATA_DIR / "real_historical_prices_state.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run build_state_dataset.py first.")
    
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["crop", "state", "ds"]).reset_index(drop=True)
    return df


def add_features_for_series(group: pd.DataFrame) -> pd.DataFrame:
    """Add state-specific lags and rolling metrics for a single (crop, state) group."""
    df = group.copy().sort_values("ds").reset_index(drop=True)
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["lag_14"] = df["y"].shift(14)
    df["lag_30"] = df["y"].shift(30)
    # Rolling windows use shift(1) to prevent leakage (only past data)
    df["rolling_7"] = df["y"].shift(1).rolling(7, min_periods=1).mean()
    df["rolling_30"] = df["y"].shift(1).rolling(30, min_periods=1).mean()
    df["rolling_std_7"] = df["y"].shift(1).rolling(7, min_periods=2).std().fillna(0.0)
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["month"] = df["ds"].dt.month
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["year"] = df["ds"].dt.year

    # price_range from real archive min/max data (valid feature)
    if "price_range" in df.columns:
        # Use pre-computed price_range from state dataset
        df["price_range"] = df["price_range"].fillna(0.0)
    elif "min_price" in df.columns and "max_price" in df.columns:
        df["price_range"] = (df["max_price"] - df["min_price"]).clip(lower=0.0)
    else:
        df["price_range"] = 0.0

    # Black swan feature
    df["black_swan"] = 0
    for event in BLACK_SWAN_EVENTS:
        mask = (df["ds"] >= pd.to_datetime(event["start"])) & (df["ds"] <= pd.to_datetime(event["end"]))
        df.loc[mask, "black_swan"] = 1

    return df


def process_crop_dataframe(crop_df: pd.DataFrame) -> pd.DataFrame:
    """Group by state and generate lagged features per state series."""
    processed_groups = []
    for state, group in crop_df.groupby("state"):
        if len(group) >= 15:  # Need minimum records for lags
            feat_group = add_features_for_series(group)
            processed_groups.append(feat_group)
    
    if not processed_groups:
        return pd.DataFrame()
        
    full_df = pd.concat(processed_groups, ignore_index=True)
    full_df = full_df.dropna(subset=["lag_1", "lag_7", "lag_14", "lag_30"]).reset_index(drop=True)
    return full_df


def train_crop_models():
    df_all = load_state_dataset()
    eval_summary = {}

    for crop in CROPS:
        logger.info(f"\n==========================================")
        logger.info(f" Training State-Aware Model for Crop: {crop.upper()}")
        logger.info(f"==========================================")

        crop_raw = df_all[df_all["crop"] == crop].copy()
        if crop_raw.empty:
            logger.warning(f"No data for crop: {crop}")
            continue

        crop_df = process_crop_dataframe(crop_raw)
        if crop_df.empty:
            logger.warning(f"Insufficient data for feature engineering on crop: {crop}")
            continue

        # Fit LabelEncoder for states
        le = LabelEncoder()
        crop_df["state_enc"] = le.fit_transform(crop_df["state"])

        # Save state encoder
        encoder_path = MODELS_DIR / f"state_encoder_{crop}.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(le, f)

        # Feature set — 12 validated features (arrival_qtl REMOVED — was all zeros)
        feature_cols = [
            "state_enc", "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_7", "rolling_30", "rolling_std_7", "price_range",
            "day_of_year", "month", "day_of_week", "year", "black_swan"
        ]

        # Chronological train-test split: train < 2024, test >= 2024
        train_mask = crop_df["ds"] < pd.to_datetime("2024-01-01")
        test_mask = crop_df["ds"] >= pd.to_datetime("2024-01-01")

        train_df = crop_df[train_mask]
        test_df = crop_df[test_mask]

        if test_df.empty:
            # Fallback split 80/20 if no 2024 data
            split_idx = int(len(crop_df) * 0.8)
            train_df = crop_df.iloc[:split_idx]
            test_df = crop_df.iloc[split_idx:]

        X_train, y_train = train_df[feature_cols], train_df["y"]
        X_test, y_test = test_df[feature_cols], test_df["y"]

        # 1. Train State-Aware XGBoost
        xgb_state = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_state.fit(X_train, y_train)

        # Predict & Evaluate State-Aware Model
        preds_state = xgb_state.predict(X_test)
        mae_state = float(mean_absolute_error(y_test, preds_state))
        rmse_state = float(np.sqrt(mean_squared_error(y_test, preds_state)))
        mape_state = float(np.mean(np.abs((y_test - preds_state) / np.maximum(y_test, 1))) * 100)
        r2_state = float(r2_score(y_test, preds_state))

        # 2. Train Baseline Crop-Only Model (No state_enc feature)
        feature_cols_baseline = [c for c in feature_cols if c != "state_enc"]
        xgb_base = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.04, random_state=42
        )
        xgb_base.fit(train_df[feature_cols_baseline], y_train)
        preds_base = xgb_base.predict(test_df[feature_cols_baseline])
        mae_base = float(mean_absolute_error(y_test, preds_base))
        rmse_base = float(np.sqrt(mean_squared_error(y_test, preds_base)))

        logger.info(f"State-Aware Model Evaluation -> MAE: {mae_state:.2f}, RMSE: {rmse_state:.2f}, MAPE: {mape_state:.2f}%, R2: {r2_state:.4f}")
        logger.info(f"Baseline Crop-Only Evaluation -> MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}")

        # Save State-Aware Model
        model_path = MODELS_DIR / f"xgboost_state_{crop}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(xgb_state, f)

        # Build data tail (most recent 60 observations per state) for online lag prediction
        data_tail_by_state = {}
        for st, st_group in crop_df.groupby("state"):
            tail_cols = ["ds", "y", "min_price", "max_price", "price_range"]
            avail_cols = [c for c in tail_cols if c in st_group.columns]
            recent = st_group.sort_values("ds").tail(60)[avail_cols].to_dict(orient="records")
            # convert dates to string
            for r in recent:
                if hasattr(r.get("ds"), "strftime"):
                    r["ds"] = r["ds"].strftime("%Y-%m-%d")
            data_tail_by_state[st] = recent

        tail_path = MODELS_DIR / f"data_tail_state_{crop}.pkl"
        with open(tail_path, "wb") as f:
            pickle.dump(data_tail_by_state, f)

        # Record metrics
        eval_summary[crop] = {
            "state_aware": {"mae": mae_state, "rmse": rmse_state, "mape": mape_state, "r2": r2_state},
            "baseline_crop_only": {"mae": mae_base, "rmse": rmse_base},
            "total_records": len(crop_df),
            "states_covered": len(le.classes_),
            "feature_cols": feature_cols
        }

        # Save evaluation JSON
        metrics_path = MODELS_DIR / f"metrics_state_{crop}.json"
        with open(metrics_path, "w") as f:
            json.dump(eval_summary[crop], f, indent=2)

    # Global summary
    summary_path = DATA_DIR / "experimental" / "state_model_evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("\nState-aware training completed for all crops!")
    return eval_summary


if __name__ == "__main__":
    train_crop_models()
