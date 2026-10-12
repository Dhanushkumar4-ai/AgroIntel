"""
feature_engineering.py — Price prediction feature engineering for AgroIntel.

Generates EXACTLY 11 features as specified in the implementation plan:

  Price-based:
    lag_1, lag_7, lag_14, lag_30
    rolling_mean_7, rolling_mean_30

  Calendar:
    month, season

  Weather (from weather_history.csv — monthly averages):
    monthly_avg_temp, monthly_total_rainfall

  Event:
    black_swan

Training NEVER calls any external API.
Weather features are read from weather_history.csv only.
Black swan events are read dynamically from black_swan_config.json.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from app.core.constants import (
    PRICE_FEATURE_COLS,
    SEASON_ENCODING,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

_WEATHER_HISTORY_PATH = settings.DATA_DIR / "weather_history.csv"
_BLACK_SWAN_CONFIG_PATH = settings.DATA_DIR / "black_swan_config.json"

# Compatibility Aliases
FEATURE_COLS = PRICE_FEATURE_COLS

# ── Caches ────────────────────────────────────────────────────────────────────
_weather_df: Optional[pd.DataFrame] = None
_black_swan_events: Optional[list[dict]] = None


def _load_weather_history() -> pd.DataFrame:
    global _weather_df
    if _weather_df is None:
        if not _WEATHER_HISTORY_PATH.exists():
            raise FileNotFoundError(
                f"weather_history.csv not found at {_WEATHER_HISTORY_PATH}. "
                "Run app/data/backfill_weather.py first."
            )
        _weather_df = pd.read_csv(
            _WEATHER_HISTORY_PATH,
            dtype={"year": int, "month": int, "avg_temp": float, "total_rainfall": float},
        )
    return _weather_df


def _load_black_swan_config() -> list[dict]:
    global _black_swan_events
    if _black_swan_events is None:
        if not _BLACK_SWAN_CONFIG_PATH.exists():
            logger.warning(
                f"black_swan_config.json not found at {_BLACK_SWAN_CONFIG_PATH}. "
                "Defaulting to empty event list."
            )
            _black_swan_events = []
        else:
            try:
                with open(_BLACK_SWAN_CONFIG_PATH, "r") as f:
                    _black_swan_events = json.load(f).get("events", [])
                logger.info(f"Loaded {len(_black_swan_events)} black swan events.")
            except Exception as e:
                logger.error(f"Failed to parse black_swan_config.json: {e}")
                _black_swan_events = []
    return _black_swan_events


BLACK_SWAN_EVENTS = _load_black_swan_config()


def is_black_swan_period(target_date: date) -> int:
    events = _load_black_swan_config()
    target_str = str(target_date)
    for event in events:
        start_str = event.get("start_date", "")
        end_str = event.get("end_date", "")
        if start_str <= target_str <= end_str:
            return 1
    return 0


def get_active_black_swan(target_date: date) -> Optional[dict]:
    events = _load_black_swan_config()
    target_str = str(target_date)
    for event in events:
        if event.get("start_date", "") <= target_str <= event.get("end_date", ""):
            return event
    return None


def get_weather_features(target_date: date) -> tuple[float, float]:
    df_w = _load_weather_history()
    year = target_date.year
    month = target_date.month

    match = df_w[(df_w["year"] == year) & (df_w["month"] == month)]
    if len(match) > 0:
        row = match.iloc[0]
        return float(row["avg_temp"]), float(row["total_rainfall"])

    month_match = df_w[df_w["month"] == month]
    if len(month_match) > 0:
        return float(month_match["avg_temp"].mean()), float(month_match["total_rainfall"].mean())

    return float(df_w["avg_temp"].mean()), float(df_w["total_rainfall"].mean())


def compute_features_for_row(
    row_idx: int,
    price_series: pd.Series,
    ds_series: pd.Series,
    weather_lookup_fn=get_weather_features,
) -> dict:
    # prices_up_to includes the current row value — we use only past prices to avoid leakage.
    # prices_up_to[-1] = current day price (target y)
    # prices_up_to[-2] = yesterday (lag_1) — this matches df["y"].shift(1) in training
    past_prices = price_series.iloc[: row_idx].values  # exclude current row
    target_dt = pd.to_datetime(ds_series.iloc[row_idx]).date()

    n = len(past_prices)
    lag_1  = float(past_prices[-1])  if n >= 1  else 0.0
    lag_7  = float(past_prices[-7])  if n >= 7  else lag_1
    lag_14 = float(past_prices[-14]) if n >= 14 else lag_7
    lag_30 = float(past_prices[-30]) if n >= 30 else lag_14

    # Rolling windows over past prices only (no leakage)
    window_7  = past_prices[-7:]  if n >= 7  else past_prices
    rolling_7 = float(np.mean(window_7))  if len(window_7) > 0 else lag_1

    window_30  = past_prices[-30:] if n >= 30 else past_prices
    rolling_30 = float(np.mean(window_30)) if len(window_30) > 0 else lag_1

    month = target_dt.month
    season_code = SEASON_ENCODING.get(month, 1)
    avg_temp, total_rain = weather_lookup_fn(target_dt)
    bs_flag = is_black_swan_period(target_dt)

    return {
        "lag_1":                  round(lag_1, 2),
        "lag_7":                  round(lag_7, 2),
        "lag_14":                 round(lag_14, 2),
        "lag_30":                 round(lag_30, 2),
        "rolling_mean_7":         round(rolling_7, 2),
        "rolling_mean_30":        round(rolling_30, 2),
        "month":                  month,
        "season":                 season_code,
        "monthly_avg_temp":       round(avg_temp, 2),
        "monthly_total_rainfall": round(total_rain, 2),
        "black_swan":             bs_flag,
    }


def build_training_features(
    df: pd.DataFrame,
    start_offset: int = 30,
) -> pd.DataFrame:
    df_sorted = df.sort_values("ds").reset_index(drop=True)
    price_series = df_sorted["y"]
    ds_series = df_sorted["ds"]

    feature_rows = []
    for idx in range(start_offset, len(df_sorted)):
        feat = compute_features_for_row(idx, price_series, ds_series)
        feat["ds"] = ds_series.iloc[idx]
        feat["y"] = float(price_series.iloc[idx])
        feature_rows.append(feat)

    result_df = pd.DataFrame(feature_rows)

    cols_order = ["ds", "y"] + PRICE_FEATURE_COLS
    result_df = result_df[cols_order]

    logger.info(
        f"Generated {len(result_df)} training feature rows "
        f"({len(PRICE_FEATURE_COLS)} features: {PRICE_FEATURE_COLS})."
    )
    return result_df


# Compatibility Alias
add_features = build_training_features


def build_inference_features(
    price_tail: list,
    target_date: date,
    monthly_avg_temp: float = 25.0,
    monthly_total_rainfall: float = 50.0,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for inference.

    price_tail: list of PAST prices (NOT including today's price).
                The last element is yesterday's price (lag_1).
                Minimum 30 values required; shorter tails are padded.
    target_date: the date being predicted (tomorrow from caller's perspective).
    """
    if len(price_tail) < 30:
        pad_val = price_tail[0] if price_tail else 1000.0
        price_tail = [pad_val] * (30 - len(price_tail)) + list(price_tail)

    # price_tail[-1] = yesterday = lag_1  (matches df["y"].shift(1) in training)
    past = np.array(price_tail, dtype=float)

    lag_1  = float(past[-1])   # yesterday
    lag_7  = float(past[-7])   # 7 days ago
    lag_14 = float(past[-14])  # 14 days ago
    lag_30 = float(past[-30])  # 30 days ago

    # Rolling means over past values (no current-day leakage)
    rolling_mean_7  = float(np.mean(past[-7:]))
    rolling_mean_30 = float(np.mean(past[-30:]))

    month       = target_date.month
    season_code = SEASON_ENCODING.get(month, 1)
    bs_flag     = is_black_swan_period(target_date)

    # Try to get actual weather for target month if not explicitly provided
    if monthly_avg_temp == 25.0 and monthly_total_rainfall == 50.0:
        try:
            monthly_avg_temp, monthly_total_rainfall = get_weather_features(target_date)
        except Exception:
            pass  # keep defaults if weather lookup fails

    feat_dict = {
        "lag_1":                  [round(lag_1, 2)],
        "lag_7":                  [round(lag_7, 2)],
        "lag_14":                 [round(lag_14, 2)],
        "lag_30":                 [round(lag_30, 2)],
        "rolling_mean_7":         [round(rolling_mean_7, 2)],
        "rolling_mean_30":        [round(rolling_mean_30, 2)],
        "month":                  [month],
        "season":                 [season_code],
        "monthly_avg_temp":       [round(monthly_avg_temp, 2)],
        "monthly_total_rainfall": [round(monthly_total_rainfall, 2)],
        "black_swan":             [bs_flag],
    }

    feat_df = pd.DataFrame(feat_dict)
    return feat_df[PRICE_FEATURE_COLS]
