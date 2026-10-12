"""
fetch_recent_prices.py -- Fill the 22-month data gap (Jan 2025 to today).

Fetches real mandi price data from data.gov.in AGMARKNET API and appends
to real_historical_prices.csv and real_historical_prices_state.csv.

Run:
    cd backend
    python -m app.data.fetch_recent_prices

Output:
    app/data/real_historical_prices.csv       -- updated national daily averages
    app/data/real_historical_prices_state.csv -- updated state-level daily averages
"""

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"

AGMARKNET_API_BASE = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aab825d38f95e74f2c"  # public demo key

# Crop name mapping: internal name -> AGMARKNET commodity name variants
CROP_COMMODITY_MAP = {
    "wheat":  ["Wheat", "WHEAT"],
    "rice":   ["Rice", "RICE", "Paddy(Dhan)(Common)", "Paddy"],
    "maize":  ["Maize", "MAIZE", "Maize(White)"],
    "potato": ["Potato", "POTATO"],
    "onion":  ["Onion", "ONION", "Onion Medium"],
}

# State name normalization: AGMARKNET spelling -> our canonical spelling
STATE_NORM = {
    "Uttar Pradesh": "Uttar Pradesh",
    "Maharashtra": "Maharashtra",
    "Punjab": "Punjab",
    "Haryana": "Haryana",
    "Madhya Pradesh": "Madhya Pradesh",
    "Rajasthan": "Rajasthan",
    "Gujarat": "Gujarat",
    "Karnataka": "Karnataka",
    "Andhra Pradesh": "Andhra Pradesh",
    "Telangana": "Telangana",
    "Bihar": "Bihar",
    "West Bengal": "West Bengal",
    "Tamil Nadu": "Tamil Nadu",
    "Odisha": "Odisha",
    "Chhattisgarh": "Chhattisgarh",
    "Jharkhand": "Jharkhand",
    "Uttarakhand": "Uttarakhand",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu and Kashmir": "Jammu and Kashmir",
    "Kerala": "Kerala",
    "Assam": "Assam",
    "Delhi": "Delhi",
    "Chandigarh": "Chandigarh",
    "Goa": "Goa",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Tripura": "Tripura",
    "Puducherry": "Puducherry",
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
}


def _fetch_page(commodity, from_date, to_date, offset=0, limit=1000):
    """Fetch one page of records from data.gov.in AGMARKNET API."""
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "filters[commodity]": commodity,
        "filters[arrival_date]": "{}:{}".format(from_date, to_date),
    }
    try:
        resp = requests.get(AGMARKNET_API_BASE, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("records", [])
    except requests.RequestException as e:
        logger.warning("API request failed for {}: {}".format(commodity, e))
        return []


def fetch_commodity_data(crop, from_date, to_date):
    """Fetch all pages for a commodity between two dates."""
    commodities = CROP_COMMODITY_MAP.get(crop, [crop.title()])
    all_records = []
    for commodity in commodities:
        logger.info("  Fetching {} from {} to {} ...".format(commodity, from_date, to_date))
        offset = 0
        while True:
            records = _fetch_page(commodity, from_date, to_date, offset=offset)
            if not records:
                break
            all_records.extend(records)
            if len(records) < 1000:
                break
            offset += 1000
            time.sleep(0.3)
    if not all_records:
        logger.warning("  No records returned for {}".format(crop))
        return pd.DataFrame()
    return pd.DataFrame(all_records)


def _parse_agmarknet_df(df, crop):
    """Normalize AGMARKNET raw records into our schema."""
    if df.empty:
        return pd.DataFrame()
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "arrival_date" in cl or (cl == "date"):
            col_map[col] = "arrival_date"
        elif "modal" in cl:
            col_map[col] = "modal_price"
        elif "min" in cl and "price" in cl:
            col_map[col] = "min_price"
        elif "max" in cl and "price" in cl:
            col_map[col] = "max_price"
        elif "state" in cl:
            col_map[col] = "state"
    df = df.rename(columns=col_map)
    if "arrival_date" not in df.columns or "modal_price" not in df.columns:
        logger.warning("Missing required columns. Available: {}".format(list(df.columns)))
        return pd.DataFrame()
    df["ds"] = pd.to_datetime(df["arrival_date"], dayfirst=True, errors="coerce")
    df["y"] = pd.to_numeric(df["modal_price"], errors="coerce")
    if "min_price" in df.columns:
        df["min_price"] = pd.to_numeric(df["min_price"], errors="coerce")
    if "max_price" in df.columns:
        df["max_price"] = pd.to_numeric(df["max_price"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[df["y"] > 0].copy()
    df["crop"] = crop
    if "state" in df.columns:
        df["state"] = df["state"].map(lambda s: STATE_NORM.get(s, s))
    return df


def build_national_daily(df, crop):
    """Aggregate to national daily modal price (median across markets)."""
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("ds")["y"].median().reset_index()
    agg["crop"] = crop
    return agg[["ds", "crop", "y"]]


def build_state_daily(df, crop):
    """Aggregate to state-level daily price with statistics."""
    if df.empty or "state" not in df.columns:
        return pd.DataFrame()
    has_min = "min_price" in df.columns
    has_max = "max_price" in df.columns
    agg_dict = {
        "y": ("y", "median"),
        "market_count": ("y", "count"),
    }
    if has_min:
        agg_dict["min_price"] = ("min_price", "min")
    else:
        agg_dict["min_price"] = ("y", "min")
    if has_max:
        agg_dict["max_price"] = ("max_price", "max")
    else:
        agg_dict["max_price"] = ("y", "max")
    grp = df.groupby(["ds", "state"]).agg(**agg_dict).reset_index()
    grp["crop"] = crop
    grp["price_range"] = (grp["max_price"] - grp["min_price"]).clip(lower=0)
    grp["rolling_std_7"] = 0.0
    return grp[["ds", "crop", "state", "y", "min_price", "max_price",
                "market_count", "price_range", "rolling_std_7"]]


def fetch_and_extend(start_date=None, end_date=None):
    """
    Main entry point. Fetch AGMARKNET data and append to CSVs.
    Args:
        start_date: 'YYYY-MM-DD'. Defaults to day after existing data ends.
        end_date:   'YYYY-MM-DD'. Defaults to today.
    """
    national_path = DATA_DIR / "real_historical_prices.csv"
    state_path = DATA_DIR / "real_historical_prices_state.csv"

    if start_date is None:
        if national_path.exists():
            existing = pd.read_csv(national_path, parse_dates=["ds"])
            latest = existing["ds"].max().date()
            start_dt = latest + timedelta(days=1)
            start_date = str(start_dt)
            logger.info("Existing national data ends {}. Fetching from {}.".format(latest, start_date))
        else:
            start_date = "2019-01-01"
    if end_date is None:
        end_date = str(date.today())

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        logger.info("No new data needed. Exiting.")
        return

    logger.info("Fetching AGMARKNET data: {} to {}".format(start_date, end_date))

    new_national_frames = []
    new_state_frames = []

    for crop in ["wheat", "rice", "maize", "potato", "onion"]:
        logger.info("\n[{}] Starting fetch ...".format(crop.upper()))
        raw = fetch_commodity_data(crop, start_date, end_date)
        parsed = _parse_agmarknet_df(raw, crop)
        if parsed.empty:
            logger.warning("[{}] No usable data fetched -- skipping.".format(crop.upper()))
            continue
        nat = build_national_daily(parsed, crop)
        if not nat.empty:
            new_national_frames.append(nat)
            logger.info("[{}] National: {} new daily rows".format(crop.upper(), len(nat)))
        st = build_state_daily(parsed, crop)
        if not st.empty:
            new_state_frames.append(st)
            logger.info("[{}] State: {} new daily rows across {} states".format(
                crop.upper(), len(st), st["state"].nunique()))

    # Update national CSV
    if new_national_frames:
        new_nat = pd.concat(new_national_frames, ignore_index=True)
        new_nat["ds"] = pd.to_datetime(new_nat["ds"])
        if national_path.exists():
            existing_nat = pd.read_csv(national_path, parse_dates=["ds"])
            combined_nat = pd.concat([existing_nat, new_nat], ignore_index=True)
        else:
            combined_nat = new_nat
        combined_nat = combined_nat.sort_values(["crop", "ds"]).drop_duplicates(
            subset=["ds", "crop"], keep="last").reset_index(drop=True)
        combined_nat.to_csv(national_path, index=False)
        logger.info("National CSV updated -> {} total rows".format(len(combined_nat)))
    else:
        logger.warning("No new national data to write.")

    # Update state CSV
    if new_state_frames:
        new_st = pd.concat(new_state_frames, ignore_index=True)
        new_st["ds"] = pd.to_datetime(new_st["ds"])
        if state_path.exists():
            existing_st = pd.read_csv(state_path, parse_dates=["ds"])
            combined_st = pd.concat([existing_st, new_st], ignore_index=True)
        else:
            combined_st = new_st
        combined_st = combined_st.sort_values(["crop", "state", "ds"]).drop_duplicates(
            subset=["ds", "crop", "state"], keep="last").reset_index(drop=True)

        def _add_rolling_std(grp):
            grp = grp.sort_values("ds").copy()
            grp["rolling_std_7"] = grp["y"].shift(1).rolling(7, min_periods=2).std().fillna(0.0)
            return grp

        combined_st = combined_st.groupby(["crop", "state"], group_keys=False).apply(_add_rolling_std)
        combined_st = combined_st.sort_values(["crop", "state", "ds"]).reset_index(drop=True)
        combined_st.to_csv(state_path, index=False)
        logger.info("State CSV updated -> {} total rows".format(len(combined_st)))
    else:
        logger.warning("No new state data to write.")

    logger.info("\n=== FETCH SUMMARY ===")
    for path, label in [(national_path, "National"), (state_path, "State")]:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["ds"])
            logger.info("  {}: {} rows | {} to {}".format(
                label, len(df), df["ds"].min().date(), df["ds"].max().date()))


if __name__ == "__main__":
    fetch_and_extend()
