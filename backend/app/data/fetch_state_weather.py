"""
fetch_state_weather.py -- Fetch state-level monthly weather data from Open-Meteo.

Replaces the single national monthly weather average with per-state monthly
averages using representative lat/lon coordinates for each state capital.

Run:
    cd backend
    python -m app.data.fetch_state_weather

Output:
    app/data/weather_history_state.csv  -- state x year x month averages
    app/data/weather_history.csv        -- updated national average (backward compatible)
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"

# Representative capital city coordinates for each Indian state
STATE_COORDS = {
    "Andhra Pradesh":      (15.9129, 79.7400),  # Amaravati
    "Andaman and Nicobar": (11.7401, 92.6586),  # Port Blair
    "Assam":               (26.2006, 92.9376),  # Guwahati
    "Bihar":               (25.5941, 85.1376),  # Patna
    "Chandigarh":          (30.7333, 76.7794),  # Chandigarh
    "Chhattisgarh":        (21.2514, 81.6296),  # Raipur
    "Delhi":               (28.6139, 77.2090),  # New Delhi
    "Goa":                 (15.2993, 74.1240),  # Panaji
    "Gujarat":             (23.0225, 72.5714),  # Ahmedabad
    "Haryana":             (29.0588, 76.0856),  # Chandigarh (shared)
    "Himachal Pradesh":    (31.1048, 77.1734),  # Shimla
    "Jammu and Kashmir":   (34.0837, 74.7973),  # Srinagar
    "Jharkhand":           (23.3441, 85.3096),  # Ranchi
    "Karnataka":           (12.9716, 77.5946),  # Bangalore
    "Kerala":              (8.5241,  76.9366),  # Thiruvananthapuram
    "Madhya Pradesh":      (23.2599, 77.4126),  # Bhopal
    "Maharashtra":         (19.0760, 72.8777),  # Mumbai
    "Manipur":             (24.8170, 93.9368),  # Imphal
    "Meghalaya":           (25.5788, 91.8933),  # Shillong
    "Mizoram":             (23.1645, 92.9376),  # Aizawl
    "Nagaland":            (25.6747, 94.1086),  # Kohima
    "Odisha":              (20.2961, 85.8245),  # Bhubaneswar
    "Puducherry":          (11.9416, 79.8083),  # Puducherry
    "Punjab":              (30.9010, 75.8573),  # Ludhiana
    "Rajasthan":           (26.9124, 75.7873),  # Jaipur
    "Tamil Nadu":          (13.0827, 80.2707),  # Chennai
    "Telangana":           (17.3850, 78.4867),  # Hyderabad
    "Tripura":             (23.8315, 91.2868),  # Agartala
    "Uttar Pradesh":       (26.8467, 80.9462),  # Lucknow
    "Uttarakhand":         (30.0668, 79.0193),  # Dehradun
    "West Bengal":         (22.5726, 88.3639),  # Kolkata
}

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_state_weather(state, lat, lon, start_year=2019, end_year=None):
    """
    Fetch monthly avg temperature and total rainfall for a state using Open-Meteo Archive API.
    Returns a DataFrame with columns: state, year, month, avg_temp, total_rainfall
    """
    if end_year is None:
        end_year = date.today().year

    end_date_str = min(f"{end_year}-12-31", (date.today() - timedelta(days=2)).isoformat())

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": "{}-01-01".format(start_year),
        "end_date":   end_date_str,
        "daily":      "temperature_2m_mean,precipitation_sum",
        "timezone":   "Asia/Kolkata",
    }
    for attempt in range(4):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
            if resp.status_code == 429:
                wait_sec = 6 * (attempt + 1)
                logger.info(f"Rate limited (429) for {state}. Waiting {wait_sec}s before retry {attempt+1}...")
                time.sleep(wait_sec)
                continue
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            if not daily or "time" not in daily:
                logger.warning("No daily data for {} ({},{})".format(state, lat, lon))
                return pd.DataFrame()
            df = pd.DataFrame({
                "date":  pd.to_datetime(daily["time"]),
                "temp":  daily.get("temperature_2m_mean", [None] * len(daily["time"])),
                "rain":  daily.get("precipitation_sum",   [None] * len(daily["time"])),
            })
            df = df.dropna(subset=["temp", "rain"])
            df["year"]  = df["date"].dt.year
            df["month"] = df["date"].dt.month
            monthly = df.groupby(["year", "month"]).agg(
                avg_temp=("temp", "mean"),
                total_rainfall=("rain", "sum"),
            ).reset_index()
            monthly["state"] = state
            return monthly[["state", "year", "month", "avg_temp", "total_rainfall"]]
        except Exception as e:
            if attempt == 3:
                logger.warning("Weather fetch failed for {}: {}".format(state, e))
                return pd.DataFrame()
            time.sleep(3)
    return pd.DataFrame()


def fetch_all_state_weather(start_year=2019, end_year=None):
    """Fetch weather for all states and save to CSV files."""
    if end_year is None:
        end_year = date.today().year

    state_path   = DATA_DIR / "weather_history_state.csv"
    national_path = DATA_DIR / "weather_history.csv"

    all_frames = []
    already_fetched = set()
    if state_path.exists():
        try:
            existing_df = pd.read_csv(state_path)
            if not existing_df.empty and "state" in existing_df.columns:
                all_frames.append(existing_df)
                already_fetched = set(existing_df["state"].unique())
                logger.info("Loaded {} existing states from {}".format(len(already_fetched), state_path))
        except Exception as e:
            logger.warning("Could not read existing state weather: {}".format(e))

    missing_states = {k: v for k, v in STATE_COORDS.items() if k not in already_fetched}
    total = len(STATE_COORDS)
    logger.info("States already have data: {}/{} | Remaining to fetch: {}".format(
        len(already_fetched), total, len(missing_states)))

    for i, (state, (lat, lon)) in enumerate(missing_states.items(), 1):
        logger.info("[{}/{}] Fetching weather for {} ({:.4f}, {:.4f}) ...".format(
            i, len(missing_states), state, lat, lon))
        df = fetch_state_weather(state, lat, lon, start_year=start_year, end_year=end_year)
        if not df.empty:
            all_frames.append(df)
            logger.info("  -> {} rows for {}".format(len(df), state))
        time.sleep(2.0)  # Rate limiting for Open-Meteo

    if not all_frames:
        logger.error("No weather data fetched for any state.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["state", "year", "month"]).reset_index(drop=True)
    combined["avg_temp"]       = combined["avg_temp"].round(2)
    combined["total_rainfall"] = combined["total_rainfall"].round(2)

    combined.to_csv(state_path, index=False)
    logger.info("\nState weather saved: {} rows -> {}".format(len(combined), state_path))

    # Update national weather_history.csv (national average across all states)
    national = combined.groupby(["year", "month"]).agg(
        avg_temp=("avg_temp", "mean"),
        total_rainfall=("total_rainfall", "mean"),
    ).reset_index()
    national["avg_temp"]       = national["avg_temp"].round(2)
    national["total_rainfall"] = national["total_rainfall"].round(2)
    national.to_csv(national_path, index=False)
    logger.info("National weather updated: {} rows -> {}".format(len(national), national_path))

    logger.info("\n=== WEATHER FETCH SUMMARY ===")
    logger.info("States fetched: {}".format(combined["state"].nunique()))
    logger.info("Year range: {} - {}".format(combined["year"].min(), combined["year"].max()))
    logger.info("National avg temp range: {:.1f}C - {:.1f}C".format(
        national["avg_temp"].min(), national["avg_temp"].max()))


if __name__ == "__main__":
    fetch_all_state_weather()
