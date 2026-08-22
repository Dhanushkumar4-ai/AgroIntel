import pandas as pd
import numpy as np
import requests
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataIngestion:
    
    @staticmethod
    def fetch_live_weather(location: str = "Delhi") -> dict:
        """
        Fetches real-time weather from OpenWeather API.
        Falls back to simulation if no API key or API fails.
        """
        api_key = settings.OPENWEATHER_API_KEY
        if api_key and api_key != "mock_weather_key":
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "rainfall": data.get("rain", {}).get("1h", 0) * 24 * 30, # Approx seasonal rainfall based on current
                    }
            except Exception as e:
                logger.warning(f"Live Weather API failed for {location}: {e}. Falling back to simulation.")
                
        # 2. Fallback Simulator ΓÇö seeded to today's date so values are stable within a day
        seed = int(datetime.now().strftime("%Y%m%d")) + hash(location) % 10000
        rng = np.random.default_rng(abs(seed))
        return {
            "temperature": round(28.5 + rng.normal(0, 3), 1),
            "humidity":    round(60.0 + rng.normal(0, 10), 1),
            "rainfall":    round(150.0 + rng.normal(0, 50), 1),
        }

    @staticmethod
    def fetch_live_market_data(crop_name: str, state: str = "All") -> dict:
        """
        Fetches mandi prices from data.gov.in (updates every ~3 days, not live-streaming).
        Multi-source strategy:
          1. data.gov.in OGD API (primary, 3-day update cycle)
          2. Cache fallback (Γëñ3 days old)
          3. Historical CSV median (informed estimate)
          4. Simulation (last resort)

        Always returns data_quality badge and data_age_hours for UI transparency.
        """
        crop_clean  = crop_name.lower().strip()
        state_clean = state.strip()

        # 1. Setup cache path
        base_dir   = Path(__file__).resolve().parent.parent.parent
        cache_path = base_dir / "app" / "data" / "mandi_cache.json"

        # Load existing cache
        cache = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read mandi cache: {e}")

        def _save_cache(key: str, price: float):
            """Helper: write a price entry to cache file."""
            cache[key] = {"price": price, "timestamp": datetime.now().isoformat()}
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except Exception as ce:
                logger.warning(f"Failed to write mandi cache: {ce}")

        def _age_hours(timestamp_str: str) -> float:
            """Return how many hours ago a timestamp was."""
            try:
                ts = datetime.fromisoformat(timestamp_str)
                return round((datetime.now() - ts).total_seconds() / 3600, 1)
            except Exception:
                return 9999.0

        # 2. Attempt Live API fetch ΓÇö data.gov.in OGD
        api_key = settings.MARKET_DATA_API_KEY
        if api_key and api_key != "mock_market_key":
            api_commodity_map = {
                "wheat":  "Wheat",
                "rice":   "Paddy (Dhan)(Common)",
                "maize":  "Maize",
                "potato": "Potato",
                "onion":  "Onion",
            }
            api_crop = api_commodity_map.get(crop_clean, crop_name.capitalize())

            try:
                url = (
                    f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
                    f"?api-key={api_key}&format=json&filters[commodity]={api_crop}&limit=100"
                )
                if state_clean.lower() != "all":
                    url += f"&filters[state]={state_clean.capitalize()}"

                logger.info(f"Fetching live mandi price for {api_crop} in {state_clean}...")
                response = requests.get(url, timeout=0.5)

                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])
                    if records:
                        # Extract modal prices from most recent arrival date
                        # Sort records by arrival_date descending to get latest batch
                        dated = [(r.get("arrival_date", ""), r) for r in records]
                        dated.sort(reverse=True)
                        # Find most recent date batch
                        latest_date = dated[0][0] if dated else ""
                        recent_records = [r for d, r in dated if d == latest_date] if latest_date else records

                        prices = [
                            float(r["modal_price"])
                            for r in recent_records
                            if str(r.get("modal_price", "")).replace(".", "", 1).isdigit()
                        ]
                        if prices:
                            avg_price = round(sum(prices) / len(prices), 2)
                            cache_key = f"{crop_clean}:{state_clean}"
                            _save_cache(cache_key, avg_price)

                            # data.gov.in updates every ~3 days ΓÇö mark with note
                            return {
                                "current_price": avg_price,
                                "data_source":   "api_data_gov_in",
                                "data_quality":  "fresh",
                                "data_age_hours": 0,
                                "note": "data.gov.in updates every 2-3 days (not real-time streaming)",
                                "arrival_date":  latest_date,
                            }
            except Exception as e:
                logger.warning(f"Live Market API failed for {crop_name} in {state}: {e}.")

        # 3. Cache Fallback ΓÇö use if within 3 days (matching API update cycle)
        cache_key          = f"{crop_clean}:{state_clean}"
        cache_fallback_key = f"{crop_clean}:All"

        for key in [cache_key, cache_fallback_key]:
            if key in cache:
                cached_data = cache[key]
                ts_str      = cached_data.get("timestamp", "")
                age_h       = _age_hours(ts_str)
                try:
                    ts = datetime.fromisoformat(ts_str)
                    # Cache valid for 3 days (data.gov.in update frequency)
                    if datetime.now() - ts < timedelta(days=3):
                        logger.info(f"Serving cached mandi price for {key} (age: {age_h:.1f}h)")
                        quality = "fresh" if age_h < 24 else ("stale" if age_h < 72 else "old")
                        return {
                            "current_price":  cached_data["price"],
                            "data_source":    "cached_api",
                            "data_quality":   quality,
                            "data_age_hours": age_h,
                            "cached_time":    ts_str,
                        }
                except Exception as te:
                    logger.warning(f"Error parsing cache timestamp: {te}")

        # 4. Yahoo Finance commodity prices (free, no API key needed)
        # Convert commodity futures to INR per quintal (100 kg)
        # ZW=F (wheat): cents/bushel, 1 bushel = 27.2155 kg
        # ZC=F (corn/maize): cents/bushel, 1 bushel = 25.4012 kg
        # ZS=F (soybean): cents/bushel, 1 bushel = 27.2155 kg
        # ZR=F (rough rice): USD/cwt, 1 cwt = 45.3592 kg
        YAHOO_TICKERS = {
            "wheat":   ("ZW=F", "cents_bushel", 27.2155),
            "rice":    ("ZR=F", "usd_cwt",      0.453592),
            "maize":   ("ZC=F", "cents_bushel", 25.4012),
            "soybean": ("ZS=F", "cents_bushel", 27.2155),
        }
        ticker_info = YAHOO_TICKERS.get(crop_clean)
        if ticker_info:
            ticker_sym, unit_type, kg_per_unit = ticker_info
            try:
                yf_url  = (
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_sym}"
                    f"?interval=1d&range=1d"
                )
                yf_resp = requests.get(yf_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if yf_resp.status_code == 200:
                    yf_data = yf_resp.json()
                    res     = yf_data.get("chart", {}).get("result", [])
                    if res:
                        meta      = res[0].get("meta", {})
                        raw_price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
                        if raw_price and raw_price > 0:
                            # Fetch live USD/INR exchange rate
                            try:
                                fx = requests.get(
                                    "https://query1.finance.yahoo.com/v8/finance/chart/USDINR=X?interval=1d&range=1d",
                                    timeout=4, headers={"User-Agent": "Mozilla/5.0"}
                                ).json()
                                fx_r = fx.get("chart", {}).get("result", [])
                                usd_inr = fx_r[0]["meta"]["regularMarketPrice"] if fx_r else 83.5
                            except Exception:
                                usd_inr = 83.5

                            if unit_type == "cents_bushel":
                                # raw_price is in US cents. Convert to USD per kg:
                                usd_per_kg = (raw_price / 100.0) / kg_per_unit
                            else:
                                # raw_price is in USD per cwt. Convert to USD per kg:
                                usd_per_kg = raw_price / (kg_per_unit * 100.0)

                            inr_q = round(usd_per_kg * 100.0 * usd_inr, 2)

                            state_adj = {
                                "punjab": 0.95, "haryana": 0.95,
                                "maharashtra": 1.08, "karnataka": 1.08,
                                "west bengal": 1.05, "uttar pradesh": 0.98,
                            }.get(state_clean.lower(), 1.0)
                            inr_q = round(inr_q * state_adj, 2)

                            logger.info(f"Yahoo {ticker_sym}: {raw_price} ΓåÆ Γé╣{inr_q}/Q ({state_clean})")
                            cache_key = f"{crop_clean}:{state_clean}"
                            _save_cache(cache_key, inr_q)
                            return {
                                "current_price":  inr_q,
                                "data_source":    "yahoo_finance",
                                "data_quality":   "fresh",
                                "data_age_hours": 0,
                                "note":           f"Global commodity ({ticker_sym}) converted to INR/quintal",
                            }
            except Exception as ye:
                logger.warning(f"Yahoo Finance failed for {crop_name}: {ye}")

        # 5. MSP-anchored estimate (India MSP 2024-25 + stable daily noise)
        MSP_PRICES = {
            "wheat":     2275,
            "rice":      2300,
            "maize":     2090,
            "cotton":    7121,
            "soybean":   4892,
            "mustard":   5650,
            "groundnut": 6783,
            "potato":    1200,
            "onion":     1800,
            "tomato":    1500,
        }
        base       = MSP_PRICES.get(crop_clean, 2200)
        today_seed = int(datetime.now().strftime("%Y%m%d"))
        crop_seed  = sum(ord(c) for c in crop_clean)
        state_seed = sum(ord(c) for c in state_clean)
        rng        = np.random.default_rng(today_seed + crop_seed + state_seed)
        state_mul  = {
            "punjab": 0.95, "haryana": 0.95, "maharashtra": 1.10,
            "karnataka": 1.08, "uttar pradesh": 0.98, "west bengal": 1.03,
        }.get(state_clean.lower(), 1.0)
        noise      = rng.normal(0, base * 0.03)
        final_price = round(max(500, base * state_mul + noise), 2)
        logger.info(f"MSP estimate: Γé╣{final_price}/Q for {crop_name} in {state_clean}")
        return {
            "current_price":  final_price,
            "data_source":    "msp_estimate",
            "data_quality":   "estimated",
            "data_age_hours": None,
            "note":           "Based on India MSP 2024-25 + regional adjustment",
        }

