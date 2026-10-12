"""
config.py — Application settings for AgroIntel.

Uses only:
  - Government Mandi API (data.gov.in) for latest market price display
  - Open-Meteo (free, no key required) for weather
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory if present
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "AgroIntel — AI Crop Advisory System"
    API_V1_STR: str = "/api"
    VERSION: str = "4.0.0"

    # ── Mandi API (data.gov.in) ─────────────────────────────────────────────
    # Used ONLY for fetching the latest available market price (display + sell/hold).
    # NOT used as a primary model input.
    MARKET_DATA_API_KEY: str = os.getenv(
        "MARKET_DATA_API_KEY",
        "579b464db66ec23bdd00000100b983ce593940d87db6f88c1a387a12",
    )
    MARKET_DATA_BASE_URL: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    # ── Open-Meteo (no API key required) ────────────────────────────────────
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    # ── File Paths ─────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "app" / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    # ── Cache Settings ─────────────────────────────────────────────────────
    # Mandi price cache TTL in seconds (1 hour = 3600s, auto-refreshes daily updates)
    MANDI_CACHE_TTL_SECONDS: int = int(os.getenv("MANDI_CACHE_TTL_SECONDS", "3600"))
    # Weather cache TTL in seconds (6 hours)
    WEATHER_CACHE_TTL_SECONDS: int = 21_600

    # ── Mandi Price Freshness Threshold ───────────────────────────────────
    # If mandi price is older than this many days, treat as stale.
    MANDI_FRESH_DAYS: int = 3


settings = Settings()
