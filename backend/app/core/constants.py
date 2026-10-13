"""
constants.py — Shared constants for AgroIntel.

Contains season definitions, crop name mappings, feature column lists,
black swan event registry, and default soil profiles.
"""

from datetime import date

# ── Indian Agricultural Seasons ───────────────────────────────────────────────

SEASON_ALIASES: dict[str, str] = {
    "rainy": "Kharif",
    "rainy season": "Kharif",
    "monsoon": "Kharif",
    "kharif": "Kharif",
    "winter": "Rabi",
    "winter season": "Rabi",
    "rabi": "Rabi",
    "summer": "Summer",
    "summer season": "Summer",
    "zaid": "Summer",
    "whole year": "Whole Year",
    "all year": "Whole Year",
    "perennial": "Whole Year",
}

SEASONS = {
    "Kharif": {"months": [6, 7, 8, 9, 10], "label": "Rainy (Monsoon, Jun–Oct)"},
    "Rabi": {"months": [11, 12, 1, 2, 3], "label": "Winter (Nov–Mar)"},
    "Zaid": {"months": [4, 5], "label": "Summer (Apr–May)"},
    "Summer": {"months": [4, 5], "label": "Summer (Apr–May)"},
    "Rainy": {"months": [6, 7, 8, 9, 10], "label": "Rainy (Monsoon, Jun–Oct)"},
    "Winter": {"months": [11, 12, 1, 2, 3], "label": "Winter (Nov–Mar)"},
    "Whole Year": {"months": list(range(1, 13)), "label": "Whole Year / Perennial"},
}

SEASON_ENCODING = {
    "Kharif": 0, "Rabi": 1, "Zaid": 2, "Summer": 2,
    "Rainy": 0, "Winter": 1, "Whole Year": 0,
}


def normalize_season(season_input: str | None) -> str:
    """Normalize any season name (e.g. Rainy, Winter, Summer, Kharif, Rabi, Zaid) to canonical season."""
    if not season_input:
        return get_current_season()
    s = season_input.strip().lower()
    return SEASON_ALIASES.get(s, season_input.strip().capitalize())


def get_current_season() -> str:
    """Return the current Indian agricultural season based on today's month."""
    month = date.today().month
    if month in (6, 7, 8, 9, 10):
        return "Kharif"
    elif month in (11, 12, 1, 2, 3):
        return "Rabi"
    else:
        return "Zaid"


# ── Crop Seasons Mapping ─────────────────────────────────────────────────────
# Which crops grow in which season(s).

CROP_SEASONS: dict[str, list[str]] = {
    "rice": ["Kharif"],
    "wheat": ["Rabi"],
    "maize": ["Kharif", "Rabi"],
    "cotton": ["Kharif"],
    "sugarcane": ["Kharif", "Rabi", "Zaid"],
    "groundnut": ["Kharif", "Rabi"],
    "soybean": ["Kharif"],
    "mustard": ["Rabi"],
    "chickpea": ["Rabi"],
    "lentil": ["Rabi"],
    "mungbean": ["Kharif", "Zaid"],
    "blackgram": ["Kharif"],
    "pigeonpeas": ["Kharif"],
    "kidneybeans": ["Rabi"],
    "mothbeans": ["Kharif"],
    "coconut": ["Kharif", "Rabi", "Zaid"],
    "banana": ["Kharif", "Rabi", "Zaid"],
    "mango": ["Kharif", "Rabi", "Zaid"],
    "coffee": ["Kharif", "Rabi", "Zaid"],
    "jute": ["Kharif"],
    "apple": ["Rabi"],
    "grapes": ["Rabi"],
    "orange": ["Rabi"],
    "papaya": ["Kharif", "Rabi", "Zaid"],
    "pomegranate": ["Kharif", "Rabi"],
    "watermelon": ["Zaid"],
    "muskmelon": ["Zaid"],
    "onion": ["Rabi", "Kharif"],
    "potato": ["Rabi"],
}


# ── Crop Name Mapping ────────────────────────────────────────────────────────
# Maps mandi / region_crop_mapping names → Kaggle RF dataset labels.

MANDI_TO_RF_LABEL: dict[str, str] = {
    # Cereals
    "paddy (dhan)(common)": "rice",
    "rice": "rice",
    "rice (paddy - husked)": "rice",
    "maize": "maize",
    "wheat": "wheat",
    # Pulses
    "bengal gram (gram)(whole)": "chickpea",
    "chickpea": "chickpea",
    "arhar (tur/red gram)(whole)": "pigeonpeas",
    "red gram": "pigeonpeas",
    "green gram (moong)(whole)": "mungbean",
    "moong": "mungbean",
    "black gram (urd beans)(whole)": "blackgram",
    "urad": "blackgram",
    "masoor": "lentil",
    "lentil": "lentil",
    "rajma": "kidneybeans",
    "kidney beans": "kidneybeans",
    "moth": "mothbeans",
    "moth beans": "mothbeans",
    # Fruits
    "banana": "banana",
    "mango": "mango",
    "mango (raw-loss green)": "mango",
    "pomegranate": "pomegranate",
    "grapes": "grapes",
    "apple": "apple",
    "orange": "orange",
    "papaya": "papaya",
    "papaya (raw)": "papaya",
    "water melon": "watermelon",
    "watermelon": "watermelon",
    "musk melon": "muskmelon",
    "muskmelon": "muskmelon",
    # Cash crops
    "cotton": "cotton",
    "cotton (kapas)": "cotton",
    "raw cotton": "cotton",
    "coconut": "coconut",
    "coconut seed": "coconut",
    "jute": "jute",
    "raw jute": "jute",
    "coffee": "coffee",
    # Oilseeds
    "groundnut": "groundnut",
    "soyabean": "soybean",
    "soybean": "soybean",
    # Vegetables
    "onion": "onion",
    "potato": "potato",
}


# ── Supported Price Prediction Crops ──────────────────────────────────────────

PRICE_PREDICTION_CROPS = ["wheat", "rice", "maize", "potato", "onion"]


# ── Official Government Minimum Support Prices (MSP) ──────────────────────────
# Source: Cabinet Committee on Economic Affairs (CCEA) & Ministry of Agriculture & Farmers Welfare
# Units: ₹/quintal (100 kg)
# NOTE: MSP is a statutory floor price, NEVER to be substituted for current market spot/modal price.
OFFICIAL_MSP: dict[str, dict] = {
    "wheat": {
        "msp_price": 2425.0,
        "season": "RMS 2025-26",
        "description": "CCEA Minimum Support Price for Wheat (FAQ Grade)",
        "source": "Cabinet Committee on Economic Affairs (CCEA)",
        "effective_year": "2025-26",
    },
    "rice": {
        "msp_price": 2300.0,
        "season": "KMS 2024-25",
        "description": "CCEA Minimum Support Price for Paddy Common (Grade A: ₹2,320)",
        "source": "Cabinet Committee on Economic Affairs (CCEA)",
        "effective_year": "2024-25",
    },
    "paddy": {
        "msp_price": 2300.0,
        "season": "KMS 2024-25",
        "description": "CCEA Minimum Support Price for Paddy Common",
        "source": "Cabinet Committee on Economic Affairs (CCEA)",
        "effective_year": "2024-25",
    },
    "maize": {
        "msp_price": 2225.0,
        "season": "KMS 2024-25",
        "description": "CCEA Minimum Support Price for Maize",
        "source": "Cabinet Committee on Economic Affairs (CCEA)",
        "effective_year": "2024-25",
    },
    "potato": {
        "msp_price": None,
        "season": "N/A",
        "description": "No Central Statutory MSP (Eligible for State Market Intervention Scheme / MIS)",
        "source": "Ministry of Agriculture (Horticulture Division)",
        "effective_year": "2024-26",
    },
    "onion": {
        "msp_price": None,
        "season": "N/A",
        "description": "No Central Statutory MSP (Procured via Price Stabilisation Fund PSF / NAFED / NCCF)",
        "source": "Department of Consumer Affairs (Price Stabilisation Fund)",
        "effective_year": "2024-26",
    },
}


def get_official_msp(crop: str) -> dict | None:
    """Retrieve official government MSP details for a commodity."""
    if not crop:
        return None
    return OFFICIAL_MSP.get(crop.strip().lower())



# ── Price Model Feature Columns ───────────────────────────────────────────────
# Exactly 11 features matching implementation plan specification.

PRICE_FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "month",
    "season",
    "monthly_avg_temp",
    "monthly_total_rainfall",
    "black_swan",
]


# ── Black Swan Events ─────────────────────────────────────────────────────────

BLACK_SWAN_EVENTS = [
    {"start": "2019-06-01", "end": "2019-09-30", "label": "2019 Drought"},
    {"start": "2020-03-15", "end": "2020-06-30", "label": "COVID-19 Lockdown"},
    {"start": "2020-07-01", "end": "2020-12-31", "label": "COVID-19 Recovery"},
    {"start": "2022-02-24", "end": "2022-06-30", "label": "Russia-Ukraine War"},
    {"start": "2022-07-01", "end": "2022-12-31", "label": "War Supply Disruption"},
    {"start": "2023-01-01", "end": "2023-09-30", "label": "Post-War Inflation"},
]


# ── Default Soil NPK Values by Soil Type ──────────────────────────────────────

DEFAULT_SOIL_VALUES: dict[str, dict[str, float]] = {
    "Alluvial Soil": {"N": 80, "P": 45, "K": 50, "pH": 6.8},
    "Black Soil": {"N": 55, "P": 30, "K": 65, "pH": 7.8},
    "Red Soil": {"N": 40, "P": 25, "K": 35, "pH": 6.2},
    "Laterite Soil": {"N": 35, "P": 20, "K": 30, "pH": 5.5},
    "Desert Soil": {"N": 20, "P": 15, "K": 25, "pH": 8.2},
    "Mountain Soil": {"N": 60, "P": 35, "K": 40, "pH": 5.8},
    "Coastal Alluvial Soil": {"N": 70, "P": 40, "K": 55, "pH": 7.2},
    "Sandy Soil": {"N": 25, "P": 18, "K": 20, "pH": 7.0},
}

DEFAULT_SOIL_FALLBACK = {"N": 50, "P": 30, "K": 40, "pH": 6.5}


# ── Open-Meteo Configuration ─────────────────────────────────────────────────

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_WEATHER_LAT = 28.6139
DEFAULT_WEATHER_LON = 77.2090
