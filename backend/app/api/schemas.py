"""
schemas.py — Pydantic Request & Response Validation Schemas for AgroIntel v4.0.

Provides strict type validation, field boundaries, and OpenAPI examples for:
  - Price Prediction Query & Response
  - Crop Recommendation Request & Response
  - System Version & Health Status
  - Model Registry Details & Error Payloads
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


# ── Request Models ────────────────────────────────────────────────────────────

class PricePredictionQuery(BaseModel):
    crop: str = Field(
        ...,
        description="Target agricultural crop name",
        example="wheat",
    )
    state: Optional[str] = Field(
        None,
        description="Optional Indian state for market filtering",
        example="Karnataka",
    )
    horizon_days: int = Field(
        30,
        description="Target forecast horizon in days (7, 15, 30, 60, 90)",
        example=30,
    )
    lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Optional latitude for weather lookup",
        example=18.52,
    )
    lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Optional longitude for weather lookup",
        example=73.85,
    )

    @validator("horizon_days")
    def validate_horizon(cls, v):
        valid_horizons = [7, 15, 30, 60, 90]
        if v not in valid_horizons:
            raise ValueError(f"horizon_days must be one of {valid_horizons}")
        return v

    @validator("crop")
    def validate_crop_name(cls, v):
        v_clean = v.lower().strip()
        supported = ["wheat", "rice", "maize", "potato", "onion"]
        if v_clean not in supported:
            raise ValueError(f"Unsupported crop '{v}'. Supported crops: {supported}")
        return v_clean


class CropRecommendationRequest(BaseModel):
    state: str = Field(
        ...,
        min_length=2,
        description="Indian state name",
        example="Maharashtra",
    )
    district: str = Field(
        ...,
        min_length=2,
        description="Target district name",
        example="Pune",
    )
    season: Optional[str] = Field(
        "Kharif",
        description="Agricultural season ('Kharif', 'Rabi', 'Zaid')",
        example="Kharif",
    )
    n: Optional[float] = Field(
        None,
        ge=0.0,
        le=300.0,
        description="Soil Nitrogen content (kg/ha)",
        example=55.0,
    )
    p: Optional[float] = Field(
        None,
        ge=0.0,
        le=300.0,
        description="Soil Phosphorus content (kg/ha)",
        example=30.0,
    )
    k: Optional[float] = Field(
        None,
        ge=0.0,
        le=300.0,
        description="Soil Potassium content (kg/ha)",
        example=65.0,
    )
    ph: Optional[float] = Field(
        None,
        ge=1.0,
        le=14.0,
        description="Soil pH level",
        example=7.2,
    )
    lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Optional GPS latitude",
        example=18.52,
    )
    lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Optional GPS longitude",
        example=73.85,
    )
    temperature: Optional[float] = Field(None, description="Optional temperature in °C", example=26.5)
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Optional relative humidity in %", example=70.0)
    rainfall: Optional[float] = Field(None, ge=0.0, description="Optional rainfall in mm", example=800.0)


# ── Response Models ───────────────────────────────────────────────────────────

class SystemVersionResponse(BaseModel):
    project: str = "AgroIntel"
    project_version: str = "4.0.0"
    api_version: str = "v1"
    ml_pipeline_version: str = "1.0.0"
    feature_version: str = "4.0.0"
    dataset_version: str = "2019-2024-v1"
    weather_version: str = "open-meteo-monthly-v1"
    model_registry_version: str = "4.0.0"
    random_forest_version: str = "RandomForestClassifier-100-trees-v4.0.0"


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    price_models: bool = True
    crop_model: bool = True
    registry_loaded: bool = True
    weather_api: str = Field(..., example="reachable")
    market_api: str = Field(..., example="reachable")
    uptime_seconds: float = Field(..., example=120.5)


class ErrorResponse(BaseModel):
    error: str = Field(..., example="Invalid district name")
    detail: str = Field(..., example="District 'InvalidDist' not found in state 'Maharashtra'")
    status_code: int = Field(..., example=422)
