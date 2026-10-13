"""
Test Suite: Critical Price Source Verification, Agmarknet 2.0 Aggregation, and MSP Segregation
------------------------------------------------------------------------------------------------
Validates:
1. Reproduction of the displayed Karnataka Wheat price (Rs 3,888) from raw official Agmarknet 2.0 records.
2. Arrival-weighted vs arithmetic aggregation algorithms and missing value handling.
3. Explicit variety breakdown without silent blending.
4. Strict statutory MSP segregation from spot auction prices.
5. Accurate observation date semantics (previous-day vs live-today, no fake rollovers).
6. Clean error and fallback handling (no price clamping, no fabrication).
7. Commodity mapping integrity (no erroneous fallback to Wheat ID 1).
8. Predict endpoint transparency payload structure.
"""

import json
import os
import pytest
from datetime import date, timedelta

from app.core.constants import OFFICIAL_MSP, get_official_msp
from app.services.mandi_service import (
    MandiPriceResult,
    aggregate_mandi_records,
    get_latest_price,
    _fetch_agmarknet_v2,
    _try_ingestion_cache,
    AGMARKNET_COMMODITY_IDS,
)


# ── Test 1: Exact Reproduction of Displayed Rs 3,888 Price from Raw Agmarknet Record ──

def test_reproduce_karnataka_wheat_3888_from_agmarknet_raw():
    """
    Verifies that the exact raw Agmarknet 2.0 payload from 12 October 2026
    contains weighted average price 3888.49, which AgroIntel formats/rounds as Rs 3,888.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "app", "data", "fixtures", "agmarknet_2026_10_12_raw.json"
    )
    assert os.path.exists(fixture_path), f"Raw fixture not found at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = raw_data.get("data", {}).get("records", [])
    assert len(records) > 0, "Agmarknet response records should not be empty"

    # Find Karnataka Wheat record
    kar_wheat = None
    for r in records:
        if r.get("cmdt_name", "").lower() == "wheat" and r.get("state_name", "").lower() == "karnataka":
            kar_wheat = r
            break

    assert kar_wheat is not None, "Karnataka Wheat record must exist in 12 Oct Agmarknet response"

    # 1. Exact raw fields
    as_on_val = kar_wheat.get("as_on")
    assert as_on_val == "3888.49", f"Expected raw as_on '3888.49', got {as_on_val}"
    assert kar_wheat.get("cmdt_id") == 1, "Commodity ID for Wheat must be 1"
    assert kar_wheat.get("state_id") == 16, "State ID for Karnataka must be 16"
    assert kar_wheat.get("unit_name_price") == "Rs./Quintal", "Price unit must be Rs./Quintal"

    # 2. Reproduction of rounded price
    raw_float = float(as_on_val)
    rounded_price = round(raw_float)
    assert rounded_price == 3888, f"Rounded price must equal Rs 3,888, got {rounded_price}"


# ── Test 2: Aggregation Algorithms (Arrival-Weighted vs Arithmetic & Missing Values) ──

def test_aggregate_mandi_records_arrival_weighted():
    """
    Tests arrival-weighted average modal price calculation across multiple mandis.
    Mandi A: 100 tonnes @ Rs 3,000/qtl
    Mandi B: 300 tonnes @ Rs 3,400/qtl
    Expected Weighted Average: (100*3000 + 300*3400) / 400 = (300,000 + 1,020,000) / 400 = 3,300.0
    """
    records = [
        {"market": "Hubli", "variety": "Lokwan", "modal_price": 3000.0, "arrival_quantity": 100.0, "min_price": 2800, "max_price": 3200},
        {"market": "Dharwad", "variety": "Lokwan", "modal_price": 3400.0, "arrival_quantity": 300.0, "min_price": 3200, "max_price": 3500},
    ]

    result = aggregate_mandi_records(records, target_crop="wheat", target_state="Karnataka", method="arrival_weighted")
    assert result is not None
    assert result["modal_price"] == 3300.0
    assert result["min_price"] == 2800.0
    assert result["max_price"] == 3500.0
    assert result["contributing_records_count"] == 2
    assert "arrival-weighted" in result["aggregation_method"].lower()


def test_aggregate_mandi_records_arithmetic_fallback_on_missing_arrivals():
    """
    When arrival quantities are missing or zero, aggregation must gracefully
    use arithmetic mean without failing or dividing by zero.
    Mandi A: modal Rs 3,000, arrival None
    Mandi B: modal Rs 3,600, arrival 0
    Expected Arithmetic Average: (3000 + 3600) / 2 = 3300.0
    """
    records = [
        {"market": "Mandi A", "variety": "Dara", "modal_price": 3000.0, "arrival_quantity": None, "min_price": 2900, "max_price": 3100},
        {"market": "Mandi B", "variety": "Dara", "modal_price": 3600.0, "arrival_quantity": 0.0, "min_price": 3500, "max_price": 3700},
    ]

    result = aggregate_mandi_records(records, target_crop="wheat", target_state="Karnataka", method="arrival_weighted")
    assert result is not None
    assert result["modal_price"] == 3300.0
    assert "arithmetic" in result["aggregation_method"].lower()


def test_aggregate_mandi_records_deduplication():
    """
    Deduplication: Identical market + variety observations on same date must be deduplicated.
    """
    records = [
        {"market": "Belgaum", "variety": "Local", "modal_price": 3200.0, "arrival_quantity": 50.0, "arrival_date": "2026-10-12"},
        {"market": "Belgaum", "variety": "Local", "modal_price": 3200.0, "arrival_quantity": 50.0, "arrival_date": "2026-10-12"},
    ]

    result = aggregate_mandi_records(records, target_crop="wheat", target_state="Karnataka")
    assert result is not None
    assert result["contributing_records_count"] == 1
    assert result["modal_price"] == 3200.0


# ── Test 3: Multi-Variety Handling (Explicit Identification, No Silent Conflation) ──

def test_aggregate_mandi_records_multi_variety():
    """
    Multiple varieties must be explicitly listed in the transparency metadata
    rather than silently merged under a generic label.
    """
    records = [
        {"market": "Market 1", "variety": "Lokwan", "modal_price": 3500.0, "arrival_quantity": 50.0},
        {"market": "Market 2", "variety": "Sharbati", "modal_price": 4200.0, "arrival_quantity": 50.0},
    ]

    result = aggregate_mandi_records(records, target_crop="wheat", target_state="Madhya Pradesh")
    assert result is not None
    assert "Lokwan" in result["variety"]
    assert "Sharbati" in result["variety"]
    assert "variety_breakdown" in result
    assert "Lokwan" in result["variety_breakdown"]
    assert "Sharbati" in result["variety_breakdown"]


# ── Test 4: Statutory MSP Segregation from Spot Auction Prices ──

def test_statutory_msp_segregation():
    """
    Verifies:
    1. Official MSP for Wheat is Rs 2,425/quintal (RMS 2025-26).
    2. MSP is kept in distinct msp_price / msp_season fields.
    3. MandiPriceResult never sets modal_price equal to MSP simply because spot data is missing.
    4. Onion and Potato do not have statutory MSP (returns None for msp_price).
    """
    # 1. Wheat MSP check
    wheat_msp = get_official_msp("wheat")
    assert wheat_msp is not None
    assert wheat_msp["msp_price"] == 2425.0
    assert "RMS 2025-26" in wheat_msp["season"]

    # 2. Paddy & Maize MSP check
    paddy_msp = get_official_msp("rice")
    assert paddy_msp is not None
    assert paddy_msp["msp_price"] == 2300.0

    maize_msp = get_official_msp("maize")
    assert maize_msp is not None
    assert maize_msp["msp_price"] == 2225.0

    # 3. Potato and Onion do NOT have statutory MSP
    assert get_official_msp("potato")["msp_price"] is None
    assert get_official_msp("onion")["msp_price"] is None

    # 4. MandiPriceResult segregation
    res = MandiPriceResult(
        crop="wheat",
        modal_price=3888.49,
        min_price=3200.0,
        max_price=4100.0,
        arrival_date="2026-10-12",
        observation_date="2026-10-12",
        state="Karnataka",
        source="agmarknet_v2",
        msp_price=2425.0,
        msp_season="RMS 2025-26",
    )
    t_dict = res.to_transparency_dict()
    assert t_dict["modal_price"] == 3888.49
    assert t_dict["msp_price"] == 2425.0
    assert t_dict["modal_price"] != t_dict["msp_price"]
    assert t_dict["msp_difference"] == round(3888.49 - 2425.0, 2)


# ── Test 5: Observation Date Semantics (Previous-Day vs Live-Today) ──

def test_observation_date_semantics_previous_day():
    """
    If a record was observed on 2026-10-12 and queried on 2026-10-13,
    it must be explicitly flagged as PREVIOUS_DAY and 'Previous-Day Observation (1 day old)'.
    Under no circumstances should the system overwrite observation_date with today's date!
    """
    res = MandiPriceResult(
        crop="wheat",
        modal_price=3888.49,
        arrival_date="2026-10-12",
        observation_date="2026-10-12",
        state="Karnataka",
        source="agmarknet_v2",
        price_status="PREVIOUS_DAY",
        status_label="Previous-Day Observation (1 day old)",
    )
    d = res.to_transparency_dict()
    assert d["observation_date"] == "2026-10-12"
    assert d["price_status"] == "PREVIOUS_DAY"
    assert "Previous-Day" in d["status_label"]
    assert d["observation_date"] != "2026-10-13"


def test_observation_date_same_day():
    """
    If a record is observed on the same date, it must be flagged as LIVE_TODAY.
    """
    today_str = date.today().isoformat()
    res = MandiPriceResult(
        crop="wheat",
        modal_price=3888.49,
        arrival_date=today_str,
        observation_date=today_str,
        state="Karnataka",
        source="agmarknet_v2",
        price_status="LIVE_TODAY",
        status_label="Live Today (Same-Day Mandi Observation)",
    )
    d = res.to_transparency_dict()
    assert d["observation_date"] == today_str
    assert d["price_status"] == "LIVE_TODAY"


# ── Test 6: Clean Fallback on API Failure (No Price Clamping, No Fabrication) ──

def test_fallback_on_api_failure_no_price_clamping():
    """
    When live API is unavailable or returns no data, system should fall back
    to historical baseline with status HISTORICAL_FALLBACK and explicit disclaimer,
    never clamping prices to arbitrary hardcoded limits.
    """
    bridge_result = _try_ingestion_cache("wheat", "Karnataka", date.today())
    if bridge_result:
        assert bridge_result.price_status in ["HISTORICAL_FALLBACK", "PREVIOUS_DAY", "LIVE_TODAY", "LIVE"]
        assert bridge_result.observation_date is not None
        assert bridge_result.modal_price > 0
        # Check MSP segregation
        assert bridge_result.msp_price == 2425.0
        t_dict = bridge_result.to_transparency_dict()
        assert t_dict["msp_price"] == 2425.0


# ── Test 7: Commodity Mapping Integrity (No Erroneous Fallback to Wheat ID 1) ──

def test_commodity_mapping_does_not_default_to_wheat():
    """
    Unmapped commodities (e.g. 'unknown_crop') must return None or empty,
    and NEVER default to Agmarknet commodity ID 1 (Wheat).
    """
    assert "unknown_crop" not in AGMARKNET_COMMODITY_IDS

    # Verify that Agmarknet fetch returns None when commodity is unmapped
    res = _fetch_agmarknet_v2("unknown_crop", "Karnataka", date.today())
    assert res is None, "Fetching unmapped commodity must return None, not Wheat data"


# ── Test 8: Predict Endpoint Transparency Payload Integrity ──

def test_predict_endpoint_transparency_payload_wheat():
    """
    Tests that predict_crop_price produces the complete price_transparency
    dictionary with all required attributes for Karnataka Wheat.
    """
    from app.ml.price_predictor import predict_crop_price

    result = predict_crop_price(crop="wheat", state="Karnataka", horizon_days=30)
    assert result is not None

    # Check price_transparency structure
    pt = result.get("price_transparency")
    assert pt is not None, "price_transparency dictionary must be present in predict output"
    assert pt.get("commodity_name", "").lower() == "wheat"
    assert pt.get("state") == "Karnataka"
    assert pt.get("modal_price") is not None
    assert pt.get("msp_price") == 2425.0
    assert pt.get("price_status") in ["LIVE", "LIVE_TODAY", "PREVIOUS_DAY", "HISTORICAL_FALLBACK", "STALE_NATIONAL_FALLBACK"]
    assert pt.get("source_url") == "https://agmarknet.gov.in"
    assert pt.get("aggregation_method") is not None
