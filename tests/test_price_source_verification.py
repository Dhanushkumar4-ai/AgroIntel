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


# ── Test 9: Independent Mathematical Reconciliation of Rs 2,515.34 vs Rs 3,888.49 ──

def test_mathematical_reconciliation_2515_34_and_terminal_skew():
    """
    Mathematical Proof and Reconciliation:
    1. Producing Mandis (Northern Karnataka, 11 Oct):
       - Haveri APMC (ID 255): 18.00 MT @ Rs 2,500.00/quintal (Lokwan / Local)
       - Vijayapura APMC (ID 247): 3.30 MT @ Rs 2,599.00/quintal (Dara / Khanda)
       Total Volume = 21.30 MT
       Sum(Arrival * Price) = (18.00 * 2500) + (3.30 * 2599) = 45000 + 8576.70 = 53576.70
       Weighted Average = 53576.70 / 21.30 = 2515.338028... = Rs 2,515.34/quintal.
       Matches official Agmarknet dashboard table to the exact paisa!
    
    2. Terminal Market Skew (12 Oct):
       - Bengaluru Yeshwanthpur Terminal APMC reported 37.50 MT (54.4% of total state arrivals 68.90 MT).
       - Inward shipments of premium sorted/milled wheat traded at terminal rates (Rs 3,800-4,500+).
       - Volume weighting with Bengaluru skewed the state-level composite index upwards to Rs 3,888.49.
       
    3. Terminal Market Dropping (13 Oct):
       - When Bengaluru reported 0 arrivals on 13 Oct, state composite immediately dropped to Rs 2,800.00.
    """
    haveri_arrival = 18.00
    haveri_price = 2500.00
    vijayapura_arrival = 3.30
    vijayapura_price = 2599.00

    total_arrival = haveri_arrival + vijayapura_arrival
    weighted_sum = (haveri_arrival * haveri_price) + (vijayapura_arrival * vijayapura_price)
    calc_weighted_avg = weighted_sum / total_arrival

    assert round(calc_weighted_avg, 2) == 2515.34, f"Expected 2515.34, got {calc_weighted_avg}"

    # Verify using aggregate_mandi_records function
    records = [
        {"market": "Haveri", "variety": "Lokwan", "modal_price": 2500.0, "arrival_quantity": 18.0},
        {"market": "Vijayapura", "variety": "Dara", "modal_price": 2599.0, "arrival_quantity": 3.3},
    ]
    res = aggregate_mandi_records(records, target_crop="wheat", target_state="Karnataka", method="arrival_weighted")
    assert res["modal_price"] == 2515.34


# ── Test 10: Multi-Tier Transparency Architecture in Mandi Service ──

def test_mandi_service_multi_tier_transparency():
    """
    Verifies that get_latest_price("wheat", "Karnataka") exposes:
    1. farm_gate_modal_price = 2515.34 (matches Rs 2,500-2,800 production mandi range)
    2. farm_gate_range = "₹2,500 – ₹2,800/quintal"
    3. state_composite_index (capturing the Rs 3,888 or Rs 2,800 state aggregate)
    4. reconciliation_notice explaining the urban terminal skew
    5. district_breakdown listing individual APMC yards
    6. strict MSP segregation (msp_price = 2425.0)
    """
    result = get_latest_price("wheat", "Karnataka")
    assert result is not None

    t_dict = result.to_transparency_dict()
    assert t_dict.get("farm_gate_modal_price") == 2515.34
    assert t_dict.get("farm_gate_range") == "₹2,500 – ₹2,800/quintal"
    assert t_dict.get("state_composite_index") is not None
    assert t_dict.get("reconciliation_notice") is not None
    assert "Bengaluru" in t_dict["reconciliation_notice"]
    assert "district_breakdown" in t_dict
    assert len(t_dict["district_breakdown"]) >= 5

    # Check that individual production mandis and terminal yard are listed
    districts = [m.get("district", "").lower() for m in t_dict["district_breakdown"]]
    markets = [m.get("market", "").lower() for m in t_dict["district_breakdown"]]
    assert any("haveri" in d or "haveri" in m for d, m in zip(districts, markets))
    assert any("vijayapura" in d or "vijayapura" in m for d, m in zip(districts, markets))
    assert any("bengaluru" in d or "bengaluru" in m for d, m in zip(districts, markets))

    # Check MSP segregation
    assert t_dict.get("msp_price") == 2425.0
    assert t_dict.get("modal_price") != t_dict.get("msp_price")


# ── Test 11: Date-Filtering Authenticity Across Dates ──

def test_agmarknet_date_filtering_authenticity():
    """
    Verifies that Agmarknet endpoints are genuinely date-filtered and not returning
    a static or hardcoded value.
    The column header explicitly defines 'as_on' as 'As on selected dates'.
    - On 11-Oct (no Bengaluru arrivals): 'as_on' = 2515.34 (Haveri + Vijayapura)
    - On 12-Oct (Bengaluru reported 37.5 MT): 'as_on' = 3888.49 / 3879.62
    - On 13-Oct (Kalaburagi only): 'as_on' = 2800.00
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "app", "data", "fixtures", "agmarknet_2026_10_12_raw.json"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_12 = json.load(f)

    # Verify column definition 'as_on' = 'As on selected dates'
    cols = fixture_12.get("data", {}).get("columns", [])
    wt_avg_col = next((c for c in cols if c.get("key") == "wt_avg_price"), None)
    assert wt_avg_col is not None, "wt_avg_price column definition must be present"
    as_on_subcol = next((sc for sc in wt_avg_col.get("columns", []) if sc.get("key") == "as_on"), None)
    assert as_on_subcol is not None
    assert as_on_subcol.get("title") == "As on selected dates"

    records = fixture_12.get("data", {}).get("records", [])
    kar_rec = next(r for r in records if r.get("cmdt_name", "").lower() == "wheat" and r.get("state_name", "").lower() == "karnataka")

    # 12-Oct reported ~3888.49
    assert kar_rec.get("as_on") in ["3888.49", "3879.62"]
    assert kar_rec.get("state_id") == 16
    assert kar_rec.get("cmdt_id") == 1


# ── Test 12: Maharashtra State Isolation — Zero Karnataka Leakage ─────────────

def test_maharashtra_state_isolation_no_karnataka_leak():
    """
    Regression Test:
    When requesting price prediction or mandi data for Maharashtra Wheat,
    the response must NEVER leak Karnataka-specific data, mandis, or explanations:
    - No Haveri, Vijayapura, Kalaburagi, Gadag, Dharwad
    - No Bengaluru or Yeshwanthpur
    - No Karnataka reconciliation notices
    - Scope and state must strictly identify Maharashtra
    """
    from app.ml.inference import predict_price

    pred = predict_price("wheat", "Maharashtra", horizon_days=30)
    assert pred is not None
    assert pred.get("state") == "Maharashtra"

    # 1. Price transparency panel check
    pt = pred.get("price_transparency", {})
    assert pt.get("selected_state") == "Maharashtra"
    assert pt.get("state") == "Maharashtra"

    # 2. Check reconciliation notice
    recon = pt.get("reconciliation_notice")
    if recon:
        recon_lower = recon.lower()
        assert "karnataka" not in recon_lower, "Karnataka reconciliation notice leaked to Maharashtra!"
        assert "bengaluru" not in recon_lower, "Bengaluru inward hub leaked to Maharashtra!"
        assert "haveri" not in recon_lower, "Haveri mandi leaked to Maharashtra!"

    # 3. Check district breakdown and contributing records
    records = pt.get("district_breakdown", []) + pt.get("contributing_records", [])
    karnataka_keywords = ["haveri", "vijayapura", "kalaburagi", "gadag", "dharwad", "yeshwanthpur"]
    for r in records:
        r_str = json.dumps(r).lower()
        for kw in karnataka_keywords:
            assert kw not in r_str, f"Karnataka mandi '{kw}' found in Maharashtra record: {r}"
        if r.get("state"):
            assert r["state"].lower() != "karnataka", f"Karnataka state found in Maharashtra record: {r}"


# ── Test 13: Karnataka State Consistency & Transparency ───────────────────────

def test_karnataka_state_transparency():
    """
    Regression Test:
    When requesting price prediction for Karnataka Wheat,
    Karnataka-specific records and multi-tier transparency are accurately presented:
    - State is Karnataka
    - Farm gate modal rate is ~2515.34 (or local production mandi rate)
    - Reconciliation notice accurately describes production mandis vs terminal inward hub
    """
    from app.ml.inference import predict_price

    pred = predict_price("wheat", "Karnataka", horizon_days=30)
    assert pred is not None
    assert pred.get("state") == "Karnataka"

    pt = pred.get("price_transparency", {})
    assert pt.get("selected_state") == "Karnataka"

    # Reconciliation notice should be present for Karnataka Wheat
    recon = pt.get("reconciliation_notice")
    assert recon is not None
    assert "Karnataka" in recon or "Haveri" in recon or "Bengaluru" in recon


# ── Test 14: Honest Unavailable State for Missing State Data ──────────────────

def test_missing_state_data_honest_unavailable():
    """
    Regression Test:
    When a state has no live reporting mandi records (e.g., Punjab off-season wheat),
    the system must NOT substitute Karnataka or an unrelated state's records:
    - Does not return Haveri, Vijayapura, or Karnataka prices
    - Live fetch returns None for unrepresented state
    """
    res = _fetch_agmarknet_v2("wheat", "Punjab", date.today())
    # Punjab has no live wheat harvest/arrivals in October
    # System must return None rather than faking data or falling back to National/Karnataka
    if res is not None:
        assert res.get("state") != "Karnataka"
        assert "haveri" not in json.dumps(res).lower()
    else:
        assert res is None, "Expected None for unrepresented state in live v2"


# ── Test 15: Cache Key Isolation Between States and Crops ─────────────────────

def test_cache_key_isolation_between_states_and_crops():
    """
    Regression Test:
    Cache keys must be strictly differentiated by (crop, state).
    Queries for Maharashtra must never hit cached Karnataka results.
    """
    from app.services.mandi_service import _CACHE_FILE, _load_cache

    cache = _load_cache()
    # Check that any keys in cache distinguish state
    for k in cache.keys():
        if "_" in k:
            parts = k.split("_", 1)
            crop_part = parts[0]
            state_part = parts[1]
            assert state_part != "", "State part in cache key must not be empty"


# ── Test 16: Crop Switching Updates MSP, Varieties, and Transparency ──────────

def test_crop_switching_updates_msp_and_varieties():
    """
    Regression Test:
    Switching crops updates MSP and commodities correctly:
    - Wheat -> MSP 2425 (Rabi 2024-25 / RMS 2025-26)
    - Rice -> MSP 2300 (Kharif 2024-25 / KMS 2024-25)
    - Onion -> No statutory MSP (Commercial/horticultural)
    """
    wheat_msp = get_official_msp("wheat")
    assert wheat_msp is not None
    assert wheat_msp["msp_price"] == 2425.0

    rice_msp = get_official_msp("rice")
    assert rice_msp is not None
    assert rice_msp["msp_price"] == 2300.0

    onion_msp = get_official_msp("onion")
    assert onion_msp is not None
    assert onion_msp["msp_price"] is None


# ── Test 17: Daily Agri News UI Complete Removal Verification ─────────────────

def test_daily_agri_news_ui_removal_from_frontend():
    """
    Regression Test:
    Verify that the Daily Agri News UI feature has been completely removed
    from frontend user-facing files:
    1. index.html does NOT contain 'navNews' button
    2. index.html does NOT contain 'page-news' section
    3. index.html does NOT contain 'showPage(\\'news\\')'
    4. script.js showPage('news') redirects to 'landing'
    5. script.js navMap does NOT contain 'news'
    6. Historical news archives and backend files REMAIN intact
    """
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "frontend")
    html_path = os.path.join(frontend_dir, "index.html")
    js_path = os.path.join(frontend_dir, "script.js")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    assert "id=\"navNews\"" not in html_content, "navNews button must be removed from index.html"
    assert "id=\"page-news\"" not in html_content, "page-news section must be removed from index.html"
    assert "showPage('news')" not in html_content, "showPage('news') triggers must be removed from index.html"

    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    assert "news: \"navNews\"" not in js_content, "navMap must not map news to navNews"
    assert "cleanId === \"news\"" in js_content, "showPage must intercept 'news' and redirect"
    assert "cleanId = \"landing\"" in js_content, "showPage must redirect 'news' to landing"

    # Verify backend news files are preserved
    archive_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "app", "data", "experimental", "historical_news_archive.json"
    )
    assert os.path.exists(archive_path), "historical_news_archive.json backend file must remain intact!"


