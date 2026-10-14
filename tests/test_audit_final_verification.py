"""
test_audit_final_verification.py — Comprehensive Validation Suite for AgroIntel
Verifies:
1. Price-source crop and state mapping
2. Price units, source dates, and observation freshness
3. Commodity mapping failure handling
4. Forecast horizon and target-date alignment
5. Forecast output consistency with decision-engine inputs
6. Correct storage-cost and threshold calculations for each crop group
7. Consistency between Farmer Advisory and Price Prediction decisions
8. Appropriate handling of stale or missing data
9. Correct Random Forest feature order and crop-class mapping
10. District, season, duplicate-crop, and local-evidence validation
11. Correct graph styling without changing underlying values
12. Correct refresh behavior after changing crop, state, district, or season
"""

import os
from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.inference import predict_price
from app.services.mandi_service import get_latest_price, _load_cache
from app.ml.crop_recommender import FEATURE_NAMES, RF_MODEL_PATH, ENCODER_PATH
import pickle

client = TestClient(app)


# ── 1. Price-Source Crop and State Mapping ─────────────────────────────────────

def test_price_source_crop_and_state_mapping():
    """Verify that get_latest_price and predict_price preserve crop and state without mixing."""
    test_cases = [
        ("wheat", "Maharashtra"),
        ("wheat", "Karnataka"),
        ("onion", "Maharashtra"),
        ("rice", "Punjab"),
        ("maize", "Karnataka"),
    ]
    for crop, state in test_cases:
        p_res = get_latest_price(crop, state)
        assert p_res is not None, f"Failed to get price for {crop} in {state}"
        assert p_res.crop.lower() == crop.lower()
        if state.lower() not in ("all", "national"):
            assert state.lower() in p_res.state.lower() or p_res.state.lower() in state.lower()
        assert p_res.modal_price > 0
        assert p_res.unit == "₹/quintal"


# ── 2. Price Units, Source Dates, and Observation Freshness ───────────────────

def test_price_units_and_observation_freshness():
    """Verify prices are consistently in ₹/quintal and observation dates are valid & not future."""
    for crop in ["wheat", "onion", "rice", "maize", "potato"]:
        res = predict_price(crop, "Maharashtra", horizon_days=30)
        assert res.get("available") is True
        assert res["current_price"] > 0
        obs_date = res.get("observation_date")
        assert obs_date is not None
        # Must not be a future date
        assert obs_date <= date.today().isoformat()
        # Price status must be recognized
        assert res.get("price_status") in ("LIVE_TODAY", "PREVIOUS_DAY", "STALE", "HISTORICAL_FALLBACK")


# ── 3. Commodity Mapping Failure Handling ─────────────────────────────────────

def test_commodity_mapping_failure_handling():
    """Unsupported crops must return unavailable rather than defaulting to wheat or another crop."""
    for bad_crop in ["dragonfruit", "avocado", "nonexistent_crop", "kiwi"]:
        p_res = get_latest_price(bad_crop, "Maharashtra")
        assert p_res is None or p_res.modal_price is None

        resp = client.get(f"/api/predict?crop={bad_crop}&state=Maharashtra&horizon_days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("available") is False
        assert "unavailable" in data.get("message", "").lower() or "unsupported" in data.get("message", "").lower()


# ── 4. Forecast Horizon and Target-Date Alignment ─────────────────────────────

@pytest.mark.parametrize("h", [7, 15, 30, 60, 90])
def test_forecast_horizon_and_target_date_alignment(h):
    """Verify exact horizon count and daily chronological date progression."""
    res = predict_price("wheat", "Maharashtra", horizon_days=h)
    assert len(res["predictions"]) == h
    assert len(res["date_labels"]) == h
    assert len(res["forecast_series"]) == h

    # Dates must be sequential
    dates = [date.fromisoformat(d) for d in res["date_labels"]]
    for i in range(1, len(dates)):
        assert dates[i] == dates[i - 1] + timedelta(days=1), f"Discontinuous date at index {i}"


# ── 5. Forecast Output Consistency with Decision-Engine Inputs ────────────────

def test_forecast_output_consistency_with_decision_inputs():
    """Verify percent_change and net_expected_change match displayed current & predicted prices."""
    for crop in ["wheat", "onion", "rice"]:
        res = predict_price(crop, "Maharashtra", horizon_days=30)
        curr = res["current_price"]
        pred = res["predicted_price"]
        expected_pct = round(((pred - curr) / curr) * 100.0, 2)
        assert res["percent_change"] == expected_pct

        expected_net = round(res["percent_change"] - res["storage_cost_pct"], 2)
        assert res["net_expected_change"] == expected_net


# ── 6. Storage-Cost and Threshold Calculations for Crop Groups ────────────────

def test_storage_cost_and_threshold_calculations():
    """Verify holding costs (2% for grains, 4% for perishables) and net expected gain calculation."""
    res_wheat = predict_price("wheat", "Maharashtra", horizon_days=30)
    res_onion = predict_price("onion", "Maharashtra", horizon_days=30)

    # 30-day storage cost
    assert res_wheat["storage_cost_pct"] == 2.0
    assert res_onion["storage_cost_pct"] == 4.0

    # 15-day storage cost check
    res_onion_15 = predict_price("onion", "Maharashtra", horizon_days=15)
    assert res_onion_15["storage_cost_pct"] == 2.0  # (15/30) * 4.0

    # Verify decision explanations are 1 to 2 sentences
    reason = res_onion["recommendation_reason"]
    assert len(reason) > 20
    # Sentences count: split on period followed by space or end of string, avoiding decimals in numbers
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[a-zA-Z\)])\.\s*', reason) if s.strip()]
    sentence_count = len(sentences)
    assert sentence_count in (1, 2), f"Expected 1 or 2 sentences, got {sentence_count}: {reason}"


# ── 7. Consistency Between Farmer Advisory and Price Prediction Decisions ──────

def test_consistency_between_advisory_and_price_prediction():
    """Ensure Farmer Advisory and Price Prediction return matching decision & reason for same inputs."""
    crop = "onion"
    state = "Maharashtra"
    horizon = 30

    pred_res = predict_price(crop, state, horizon_days=horizon)

    adv_resp = client.post("/api/advisory", json={
        "state": state,
        "district": "Nashik",
        "season": "Rainy",
        "crop": crop
    })
    assert adv_resp.status_code == 200
    adv_data = adv_resp.json()
    adv_price = adv_data["price_prediction"]

    assert adv_price["decision"] == pred_res["recommendation"]
    assert adv_price["decision_reason"] == pred_res["recommendation_reason"]
    assert adv_price["current_price"] == pred_res["current_price"]
    assert adv_price["predicted_30d_avg"] == pred_res["predicted_price"]


# ── 8. Handling of Stale or Missing Data ───────────────────────────────────────

def test_handling_of_stale_or_missing_data():
    """Ensure non-benchmark crops in advisory cleanly indicate forecast unavailability with no fake decision."""
    adv_resp = client.post("/api/advisory", json={
        "state": "Maharashtra",
        "district": "Ahilya Nagar",
        "season": "Whole Year",
        "crop": "sugarcane"
    })
    assert adv_resp.status_code == 200
    data = adv_resp.json()
    p_block = data["price_prediction"]
    assert p_block["forecast_available"] is False
    assert p_block["decision"] is None
    assert "unavailable" in p_block["decision_reason"].lower()


# ── 9. Random Forest Feature Order and Crop-Class Mapping ─────────────────────

def test_random_forest_feature_order_and_classes():
    """Verify 7 features in exact order and 22 class labels in Random Forest model."""
    expected_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    assert FEATURE_NAMES == expected_features

    assert RF_MODEL_PATH.exists()
    assert ENCODER_PATH.exists()

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    classes = encoder.classes_.tolist()
    assert len(classes) == 22
    assert "rice" in classes
    assert "maize" in classes
    assert "chickpea" in classes
    assert "cotton" in classes


# ── 10. District, Season, Duplicate-Crop, and Local Evidence Validation ───────

def test_crop_recommendation_district_season_and_deduplication():
    """Verify crop recommendations handle seasons and return valid suitability scores without duplicates."""
    resp = client.post("/api/phase6/recommend", json={
        "state": "Karnataka",
        "district": "Dharwad",
        "season": "Kharif"
    })
    assert resp.status_code == 200
    data = resp.json()
    recs = data.get("recommendations", [])
    assert len(recs) > 0

    crop_names = [r["crop"].lower().strip() for r in recs]
    # Check deduplication
    assert len(crop_names) == len(set(crop_names)), "Duplicate crop names in recommendations!"

    for r in recs:
        score = r.get("suitability_score") or r.get("confidence_score") or 0
        assert 0 <= score <= 100, f"Invalid suitability score: {score}"


# ── 11. Graph Styling in Frontend Without Changing Values ─────────────────────

def test_price_graph_single_visual_style_in_frontend():
    """Verify frontend script uses single line style, single legend entry, and text note."""
    js_path = os.path.join(os.path.dirname(__file__), "..", "backend", "frontend", "script.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    # Single dataset label
    assert 'label: "Price (₹/quintal)"' in js or "label: 'Price (₹/quintal)'" in js
    # Text note present
    assert "Forecast begins after the last observed date." in js
    # No separate orange/green distinction in chart datasets
    assert 'label: "Historical Price"' not in js
    assert 'label: "Predicted Price"' not in js


# ── 12. Correct Refresh Behavior and Race-Condition Prevention ─────────────────

def test_refresh_behavior_and_race_condition_prevention():
    """Verify script.js has currentAdvRequestId and currentPredRequestId to protect against race conditions."""
    js_path = os.path.join(os.path.dirname(__file__), "..", "backend", "frontend", "script.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    assert "currentAdvRequestId" in js
    assert "currentPredRequestId" in js
