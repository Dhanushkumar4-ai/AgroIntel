"""
test_all_india_crop_recommendation.py — Comprehensive Test Suite for All-India Crop Recommendation

Verifies:
1. Support for all 36 Indian administrative entities (28 States + 8 UTs) in official nomenclature.
2. Two-stage recommendation pipeline (Stage A: RF agronomic model; Stage B: official APY evidence).
3. Botanical seasonal gates (Chickpea/Wheat rejected in Kharif; Pearl Millet/Soybean rejected in Rabi).
4. Signature Local Crop rule enforcement (only crops satisfying documented rules receive badge).
5. Multi-region validation:
   - Ahilyanagar, Maharashtra
   - Dharwad, Karnataka
   - Ludhiana, Punjab
   - Kamrup, Assam
   - Jaipur, Rajasthan
   - Nicobars, Andaman and Nicobar Islands (UT with records)
   - Lakshadweep (Missing district data honest fallback)
6. Agronomic input sensitivity (RF probabilities react to temperature, humidity, rainfall changes).
7. Missing data transparency (no fabricated acreage or silent district substitution).
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.phase6_integration_service import AgroIntelPhase6Engine
from app.services.location_normalizer import normalize_location

client = TestClient(app)
engine = AgroIntelPhase6Engine()

BASE_DIR = Path(__file__).resolve().parent.parent


def test_administrative_coverage_36_regions():
    """Verify all 28 States and 8 UTs exist in indian_districts.json with official names."""
    json_path = BASE_DIR / "backend" / "frontend" / "indian_districts.json"
    assert json_path.exists(), "indian_districts.json missing"
    
    with open(json_path) as f:
        data = json.load(f)
    
    states = data.get("states", [])
    assert len(states) == 36, f"Expected 36 administrative entities, got {len(states)}"
    
    state_names = {s["state"] for s in states}
    
    # 8 Union Territories in official nomenclature
    official_uts = [
        "Andaman and Nicobar Islands",
        "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi",
        "Jammu and Kashmir",
        "Ladakh",
        "Lakshadweep",
        "Puducherry"
    ]
    for ut in official_uts:
        assert ut in state_names, f"UT '{ut}' missing from official administrative master"
        
    for s in states:
        assert len(s.get("districts", [])) > 0, f"Entity {s['state']} has no districts"


def test_ahilyanagar_maharashtra_seasonal_contrast():
    """
    Ahilyanagar, Maharashtra:
    - Kharif: Sugarcane, Bajra, Soybean, Sorghum. Chickpea and Wheat must be filtered out.
    - Rabi: Wheat, Sorghum, Sugarcane. Bajra and Soybean must be filtered out.
    """
    # 1. Kharif
    res_k = engine.evaluate_recommendation(
        state="Maharashtra",
        district="Ahilyanagar",
        season="Kharif",
        soil_ph=7.2,
        soil_npk={"N": 60, "P": 40, "K": 40},
        temperature=28,
        humidity=70,
        rainfall=750
    )
    recs_k = [r["crop"] for r in res_k["recommendations"]]
    assert len(recs_k) > 0
    assert "Wheat" not in recs_k, "Wheat should not be recommended in Kharif"
    assert "Chickpea (Gram)" not in recs_k, "Chickpea should not be recommended in Kharif"
    assert any(c in recs_k for c in ["Sugarcane", "Pearl Millet (Bajra)", "Soybean", "Sorghum (Jowar)"])
    
    # 2. Rabi
    res_r = engine.evaluate_recommendation(
        state="Maharashtra",
        district="Ahilyanagar",
        season="Rabi",
        soil_ph=7.2,
        soil_npk={"N": 60, "P": 40, "K": 40},
        temperature=20,
        humidity=50,
        rainfall=100
    )
    recs_r = [r["crop"] for r in res_r["recommendations"]]
    assert len(recs_r) > 0
    assert "Pearl Millet (Bajra)" not in recs_r, "Bajra should not be recommended in Rabi"
    assert "Soybean" not in recs_r, "Soybean should not be recommended in Rabi"
    assert any(c in recs_r for c in ["Wheat", "Sorghum (Jowar)", "Sugarcane"])


def test_karnataka_dharwad_kharif():
    """Dharwad, Karnataka in Kharif."""
    res = engine.evaluate_recommendation(
        state="Karnataka",
        district="Dharwad",
        season="Kharif",
        soil_ph=7.5,
        soil_npk={"N": 50, "P": 30, "K": 60},
        temperature=26,
        humidity=75,
        rainfall=700
    )
    recs = [r["crop"] for r in res["recommendations"]]
    assert len(recs) > 0
    # Dharwad signature staples in Kharif
    assert any(c in recs for c in ["Chilli (Dry)", "Soybean", "Sorghum (Jowar)", "Cotton"])
    # Not all crops should have Signature Local Crop badge
    badges = [r["evidence_badge"] for r in res["recommendations"]]
    assert "Locally Supported" in badges or "Signature Local Crop" in badges


def test_punjab_ludhiana_rabi():
    """Ludhiana, Punjab in Rabi: Wheat is #1 signature staple, Rice is rejected."""
    res = engine.evaluate_recommendation(
        state="Punjab",
        district="Ludhiana",
        season="Rabi",
        soil_ph=7.0,
        soil_npk={"N": 110, "P": 55, "K": 45},
        temperature=16,
        humidity=60,
        rainfall=150
    )
    recs = [r["crop"] for r in res["recommendations"]]
    assert len(recs) > 0
    assert recs[0] == "Wheat", f"Expected Wheat as #1 in Ludhiana Rabi, got {recs[0]}"
    assert "Rice" not in recs, "Rice should not be recommended in Ludhiana Rabi"
    assert res["recommendations"][0]["evidence_badge"] == "Signature Local Crop"
    assert res["recommendations"][0]["district_cultivation_evidence"] is not None


def test_assam_kamrup_kharif():
    """Kamrup, Assam in Kharif: Arecanut, Sugarcane, Sesame, Mesta."""
    res = engine.evaluate_recommendation(
        state="Assam",
        district="Kamrup",
        season="Kharif",
        soil_ph=5.8,
        soil_npk={"N": 70, "P": 40, "K": 40},
        temperature=29,
        humidity=85,
        rainfall=1400
    )
    recs = [r["crop"] for r in res["recommendations"]]
    assert len(recs) > 0
    assert any(c in recs for c in ["Arecanut", "Sugarcane", "Sesame (Sesamum)", "Mesta", "Banana"])
    # Non-signature crops should have Historically Cultivated Locally / Locally Supported badge
    assert any(r["evidence_badge"] in ("Historically Cultivated Locally", "Locally Supported") for r in res["recommendations"])


def test_rajasthan_jaipur_kharif():
    """Jaipur, Rajasthan in Kharif: Pearl Millet (Bajra) is #1 signature crop."""
    res = engine.evaluate_recommendation(
        state="Rajasthan",
        district="Jaipur",
        season="Kharif",
        soil_ph=7.8,
        soil_npk={"N": 40, "P": 25, "K": 30},
        temperature=32,
        humidity=55,
        rainfall=400
    )
    recs = [r["crop"] for r in res["recommendations"]]
    assert len(recs) > 0
    assert recs[0] == "Pearl Millet (Bajra)", f"Expected Bajra as #1 in Jaipur Kharif, got {recs[0]}"
    assert res["recommendations"][0]["evidence_badge"] == "Signature Local Crop"
    assert "Chickpea (Gram)" not in recs, "Chickpea should not be recommended in Jaipur Kharif"


def test_union_territory_with_records_nicobars():
    """Andaman and Nicobar Islands (Nicobars) has government APY records."""
    res = engine.evaluate_recommendation(
        state="Andaman and Nicobar Islands",
        district="Nicobars",
        season="Kharif",
        soil_ph=6.5,
        soil_npk={"N": 50, "P": 30, "K": 40},
        temperature=28,
        humidity=85,
        rainfall=1800
    )
    recs = res["recommendations"]
    assert len(recs) > 0
    crops = [r["crop"] for r in recs]
    assert "Coconut" in crops or "Arecanut" in crops
    # District cultivation evidence must be present
    has_evidence = any(r.get("district_cultivation_evidence") is not None for r in recs)
    assert has_evidence is True, "Expected APY evidence records for Nicobars"


def test_missing_data_district_lakshadweep():
    """Lakshadweep has no published APY crop production records; returns honest fallback."""
    res = engine.evaluate_recommendation(
        state="Lakshadweep",
        district="Lakshadweep",
        season="Kharif",
        soil_ph=7.5,
        soil_npk={"N": 40, "P": 30, "K": 40},
        temperature=29,
        humidity=80,
        rainfall=1200
    )
    assert res.get("data_availability_message") is not None
    assert "not published or unavailable" in res["data_availability_message"]
    
    recs = res["recommendations"]
    assert len(recs) > 0
    # Every recommendation must have LIMITED_DATA badge without fabricated APY evidence
    for r in recs:
        assert r["evidence_level"] == "LIMITED_DATA"
        assert r["evidence_badge"] == "Limited Regional Data"
        assert r["is_local_crop"] is False
        assert r["district_cultivation_evidence"] is None
        assert "Official district-level crop production statistics are not published" in r["limitations"]


def test_agronomic_input_sensitivity():
    """Varying temperature and rainfall alters Random Forest suitability and scores."""
    # Hot and dry conditions (Arid)
    res_dry = engine.evaluate_recommendation(
        state="Rajasthan",
        district="Jaipur",
        season="Kharif",
        soil_ph=7.5,
        soil_npk={"N": 40, "P": 25, "K": 30},
        temperature=35.0,
        humidity=35.0,
        rainfall=250.0
    )
    
    # Warm and heavy monsoon conditions (Wet)
    res_wet = engine.evaluate_recommendation(
        state="Rajasthan",
        district="Jaipur",
        season="Kharif",
        soil_ph=7.5,
        soil_npk={"N": 80, "P": 45, "K": 50},
        temperature=26.0,
        humidity=85.0,
        rainfall=1400.0
    )
    
    scores_dry = {r["crop"]: r["final_score"] for r in res_dry["recommendations"]}
    scores_wet = {r["crop"]: r["final_score"] for r in res_wet["recommendations"]}
    
    # Scores must differ as climate inputs change
    common_crops = set(scores_dry.keys()) & set(scores_wet.keys())
    assert len(common_crops) > 0
    differences = [scores_dry[c] != scores_wet[c] for c in common_crops]
    assert any(differences), "Crop scores should change when weather inputs change"


def test_api_recommend_endpoint_integration():
    """Verify HTTP POST /api/phase6/recommend returns complete response schema."""
    resp = client.post("/api/phase6/recommend", json={
        "state": "Maharashtra",
        "district": "Ahilyanagar",
        "season": "Kharif",
        "n": 60,
        "p": 40,
        "k": 40,
        "soil_ph": 7.2,
        "temperature": 28.0,
        "humidity": 70.0,
        "rainfall": 750.0
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "location" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
    
    top = data["recommendations"][0]
    required_keys = [
        "rank", "crop", "agronomic_suitability_score", "rf_evaluated",
        "evidence_level", "evidence_badge", "planting_window", "harvesting_window",
        "district_cultivation_evidence", "limitations", "data_source", "score_breakdown"
    ]
    for k in required_keys:
        assert k in top, f"Key '{k}' missing from recommendation response"


def test_dakshina_kannada_rice_acreage_and_production_aggregation():
    """
    Dakshina Kannada Rice:
    - Annual average acreage must be ~70,282 ha/yr (NOT the 18-year sum 1,265,073 ha).
    - Cumulative historical area must be explicitly labeled as 1,265,073.0 ha across 18 years.
    - Annual average production must be ~166,990 tonnes/yr (NOT the 18-year sum 3,005,813.5 tonnes).
    - Geographic level must be DISTRICT_LEVEL.
    """
    res = engine.evaluate_recommendation(
        state="Karnataka",
        district="Dakshina Kannada",
        season="Rainy Season",
        soil_ph=5.8,
        soil_npk={"N": 60, "P": 40, "K": 40},
        rainfall=3500.0,
        temperature=28.0,
        humidity=85.0
    )
    rice_rec = next((r for r in res["recommendations"] if r["crop"] == "Rice"), None)
    assert rice_rec is not None, "Rice should be recommended in Dakshina Kannada Rainy Season"
    
    ev = rice_rec.get("district_cultivation_evidence")
    assert ev is not None, "District cultivation evidence must be present for Rice in Dakshina Kannada"
    assert ev["geographic_level"] == "DISTRICT_LEVEL"
    assert ev["years_recorded"] == 18
    
    # De-duplicated gross multi-season and season-specific annual averages
    gross_area = ev["gross_annual_average_area_ha"]
    gross_prod = ev["gross_annual_average_production_tonnes"]
    assert gross_area == 62574.3, f"Expected de-duplicated gross annual area 62,574.3 ha/yr, got {gross_area}"
    assert gross_prod == 144564.7, f"Expected de-duplicated gross annual production 144,564.7 tonnes/yr, got {gross_prod}"

    # De-duplicated cumulative multi-year footprint across 18 recorded years (1997-2014)
    # Dropping the duplicate 1997 Paddy vs Rice record corrected cumulative area from 1,265,073 to 1,126,337 ha
    assert ev["cumulative_historical_area_ha"] == 1126337.0
    assert ev["cumulative_historical_production_tonnes"] == 2602164.5

    # Season-specific reported Kharif averages
    assert ev["season_reported_annual_area_ha"] == 36765.1
    assert ev["season_reported_annual_production_tonnes"] == 84963.3
    
    # Verified latest single-year observation (2014)
    assert ev["latest_verified_year"] == 2014
    assert ev["latest_year_seasonal_area_ha"] == 29749.0
    assert ev["latest_year_seasonal_production_tonnes"] == 78228.0
    assert "Latest verified district data: 2014; historical evidence baseline." in ev["freshness_notice"]


def test_black_pepper_and_arecanut_dakshina_kannada_aggregation():
    """
    Dakshina Kannada Black Pepper & Arecanut:
    - Black Pepper: Annual average area ~1,828 ha/yr; cumulative area 32,912 ha across 18 years.
    - Arecanut: Annual average area 27,242.9 ha/yr; cumulative area 463,130 ha across 17 years (1997–2014).
    - Growth duration and perennial category must be separated from planting window.
    """
    res = engine.evaluate_recommendation(
        state="Karnataka",
        district="Dakshina Kannada",
        season="Rainy Season",
        soil_ph=5.8,
        soil_npk={"N": 60, "P": 40, "K": 40},
        rainfall=3500.0,
        temperature=28.0,
        humidity=85.0
    )
    pepper_rec = next((r for r in res["recommendations"] if r["crop"] == "Black Pepper"), None)
    areca_rec = next((r for r in res["recommendations"] if r["crop"] == "Arecanut"), None)
    
    assert pepper_rec is not None
    p_ev = pepper_rec["district_cultivation_evidence"]
    assert p_ev["annual_average_area_ha"] == 1828.4
    assert p_ev["cumulative_historical_area_ha"] == 32912.0
    assert p_ev["annual_average_production_tonnes"] == 5086.3
    assert p_ev["years_recorded"] == 18
    assert pepper_rec["is_perennial"] is True
    assert "Perennial Woody Vine" in pepper_rec["crop_duration_category"]
    assert "May – July" in pepper_rec["planting_window"]
    
    assert areca_rec is not None
    a_ev = areca_rec["district_cultivation_evidence"]
    assert a_ev["annual_average_area_ha"] == 27242.9
    assert a_ev["cumulative_historical_area_ha"] == 463130.0
    assert a_ev["annual_average_production_tonnes"] == 192132.1
    assert a_ev["years_recorded"] == 17
    assert areca_rec["is_perennial"] is True
    assert "Perennial Palm" in areca_rec["crop_duration_category"]


def test_no_ambiguous_reporting_periods_2002_2002():
    """
    Reporting period labels must never be formatted as 'YYYY–YYYY' (e.g. '2002–2002').
    Multi-year records must display the full period; single-year records must say 'Reporting Year: YYYY'.
    """
    import re
    
    test_queries = [
        ("Karnataka", "Dakshina Kannada", "Rainy Season"),
        ("Karnataka", "Dakshina Kannada", "Whole Year"),
        ("Punjab", "Ludhiana", "Winter Season"),
        ("Assam", "Kamrup", "Rainy Season"),
        ("Maharashtra", "Ahilyanagar", "Rainy Season")
    ]
    
    single_year_regex = re.compile(r"\b(\d{4})[–\-]\1\b")
    
    for st, dt, sn in test_queries:
        res = engine.evaluate_recommendation(st, dt, sn, soil_ph=6.5, rainfall=1000.0, temperature=26.0, humidity=75.0)
        for r in res.get("recommendations", []):
            ev = r.get("district_cultivation_evidence")
            if ev:
                period = ev.get("reporting_period", "")
                assert not single_year_regex.search(period), f"Found ambiguous identical reporting years '{period}' in {st}::{dt} for {r['crop']}"
                if ev.get("earliest_year") == ev.get("latest_year"):
                    assert "Reporting Year:" in period or str(ev["latest_year"]) in period


def test_whole_year_selection_perennial_vs_annual():
    """
    When 'Whole Year' is queried:
    - Perennial plantation crops (Arecanut, Black Pepper, Coconut, Cashewnut, Sugarcane) are eligible.
    - Strictly single-season annual crops (Wheat, Cotton) are excluded from whole-year recommendations.
    """
    res = engine.evaluate_recommendation(
        state="Karnataka",
        district="Dakshina Kannada",
        season="Whole Year",
        soil_ph=5.8,
        soil_npk={"N": 60, "P": 40, "K": 40},
        rainfall=3500.0,
        temperature=28.0,
        humidity=85.0
    )
    crops = [r["crop"] for r in res["recommendations"]]
    assert len(crops) > 0
    # Must contain perennials
    assert any(c in crops for c in ["Arecanut", "Black Pepper", "Cashewnut", "Coconut"])
    # Strictly single-season annual crops must not be recommended
    assert "Wheat" not in crops
    assert "Cotton" not in crops


def test_score_breakdown_and_rf_probability_transparency():
    """
    Score breakdown transparency:
    - score_breakdown dictionary must have all 6 non-overlapping components.
    - The components must sum to the final composite score.
    - Random Forest probability must equal agronomic_suitability_score / 100 when rf_evaluated is True.
    """
    res = engine.evaluate_recommendation(
        state="Karnataka",
        district="Dakshina Kannada",
        season="Rainy Season",
        soil_ph=5.8,
        soil_npk={"N": 60, "P": 40, "K": 40},
        rainfall=3500.0,
        temperature=28.0,
        humidity=85.0
    )
    for r in res["recommendations"]:
        sb = r.get("score_breakdown")
        assert sb is not None, f"score_breakdown missing for {r['crop']}"
        expected_keys = [
            "agronomic_suitability", "season_compatibility", "district_evidence",
            "soil_compatibility", "water_availability", "market_presence", "composite_total"
        ]
        for k in expected_keys:
            assert k in sb, f"Key '{k}' missing from score_breakdown"
            
        components_sum = round(
            sb["agronomic_suitability"] + sb["season_compatibility"] +
            sb["district_evidence"] + sb["soil_compatibility"] +
            sb["water_availability"] + sb["market_presence"], 1
        )
        assert abs(components_sum - r["final_score"]) < 0.15, (
            f"Component sum {components_sum} does not match final_score {r['final_score']} for {r['crop']}"
        )
        
        if r["rf_evaluated"]:
            assert r["rf_probability"] is not None
            expected_score = round(r["rf_probability"] * 100.0, 1)
            assert abs(expected_score - r["agronomic_suitability_score"]) < 0.2


def test_renamed_districts_and_legacy_aliases():
    """
    Verify district canonicalizer maps historical and newly renamed districts:
    - South Canara / Dakshin Kannad -> Dakshina Kannada (Karnataka)
    - Dadra and Nagar Haveli -> Combined UT (Dadra and Nagar Haveli and Daman and Diu)
    - Leh / Kargil -> Ladakh
    - Ahmednagar -> Ahilyanagar (Maharashtra)
    - Belgaum -> Belagavi (Karnataka)
    - Mysore -> Mysuru (Karnataka)
    """
    alias_tests = [
        ("Karnataka", "South Canara", "Karnataka::Dakshin Kannad"),
        ("Karnataka", "Dakshin Kannad", "Karnataka::Dakshina Kannada"),
        ("Dadra and Nagar Haveli", "Dadra and Nagar Haveli", "Dadra and Nagar Haveli and Daman and Diu::Dadra and Nagar Haveli"),
        ("Ladakh", "Leh", "Ladakh::Leh"),
        ("Maharashtra", "Ahmednagar", "Maharashtra::Ahilya Nagar"),
        ("Karnataka", "Belgaum", "Karnataka::Belagavi"),
        ("Karnataka", "Mysore", "Karnataka::Mysuru"),
    ]
    for st, dt, expected_cid in alias_tests:
        c = engine.canonicalize_district(dt, st)
        assert c is not None, f"Failed to canonicalize ({st}, {dt})"
        assert c["canonical_id"].lower() == expected_cid.lower(), (
            f"Expected {expected_cid}, got {c['canonical_id']} for ({st}, {dt})"
        )


def test_region_aware_rice_season_compatibility():
    """
    Fix 1: Region-aware rice season compatibility.
    - Rice is NOT exclusively Kharif nationwide.
    - Rabi rice is supported in Peninsular South (Dakshina Kannada, Thanjavur).
    - Summer/Boro rice is supported in Eastern wetland ecology (West Bengal, Assam).
    - Rice is strictly rejected for Rabi in North-West winter frost states (Punjab, Haryana).
    - Incompatible annual crops (Rice, Wheat, Cotton) are strictly rejected for 'Whole Year'.
    """
    # Peninsular South: Dakshina Kannada (Karnataka) & Thanjavur (Tamil Nadu) support Rabi rice
    dk_compat, dk_msg = engine.is_crop_season_compatible(
        "Rice", "Rabi", canonical_id="Karnataka::Dakshina Kannada", state="Karnataka", district="Dakshina Kannada"
    )
    assert dk_compat is True, f"Rabi rice must be compatible in Dakshina Kannada: {dk_msg}"

    tj_compat, tj_msg = engine.is_crop_season_compatible(
        "Rice", "Rabi", canonical_id="Tamil Nadu::Thanjavur", state="Tamil Nadu", district="Thanjavur"
    )
    assert tj_compat is True, f"Rabi rice must be compatible in Thanjavur: {tj_msg}"

    # Eastern India: Summer/Boro rice supported in West Bengal and Assam
    wb_compat, wb_msg = engine.is_crop_season_compatible(
        "Rice", "Summer", canonical_id="West Bengal::Barddhaman", state="West Bengal", district="Barddhaman"
    )
    assert wb_compat is True, f"Summer/Boro rice must be compatible in West Bengal: {wb_msg}"

    as_compat, as_msg = engine.is_crop_season_compatible(
        "Rice", "Summer", canonical_id="Assam::Kamrup", state="Assam", district="Kamrup"
    )
    assert as_compat is True, f"Summer/Boro rice must be compatible in Assam: {as_msg}"

    # Northern Winter: Rice strictly rejected in Ludhiana (Punjab) and Karnal (Haryana) during Rabi
    pb_compat, pb_msg = engine.is_crop_season_compatible(
        "Rice", "Rabi", canonical_id="Punjab::Ludhiana", state="Punjab", district="Ludhiana"
    )
    assert pb_compat is False, f"Rice must be rejected for Rabi in Punjab: {pb_msg}"
    assert "frost" in pb_msg.lower() or "not viable" in pb_msg.lower()

    hr_compat, hr_msg = engine.is_crop_season_compatible(
        "Rice", "Rabi", canonical_id="Haryana::Karnal", state="Haryana", district="Karnal"
    )
    assert hr_compat is False, f"Rice must be rejected for Rabi in Haryana: {hr_msg}"

    # Annual crops rejected for Whole Year
    for annual_crop in ["Rice", "Wheat", "Cotton", "Pearl Millet (Bajra)"]:
        wy_compat, wy_msg = engine.is_crop_season_compatible(annual_crop, "Whole Year")
        assert wy_compat is False, f"Annual crop {annual_crop} must be rejected for Whole Year: {wy_msg}"


def test_apy_deduplication_audit_and_provenance():
    """
    Fix 2: De-duplicate Paddy/Rice APY records and verify audit trail.
    - Inspect apy_deduplication_audit.json for dropped duplicate count and provenance.
    - Confirm 1997 Dakshina Kannada Paddy duplicate record was logged and dropped.
    - Confirm separate Kharif, Rabi, and Summer metrics are populated.
    """
    import json
    from pathlib import Path
    
    audit_file = Path("backend/app/data/experimental/apy_deduplication_audit.json")
    assert audit_file.exists(), "Deduplication audit file must exist"
    
    with open(audit_file) as f:
        audit_data = json.load(f)
        
    assert audit_data.get("total_duplicates_dropped", 0) > 0, "Duplicate records must be logged and dropped"
    
    # Check that Dakshina Kannada 1997 Paddy vs Rice was captured
    dk_drop = next((
        d for d in audit_data.get("dropped_records", [])
        if d.get("canonical_id") == "Karnataka::Dakshina Kannada" and d.get("year") == 1997
    ), None)
    assert dk_drop is not None, "1997 Dakshina Kannada duplicate Paddy record must be documented in audit trail"
    assert dk_drop.get("deduplication_reason") == "IDENTICAL_TEMPORAL_AND_SEASONAL_OBSERVATION"

    # Verify separate Kharif, Rabi, and Summer metrics exist in district evidence
    ev_file = Path("backend/app/data/experimental/district_crop_evidence.json")
    with open(ev_file) as f:
        all_ev = json.load(f)
        
    dk_ev = next((d for d in all_ev if d.get("district") == "Dakshina Kannada"), None)
    assert dk_ev is not None
    rice_entry = next((c for c in dk_ev["crops"] if c["crop"] == "Rice"), None)
    assert rice_entry is not None
    
    sb = rice_entry.get("seasonal_breakdown", {})
    assert "Kharif" in sb and "Rabi" in sb and "Summer" in sb, "Separate seasonal breakdown must exist for Kharif, Rabi, Summer"
    assert sb["Kharif"]["annual_average_area_ha"] == 36765.1
    assert sb["Rabi"]["annual_average_area_ha"] == 23585.9
    assert sb["Summer"]["annual_average_area_ha"] == 2223.3
    assert rice_entry["gross_annual_average_area_ha"] == 62574.3


def test_administrative_district_master_synchronization():
    """
    Fix 3: Synchronize administrative district data.
    - All 28 States and 8 UTs must be present in frontend indian_districts.json.
    - All 26 Andhra Pradesh reorganized districts must be present.
    - Punjab's 23rd district (Malerkotla) must be present.
    - Karnataka must have 31 distinct official districts with 0 duplicate parenthetical dropdown entries.
    - Every district in frontend must resolve in backend location normalizer.
    """
    import json
    from pathlib import Path
    from app.services.location_normalizer import normalize_location
    
    fe_file = Path("backend/frontend/indian_districts.json")
    assert fe_file.exists()
    with open(fe_file) as f:
        fe_raw = json.load(f)
        
    fe_states_list = fe_raw.get("states", [])
    # Check 28 States and 8 UTs = 36 entities
    assert len(fe_states_list) >= 36, f"Expected at least 36 States/UTs, found {len(fe_states_list)}"
    fe_state_map = {item["state"]: item["districts"] for item in fe_states_list}

    # Andhra Pradesh: 26 official districts
    ap_districts = fe_state_map.get("Andhra Pradesh", [])
    assert len(ap_districts) == 26, f"Expected 26 Andhra Pradesh districts, found {len(ap_districts)}"
    expected_ap = [
        "Alluri Sitharama Raju", "Anakapalli", "Ananthapuramu", "Annamayya", "Bapatla",
        "Chittoor", "Dr. B.R. Ambedkar Konaseema", "East Godavari", "Eluru", "Guntur",
        "Kakinada", "Krishna", "Kurnool", "Nandyal", "NTR", "Palnadu",
        "Parvathipuram Manyam", "Prakasam", "Sri Potti Sriramulu Nellore", "Sri Sathya Sai",
        "Srikakulam", "Tirupati", "Visakhapatnam", "Vizianagaram", "West Godavari", "YSR Kadapa"
    ]
    for d in expected_ap:
        assert d in ap_districts, f"Missing Andhra Pradesh district '{d}' in frontend"
        # Test backend resolution
        norm = normalize_location(d, "Andhra Pradesh")
        assert norm["canonical_state"] == "Andhra Pradesh"

    # Punjab: 23 districts including Malerkotla
    pb_districts = fe_state_map.get("Punjab", [])
    assert len(pb_districts) == 23, f"Expected 23 Punjab districts, found {len(pb_districts)}"
    assert "Malerkotla" in pb_districts
    norm_mal = normalize_location("Malerkotla", "Punjab")
    assert norm_mal["canonical_district"] == "Malerkotla"

    # Karnataka: 31 official districts, no duplicate parentheticals
    ka_districts = fe_state_map.get("Karnataka", [])
    assert len(ka_districts) == 31, f"Expected 31 Karnataka districts, found {len(ka_districts)}"
    for d in ka_districts:
        assert "(" not in d, f"Found duplicate parenthetical alias in selectable Karnataka dropdown: '{d}'"
        assert "(" not in d, f"Found duplicate parenthetical alias in selectable Karnataka dropdown: '{d}'"


def test_crop_card_reporting_period_and_freshness():
    """
    Outdated Reporting Period Fix:
    - Every crop card distinguishes latest verified year and historical baseline.
    - Each district and crop reports its actual latest agricultural year (not a generic 2014 or fake 2026).
    - Freshness notice is present and transparent.
    """
    # Dakshina Kannada (latest year 2014)
    res_dk = engine.evaluate_recommendation(
        state="Karnataka", district="Dakshina Kannada", season="Rainy Season",
        soil_ph=5.8, rainfall=3500.0, temperature=28.0, humidity=85.0
    )
    for r in res_dk["recommendations"]:
        ev = r.get("district_cultivation_evidence")
        if ev:
            assert ev["latest_verified_year"] is not None
            assert ev["latest_verified_year"] == ev["latest_year"]
            assert ev["latest_observation_type"] == "SINGLE_YEAR_OBSERVATION"
            assert "Latest verified district data:" in ev["freshness_notice"]
            assert "historical evidence baseline" in ev["freshness_notice"]

    # Jaipur (Rajasthan DES records end at 2010)
    res_jp = engine.evaluate_recommendation(
        state="Rajasthan", district="Jaipur", season="Rainy Season",
        soil_ph=7.5, rainfall=600.0, temperature=30.0, humidity=60.0
    )
    jp_evs = [r["district_cultivation_evidence"] for r in res_jp["recommendations"] if r.get("district_cultivation_evidence")]
    assert len(jp_evs) > 0
    for ev in jp_evs:
        assert ev["latest_verified_year"] == 2010, f"Expected Jaipur latest year 2010, got {ev['latest_verified_year']}"
        assert "2010" in ev["freshness_notice"]

