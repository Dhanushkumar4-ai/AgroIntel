import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ahilya_nagar_canonicalization():
    aliases = ["Ahmednagar", "Ahmed Nagar", "Ahilyanagar", "Ahmadnagar", "Ahilya Nagar"]
    for alias in aliases:
        resp = client.post("/api/phase6/recommend", json={"state": "Maharashtra", "district": alias, "season": "Kharif"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"]["district"] == "Ahilya Nagar"
        assert data["location"]["canonical_id"] == "Maharashtra::Ahilya Nagar"

def test_advisory_same_crop_consistency():
    # When advisory recommends Sugarcane for Ahilya Nagar (Whole Year crop), target_price_crop must be Sugarcane
    resp = client.post("/api/advisory", json={"state": "Maharashtra", "district": "Ahilya Nagar", "season": "Whole Year"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_price_crop"] == "Sugarcane"
    assert data["district"] == "Ahilya Nagar"
    assert "25.0/25" not in data["combined_summary"]
    assert "UNKNOWN" not in data["combined_summary"]

def test_weather_provider_live_or_graceful():
    resp = client.get("/api/weather/current?lat=19.09&lon=74.74")
    assert resp.status_code in [200, 404]  # If endpoint mounted or direct service
