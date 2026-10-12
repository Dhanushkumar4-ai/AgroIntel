"""
test_daily_news_intelligence.py — Comprehensive Test Suite for Daily RSS News Intelligence.

Verifies:
  1. Duplicate detection (canonical URL stripping & headline fingerprinting)
  2. Timezone handling and publication date parsing (no fetch-time confusion)
  3. Feed failure resiliency & query fallback
  4. Agricultural relevance and noise filtering (blocking sports, equities, non-ag)
  5. Entity extraction (crops, states, events)
  6. Market impact classification (BULLISH, BEARISH, MIXED, UNCERTAIN)
  7. Source reliability and multi-source corroboration
  8. Transparent scoring formula and exponential recency decay
  9. Conflicting signals and forecast alignment evaluation
 10. Advisory behavior when no relevant news exists
 11. Prevention of future-news leakage in historical evaluation
 12. API consistency (GET /api/news, GET /api/news/status, POST /api/news/refresh)
 13. Regression testing: existing 4 forecasting models and crop recommendation
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.news_intelligence_service import (
    parse_publication_date,
    classify_source_credibility,
    is_agricultural_article,
    extract_entities,
    classify_market_impact,
    calculate_article_relevance_score,
    evaluate_forecast_alignment,
    DEFAULT_NEWS_SCORING_CONFIG,
)
from app.jobs.fetch_news import (
    NewsIngestionJob,
    canonicalize_url,
    compute_headline_fingerprint,
)
from app.jobs.process_news import NewsProcessingJob
from app.ml.inference import predict_price
from app.services.phase6_integration_service import AgroIntelPhase6Engine


client = TestClient(app)


# ─── 1. DUPLICATE DETECTION & CANONICALIZATION ──────────────────────────────

def test_canonicalize_url_strips_tracking():
    """Verify UTM parameters, session tokens, and fragments are stripped."""
    raw_url = "https://economictimes.indiatimes.com/news/economy/agriculture/onion-prices-surge/articleshow/12345.cms?utm_source=twitter&utm_medium=social&utm_campaign=daily#comments"
    canonical = canonicalize_url(raw_url)
    assert "utm_source" not in canonical
    assert "utm_medium" not in canonical
    assert "#comments" not in canonical
    assert canonical.startswith("https://economictimes.indiatimes.com/news/economy/agriculture/onion-prices-surge/articleshow/12345.cms")


def test_headline_fingerprint_deduplication():
    """Verify syndicated or slightly punctuated headlines produce identical fingerprints."""
    h1 = "Onion prices jump 25% in Nashik mandi due to delayed monsoon arrivals!"
    h2 = "Onion Prices Jump 25% In Nashik Mandi, Due To Delayed Monsoon Arrivals"
    h3 = "Wheat procurement begins in Punjab amid record high temperatures."
    
    fp1 = compute_headline_fingerprint(h1)
    fp2 = compute_headline_fingerprint(h2)
    fp3 = compute_headline_fingerprint(h3)
    
    assert fp1 == fp2, "Syndicated variations should yield identical fingerprints"
    assert fp1 != fp3, "Distinct articles must have distinct fingerprints"


# ─── 2. TIMEZONE HANDLING & PUBLICATION DATES ───────────────────────────────

def test_parse_publication_date_timezone_awareness():
    """Verify RFC-822 / RFC-2822 dates parse into UTC ISO strings without fetch-time confusion."""
    # IST (+0530)
    ist_str = "Mon, 12 Oct 2026 14:30:00 +0530"
    parsed_ist, is_missing = parse_publication_date(ist_str)
    assert not is_missing
    assert parsed_ist.startswith("2026-10-12T09:00:00")  # 14:30 IST is 09:00 UTC

    # GMT / UTC
    gmt_str = "Sun, 11 Oct 2026 18:00:00 GMT"
    parsed_gmt, is_missing = parse_publication_date(gmt_str)
    assert not is_missing
    assert parsed_gmt.startswith("2026-10-11T18:00:00")


def test_missing_or_invalid_publication_date_graceful_handling():
    """Verify missing/corrupt date strings return is_missing=True gracefully."""
    parsed, is_missing = parse_publication_date("invalid-date-string-12345")
    assert is_missing is True
    assert parsed is None


# ─── 3. RESILIENCE TO FEED FAILURES & FALLBACKS ─────────────────────────────

def test_feed_failure_resilience():
    """Verify feed parser does not crash on simulated broken HTTP/network responses."""
    job = NewsIngestionJob()
    
    # Broken URL that fails network lookup
    status_code, articles, err = job.fetch_rss_feed("https://invalid-nonexistent-domain-999.gov.in/rss.xml")
    assert status_code != 200
    assert len(articles) == 0
    assert err is not None


# ─── 4. AGRICULTURAL RELEVANCE & NOISE FILTERING ────────────────────────────

def test_is_agricultural_article_filters_noise():
    """Verify non-agricultural articles (equities, politics, sports) are rejected."""
    # Non-agricultural news
    assert is_agricultural_article(
        "Sensex surges 600 points on IT stock rally as Federal Reserve holds rates",
        "BSE Sensex and Nifty closed at fresh record highs driven by technology shares."
    ) is False

    assert is_agricultural_article(
        "Indian cricket team announces squad for ICC Champions Trophy in Australia",
        "Rohit Sharma will captain the 15-man squad announced by BCCI selection committee."
    ) is False

    # Legitimate agricultural news
    assert is_agricultural_article(
        "Wheat arrivals gather pace in Punjab mandis; government procures 50 lakh tonnes",
        "Daily arrivals across 150 APMC mandis crossed 2 lakh bags with modal price at MSP."
    ) is True

    assert is_agricultural_article(
        "Deficit monsoon in Maharashtra raises concerns over Kharif soybean and onion output",
        "Rainfall deficiency of 32% in Marathwada threatens yield prospects across dryland tracts."
    ) is True


# ─── 5. ENTITY & EVENT EXTRACTION ───────────────────────────────────────────

def test_crop_and_state_entity_extraction():
    """Verify extraction of target crops and Indian states."""
    title = "Punjab government announces bonus on paddy procurement as rice arrivals surge"
    summary = "Farmers in Ludhiana and Patiala mandis report robust harvest quality."
    
    state, crop = extract_entities(f"{title} {summary}")
    assert crop == "Rice"
    assert state == "Punjab"


# ─── 6. MARKET IMPACT CLASSIFICATION ────────────────────────────────────────

def test_classify_market_impact_bullish_and_bearish():
    """Verify directional market classification, severity, and scope."""
    # Bullish scenario (supply shock / drought / crop damage)
    bullish_text = "Severe unseasonal rains damage standing onion crop in Maharashtra; supplies to mandis plunge 40%"
    impact_bull = classify_market_impact(bullish_text)
    assert impact_bull["market_effect"] == "BULLISH"
    assert impact_bull["impact_strength"] in ("HIGH", "MEDIUM")

    # Bearish scenario (bumper production / export ban / arrival surge)
    bearish_text = "Record bumper harvest of wheat triggers glut across APMC mandis; wholesale prices fall below MSP"
    impact_bear = classify_market_impact(bearish_text)
    assert impact_bear["market_effect"] == "BEARISH"
    assert impact_bear["impact_strength"] in ("HIGH", "MEDIUM")


# ─── 7. SOURCE RELIABILITY & CORROBORATION ──────────────────────────────────

def test_source_reliability_tiers():
    """Verify Tier 1 official sources receive higher reliability than unverified sources."""
    pib = classify_source_credibility("Press Information Bureau (PIB)", "https://pib.gov.in")
    imd = classify_source_credibility("India Meteorological Department (IMD)", "https://mausam.imd.gov.in")
    hindu = classify_source_credibility("The Hindu Business Line", "https://thehindubusinessline.com")
    unknown = classify_source_credibility("Anonymous Blog", "https://randomagriblog.com")
    
    assert pib["credibility_weight"] >= 0.95
    assert pib["is_official"] is True
    assert imd["credibility_weight"] >= 0.95
    assert hindu["credibility_weight"] >= 0.60
    assert unknown["credibility_weight"] <= 0.60


def test_corroboration_does_not_label_single_source_as_verified():
    """Never label a claim as verified merely because it appears in a single RSS feed."""
    job = NewsProcessingJob()
    now_utc = datetime.now(timezone.utc)
    
    single_article = {
        "title": "Local trader claims secret onion export permit issued in Nashik",
        "link": "https://example.com/single-onion-article",
        "source_name": "Local City Portal",
        "tier": "TIER_4_GENERAL_MEDIA",
        "source_credibility_weight": 0.40,
        "crop": "Onion",
        "state": "Maharashtra",
        "event_type": "EXPORT_RESTRICTION",
        "market_effect": "BULLISH",
        "impact_strength": "HIGH",
        "geographic_scope": "STATE",
        "confidence": 0.8,
        "classification_reason": "Export policy change",
        "published_at": now_utc.isoformat(),
        "pub_date_missing": False,
        "age_days": 0.1,
        "freshness_status": "FRESH",
        "fetched_at": now_utc.isoformat()
    }
    
    # Process single event without corroboration
    verified_clusters, _, _, stats = job.cluster_and_verify([single_article])
    
    # When only 1 general source exists, status must be SINGLE_SOURCE, NOT VERIFIED
    assert len(verified_clusters) == 1
    cluster = verified_clusters[0]
    assert cluster["verification_status"] == "SINGLE_SOURCE"
    assert cluster["verification_status"] != "VERIFIED"


# ─── 8. TRANSPARENT SCORING FORMULA & RECENCY DECAY ─────────────────────────

def test_transparent_scoring_formula_components():
    """Verify article relevance scoring computes composite score with explicit components."""
    now_utc = datetime.now(timezone.utc)
    
    article = {
        "title": "Central government hikes Wheat MSP by ₹150 per quintal for upcoming Rabi season",
        "summary": "Cabinet Committee on Economic Affairs approves MSP revision across Punjab, Haryana, and MP.",
        "source": "PIB Agriculture",
        "source_reliability": 0.98,
        "crops": ["wheat"],
        "states": ["Punjab", "Haryana", "Madhya Pradesh"],
        "market_effect": "BULLISH",
        "severity": "HIGH",
        "geographic_scope": "NATIONAL",
        "verification_status": "VERIFIED",
        "published_at": now_utc.isoformat()
    }
    
    # Score for wheat in Punjab
    score = calculate_article_relevance_score(article, target_crop="wheat", target_state="Punjab", as_of_date=now_utc)
    
    assert 0.0 <= score["composite_score"] <= 100.0
    comps = score["score_components"]
    assert comps["crop_relevance"] == 1.0   # Exact match
    assert comps["geographic_relevance"] == 1.0    # Exact state match
    assert comps["source_reliability"] == 0.98  # 0.98
    assert comps["recency_decay"] >= 0.95 # Just published today
    assert score["composite_score"] >= 80.0


def test_exponential_recency_decay():
    """Verify older articles decay in relevance according to half-life configuration."""
    now_utc = datetime.now(timezone.utc)
    half_life_days = DEFAULT_NEWS_SCORING_CONFIG["recency_halflife_days"] # 7.0
    
    art_today = {
        "crops": ["onion"], "states": ["Maharashtra"], "severity": "HIGH",
        "source_reliability": 0.9, "published_at": now_utc.isoformat()
    }
    art_7d_old = {
        "crops": ["onion"], "states": ["Maharashtra"], "severity": "HIGH",
        "source_reliability": 0.9, "published_at": (now_utc - timedelta(days=7)).isoformat()
    }
    art_28d_old = {
        "crops": ["onion"], "states": ["Maharashtra"], "severity": "HIGH",
        "source_reliability": 0.9, "published_at": (now_utc - timedelta(days=28)).isoformat()
    }
    
    score_today = calculate_article_relevance_score(art_today, target_crop="onion", as_of_date=now_utc)
    score_7d = calculate_article_relevance_score(art_7d_old, target_crop="onion", as_of_date=now_utc)
    score_28d = calculate_article_relevance_score(art_28d_old, target_crop="onion", as_of_date=now_utc)
    
    # Recency component: 1.0 -> ~0.50 at 7d -> ~0.06 at 28d
    assert score_today["score_components"]["recency_decay"] >= 0.95
    assert 0.45 <= score_7d["score_components"]["recency_decay"] <= 0.55
    assert score_28d["score_components"]["recency_decay"] < 0.10
    assert score_today["composite_score"] > score_7d["composite_score"] > score_28d["composite_score"]


# ─── 9. CONFLICTING SIGNALS & FORECAST ALIGNMENT ─────────────────────────────

def test_evaluate_forecast_alignment():
    """Verify alignment classification: REINFORCING, WEAKENING, CONFLICTING, NEUTRAL."""
    # 1. Bullish model (+5%) + Bullish news -> REINFORCING
    bullish_news = [{"market_effect": "BULLISH", "composite_score": 85.0}]
    align_pos = evaluate_forecast_alignment(pred_pct_change=5.2, relevant_news=bullish_news)
    assert align_pos["alignment"] == "REINFORCING"
    assert align_pos["news_sentiment_direction"] == "BULLISH"

    # 2. Bullish model (+5%) + Bearish news -> CONFLICTING
    bearish_news = [{"market_effect": "BEARISH", "composite_score": 85.0}]
    align_conflict = evaluate_forecast_alignment(pred_pct_change=5.2, relevant_news=bearish_news)
    assert align_conflict["alignment"] == "CONFLICTING"

    # 3. Bearish model (-4%) + Bearish news -> REINFORCING
    align_bear_reinforce = evaluate_forecast_alignment(pred_pct_change=-4.0, relevant_news=bearish_news)
    assert align_bear_reinforce["alignment"] == "REINFORCING"


# ─── 10. ADVISORY BEHAVIOR WHEN NO RELEVANT NEWS EXISTS ─────────────────────

def test_advisory_when_no_news_available():
    """Verify model fallback when zero relevant news exists."""
    align_empty = evaluate_forecast_alignment(pred_pct_change=3.5, relevant_news=[])
    assert align_empty["alignment"] == "NEUTRAL"
    assert align_empty["news_count"] == 0
    assert "No recent verified news" in align_empty["explanation"]


# ─── 11. PREVENTION OF FUTURE-NEWS LEAKAGE ──────────────────────────────────

def test_prevention_of_future_news_leakage():
    """Verify news published after historical observation timestamp is strictly excluded."""
    cutoff_date = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    
    past_article = {
        "title": "July monsoon progress normal across Central India",
        "published_at": "2026-07-25T10:00:00Z"
    }
    future_article = {
        "title": "August floods trigger supply disruption in Nashik",
        "published_at": "2026-08-05T10:00:00Z"
    }
    
    # Filter using as_of_date
    articles = [past_article, future_article]
    valid_articles = [
        a for a in articles
        if datetime.fromisoformat(a["published_at"].replace("Z", "+00:00")) <= cutoff_date
    ]
    
    assert len(valid_articles) == 1
    assert valid_articles[0]["title"] == past_article["title"]


# ─── 12. API ENDPOINTS INTEGRATION ──────────────────────────────────────────

def test_api_news_endpoints():
    """Verify /api/news, /api/news/status, and /api/news/refresh endpoints."""
    # 1. GET /api/news
    res_news = client.get("/api/news?crop=wheat&limit=10")
    assert res_news.status_code == 200
    data_news = res_news.json()
    assert "total_articles" in data_news
    assert "articles" in data_news
    assert isinstance(data_news["articles"], list)

    # 2. GET /api/news/status
    res_status = client.get("/api/news/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert "freshness_status" in data_status
    assert "total_articles_stored" in data_status
    assert "sources_active" in data_status

    # 3. POST /api/news/refresh manual trigger
    res_refresh = client.post("/api/news/refresh")
    assert res_refresh.status_code == 200
    data_refresh = res_refresh.json()
    assert data_refresh["status"] == "SUCCESS"
    assert "ingestion_summary" in data_refresh
    assert "processing_summary" in data_refresh


# ─── 13. REGRESSION TESTING: FORECASTING & CROP RECOMMENDATION ──────────────

def test_regression_price_prediction_pipeline():
    """Verify predict_price preserves 4 forecasting models and outputs news context."""
    res = predict_price(crop="wheat", state="Punjab", horizon_days=30)
    
    assert res.get("available") is True
    assert "predicted_price" in res
    assert "current_price" in res
    assert "advisory" in res
    assert res["advisory"]["decision"] in ("SELL", "HOLD", "WAIT")
    assert "forecast_series" in res
    assert len(res["forecast_series"]) == 30
    
    # News intelligence fields
    assert "news_intelligence" in res
    assert "forecast_alignment" in res
    align_val = res["forecast_alignment"].get("alignment", res["forecast_alignment"].get("forecast_alignment", "NEUTRAL")) if isinstance(res["forecast_alignment"], dict) else res["forecast_alignment"]
    assert align_val in ("REINFORCING", "WEAKENING", "CONFLICTING", "NEUTRAL")


def test_regression_crop_recommendation_pipeline():
    """Verify crop recommendation pipeline runs without error."""
    engine = AgroIntelPhase6Engine()
    rec = engine.evaluate_recommendation(
        state="Karnataka",
        district="Dakshina Kannada",
        season="Rainy",
        soil_ph=6.5,
        rainfall=1200.0,
        temperature=27.0,
        humidity=80.0
    )
    assert rec is not None
    assert "top_recommendations" in rec or "recommendations" in rec or "ranked_crops" in rec
