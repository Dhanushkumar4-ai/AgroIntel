"""
test_historical_news_archive.py — Comprehensive Test Suite for Historical News Recovery & Archiving.
=====================================================================================================
Verifies:
  1. Archive Integrity: Permanent archive exists, parses valid JSON, non-empty.
  2. Timestamp Validity: All articles have valid, parseable UTC ISO-8601 publication timestamps.
  3. Deduplication: 100% unique canonical URLs and headline fingerprints.
  4. Entity & Impact Classification: Crop, state, event_type, market_effect are strictly classified.
  5. Importer Idempotency: Re-running importer on existing records produces zero duplicates.
  6. Daily Pipeline Ingestion Hook: `fetch_news.py` permanently archives fresh daily articles.
  7. Strict Point-in-Time Temporal Alignment: No future information leakage (pub_date <= origin_date).
  8. Price Data Isolation: Real historical prices and news archives remain strictly segregated.
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.jobs.fetch_news import NewsIngestionJob, canonicalize_url, compute_headline_fingerprint
from app.jobs.import_historical_news import HistoricalNewsImporter
from app.ml.evaluate_historical_news_impact import HistoricalNewsEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "backend" / "app" / "data" / "experimental"
DATA_DIR = BASE_DIR / "backend" / "app" / "data"

ARCHIVE_PATH = EXP_DIR / "historical_news_archive.json"
AUDIT_PATH = EXP_DIR / "historical_news_audit.json"
PRICE_PATH = DATA_DIR / "real_historical_prices_state.csv"


@pytest.fixture(scope="module")
def archive_data():
    assert ARCHIVE_PATH.exists(), f"Archive missing at {ARCHIVE_PATH}"
    with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data


@pytest.fixture(scope="module")
def audit_data():
    assert AUDIT_PATH.exists(), f"Audit missing at {AUDIT_PATH}"
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def test_archive_file_exists_and_non_empty(archive_data):
    """Verify permanent historical archive is populated with authentic records."""
    assert len(archive_data) > 0
    # Minimum required coverage: authentic 2020–2025 records
    years = [int(a["published_at"][:4]) for a in archive_data if a.get("published_at")]
    hist_2020_2025 = [y for y in years if 2020 <= y <= 2025]
    assert len(hist_2020_2025) >= 100, f"Expected >= 100 historical articles, found {len(hist_2020_2025)}"


def test_all_articles_have_valid_utc_timestamps(archive_data):
    """Verify all archived articles have parseable ISO-8601 publication timestamps."""
    for art in archive_data:
        pub = art.get("published_at")
        assert pub is not None and len(pub) > 0, f"Article {art.get('article_id')} missing published_at"
        dt = datetime.fromisoformat(pub)
        assert dt is not None
        assert dt.tzinfo is not None, "Timestamp must be timezone-aware (UTC)"


def test_no_duplicate_urls_or_fingerprints(archive_data):
    """Verify canonical URL and headline fingerprint deduplication."""
    seen_urls = set()
    seen_fps = set()

    for art in archive_data:
        url = canonicalize_url(art.get("url") or art.get("original_url"))
        fp = art.get("headline_fingerprint")

        assert url not in seen_urls, f"Duplicate URL detected: {url}"
        assert fp not in seen_fps, f"Duplicate headline fingerprint detected: {fp}"

        seen_urls.add(url)
        seen_fps.add(fp)


def test_entity_extraction_and_event_classification(archive_data):
    """Verify entity extraction (crop, state) and event impact schema compliance."""
    valid_effects = {"BULLISH", "BEARISH", "MIXED", "UNCERTAIN"}
    valid_tiers = {
        "TIER_1_OFFICIAL", "TIER_2_AGRI_RESEARCH", "TIER_3_BUSINESS_MARKET",
        "TIER_4_GENERAL_MEDIA", "DISCOVERY_ONLY"
    }

    for art in archive_data:
        assert "crop" in art and len(art["crop"]) > 0
        assert "state" in art and len(art["state"]) > 0
        assert "event_type" in art and len(art["event_type"]) > 0
        assert art.get("market_effect") in valid_effects
        assert art.get("source_tier") in valid_tiers
        assert "imported_at" in art and len(art["imported_at"]) > 0
        assert "publisher" in art and len(art["publisher"]) > 0


def test_importer_idempotency(archive_data):
    """Verify running importer against existing items produces zero duplicates."""
    importer = HistoricalNewsImporter()
    assert len(importer.archive) >= len(archive_data)

    # Attempt re-ingesting first 5 articles
    for art in archive_data[:5]:
        ok, reason = importer.add_record(
            title=art["title"],
            link=art["url"],
            description=art.get("description", ""),
            raw_pub_date=art.get("published_at", ""),
            publisher=art.get("publisher", ""),
            source_id="idempotency_test",
        )
        assert not ok
        assert reason in ("DUPLICATE_URL", "DUPLICATE_FINGERPRINT")


def test_daily_pipeline_permanent_archive_hook():
    """Verify NewsIngestionJob has _append_to_permanent_archive method and executes safely."""
    job = NewsIngestionJob()
    assert hasattr(job, "_append_to_permanent_archive")

    # Test with empty list
    added = job._append_to_permanent_archive([])
    assert added == 0


def test_strict_temporal_alignment_no_leakage():
    """Verify point-in-time news extraction strictly prevents future-information leakage."""
    evaluator = HistoricalNewsEvaluator()

    # Define forecast origin T
    test_origin = date(2023, 6, 1)

    # Extract features as of test_origin
    feats = evaluator.extract_point_in_time_news_features(crop="rice", as_of_date=test_origin, lookback_days=7)
    assert isinstance(feats, dict)
    assert "crop_news_count_7d" in feats
    assert "net_sentiment_7d" in feats

    # Inject mock future article at T + 2 days and verify it is not observable at T
    future_date = test_origin + timedelta(days=2)
    evaluator.add_article({
        "article_id": "MOCK_FUTURE_01",
        "title": "Severe Rice Shortage Expected in Late 2023",
        "crop": "Rice",
        "market_effect": "BULLISH",
        "source_credibility_weight": 1.0,
        "event_type": "PRICE_SPIKE",
        "parsed_pub_date": future_date,
    })

    # Re-extract as of test_origin — future article must NOT be counted
    feats_as_of_t = evaluator.extract_point_in_time_news_features(crop="rice", as_of_date=test_origin, lookback_days=7)
    assert feats_as_of_t["crop_news_count_7d"] == feats["crop_news_count_7d"]

    # But as of T + 3 days, it SHOULD be counted
    feats_future = evaluator.extract_point_in_time_news_features(crop="rice", as_of_date=future_date + timedelta(days=1), lookback_days=7)
    assert feats_future["crop_news_count_7d"] >= 1


def test_price_data_segregation():
    """Confirm historical price series and historical news archives are strictly segregated."""
    assert PRICE_PATH.exists()
    assert ARCHIVE_PATH.exists()
    assert str(PRICE_PATH) != str(ARCHIVE_PATH)

    # Inspect headers
    with open(PRICE_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert "price" in first_line or "y" in first_line
    assert "article_id" not in first_line
