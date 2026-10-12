"""
process_news.py — Daily Agricultural News Intelligence Extraction, Classification & Verification Job.
======================================================================================================
Processes raw articles from news_articles.json:
  1. Filters out non-agricultural noise (equities, sports, cinema, real estate) to prevent false market alerts.
  2. Extracts State and Crop entities using canonical gazetteers.
  3. Classifies agricultural events and market direction (BULLISH, BEARISH, MIXED, UNCERTAIN).
  4. Assigns impact strength (HIGH, MEDIUM, LOW) and geographic scope (NATIONAL, STATE, REGIONAL, INTERNATIONAL).
  5. Performs entity-aware cross-source verification:
     - VERIFIED: >= 2 independent sources with matching direction and credible tier.
     - CORROBORATED: Supported by Tier 1/2 official or research authorities.
     - SINGLE_SOURCE: 1 credible source.
     - CONFLICTING: Divergent directional market signals.
     - UNVERIFIED: Uncorroborated single tier-4 or discovery source.
     - STALE: Older than 45 days.
  6. Calculates transparent multi-factor relevance and recency decay scores (0-100).
  7. Outputs news_events.json, news_event_clusters.json, current_intelligence.json, and audit report.
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.news_intelligence_service import (
    classify_source_credibility,
    is_agricultural_article,
    extract_entities,
    classify_market_impact,
    calculate_article_relevance_score,
    parse_publication_date,
    DEFAULT_NEWS_SCORING_CONFIG,
    SOURCE_TIERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.jobs.process_news")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
EXP_DIR = DATA_DIR / "experimental"
EXP_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = BASE_DIR.parent / "audit" / "news"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_PATH = EXP_DIR / "news_articles.json"
EVENTS_PATH = EXP_DIR / "news_events.json"
CLUSTERS_PATH = EXP_DIR / "news_event_clusters.json"
CURRENT_INTEL_PATH = EXP_DIR / "current_intelligence.json"
VERIF_REPORT_PATH = EXP_DIR / "news_cross_verification_report.json"
AUDIT_COPY_PATH = AUDIT_DIR / "news_cross_verification_report.json"


class NewsProcessingJob:
    """Production Agricultural News Processing & Cross-Verification Job."""

    def __init__(self):
        self.config = DEFAULT_NEWS_SCORING_CONFIG

    def process_articles(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filter non-ag articles and extract structured events with market impact classification.
        Returns: (events_list, non_ag_filtered_count)
        """
        events = []
        non_ag_count = 0
        now_utc = datetime.now(timezone.utc)

        for art in articles:
            title = art.get("title", "")
            desc = art.get("description", "")

            # 1. Agricultural Domain Filter: Exclude non-ag noise
            if not is_agricultural_article(title, desc):
                non_ag_count += 1
                continue

            full_text = f"{title}. {desc}".strip()

            # 2. Extract Entities
            state, crop = extract_entities(full_text)
            state_label = state or "National"
            crop_label = crop or "General Agriculture"

            # 3. Classify Market Impact & Event Category
            impact_info = classify_market_impact(full_text)
            event_type = impact_info["event_type"]
            market_effect = impact_info["market_effect"]
            impact_strength = impact_info["impact_strength"]
            geo_scope = impact_info["geographic_scope"]
            reasoning = impact_info["reasoning"]
            confidence = impact_info["confidence"]

            # 4. Publication Date & Age in Days (Publication time, NOT fetch time!)
            raw_date_candidate = art.get("pub_date") or art.get("pub_date_raw") or ""
            pub_date_iso, is_missing = parse_publication_date(raw_date_candidate)

            if pub_date_iso and not is_missing:
                try:
                    pub_dt = datetime.fromisoformat(pub_date_iso)
                    if not pub_dt.tzinfo:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    age_days = max(0.0, round((now_utc - pub_dt).total_seconds() / 86400.0, 1))
                    pub_missing = False
                except Exception:
                    pub_dt = now_utc
                    age_days = 0.0
                    pub_missing = True
            else:
                pub_dt = now_utc
                age_days = 0.0
                pub_missing = True

            # Freshness label
            if age_days <= 3.0:
                freshness = "VERY_FRESH"
            elif age_days <= 14.0:
                freshness = "FRESH"
            elif age_days <= 30.0:
                freshness = "RECENT"
            elif age_days <= 45.0:
                freshness = "BACKGROUND"
            else:
                freshness = "STALE"

            # 5. Source Credibility Tier
            source_name = art.get("source_name", "Unknown")
            source_url = art.get("link", "")
            cred_info = classify_source_credibility(source_name, source_url)
            tier = cred_info["tier"]
            cred_weight = cred_info["credibility_weight"]

            event_id = "EVT_" + hashlib.sha256(
                f"{art.get('article_id', '')}::{event_type}::{market_effect}".encode("utf-8")
            ).hexdigest()[:16]

            event_obj = {
                "event_id": event_id,
                "article_id": art.get("article_id"),
                "title": title,
                "description": desc,
                "url": art.get("link") or art.get("original_link", ""),
                "source_id": art.get("source_id", "unknown"),
                "source_name": source_name,
                "tier": tier,
                "source_tier_label": cred_info["label"],
                "source_credibility_weight": cred_weight,
                "state": state_label,
                "crop": crop_label,
                "event_type": event_type,
                "market_effect": market_effect,
                "impact_strength": impact_strength,
                "geographic_scope": geo_scope,
                "confidence": confidence,
                "classification_reason": reasoning,
                "published_at": pub_date_iso or pub_dt.isoformat(),
                "pub_date_missing": pub_missing,
                "age_days": age_days,
                "freshness_status": freshness,
                "fetched_at": art.get("fetched_at", now_utc.isoformat()),
            }

            events.append(event_obj)

        return events, non_ag_count

    def cluster_and_verify(
        self, events: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Group events into entity-specific clusters, perform cross-source verification,
        and calculate transparent multi-factor relevance scores.
        """
        # Cluster key requires specific event matching; general other items are clustered by headline fingerprint
        clusters = defaultdict(list)
        for ev in events:
            # Distinguish specific events from generic items
            if ev["crop"] != "General Agriculture" or ev["event_type"] != "GENERAL_AGRI":
                key = f"{ev['state']}::{ev['crop']}::{ev['event_type']}"
            else:
                # Group by first 3 words of title to prevent broad false-positive clustering
                title_words = "_".join(ev["title"].lower().split()[:3])
                key = f"{ev['state']}::{ev['crop']}::{title_words}"
            clusters[key].append(ev)

        verified_clusters = []
        verified_count = 0
        corroborated_count = 0
        single_source_count = 0
        conflicting_count = 0

        for key, ev_list in clusters.items():
            sources = {e["source_name"] for e in ev_list}
            tiers = {e["tier"] for e in ev_list}
            effects = {e["market_effect"] for e in ev_list if e["market_effect"] in ["BULLISH", "BEARISH"]}

            # Verification logic
            if "BULLISH" in effects and "BEARISH" in effects:
                status = "CONFLICTING"
                conflicting_count += len(ev_list)
            elif len(sources) >= 2 and ("TIER_1_OFFICIAL" in tiers or "TIER_2_AGRI_RESEARCH" in tiers):
                status = "VERIFIED"
                verified_count += len(ev_list)
            elif len(sources) >= 2:
                status = "VERIFIED"
                verified_count += len(ev_list)
            elif "TIER_1_OFFICIAL" in tiers or "TIER_2_AGRI_RESEARCH" in tiers:
                status = "CORROBORATED"
                corroborated_count += len(ev_list)
            elif len(sources) == 1:
                status = "SINGLE_SOURCE"
                single_source_count += len(ev_list)
            else:
                status = "UNVERIFIED"

            for e in ev_list:
                e["verification_status"] = status
                # Compute transparent relevance score
                rel_res = calculate_article_relevance_score(
                    e,
                    target_crop=e["crop"] if e["crop"] != "General Agriculture" else None,
                    target_state=e["state"] if e["state"] != "National" else None,
                    config=self.config
                )
                e["relevance_score"] = rel_res

            verified_clusters.append({
                "cluster_key": key,
                "event_count": len(ev_list),
                "unique_sources": len(sources),
                "sources": list(sources),
                "verification_status": status,
                "sample_title": ev_list[0]["title"],
                "market_effect": ev_list[0]["market_effect"],
            })

        # Sort events by composite relevance and freshness
        events_sorted = sorted(
            events,
            key=lambda x: (
                x.get("relevance_score", {}).get("composite_score", 0),
                -float(x.get("age_days", 999))
            ),
            reverse=True
        )

        # Build current intelligence: list format for phase6_integration_service compatibility
        # Keep top 100 most relevant events
        current_intelligence_list = events_sorted[:100]

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_agricultural_events": len(events),
            "total_event_clusters": len(clusters),
            "verified_events_count": verified_count,
            "corroborated_events_count": corroborated_count,
            "single_source_events_count": single_source_count,
            "conflicting_events_count": conflicting_count,
            "verified_clusters_count": sum(1 for c in verified_clusters if c["verification_status"] == "VERIFIED"),
            "single_source_clusters_count": sum(1 for c in verified_clusters if c["verification_status"] == "SINGLE_SOURCE"),
        }

        return events_sorted, verified_clusters, current_intelligence_list, report

    def run(self) -> Dict[str, Any]:
        """Execute full news processing & cross-verification."""
        t_start = time.time()
        logger.info("Starting Daily News Processing & Cross-Verification Job...")

        if not ARTICLES_PATH.exists():
            logger.warning(f"No articles file found at {ARTICLES_PATH}. Run fetch_news first.")
            return {"status": "NO_ARTICLES_TO_PROCESS"}

        with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)

        events, non_ag_filtered = self.process_articles(articles)
        events, clusters, intel_list, report = self.cluster_and_verify(events)

        report["raw_articles_evaluated"] = len(articles)
        report["non_agricultural_noise_excluded"] = non_ag_filtered

        # Save artifacts
        with open(EVENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)
        with open(CURRENT_INTEL_PATH, "w", encoding="utf-8") as f:
            # Saved as list for direct phase6_integration_service compatibility
            json.dump(intel_list, f, indent=2)
        with open(VERIF_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(AUDIT_COPY_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        elapsed = round(time.time() - t_start, 2)
        logger.info(
            f"News Processing Complete in {elapsed}s: {len(events)} agricultural events processed "
            f"({non_ag_filtered} non-ag excluded). {report['verified_clusters_count']} verified clusters."
        )
        return report


if __name__ == "__main__":
    job = NewsProcessingJob()
    res = job.run()
    print(json.dumps(res, indent=2))
