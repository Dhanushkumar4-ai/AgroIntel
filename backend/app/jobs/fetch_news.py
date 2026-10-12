"""
fetch_news.py — Daily Multi-Tier RSS News Ingestion Job for AgroIntel.
====================================================================
Fetches agricultural news from configured sources across credibility tiers:
  - TIER 1: Official Government & Research (ICAR, PIB, DA&FW, IMD, MoES, KVKs, State Depts)
  - TIER 2: Agri-Research & Trade (Krishi Jagran, Rural Voice, AgroSpectrum, AgriWatch, FAO, Down to Earth, etc.)
  - TIER 3: Business & Commodity Media (Economic Times, Business Standard, Financial Express, Reuters, etc.)
  - TIER 4: National & Regional Media (The Hindu, Indian Express, Times of India, Hindustan Times, etc.)
  - DISCOVERY: Targeted Google News RSS queries for key agricultural commodities and events.

Robust Resilience:
  - Canonical URL normalization (stripping UTM tracking & query parameters).
  - Normalized headline/content fingerprinting to eliminate syndication duplicates.
  - Strict publication timestamp parsing (RFC 822/2822/ISO) into UTC ISO-8601. Never treats fetch time as publication time.
  - Graceful fallback: If direct RSS returns 403/404 or errors, automatically queries Google News RSS with source site query.
  - Resilient to individual feed outages, malformed XML, and network timeouts.
  - Generates comprehensive diagnostic runtime audit: news_ingestion_audit.json.
"""

import email.utils
import hashlib
import json
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.jobs.fetch_news")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
EXP_DIR = DATA_DIR / "experimental"
EXP_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = BASE_DIR.parent / "audit" / "news"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = EXP_DIR / "news_source_registry.json"
ARTICLES_JSON_PATH = EXP_DIR / "news_articles.json"
SOURCE_STATUS_PATH = EXP_DIR / "news_source_runtime_audit.json"
INGESTION_AUDIT_PATH = EXP_DIR / "news_ingestion_audit.json"
AUDIT_COPY_PATH = AUDIT_DIR / "news_source_runtime_audit.json"


class NewsIngestionJob:
    """Production Multi-Source Daily News Ingestion Job."""

    def __init__(self):
        self.registry = self._load_registry()
        self.client_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 (AgroIntelBot/4.1)"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, text/html, */*",
        }

    def _load_registry(self) -> Dict[str, Any]:
        if not REGISTRY_PATH.exists():
            logger.warning(f"Source registry missing at {REGISTRY_PATH}. Using empty registry.")
            return {}
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading source registry: {e}")
            return {}

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Strip HTML tags, decode entities, and normalize whitespace."""
        if not raw_text:
            return ""
        # Strip CDATA markers
        text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw_text, flags=re.DOTALL)
        # Strip HTML tags
        clean = re.sub(r"<[^>]+>", " ", text)
        # Decode HTML entities
        import html
        clean = html.unescape(clean)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def canonicalize_url(raw_url: str) -> str:
        """Normalize URL by stripping tracking parameters, anchors, and protocol variants."""
        if not raw_url:
            return ""
        try:
            parsed = urllib.parse.urlparse(raw_url.strip())
            # For Google News RSS links, preserve article path
            if "news.google.com" in parsed.netloc:
                return urllib.parse.urlunparse((
                    "https", parsed.netloc.lower(), parsed.path, "", "", ""
                ))

            # Filter out standard marketing / analytics query params
            tracking_params = {
                "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                "ref", "fbclid", "gclid", "_ga", "amp", "outputtype", "ved", "usqp"
            }
            q_pairs = urllib.parse.parse_qsl(parsed.query)
            clean_q = [(k, v) for k, v in q_pairs if k.lower() not in tracking_params]
            new_query = urllib.parse.urlencode(clean_q)

            clean_url = urllib.parse.urlunparse((
                parsed.scheme.lower() or "https",
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                new_query,
                ""
            ))
            return clean_url
        except Exception:
            return raw_url.strip()

    @staticmethod
    def compute_headline_fingerprint(title: str) -> str:
        """Generate normalized alphanumeric fingerprint for detecting syndicated cross-postings."""
        clean = re.sub(r"[^a-z0-9]", "", (title or "").lower())
        return clean[:80]

    @staticmethod
    def parse_publication_date(raw_date_str: str) -> Tuple[Optional[str], bool]:
        """
        Parse raw RSS date string into timezone-aware ISO-8601 UTC string.
        Returns: (iso_utc_string or None, is_missing_flag)
        """
        if not raw_date_str or not raw_date_str.strip():
            return None, True
        raw_clean = raw_date_str.strip()

        # Try RFC 2822 / email parsing first (most standard for RSS)
        try:
            parsed_tuple = email.utils.parsedate_to_datetime(raw_clean)
            if parsed_tuple:
                if not parsed_tuple.tzinfo:
                    parsed_tuple = parsed_tuple.replace(tzinfo=timezone.utc)
                return parsed_tuple.astimezone(timezone.utc).isoformat(), False
        except Exception:
            pass

        # Try standard datetime formats
        date_formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(raw_clean, fmt)
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat(), False
            except Exception:
                continue

        return None, True

    def fetch_rss_feed(self, url: str, timeout: float = 6.0) -> Tuple[int, List[Dict[str, Any]], Optional[str]]:
        """Fetch and parse an RSS/Atom XML feed with timeout protection."""
        try:
            with httpx.Client(timeout=timeout, headers=self.client_headers, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    return resp.status_code, [], f"HTTP {resp.status_code}"

                content = resp.text
                articles = self._parse_xml_feed(content, url)
                return 200, articles, None
        except httpx.TimeoutException:
            return 408, [], f"Request Timeout (> {timeout}s)"
        except httpx.ConnectError:
            return 503, [], "Connection Failed / Server Unreachable"
        except Exception as e:
            return 500, [], str(e)

    def _parse_xml_feed(self, xml_content: str, source_url: str) -> List[Dict[str, Any]]:
        """Parse XML items into standardized article dictionaries."""
        articles = []
        if not xml_content or not xml_content.strip():
            return articles

        try:
            # Strip potential leading garbage before <?xml
            idx = xml_content.find("<?xml")
            if idx > 0:
                xml_content = xml_content[idx:]
            elif xml_content.find("<rss") > 0:
                xml_content = xml_content[xml_content.find("<rss"):]
            elif xml_content.find("<feed") > 0:
                xml_content = xml_content[xml_content.find("<feed"):]

            root = ET.fromstring(xml_content)
            # RSS channel -> item
            items = root.findall(".//item")
            if not items:
                # Atom feed entry
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for it in items:
                title = (
                    it.findtext("title") or
                    it.findtext("{http://www.w3.org/2005/Atom}title") or ""
                )
                desc = (
                    it.findtext("description") or
                    it.findtext("summary") or
                    it.findtext("{http://www.w3.org/2005/Atom}summary") or
                    it.findtext("{http://www.w3.org/2005/Atom}content") or ""
                )
                link = it.findtext("link") or ""
                if not link and it.find("{http://www.w3.org/2005/Atom}link") is not None:
                    link = it.find("{http://www.w3.org/2005/Atom}link").attrib.get("href", "")

                raw_pub_date = (
                    it.findtext("pubDate") or
                    it.findtext("published") or
                    it.findtext("{http://www.w3.org/2005/Atom}published") or
                    it.findtext("updated") or
                    it.findtext("{http://www.w3.org/2005/Atom}updated") or
                    it.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
                )

                title_clean = self._clean_text(title)
                desc_clean = self._clean_text(desc)
                canonical_link = self.canonicalize_url(link)
                fingerprint = self.compute_headline_fingerprint(title_clean)
                pub_iso, is_missing_date = self.parse_publication_date(raw_pub_date)

                if title_clean:
                    # Generate deterministic ID based on canonical link and fingerprint
                    hash_key = f"{canonical_link}::{fingerprint}"
                    art_id = "ART_" + hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:16]

                    articles.append({
                        "article_id": art_id,
                        "title": title_clean,
                        "description": desc_clean,
                        "link": canonical_link or link,
                        "original_link": link,
                        "pub_date_raw": raw_pub_date,
                        "pub_date": pub_iso,
                        "pub_date_missing": is_missing_date,
                        "headline_fingerprint": fingerprint,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "source_url": source_url,
                    })
        except ET.ParseError as pe:
            logger.debug(f"XML parse error for {source_url}: {pe}")
        except Exception as e:
            logger.debug(f"XML processing note for {source_url}: {e}")
        return articles

    def _append_to_permanent_archive(self, new_articles: List[Dict[str, Any]]) -> int:
        """
        Phase 4: Permanently append newly fetched daily articles into historical_news_archive.json
        so the historical archive grows continuously over time without being lost when the rolling
        buffer slides.
        """
        if not new_articles:
            return 0
        try:
            from app.jobs.import_historical_news import HistoricalNewsImporter
            importer = HistoricalNewsImporter()
            added = 0
            for art in new_articles:
                ok, _ = importer.add_record(
                    title=art.get("title", ""),
                    link=art.get("link", ""),
                    description=art.get("description", ""),
                    raw_pub_date=art.get("pub_date") or art.get("pub_date_raw", ""),
                    publisher=art.get("source_name", "Unknown"),
                    source_id=art.get("source_id", "daily_rss"),
                    source_provenance="daily_rss_scheduled_job",
                )
                if ok:
                    added += 1
            if added > 0:
                importer.save_and_audit()
                logger.info(f"Permanently appended {added} new daily articles to historical_news_archive.json")
            return added
        except Exception as e:
            logger.warning(f"Could not append to permanent historical news archive: {e}")
            return 0

    def run(self) -> Dict[str, Any]:
        """Execute full news ingestion across all configured tiers."""
        t_start = time.time()
        logger.info("Starting Daily News RSS Ingestion Job...")

        source_runtime_audit = []
        all_articles = []
        seen_article_ids: Set[str] = set()
        seen_fingerprints: Set[str] = set()
        seen_urls: Set[str] = set()

        # 1. Load existing articles to preserve history and deduplicate
        initial_count = 0
        if ARTICLES_JSON_PATH.exists():
            try:
                with open(ARTICLES_JSON_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        for a in existing:
                            aid = a.get("article_id")
                            url = a.get("link")
                            fp = a.get("headline_fingerprint")
                            if aid:
                                seen_article_ids.add(aid)
                            if url:
                                seen_urls.add(url)
                            if fp:
                                seen_fingerprints.add(fp)
                            all_articles.append(a)
                        initial_count = len(all_articles)
            except Exception as e:
                logger.warning(f"Could not load existing articles: {e}")

        # 2. Iterate through tiers in registry
        tiers = self.registry.get("credibility_tiers", {})
        total_sources_checked = 0
        successful_fetches = 0
        total_duplicates_skipped = 0

        for tier_key, tier_info in tiers.items():
            sources = tier_info.get("sources", [])
            tier_default = "TIER_4"
            if "TIER_1" in tier_key:
                tier_default = "TIER_1"
            elif "TIER_2" in tier_key:
                tier_default = "TIER_2"
            elif "TIER_3" in tier_key:
                tier_default = "TIER_3"
            elif "TIER_5" in tier_key:
                tier_default = "TIER_2"

            for src in sources:
                total_sources_checked += 1
                src_id = src.get("source_id", "unknown")
                src_name = src.get("source_name", "Unknown Source")
                tier_label = src.get("tier", tier_default)
                rss_url = src.get("rss_url")
                query = src.get("google_news_query")

                # Build target URL
                target_url = rss_url
                used_fallback = False
                fallback_reason = None

                if not target_url and query:
                    encoded_q = urllib.parse.quote(query)
                    target_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"

                if not target_url:
                    source_runtime_audit.append({
                        "source_id": src_id,
                        "source_name": src_name,
                        "tier": tier_label,
                        "status": "NO_FEED_CONFIGURED",
                        "http_status": None,
                        "articles_fetched": 0,
                        "new_articles_added": 0,
                        "duplicates_skipped": 0,
                        "latest_date": None,
                        "error": "No RSS URL or Query configured"
                    })
                    continue

                # Fetch feed
                http_code, fetched, err = self.fetch_rss_feed(target_url, timeout=5.0)

                # Fallback to Google News search if direct RSS failed (403, 404, etc.)
                if (http_code != 200 or not fetched) and query and not used_fallback:
                    encoded_q = urllib.parse.quote(query)
                    fallback_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
                    logger.info(f"Direct feed failed for {src_id} ({err}). Falling back to Google News search query...")
                    fb_code, fb_fetched, fb_err = self.fetch_rss_feed(fallback_url, timeout=5.0)
                    if fb_code == 200 and fb_fetched:
                        fallback_reason = f"Direct RSS ({http_code}) failed; recovered via query fallback"
                        target_url = fallback_url
                        http_code = fb_code
                        fetched = fb_fetched
                        err = None
                        used_fallback = True

                new_for_src = 0
                dup_for_src = 0
                latest_dt = None

                for item in fetched:
                    item["source_id"] = src_id
                    item["source_name"] = src_name
                    item["tier"] = tier_label

                    art_id = item["article_id"]
                    link = item.get("link")
                    fp = item.get("headline_fingerprint")

                    # Deduplication check: ID, URL, or headline fingerprint
                    if art_id in seen_article_ids or (link and link in seen_urls) or (fp and fp in seen_fingerprints):
                        dup_for_src += 1
                        total_duplicates_skipped += 1
                        continue

                    # Fresh new article
                    seen_article_ids.add(art_id)
                    if link:
                        seen_urls.add(link)
                    if fp:
                        seen_fingerprints.add(fp)

                    all_articles.append(item)
                    new_for_src += 1

                    if not latest_dt and item.get("pub_date"):
                        latest_dt = item.get("pub_date")

                if http_code == 200 and fetched:
                    successful_fetches += 1
                    status_label = "SUCCESS"
                elif http_code == 200 and not fetched:
                    status_label = "EMPTY_FEED"
                else:
                    status_label = "FAILED"

                source_runtime_audit.append({
                    "source_id": src_id,
                    "source_name": src_name,
                    "tier": tier_label,
                    "target_url": target_url,
                    "status": status_label,
                    "http_status": http_code,
                    "articles_fetched": len(fetched),
                    "new_articles_added": new_for_src,
                    "duplicates_skipped": dup_for_src,
                    "latest_date": latest_dt,
                    "used_fallback": used_fallback,
                    "note": fallback_reason or err
                })

        new_total_added = len(all_articles) - initial_count
        newly_ingested = all_articles[initial_count:] if initial_count < len(all_articles) else []

        # Cap working buffer at latest 5,000 to maintain high performance
        all_articles = all_articles[-5000:]
        with open(ARTICLES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=2)

        # Phase 4: Permanently archive new daily articles so historical archive grows over time
        permanently_archived = self._append_to_permanent_archive(newly_ingested)

        elapsed = round(time.time() - t_start, 2)
        now_utc = datetime.now(timezone.utc).isoformat()

        audit_summary = {
            "job_name": "fetch_news",
            "timestamp": now_utc,
            "last_successful_fetch_time": now_utc,
            "execution_time_seconds": elapsed,
            "total_sources_audited": total_sources_checked,
            "sources_active": successful_fetches,
            "sources_failed": total_sources_checked - successful_fetches,
            "new_articles_ingested": new_total_added,
            "permanently_archived_articles": permanently_archived,
            "duplicates_skipped": total_duplicates_skipped,
            "total_stored_articles": len(all_articles),
            "sources_status": source_runtime_audit,
        }

        with open(SOURCE_STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(source_runtime_audit, f, indent=2)
        with open(INGESTION_AUDIT_PATH, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)
        with open(AUDIT_COPY_PATH, "w", encoding="utf-8") as f:
            json.dump(source_runtime_audit, f, indent=2)

        logger.info(
            f"News Ingestion Complete in {elapsed}s: {new_total_added} new articles, "
            f"{total_duplicates_skipped} duplicates skipped across {total_sources_checked} sources."
        )
        return audit_summary

# Module-level aliases
canonicalize_url = NewsIngestionJob.canonicalize_url
compute_headline_fingerprint = NewsIngestionJob.compute_headline_fingerprint


if __name__ == "__main__":
    job = NewsIngestionJob()
    res = job.run()
    print(json.dumps({k: v for k, v in res.items() if k != "sources_status"}, indent=2))
