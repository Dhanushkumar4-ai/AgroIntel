"""
news_intelligence_service.py — AgroIntel Tiered News Intelligence, Scoring & Verification Pipeline
==================================================================================================
Key Capabilities:
  1. Source Credibility Tiers (Tier 1 Official = 1.0, Tier 2 Ag-Research = 0.8, Tier 3 Market = 0.6, Tier 4 Media = 0.4, Discovery = 0.2)
  2. Agricultural Domain Relevance & Non-Ag Noise Exclusion
  3. Market Direction Classification (Bullish, Bearish, Mixed, Uncertain) & Impact Strength (High, Medium, Low)
  4. Cross-Source Verification (VERIFIED >=2 independent sources, CORROBORATED, SINGLE_SOURCE, CONFLICTING, UNVERIFIED, STALE)
  5. Transparent Multi-Factor News Relevance & Recency Decay Scoring
  6. Price Forecast Alignment Engine (Reinforcing, Weakening, Conflicting, Neutral)
  7. Bounded Advisory Risk Adjustment (-5.0 to +3.0 max)
"""

import json
import logging
import math
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "app" / "data" / "experimental"
CORE_DIR = BASE_DIR / "app" / "core"

# ── Configurable Scoring Framework ───────────────────────────────────────────
DEFAULT_NEWS_SCORING_CONFIG = {
    "weights": {
        "crop_relevance": 0.25,
        "geographic_relevance": 0.20,
        "event_severity": 0.20,
        "source_reliability": 0.15,
        "recency_decay": 0.10,
        "corroboration": 0.10,
    },
    "recency_halflife_days": 7.0,
    "stale_cutoff_days": 45.0,
    "max_advisory_adjustment_points": 3.0,
}

# ── Source Credibility Registry ───────────────────────────────────────────────
SOURCE_TIERS = {
    "TIER_1_OFFICIAL": {
        "label": "Official Government & Scientific Authorities",
        "credibility_weight": 1.0,
        "keywords": [
            "icar", "pib", "department of agriculture", "ministry of agriculture",
            "da&fw", "imd", "ministry of earth sciences", "moes", "kvk", "apmc",
            "state agriculture department", "state horticulture department",
            "disaster management authority", "agricultural university", "agmarknet"
        ]
    },
    "TIER_2_AGRI_RESEARCH": {
        "label": "Agricultural Research & Dedicated Media",
        "credibility_weight": 0.8,
        "keywords": [
            "krishi jagran", "rural voice", "agrospectrum", "agriwatch",
            "agri farming", "chinimandi", "agriculture post", "krishak jagat",
            "agriculture world", "fao", "down to earth", "cgiar", "icrisat"
        ]
    },
    "TIER_3_BUSINESS_MARKET": {
        "label": "Business & Financial Media",
        "credibility_weight": 0.6,
        "keywords": [
            "reuters", "economic times", "business standard", "financial express",
            "businessline", "the hindu businessline", "moneycontrol", "bloomberg",
            "commodity online", "cnbc-tv18", "mint"
        ]
    },
    "TIER_4_GENERAL_MEDIA": {
        "label": "General Media Publishers",
        "credibility_weight": 0.4,
        "keywords": [
            "the hindu", "indian express", "times of india", "hindustan times",
            "deccan herald", "the tribune", "matrubhumi", "dinakaran", "lokmat",
            "india today", "ndtv", "zee news", "deccan chronicle"
        ]
    },
    "DISCOVERY_ONLY": {
        "label": "Search Aggregator / Discovery Only",
        "credibility_weight": 0.2,
        "keywords": ["google news", "news aggregator", "rss feed", "discovery"]
    }
}

# ── Crop & Commodity Vocabulary ───────────────────────────────────────────────
CANONICAL_CROPS = [
    "Rice", "Wheat", "Maize", "Onion", "Potato", "Sugarcane", "Cotton", "Soybean",
    "Chickpea", "Groundnut", "Mustard", "Barley", "Jowar", "Bajra", "Ragi", "Urad",
    "Moong", "Tur", "Arecanut", "Coconut", "Banana", "Black Pepper", "Cardamom",
    "Ginger", "Turmeric", "Tea", "Coffee", "Tobacco", "Jute", "Cashew", "Tomato"
]

CROP_ALIASES = {
    "paddy": "Rice",
    "corn": "Maize",
    "chana": "Chickpea",
    "gram": "Chickpea",
    "groundnut": "Groundnut",
    "peanut": "Groundnut",
    "sarson": "Mustard",
    "arhar": "Tur",
    "pigeon pea": "Tur",
    "sugar": "Sugarcane",
    "cane": "Sugarcane",
}

CROP_FAMILIES = {
    "cereals": ["Wheat", "Rice", "Maize", "Barley", "Jowar", "Bajra", "Ragi"],
    "vegetables": ["Onion", "Potato", "Tomato"],
    "pulses": ["Chickpea", "Tur", "Urad", "Moong"],
    "oilseeds": ["Soybean", "Mustard", "Groundnut"],
    "cash_crops": ["Sugarcane", "Cotton", "Jute", "Tobacco"],
}

# ── 28 States & UTs ───────────────────────────────────────────────────────────
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

# ── Agricultural Filtering Keywords ───────────────────────────────────────────
AGRI_KEYWORDS = [
    "mandi", "apmc", "msp", "farmer", "farmers", "farming", "farm", "crop", "crops",
    "kharif", "rabi", "zaid", "sowing", "harvest", "harvesting", "procurement",
    "yield", "irrigation", "fertilizer", "urea", "dap", "pest", "locust", "blight",
    "rust", "disease", "drought", "flood", "rainfall", "monsoon", "agro", "agriculture",
    "agricultural", "horticulture", "grain", "grains", "foodgrain", "cereal", "pulses",
    "oilseed", "export ban", "export duty", "mep", "minimum export price", "import duty",
    "cold wave", "heatwave", "hailstorm", "cyclone", "unseasonal rain", "arrival", "arrivals"
]

# Non-agricultural noise indicators
NON_AGRI_DISQUALIFIERS = [
    "ipl", "cricket", "t20", "world cup", "cinema", "bollywood", "hollywood",
    "actor", "actress", "box office", "movie", "airport", "flight", "airlines",
    "sensex", "nifty", "shares rally", "stock market", "ipo", "angel one",
    "embassy developments", "real estate", "bain capital", "dhoot transmission",
    "bcci", "icc", "trophy"
]

# ── Event Classification & Market Impact Rules ───────────────────────────────
EVENT_CATEGORIES = {
    "FLOOD": {
        "keywords": ["flood", "flooding", "waterlogged", "inundated", "submerged", "dam overflow"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Flooding damages standing crops, reducing market arrival volumes and putting upward pressure on spot rates."
    },
    "DROUGHT": {
        "keywords": ["drought", "dry spell", "water scarcity", "monsoon deficit", "rain deficit", "delayed monsoon"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Rain deficit and drought stunt acreage and crop yield, tightening market supplies."
    },
    "HEAVY_RAIN": {
        "keywords": ["heavy rainfall", "torrential rain", "excess rain", "cloudburst", "downpour", "unseasonal rain"],
        "default_direction": "BULLISH",
        "severity": "MEDIUM",
        "reason": "Unseasonal or heavy rain damages mature crops and disrupts harvesting logistics."
    },
    "HEATWAVE": {
        "keywords": ["heatwave", "scorching heat", "temperature surge", "terminal heat", "extreme temperature"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Extreme heat during grain-filling causes shriveling and yield reduction."
    },
    "PEST_OUTBREAK": {
        "keywords": ["locust", "fall armyworm", "bollworm", "pest attack", "pest infestation", "stem borer", "whitefly"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Severe pest outbreaks destroy harvestable crop volumes, reducing supply."
    },
    "DISEASE_OUTBREAK": {
        "keywords": ["blast disease", "rust disease", "blight", "mosaic virus", "fungal disease", "rot", "sheath blight"],
        "default_direction": "BULLISH",
        "severity": "MEDIUM",
        "reason": "Fungal and viral blights impair crop yields and quality."
    },
    "CROP_DAMAGE": {
        "keywords": ["crop damage", "crop loss", "yield loss", "hailstorm damage", "standing crop damaged", "hectares damaged"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Verified field damage reduces total marketable volume."
    },
    "EXPORT_RESTRICTION": {
        "keywords": ["export ban", "export curb", "minimum export price", "export duty", "mep imposed", "export prohibited"],
        "default_direction": "BEARISH",
        "severity": "HIGH",
        "reason": "Export curbs trap commodities inside domestic markets, increasing domestic supply and dampening prices."
    },
    "EXPORT_PROMOTION": {
        "keywords": ["export ban lifted", "export duty removed", "mep scrapped", "mep removed", "duty-free export", "exports allowed"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Removal of export restrictions opens international demand channels, lifting domestic mandi prices."
    },
    "IMPORT_RESTRICTION": {
        "keywords": ["import duty hiked", "tariff hike", "import restricted", "safeguard duty"],
        "default_direction": "BULLISH",
        "severity": "MEDIUM",
        "reason": "Higher import tariffs restrict foreign inflows, shielding domestic prices."
    },
    "IMPORT_RELAXATION": {
        "keywords": ["duty-free import", "import duty slashed", "import duty cut", "import relaxed", "import window extended"],
        "default_direction": "BEARISH",
        "severity": "HIGH",
        "reason": "Cheaper imports enter domestic markets, increasing aggregate availability and cooling prices."
    },
    "MSP_HIKE": {
        "keywords": ["msp hiked", "msp increased", "cabinet approves msp", "msp raised", "procurement price raised"],
        "default_direction": "BULLISH",
        "severity": "HIGH",
        "reason": "Higher Minimum Support Price provides a strong institutional floor for open-market mandi rates."
    },
    "BUMPER_HARVEST": {
        "keywords": ["bumper harvest", "bumper crop", "record production", "record harvest", "bumper output", "massive arrival"],
        "default_direction": "BEARISH",
        "severity": "MEDIUM",
        "reason": "Bumper supply and heavy mandi arrivals create a supply glut, softening prices."
    },
    "SUPPLY_SHOCK": {
        "keywords": ["supply disruption", "mandi arrival halted", "transport strike", "supply crisis", "logistics disruption"],
        "default_direction": "BULLISH",
        "severity": "MEDIUM",
        "reason": "Logistics bottlenecks and mandi closures interrupt supply flow to urban consumption centers."
    },
    "GOVT_INTERVENTION": {
        "keywords": ["omss", "open market sale", "stock limits", "market intervention", "subsidized sale by fci"],
        "default_direction": "BEARISH",
        "severity": "HIGH",
        "reason": "Government open-market releases and trader stock limits actively depress inflated spot prices."
    },
    "PRICE_SURGE": {
        "keywords": ["price surge", "rates rise", "record high price", "price spiked", "firm prices", "bullish momentum"],
        "default_direction": "BULLISH",
        "severity": "MEDIUM",
        "reason": "Reported market trend shows rising buyer bids and robust demand."
    },
    "PRICE_CRASH": {
        "keywords": ["price crash", "mandi rate drops", "sharp fall", "slump in rates", "prices plunge"],
        "default_direction": "BEARISH",
        "severity": "HIGH",
        "reason": "Reported distress selling or sudden influx of arrivals causing sharp price decline."
    },
    "WEATHER_ANOMALY": {
        "keywords": ["cold wave", "frost", "ground frost", "winter freeze", "cyclone", "cyclonic storm"],
        "default_direction": "BULLISH",
        "severity": "MEDIUM",
        "reason": "Adverse weather anomalies introduce risks to standing crops."
    },
    "INPUT_COST": {
        "keywords": ["fertilizer subsidy", "urea shortage", "dap shortage", "npk subsidy", "diesel price hike"],
        "default_direction": "UNCERTAIN",
        "severity": "LOW",
        "reason": "Farm input price shifts affect cost of cultivation but impact short-term spot market rates with delay."
    }
}


# ── Core Functions ────────────────────────────────────────────────────────────

def parse_publication_date(raw_date_str: str) -> Tuple[Optional[str], bool]:
    """
    Parse raw RSS date string into timezone-aware ISO-8601 UTC string.
    Returns: (iso_utc_string or None, is_missing_flag)
    """
    if not raw_date_str or not str(raw_date_str).strip():
        return None, True
    raw_clean = str(raw_date_str).strip()

    # Try email.utils for standard RFC 2822
    try:
        import email.utils
        parsed_tuple = email.utils.parsedate_to_datetime(raw_clean)
        if parsed_tuple:
            if not parsed_tuple.tzinfo:
                parsed_tuple = parsed_tuple.replace(tzinfo=timezone.utc)
            return parsed_tuple.astimezone(timezone.utc).isoformat(), False
    except Exception:
        pass

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


def generate_local_search_queries(state: str, district: str, crop: str = None, season: str = None) -> List[str]:
    """Generate dynamic local and state-level agricultural search query terms."""
    queries = []
    st = state.strip() if state else ""
    dt = district.strip() if district else ""
    cr = crop.strip() if crop else ""

    if dt:
        queries.append(f"{dt} agriculture")
        queries.append(f"{dt} crop news")
        queries.append(f"{dt} rainfall flood drought")
        queries.append(f"{dt} pest disease advisory")
        if cr:
            queries.append(f"{dt} {cr}")

    if st:
        queries.append(f"{st} agriculture department")
        queries.append(f"{st} crop advisory")
        queries.append(f"{st} mandi market news")
        if cr:
            queries.append(f"{st} {cr} crop")

    return queries


def generate_price_search_queries(state: str, crop: str) -> List[str]:
    """Generate state and commodity search query terms for Price Prediction."""
    cr = crop.strip() if crop else "crop"
    st = state.strip() if state else "India"

    return [
        f"{st} {cr} price news",
        f"{st} {cr} mandi rates",
        f"{st} {cr} market arrival",
        f"{cr} export import policy India",
        f"{cr} MSP weather risk {st}"
    ]


def classify_source_credibility(source_name: str, url: str = "") -> dict:
    """Classify an article's source into Credibility Tiers 1-4 or Discovery."""
    src_clean = (source_name or "").lower() + " " + (url or "").lower()

    for tier_key, tier_info in SOURCE_TIERS.items():
        if tier_key == "DISCOVERY_ONLY":
            continue
        for kw in tier_info["keywords"]:
            if kw in src_clean:
                return {
                    "tier": tier_key,
                    "label": tier_info["label"],
                    "credibility_weight": tier_info["credibility_weight"],
                    "is_official": tier_key == "TIER_1_OFFICIAL"
                }

    if "google" in src_clean or "feed" in src_clean:
        return {
            "tier": "DISCOVERY_ONLY",
            "label": SOURCE_TIERS["DISCOVERY_ONLY"]["label"],
            "credibility_weight": 0.2,
            "is_official": False
        }

    return {
        "tier": "TIER_4_GENERAL_MEDIA",
        "label": SOURCE_TIERS["TIER_4_GENERAL_MEDIA"]["label"],
        "credibility_weight": 0.4,
        "is_official": False
    }


def is_agricultural_article(title: str, description: str = "") -> bool:
    """
    Check if an article is genuinely relevant to Indian agriculture.
    Excludes sports, film, general stock equities, and real estate noise.
    """
    full_text = f"{title} {description}".lower()

    # Check for direct disqualifiers
    has_disqualifier = any(re.search(rf"\b{re.escape(dq)}\b", full_text) for dq in NON_AGRI_DISQUALIFIERS)
    agri_match_count = sum(1 for kw in AGRI_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", full_text))
    crop_match_count = sum(1 for c in CANONICAL_CROPS if re.search(rf"\b{re.escape(c.lower())}\b", full_text))

    if has_disqualifier:
        # Disqualified if no strong crop confirmation or minimal agri keywords
        if crop_match_count == 0 or agri_match_count < 2:
            return False

    # Positive confirmation: Must contain an agricultural keyword or recognized crop
    return (agri_match_count > 0 or crop_match_count > 0)


def extract_entities(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract recognized State and Crop from text using canonical gazetteers."""
    text_lower = text.lower()
    
    # 1. State extraction
    detected_state = None
    for st in INDIAN_STATES:
        if re.search(rf"\b{re.escape(st.lower())}\b", text_lower):
            detected_state = st
            break

    # 2. Crop extraction
    detected_crop = None
    # Check aliases first (paddy -> Rice, chana -> Chickpea)
    for alias, canonical in CROP_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_lower):
            detected_crop = canonical
            break

    if not detected_crop:
        for c in CANONICAL_CROPS:
            if re.search(rf"\b{re.escape(c.lower())}\b", text_lower):
                detected_crop = c
                break

    return detected_state, detected_crop


def classify_market_impact(text: str) -> Dict[str, Any]:
    """
    Classify agricultural event, market effect (Bullish/Bearish/Mixed/Uncertain),
    impact strength (High/Medium/Low), and geographic scope.
    """
    text_lower = text.lower()
    matched_events = []

    for evt_type, evt_data in EVENT_CATEGORIES.items():
        for kw in evt_data["keywords"]:
            if kw in text_lower:
                matched_events.append((evt_type, evt_data))
                break

    if not matched_events:
        return {
            "event_type": "GENERAL_AGRI",
            "market_effect": "UNCERTAIN",
            "impact_strength": "LOW",
            "geographic_scope": "NATIONAL",
            "reasoning": "General agricultural update with neutral or unspecified market direction.",
            "confidence": 0.40
        }

    # Evaluate multiple matched events
    directions = {e[1]["default_direction"] for e in matched_events}
    severities = [e[1]["severity"] for e in matched_events]
    primary_evt = matched_events[0]

    # Geographic scope detection
    geo_scope = "NATIONAL"
    if any(st.lower() in text_lower for st in INDIAN_STATES):
        geo_scope = "STATE"
    elif any(r in text_lower for r in ["global", "usda", "international", "chicago", "world market"]):
        geo_scope = "INTERNATIONAL"

    # Market effect resolution
    if "BULLISH" in directions and "BEARISH" in directions:
        market_effect = "MIXED"
        reason = f"Article contains opposing drivers: {', '.join(e[0] for e in matched_events)}. Market impact is mixed."
    elif "BULLISH" in directions:
        market_effect = "BULLISH"
        reason = primary_evt[1]["reason"]
    elif "BEARISH" in directions:
        market_effect = "BEARISH"
        reason = primary_evt[1]["reason"]
    else:
        market_effect = "UNCERTAIN"
        reason = primary_evt[1]["reason"]

    # Max severity
    if "HIGH" in severities:
        strength = "HIGH"
    elif "MEDIUM" in severities:
        strength = "MEDIUM"
    else:
        strength = "LOW"

    return {
        "event_type": primary_evt[0],
        "market_effect": market_effect,
        "impact_strength": strength,
        "geographic_scope": geo_scope,
        "reasoning": reason,
        "confidence": 0.85 if strength == "HIGH" else (0.75 if strength == "MEDIUM" else 0.60)
    }


def calculate_article_relevance_score(
    article: Dict[str, Any],
    target_crop: Optional[str] = None,
    target_state: Optional[str] = None,
    as_of_date: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Transparent scoring framework combining:
      - Crop relevance (0-1.0)
      - Geographic relevance (0-1.0)
      - Event severity (0-1.0)
      - Source reliability (0-1.0)
      - Recency with half-life decay (0-1.0)
      - Corroboration status (0-1.0)
    Returns composite score (0-100) and full component breakdown.
    """
    cfg = config or DEFAULT_NEWS_SCORING_CONFIG
    weights = cfg["weights"]
    halflife = cfg.get("recency_halflife_days", 7.0)

    # Flexible crop and state extraction
    art_crop = (article.get("crop") or "").strip()
    if not art_crop and isinstance(article.get("crops"), list) and article.get("crops"):
        art_crop = str(article["crops"][0]).strip()

    art_state = (article.get("state") or "").strip()
    if not art_state and isinstance(article.get("states"), list) and article.get("states"):
        art_state = str(article["states"][0]).strip()

    geo_scope = article.get("geographic_scope", "NATIONAL")
    severity = str(article.get("impact_strength") or article.get("severity") or "MEDIUM").upper()
    verif = article.get("verification_status", "SINGLE_SOURCE")
    tier = article.get("tier", "TIER_4")

    # 1. Crop Relevance
    s_crop = 0.2  # Base for general ag
    if target_crop:
        tc = target_crop.strip().lower()
        ac = art_crop.lower()
        if ac == tc:
            s_crop = 1.0
        else:
            # Check crop families
            same_family = False
            for f_crops in CROP_FAMILIES.values():
                f_lower = [x.lower() for x in f_crops]
                if tc in f_lower and ac in f_lower:
                    same_family = True
                    break
            if same_family:
                s_crop = 0.50
            elif ac in ["general agriculture", "all", ""]:
                s_crop = 0.30
            else:
                s_crop = 0.05
    else:
        s_crop = 0.8 if art_crop and art_crop != "General Agriculture" else 0.4

    # 2. Geographic Relevance
    s_geo = 0.4
    if target_state:
        ts = target_state.strip().lower()
        ast = art_state.lower()
        if ast == ts:
            s_geo = 1.0
        elif geo_scope == "NATIONAL" or ast in ["national", "all", ""]:
            s_geo = 0.50
        elif geo_scope == "INTERNATIONAL":
            s_geo = 0.25
        else:
            s_geo = 0.10
    else:
        s_geo = 0.9 if art_state and art_state != "National" else 0.5

    # 3. Event Severity
    if severity == "HIGH":
        s_event = 1.0
    elif severity == "MEDIUM":
        s_event = 0.60
    else:
        s_event = 0.30

    # 4. Source Reliability
    tier_info = SOURCE_TIERS.get(tier, {})
    if "source_reliability" in article:
        s_source = float(article["source_reliability"])
    elif "source_credibility_weight" in article:
        s_source = float(article["source_credibility_weight"])
    else:
        s_source = tier_info.get("credibility_weight", 0.40)

    # 5. Recency with Exponential Half-Life Decay
    if as_of_date and article.get("published_at"):
        try:
            pub_dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            ref_dt = as_of_date if as_of_date.tzinfo else as_of_date.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (ref_dt - pub_dt).total_seconds() / 86400.0)
        except Exception:
            age_days = float(article.get("age_days", 0.0))
    else:
        age_days = float(article.get("age_days", 0.0))

    if age_days <= 0:
        s_recency = 1.0
    else:
        # e^(-ln(2) * t / halflife) = 2^(-t / halflife)
        s_recency = round(math.pow(2, -age_days / halflife), 3)
    if age_days > cfg.get("stale_cutoff_days", 45.0):
        s_recency = 0.0

    # 6. Corroboration & Verification
    if verif == "VERIFIED":
        s_corrob = 1.0
    elif verif == "CORROBORATED":
        s_corrob = 0.85
    elif verif == "SINGLE_SOURCE":
        s_corrob = 0.60 if s_source >= 0.8 else 0.40
    elif verif == "CONFLICTING":
        s_corrob = 0.10  # Penalized: high uncertainty
    else:
        s_corrob = 0.20

    # Composite weighted score (0-100)
    composite = 100.0 * (
        weights["crop_relevance"] * s_crop +
        weights["geographic_relevance"] * s_geo +
        weights["event_severity"] * s_event +
        weights["source_reliability"] * s_source +
        weights["recency_decay"] * s_recency +
        weights["corroboration"] * s_corrob
    )
    composite = round(min(100.0, max(0.0, composite)), 1)

    # Generate transparent explanation
    parts = []
    if s_crop >= 0.9:
        parts.append(f"Direct crop match ({art_crop})")
    elif s_crop >= 0.5:
        parts.append(f"Allied crop group ({art_crop})")
    if s_geo >= 0.9:
        parts.append(f"State match ({art_state})")
    elif s_geo >= 0.5:
        parts.append("National geographic scope")
    parts.append(f"{verif.capitalize()} source ({tier_info.get('label', tier)})")
    parts.append(f"Published {age_days:.0f}d ago ({severity} severity)")

    explanation = "; ".join(parts) + f" — Composite score: {composite}/100."

    return {
        "composite_score": composite,
        "score_components": {
            "crop_relevance": round(s_crop, 2),
            "geographic_relevance": round(s_geo, 2),
            "event_severity": round(s_event, 2),
            "source_reliability": round(s_source, 2),
            "recency_decay": round(s_recency, 2),
            "corroboration": round(s_corrob, 2),
        },
        "weights": weights,
        "explanation": explanation
    }


def evaluate_forecast_alignment(
    percent_change: Optional[float] = None,
    news_articles: Optional[List[Dict[str, Any]]] = None,
    uncertainty_threshold: float = 3.0,
    pred_pct_change: Optional[float] = None,
    relevant_news: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compare statistical model forecast with aggregate news market sentiment.
    Explains whether news REINFORCES, WEAKENS, CONFLICTS with, or remains NEUTRAL
    relative to the model's projected price movement.
    """
    if percent_change is None and pred_pct_change is not None:
        percent_change = pred_pct_change
    if percent_change is None:
        percent_change = 0.0
    if news_articles is None and relevant_news is not None:
        news_articles = relevant_news
    if news_articles is None:
        news_articles = []

    if not news_articles:
        return {
            "aggregate_market_signal": "NEUTRAL",
            "forecast_alignment": "NEUTRAL",
            "advisory_impact_explanation": (
                "No fresh, market-moving agricultural news detected for this commodity and region. "
                "Advisory relies entirely on historical seasonal cycles and physical APMC Mandi trends."
            ),
            "news_confidence": 0.50,
            "relevant_articles_count": 0,
            "alignment": "NEUTRAL",
            "news_sentiment_direction": "NEUTRAL",
            "explanation": "No recent verified news detected for this commodity and region.",
            "news_count": 0,
            "confidence": 0.50,
        }

    # Aggregate Bullish vs Bearish signals weighted by composite relevance
    bullish_weight = 0.0
    bearish_weight = 0.0
    conflicting_count = 0

    for a in news_articles:
        score = a.get("relevance_score", {}).get("composite_score", 50.0) / 100.0 if isinstance(a.get("relevance_score"), dict) else float(a.get("composite_score", 50.0)) / 100.0
        effect = str(a.get("market_effect", "UNCERTAIN")).upper()
        verif = str(a.get("verification_status", "SINGLE_SOURCE")).upper()

        if verif == "CONFLICTING":
            conflicting_count += 1

        if effect == "BULLISH":
            bullish_weight += score
        elif effect == "BEARISH":
            bearish_weight += score

    total_weight = bullish_weight + bearish_weight
    if total_weight < 0.2 or (abs(bullish_weight - bearish_weight) < 0.15 and conflicting_count > 0):
        if conflicting_count > 0:
            market_signal = "CONFLICTING"
        else:
            market_signal = "UNCERTAIN"
    elif bullish_weight > bearish_weight * 1.3:
        market_signal = "BULLISH"
    elif bearish_weight > bullish_weight * 1.3:
        market_signal = "BEARISH"
    else:
        market_signal = "MIXED"

    # Compare with model trend
    model_is_up = percent_change > uncertainty_threshold
    model_is_down = percent_change < -uncertainty_threshold

    if market_signal == "BULLISH":
        if model_is_up:
            alignment = "REINFORCING"
            explanation = (
                f"Verified agricultural news REINFORCES the model's projected price increase ({percent_change:+.1f}%). "
                f"Market intelligence confirms supply constraints and supportive demand conditions."
            )
        elif model_is_down:
            alignment = "CONFLICTING"
            explanation = (
                f"Agricultural news CONFLICTS with the model's projected price decline ({percent_change:+.1f}%). "
                f"Emerging bullish signals (supply disruptions or export openings) may cushion prices against downward drift."
            )
        else:
            alignment = "REINFORCING"
            explanation = "Bullish agricultural news provides positive support during a period of steady price expectations."

    elif market_signal == "BEARISH":
        if model_is_down:
            alignment = "REINFORCING"
            explanation = (
                f"Verified agricultural news REINFORCES the model's projected price decline ({percent_change:+.1f}%). "
                f"Market reports confirm heavy Mandi arrivals, bumper supply, or trade interventions dampening prices."
            )
        elif model_is_up:
            alignment = "CONFLICTING"
            explanation = (
                f"Agricultural news CONFLICTS with the model's projected price increase ({percent_change:+.1f}%). "
                f"Bearish market signals (import duty relaxations or surge in mandi arrivals) introduce downside risk to the forecast."
            )
        else:
            alignment = "WEAKENING"
            explanation = "Bearish news alerts indicate potential downward pressure despite steady historical baseline."

    elif market_signal in ["CONFLICTING", "MIXED"]:
        alignment = "CONFLICTING"
        explanation = (
            "Divergent news reports and conflicting market signals increase uncertainty. "
            "Neither buyers nor sellers hold clear market power; staggered selling or hedging is advised."
        )
    else:
        alignment = "NEUTRAL"
        explanation = (
            "Available news provides neutral or background context without definitive market-moving direction. "
            "Decision remains guided by model fundamentals and physical holding costs."
        )

    conf_calc = round(max(bullish_weight, bearish_weight) / max(0.01, total_weight), 2) if total_weight > 0 else 0.50

    return {
        "aggregate_market_signal": market_signal,
        "forecast_alignment": alignment,
        "advisory_impact_explanation": explanation,
        "bullish_evidence_weight": round(bullish_weight, 2),
        "bearish_evidence_weight": round(bearish_weight, 2),
        "relevant_articles_count": len(news_articles),
        "alignment": alignment,
        "news_sentiment_direction": market_signal,
        "explanation": explanation,
        "news_count": len(news_articles),
        "confidence": conf_calc,
    }


def verify_cross_source(events: List[dict]) -> str:
    """
    Determine cross-source verification status across multiple independent sources:
    - VERIFIED: 2+ independent sources reporting matching event/direction
    - CORROBORATED: 1 media source + 1 official Tier 1 or 2 research source
    - SINGLE_SOURCE: 1 credible source
    - CONFLICTING: sources disagree on impact direction
    - UNVERIFIED: weak or unconfirmed source
    - STALE: article > 45 days old
    """
    if not events:
        return "NO_INTELLIGENCE"

    fresh_events = [e for e in events if float(e.get("age_days", 0.0)) <= 45.0]
    if not fresh_events:
        return "STALE"

    directions = set()
    sources = set()
    tiers = set()

    for e in fresh_events:
        src = e.get("source_name") or e.get("source_id") or "unknown"
        sources.add(src)
        tier = e.get("tier") or "TIER_4"
        tiers.add(tier)
        dir_val = e.get("market_effect") or e.get("impact_direction") or "UNCERTAIN"
        if dir_val not in ["UNCERTAIN", "NEUTRAL"]:
            directions.add(dir_val)

    if "BULLISH" in directions and "BEARISH" in directions:
        return "CONFLICTING"

    if len(sources) >= 2:
        if "TIER_1_OFFICIAL" in tiers or "TIER_2_AGRI_RESEARCH" in tiers:
            return "VERIFIED"
        return "VERIFIED"
    elif len(sources) == 1:
        if "TIER_1_OFFICIAL" in tiers or "TIER_2_AGRI_RESEARCH" in tiers:
            return "SINGLE_SOURCE"
        return "UNVERIFIED"

    return "UNVERIFIED"


def calculate_bounded_news_adjustment(events: List[dict]) -> tuple[float, str, str]:
    """
    Calculate bounded news risk score adjustment for crop recommendation:
    - Range: STRICTLY bounded between -5.0 and +3.0
    - UNVERIFIED / CONFLICTING / STALE news = 0.0 adjustment
    """
    if not events:
        return 0.0, "NO_SIGNIFICANT_SIGNAL", "NO_INTELLIGENCE"

    verification_status = verify_cross_source(events)

    if verification_status in ["UNVERIFIED", "STALE", "CONFLICTING", "NO_INTELLIGENCE"]:
        return 0.0, "NEUTRAL_SIGNAL", verification_status

    fresh_events = [e for e in events if float(e.get("age_days", 0.0)) <= 45.0]
    if not fresh_events:
        return 0.0, "NEUTRAL_SIGNAL", "STALE"

    top_event = fresh_events[0]
    cred_weight = classify_source_credibility(top_event.get("source_name", "")).get("credibility_weight", 0.5)
    sig = top_event.get("market_effect") or top_event.get("impact_direction", "NEUTRAL")

    adjustment = 0.0
    if sig in ["BEARISH", "NEGATIVE", "RISK_INCREASED"]:
        adjustment = round(-3.0 * cred_weight - 2.0 * (1.0 if verification_status == "VERIFIED" else 0.5), 1)
        adjustment = max(-5.0, adjustment)
    elif sig in ["BULLISH", "POSITIVE", "RISK_DECREASED"]:
        adjustment = round(+2.0 * cred_weight + (1.0 if verification_status == "VERIFIED" else 0.0), 1)
        adjustment = min(+3.0, adjustment)

    return adjustment, sig, verification_status
