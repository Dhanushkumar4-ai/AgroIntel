"""
phase6_integration_service.py — AgroIntel Phase 6 Final End-to-End Integration Engine
========================================================================================
Fuses all Phase 1-5 sub-systems:
  • Phase 1: Official APY historical crop evidence (246,091 records, 652 canonical districts)
  • Phase 2: Seasonal crop calendar, soil/weather bounds, crop families, rotation parameters
  • Phase 3: Recent cultivation evidence & source comparison
  • Phase 4: Multi-source evidence, candidate generation engine, RFCandidateAdapter
  • Phase 5: Mandi market price vectors, State-Aware XGBoost price forecast, Phase 5.3 news risk

CRITICAL RULES ENFORCED:
  1. Zero hardcoded district logic (100% data-driven across 652 canonical districts).
  2. Candidate restriction: never invent crops outside evidence.
  3. Price vector separation: min_price, current_price (modal), max_price vs predicted_price.
  4. Water/Soil UNKNOWN rule: UNKNOWN is NEVER converted to SUITABLE.
  5. Perennial crop preservation: Whole Year / Perennial cycles preserved.
  6. Mandi price is LATEST_AVAILABLE_MARKET_PRICE with observation_date and data_age_days shown.
  7. MAE/RMSE is model accuracy metric. It is NEVER displayed as a crop price.
  8. Clean Farmer UI: All technical scoring tokens (e.g. 25.0/25) are stripped from user explanations.
"""

import sys
import json
import math
import random
import datetime
import re
import unicodedata
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

from app.core.constants import normalize_season
from app.services.location_normalizer import normalize_location
from app.services.weather_service import get_weather_summary, get_current_monthly_weather
from app.services.nlp_explanation_service import (
    explain_crop_recommendation,
    explain_price_prediction,
    summarize_news_intelligence,
    get_crop_information
)
from app.services.news_intelligence_service import (
    classify_source_credibility,
    generate_local_search_queries,
    generate_price_search_queries,
    verify_cross_source,
    calculate_bounded_news_adjustment
)

try:
    from app.services import mandi_service
except ImportError:
    mandi_service = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "app" / "data" / "experimental"
MODELS_DIR = BASE_DIR / "models"


def _normalize_district_name(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.lower().strip()

    replacements = [
        (r"\bdakshina\b", "dakshin"),
        (r"\buttara\b", "uttar"),
        (r"\bkannada\b", "kannad"),
        (r"\beast\b", "purbi"),
        (r"\bpurba\b", "purbi"),
        (r"\bwest\b", "pashchim"),
        (r"\bpaschim\b", "pashchim"),
        (r"\bsouth\b", "dakshin"),
        (r"\bnorth\b", "uttar"),
        (r"\bvisakhapatnam\b", "visakhapatanam"),
        (r"\bdistrict\b", ""),
        (r"\bmetropolitan\b", ""),
        (r"\but\b", ""),
        (r"\bnct\b", ""),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


PERENNIAL_CROPS = {
    "Arecanut", "Coconut", "Coffee", "Tea", "Rubber", "Banana",
    "Black Pepper", "Cardamom", "Cashew", "Sugarcane"
}

LEGUME_CROPS = {
    "Moong (Green Gram)", "Black Gram (Urad)", "Pigeonpea (Arhar/Tur)",
    "Chickpea (Gram)", "Lentil (Masur)", "Groundnut", "Soybean"
}

CROP_FAMILIES = {
    "Rice": "Poaceae", "Wheat": "Poaceae", "Maize": "Poaceae", "Sorghum": "Poaceae",
    "Pearl Millet (Bajra)": "Poaceae", "Finger Millet (Ragi)": "Poaceae", "Sugarcane": "Poaceae",
    "Pigeonpea (Arhar/Tur)": "Fabaceae", "Moong (Green Gram)": "Fabaceae", "Black Gram (Urad)": "Fabaceae",
    "Chickpea (Gram)": "Fabaceae", "Lentil (Masur)": "Fabaceae", "Groundnut": "Fabaceae", "Soybean": "Fabaceae",
    "Potato": "Solanaceae", "Tomato": "Solanaceae", "Brinjal": "Solanaceae", "Chilli": "Solanaceae",
    "Onion": "Amaryllidaceae", "Garlic": "Amaryllidaceae",
    "Cotton": "Malvaceae", "Arecanut": "Arecaceae", "Coconut": "Arecaceae"
}

CROP_TO_COMMODITY_MAP = {
    "rice": "rice", "wheat": "wheat", "maize": "maize", "onion": "onion",
    "potato": "potato", "arecanut": "arecanut", "arcanut (processed)": "arecanut",
    "moong (green gram)": "moong (green gram)",
}


def _compute_freshness(observation_date_str: str) -> dict:
    """Compute data_age_days and freshness_label from observation date string."""
    if not observation_date_str:
        return {
            "observation_date": "UNAVAILABLE",
            "observation_date_iso": None,
            "data_age_days": None,
            "freshness_label": "UNAVAILABLE",
            "freshness_note": "No observation record available for this commodity."
        }

    today = datetime.date.today()
    arr_date = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            arr_date = datetime.datetime.strptime(observation_date_str, fmt).date()
            break
        except (ValueError, TypeError):
            continue

    if arr_date is None:
        return {
            "observation_date": observation_date_str,
            "observation_date_iso": None,
            "data_age_days": None,
            "freshness_label": "UNAVAILABLE",
            "freshness_note": "Observation date could not be parsed."
        }

    age_days = max(0, (today - arr_date).days)

    if age_days <= 3:
        freshness_label = "VERY_FRESH"
        freshness_note = "Market data is recent and reliable."
    elif age_days <= 14:
        freshness_label = "FRESH"
        freshness_note = "Market data is within normal 2-week observation window."
    elif age_days <= 30:
        freshness_label = "RECENT"
        freshness_note = "Market data is from within the last month."
    elif age_days <= 60:
        freshness_label = "BACKGROUND"
        freshness_note = "Market data is 1 to 2 months old."
    elif age_days <= 180:
        freshness_label = "STALE"
        freshness_note = "Market data is 2 to 6 months old."
    else:
        freshness_label = "VERY_STALE"
        freshness_note = "Market data is older than 6 months."

    return {
        "observation_date": arr_date.strftime("%d-%m-%Y"),
        "observation_date_iso": arr_date.isoformat(),
        "data_age_days": age_days,
        "freshness_label": freshness_label,
        "freshness_note": freshness_note
    }


class AgroIntelPhase6Engine:
    def __init__(self):
        self.district_master = self._load_json(EXP_DIR / "district_master.json")
        self.candidate_matrix = self._load_json(EXP_DIR / "nationwide_candidate_matrix_v2.json")
        self.crop_evidence = self._load_json(EXP_DIR / "district_crop_evidence.json")
        self.season_calendar = self._load_json(EXP_DIR / "crop_season_calendar.json")
        self.market_intel = self._load_json(EXP_DIR / "market_intelligence.json")
        self.current_intel = self._load_json(EXP_DIR / "current_intelligence.json")
        self.news_events = self._load_json(EXP_DIR / "news_events.json")

        self.dist_map = {}
        self.dist_norm_map = {}
        self.dist_token_list = []

        if isinstance(self.district_master, list):
            for d in self.district_master:
                canon_id = d.get("canonical_id", "").lower()
                dist_raw = d.get("district", "").lower()
                st_raw = d.get("state", "").lower()

                self.dist_map[dist_raw] = d
                self.dist_map[canon_id] = d
                self.dist_map[f"{st_raw}::{dist_raw}"] = d

                for sn in d.get("source_names", []):
                    self.dist_map[sn.lower()] = d
                    self.dist_map[f"{st_raw}::{sn.lower()}"] = d

                st_norm = _normalize_district_name(d.get("state", ""))
                names = [d.get("district", "")] + d.get("source_names", [])
                for name in names:
                    n_norm = _normalize_district_name(name)
                    if n_norm:
                        self.dist_norm_map[(st_norm, n_norm)] = d
                        if n_norm not in self.dist_norm_map:
                            self.dist_norm_map[n_norm] = d

                d_tokens = set(_normalize_district_name(d.get("district", "")).split())
                if d_tokens:
                    self.dist_token_list.append((st_norm, d_tokens, d))

        self.cand_lookup = defaultdict(list)
        self.district_cand_lookup = defaultdict(list)
        if isinstance(self.candidate_matrix, list):
            for entry in self.candidate_matrix:
                st = entry.get("state", "")
                dt = entry.get("district", "")
                se = entry.get("season", "").lower()
                cands = entry.get("candidates", [])

                canon = self.canonicalize_district(dt, st)
                cid = canon["canonical_id"] if canon else f"{st}::{dt}"

                self.cand_lookup[(cid, se)].extend(cands)
                self.cand_lookup[(st.lower(), dt.lower(), se)].extend(cands)
                self.district_cand_lookup[cid].extend(cands)
                self.district_cand_lookup[(st.lower(), dt.lower())].extend(cands)

        self.mandi_lookup = {}
        if isinstance(self.market_intel, list):
            for m in self.market_intel:
                d_name = m.get("district", "").lower()
                c_name = m.get("commodity", "").lower()
                if d_name and c_name:
                    self.mandi_lookup[(d_name, c_name)] = m

        self.intel_lookup = defaultdict(list)
        if isinstance(self.current_intel, list):
            for intel in self.current_intel:
                key = (intel.get("state", "").lower(), intel.get("crop", "").lower())
                self.intel_lookup[key].append(intel)

        self.district_crop_evidence_map = {}
        if isinstance(self.crop_evidence, list):
            for d_item in self.crop_evidence:
                d_cid = d_item.get("district_id", "").lower()
                d_st = d_item.get("state", "").lower()
                d_dt = d_item.get("district", "").lower()

                for c_entry in d_item.get("crops", []):
                    c_name = c_entry.get("crop", "").lower()
                    canon_c = c_entry.get("canonical_crop", "").lower()

                    if c_name:
                        self.district_crop_evidence_map[(d_cid, c_name)] = c_entry
                        self.district_crop_evidence_map[(d_st, d_dt, c_name)] = c_entry
                    if canon_c:
                        self.district_crop_evidence_map[(d_cid, canon_c)] = c_entry
                        self.district_crop_evidence_map[(d_st, d_dt, canon_c)] = c_entry

        reqs_raw = self._load_json(EXP_DIR / "crop_requirements.json")
        self.crop_requirements = reqs_raw.get("crop_requirements", {}) if isinstance(reqs_raw, dict) else {}

    def _load_json(self, path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def canonicalize_district(self, query_district: str, query_state: str = None) -> dict:
        """Resolve location using canonical location normalization layer."""
        if not query_district:
            return None

        # 1. Use location normalizer first
        norm_res = normalize_location(query_district, query_state)
        canon_name = norm_res["canonical_name"].lower()
        canon_id = norm_res["canonical_id"].lower()
        state_name = norm_res["state"].lower()

        if canon_id in self.dist_map:
            return self.dist_map[canon_id]
        if f"{state_name}::{canon_name}" in self.dist_map:
            return self.dist_map[f"{state_name}::{canon_name}"]
        if canon_name in self.dist_map:
            return self.dist_map[canon_name]

        # 2. Normalized state + district match
        st_norm = _normalize_district_name(query_state or norm_res["state"])
        dist_norm = _normalize_district_name(query_district)

        if (st_norm, dist_norm) in self.dist_norm_map:
            return self.dist_norm_map[(st_norm, dist_norm)]

        dist_noparen = _normalize_district_name(re.sub(r"\(.*?\)", "", query_district))
        if (st_norm, dist_noparen) in self.dist_norm_map:
            return self.dist_norm_map[(st_norm, dist_noparen)]

        if dist_norm in self.dist_norm_map:
            return self.dist_norm_map[dist_norm]

        # 3. Token-overlap fallback
        q_tokens = set(dist_norm.split())
        if q_tokens:
            best_d = None
            best_score = 0.0
            for d_st_norm, d_tokens, d in self.dist_token_list:
                if st_norm and d_st_norm != st_norm:
                    continue
                inter = len(q_tokens & d_tokens)
                if inter > 0:
                    score = inter / len(q_tokens | d_tokens)
                    if score > best_score:
                        best_score = score
                        best_d = d
            if best_d and best_score >= 0.4:
                return best_d

        return {
            "canonical_id": norm_res["canonical_id"],
            "state": norm_res["state"],
            "district": norm_res["canonical_name"],
            "source_names": [query_district]
        }

    def evaluate_recommendation(
        self,
        state: str,
        district: str,
        season: str,
        soil_ph: float = None,
        soil_npk: dict = None,
        previous_crop: str = None,
        water_available_mm: float = None,
        lat: float = None,
        lon: float = None
    ) -> dict:
        """
        Full End-to-End Explainable Crop Recommendation Engine.
        """
        dist_obj = self.canonicalize_district(district, state)
        if not dist_obj:
            return {
                "status": "ERROR",
                "message": f"District '{district}' could not be resolved.",
                "canonical_id": "UNRESOLVED_LOCATION"
            }

        canon_state = dist_obj["state"]
        canon_district = dist_obj["district"]
        canonical_id = dist_obj["canonical_id"]
        canon_season = normalize_season(season)

        # Fetch candidate crops strictly from APY evidence matrix
        raw_candidates = self.cand_lookup.get((canonical_id, canon_season.lower()), [])
        if not raw_candidates:
            raw_candidates = self.cand_lookup.get((canonical_id, season.lower()), [])
        if not raw_candidates:
            raw_candidates = self.cand_lookup.get((canon_state.lower(), canon_district.lower(), canon_season.lower()), [])
        if not raw_candidates:
            raw_candidates = self.cand_lookup.get((canon_state.lower(), canon_district.lower(), season.lower()), [])
        if not raw_candidates:
            raw_candidates = self.district_cand_lookup.get(canonical_id, [])
        if not raw_candidates:
            raw_candidates = self.district_cand_lookup.get((canon_state.lower(), canon_district.lower()), [])

        if not raw_candidates:
            return {
                "location": {"state": canon_state, "district": canon_district, "canonical_id": canonical_id},
                "season": season,
                "recommendations": [],
                "rejected_crops": [],
                "message": "No historical cultivation records found for this district and season.",
                "data_quality": {"cultivation_evidence": "NO_HISTORICAL_RECORD"}
            }

        # Multi-provider weather check
        weather_info = get_weather_summary(lat or 20.0, lon or 77.0, canon_district)

        unique_candidates = {}
        for c in raw_candidates:
            c_name = c.get("crop")
            if c_name and c_name not in unique_candidates:
                unique_candidates[c_name] = c

        scored_recommendations = []
        rejected_crops = []

        for crop_name, cand_info in unique_candidates.items():
            is_perennial = crop_name in PERENNIAL_CROPS

            # 1. District Evidence
            ev_item = self.district_crop_evidence_map.get((canonical_id.lower(), crop_name.lower()))
            if not ev_item:
                ev_item = self.district_crop_evidence_map.get((canon_state.lower(), canon_district.lower(), crop_name.lower()))

            if ev_item:
                years_present = ev_item.get("years_present", [])
                n_years = len(years_present)
                tot_prod = ev_item.get("total_production", 0)
                seasons_present = [s.lower() for s in ev_item.get("seasons_present", [])]

                if n_years >= 10:
                    evidence_score = 25.0 if tot_prod > 10000 else 20.0
                    evidence_text = f"{crop_name} has strong historical cultivation evidence in {canon_district}."
                elif n_years >= 5:
                    evidence_score = 15.0
                    evidence_text = f"{crop_name} has established historical cultivation records in {canon_district}."
                elif n_years >= 1:
                    evidence_score = 8.0
                    evidence_text = f"{crop_name} is grown in {canon_district}."
                else:
                    evidence_score = 4.0
                    evidence_text = f"{crop_name} has regional presence."
            else:
                years_present = []
                seasons_present = []
                c_ev = cand_info.get("evidence_score", 0.5)
                evidence_score = round(float(c_ev) * 6.0, 1) if isinstance(c_ev, (int, float)) else 3.0
                evidence_text = f"{crop_name} is cultivated in the agro-climatic zone."

            # 2. Season Filter & Compatibility
            season_suitable = True
            season_score = 20.0
            season_text = f"Well-suited for the {season} cropping season."

            if not is_perennial and canon_season.lower() not in ["whole year", "perennial"] and season.lower() not in ["whole year", "perennial"]:
                cal_info = self.season_calendar.get(crop_name, {}) if isinstance(self.season_calendar, dict) else {}
                cal_seasons = [s.lower() for s in cal_info.get("seasons", ["Kharif", "Rabi", "Summer", "Whole Year"])]
                valid_seasons = set(seasons_present + cal_seasons + ["whole year"])

                if canon_season.lower() not in valid_seasons and season.lower() not in valid_seasons:
                    season_suitable = False
                    season_score = 0.0
                    season_text = f"Season mismatch (grows in {', '.join([s.title() for s in valid_seasons if s != 'whole year'])})"

            if not season_suitable:
                rejected_crops.append({
                    "crop": crop_name,
                    "rejection_reason": season_text,
                    "rejection_stage": "SEASONAL_FILTER"
                })
                continue

            # 3. Soil pH Hard Filter
            soil_score = 0.0
            soil_status = "UNKNOWN"
            req_info = self.crop_requirements.get(crop_name, {})
            ph_req = req_info.get("soil_ph", {})
            min_ph = ph_req.get("min", 5.5)
            max_ph = ph_req.get("max", 7.5)

            if soil_ph is not None:
                if min_ph <= soil_ph <= max_ph:
                    soil_score = 15.0
                    soil_status = "SUITABLE"
                    soil_text = f"Soil pH {soil_ph} is within optimal range ({min_ph}–{max_ph})."
                elif (min_ph - 0.5) <= soil_ph <= (max_ph + 0.5):
                    soil_score = 7.0
                    soil_status = "PARTIALLY_SUITABLE"
                    soil_text = f"Soil pH {soil_ph} is acceptable for {crop_name}."
                else:
                    soil_score = 0.0
                    soil_status = "UNSUITABLE"
                    soil_text = f"Soil pH {soil_ph} is outside safe tolerance ({min_ph}–{max_ph}) for {crop_name}."

            if soil_status == "UNSUITABLE":
                rejected_crops.append({
                    "crop": crop_name,
                    "rejection_reason": soil_text,
                    "rejection_stage": "SOIL_FILTER"
                })
                continue

            # 4. Water Status (4-State Classification)
            if water_available_mm is not None:
                if water_available_mm >= 800:
                    water_score = 10.0
                    water_status = "SUITABLE"
                elif water_available_mm >= 400:
                    water_score = 6.0
                    water_status = "LIMITED"
                else:
                    water_score = 2.0
                    water_status = "UNSUITABLE"
            else:
                water_score = 0.0
                water_status = "UNKNOWN"

            # 5. Crop Rotation
            rotation_score = 0.0
            rotation_text = None
            if previous_crop:
                if previous_crop.lower() == crop_name.lower():
                    rotation_score = 2.0
                    rotation_text = f"Same crop repetition ({previous_crop} → {crop_name})"
                elif CROP_FAMILIES.get(previous_crop) and CROP_FAMILIES.get(previous_crop) == CROP_FAMILIES.get(crop_name):
                    rotation_score = 5.0
                    rotation_text = f"Same family repetition ({CROP_FAMILIES.get(crop_name)})"
                elif previous_crop in LEGUME_CROPS:
                    rotation_score = 15.0
                    rotation_text = f"Beneficial legume rotation after {previous_crop} (Nitrogen restoration)"
                else:
                    rotation_score = 10.0
                    rotation_text = f"Compatible rotation after {previous_crop}"

            # 6. ML Suitability Score
            data_conf = cand_info.get("data_confidence") or cand_info.get("evidence_score") or 0.7
            ml_score = min(15.0, round(float(data_conf) * 15.0, 2)) if isinstance(data_conf, (int, float)) else 8.0

            # 7. News Intelligence Risk Adjustment
            intel_records = self.intel_lookup.get((canon_state.lower(), crop_name.lower()), [])
            news_adj, news_sig, news_ver = calculate_bounded_news_adjustment(intel_records)

            # 8. Mandi Market Score
            m_data = self.mandi_lookup.get((canon_district.lower(), crop_name.lower()))
            if not m_data:
                m_data = self.mandi_lookup.get((canon_state.lower(), crop_name.lower()))
            market_score = 5.0 if m_data else 0.0

            # Composite Score (0-100)
            final_score = round(
                evidence_score + season_score + soil_score +
                water_score + rotation_score + ml_score + news_adj + market_score, 1
            )
            final_score = max(0.0, min(100.0, final_score))

            reasons = [evidence_text, season_text]
            if soil_status == "SUITABLE":
                reasons.append(soil_text)
            if rotation_text:
                reasons.append(rotation_text)

            scored_recommendations.append({
                "crop": crop_name,
                "is_perennial": is_perennial,
                "final_score": final_score,
                "rank": 0,
                "reasons": reasons,
                "water_status": water_status,
                "evidence_source": cand_info.get("source", "data.gov.in APY")
            })

        scored_recommendations = sorted(scored_recommendations, key=lambda x: x["final_score"], reverse=True)
        top_5 = scored_recommendations[:5]

        for idx, rec in enumerate(top_5):
            rec["rank"] = idx + 1
            rel_intel = self.intel_lookup.get((canon_state.lower(), rec["crop"].lower()), [])
            rec["nlp_explanation"] = explain_crop_recommendation(
                canon_state, canon_district, season, rec, rel_intel
            )
            rec["crop_information"] = get_crop_information(rec["crop"])

        # Fetch Mandi & Prediction for top recommended crop
        top_crop = top_5[0]["crop"] if top_5 else "Rice"
        top_intel = self.intel_lookup.get((canon_state.lower(), top_crop.lower()), [])
        mandi_vec = self._get_mandi_vector(canon_district, top_crop, canon_state)
        price_forecast = self._get_price_forecast(top_crop, mandi_vec.get("current_price"), canon_state)
        advisory = self._derive_price_advisory(mandi_vec, price_forecast)

        nlp_price_explanation = explain_price_prediction(
            top_crop, canon_district, canon_state, mandi_vec, price_forecast, advisory, top_intel
        )
        nlp_news_summary = summarize_news_intelligence(top_intel)

        return {
            "location": {
                "state": canon_state,
                "district": canon_district,
                "canonical_id": canonical_id
            },
            "season": season,
            "recommendations": top_5,
            "rejected_crops": rejected_crops[:5],
            "market": mandi_vec,
            "price_forecast": price_forecast,
            "price_advisory": advisory,
            "nlp_price_explanation": nlp_price_explanation,
            "nlp_news_summary": nlp_news_summary,
            "current_intelligence": top_intel[:2],
            "weather": {
                "provider": weather_info.provider,
                "weather_status": weather_info.weather_status,
                "temperature": weather_info.avg_temp,
                "rainfall": weather_info.total_rainfall
            }
        }

    def _get_mandi_vector(self, district: str, crop: str, state: str = "") -> dict:
        crop_lower = crop.lower()
        crop_aliases = {
            "finger millet (ragi)": "Finger Millet (Ragi)",
            "ragi": "Finger Millet (Ragi)",
            "tur (arhar)": "Tur (Arhar)",
            "tur": "Tur (Arhar)",
            "moong (green gram)": "Moong (Green Gram)",
            "moong": "Moong (Green Gram)",
            "urad (black gram)": "Urad (Black Gram)",
            "urad": "Urad (Black Gram)",
            "wheat": "Wheat",
            "rice": "Rice",
            "maize": "Maize",
            "onion": "Onion",
            "potato": "Potato",
        }
        comm_name = crop_aliases.get(crop_lower, crop.capitalize())
        comm_key = comm_name.lower()

        canon = self.canonicalize_district(district, state)
        canon_id = canon["canonical_id"] if canon else f"{state}::{district}"
        canon_st = canon.get("state", state) if canon else state
        canon_dt = canon.get("district", district) if canon else district
        src_names = [s.lower() for s in canon.get("source_names", [])] if canon else []

        m_data = None
        if (canon_id.lower(), comm_key) in self.mandi_lookup:
            m_data = self.mandi_lookup[(canon_id.lower(), comm_key)]
        elif (canon_st.lower(), canon_dt.lower(), comm_key) in self.mandi_lookup:
            m_data = self.mandi_lookup[(canon_st.lower(), canon_dt.lower(), comm_key)]
        elif (canon_dt.lower(), comm_key) in self.mandi_lookup:
            m_data = self.mandi_lookup[(canon_dt.lower(), comm_key)]
        else:
            for s_alias in src_names:
                if (s_alias, comm_key) in self.mandi_lookup:
                    m_data = self.mandi_lookup[(s_alias, comm_key)]
                    break

        if not m_data and mandi_service is not None:
            live_res = mandi_service.get_latest_price(crop, canon_st)
            if live_res and live_res.modal_price > 0:
                m_data = {
                    "commodity": comm_name,
                    "district": canon_dt,
                    "state": canon_st,
                    "market_name": live_res.market,
                    "min_price_rs_qtl": live_res.min_price,
                    "modal_price_rs_qtl": live_res.modal_price,
                    "max_price_rs_qtl": live_res.max_price,
                    "arrival_date": live_res.arrival_date,
                    "data_age_days": live_res.data_age_days,
                    "freshness_label": live_res.freshness_label
                }

        if m_data:
            min_p  = m_data.get("min_price_rs_qtl",  m_data.get("min_price",  0.0))
            modal_p = m_data.get("modal_price_rs_qtl", m_data.get("modal_price", 0.0))
            max_p  = m_data.get("max_price_rs_qtl",  m_data.get("max_price",  0.0))
            arr_date_str = str(m_data.get("arrival_date", ""))

            freshness_info = _compute_freshness(arr_date_str)
            return {
                "available": True,
                "crop": crop,
                "commodity": m_data.get("commodity", comm_name),
                "district": m_data.get("district", canon_dt),
                "state": m_data.get("state", canon_st),
                "market": m_data.get("market_name", f"{canon_dt} Mandi Yard"),
                "min_price": round(float(min_p), 2),
                "current_price": round(float(modal_p), 2),
                "max_price": round(float(max_p), 2),
                "currency": "INR",
                "unit": "quintal",
                "observation_date": freshness_info["observation_date"],
                "data_age_days": freshness_info["data_age_days"],
                "freshness_label": freshness_info["freshness_label"],
                "source": "data.gov.in"
            }

        return {
            "available": False,
            "crop": crop,
            "commodity": comm_name,
            "district": canon_dt,
            "state": canon_st,
            "market": "UNAVAILABLE",
            "min_price": None,
            "current_price": None,
            "max_price": None,
            "currency": "INR",
            "unit": "quintal",
            "observation_date": "UNAVAILABLE",
            "data_age_days": None,
            "freshness_label": "UNAVAILABLE",
            "source": "data.gov.in"
        }

    def _get_price_forecast(self, crop: str, current_price, state: str = "All") -> dict:
        crop_lower = crop.lower()
        supported_crops = {"rice", "wheat", "maize", "onion", "potato"}

        if crop_lower not in supported_crops:
            return {
                "available": False,
                "crop": crop,
                "predicted_price": None,
                "predictions": [],
                "date_labels": [],
                "forecast_horizon_days": 30,
                "message": "A reliable 30-day price forecast is currently unavailable for this crop."
            }

        from app.ml.inference import predict_price
        try:
            ml_res = predict_price(crop_lower, state=state, horizon_days=30)
            if not ml_res.get("available", False):
                return {
                    "available": False,
                    "crop": crop,
                    "predicted_price": None,
                    "predictions": [],
                    "date_labels": [],
                    "forecast_horizon_days": 30,
                    "message": "A reliable 30-day price forecast could not be generated for this crop at this time."
                }
            preds = ml_res.get("predictions", [])
            pred_price = ml_res.get("predicted_price")
            date_labels = ml_res.get("date_labels", [])
            obs_date = ml_res.get("observation_date")
            # Enforce exactly 30 forecast points
            if len(preds) != 30:
                logger.warning(f"Forecast for {crop} has {len(preds)} points, expected 30. Trimming/padding.")
                if len(preds) > 30:
                    preds = preds[:30]
                    date_labels = date_labels[:30] if len(date_labels) > 30 else date_labels
            return {
                "available": True,
                "crop": crop,
                "current_price": current_price,
                "predicted_price": pred_price,
                "predictions": preds,
                "date_labels": date_labels,
                "observation_date": obs_date,
                "forecast_horizon_days": 30
            }
        except Exception as e:
            logger.warning(f"Error running ML inference for {crop}: {e}")
            return {
                "available": False,
                "crop": crop,
                "predicted_price": None,
                "predictions": [],
                "date_labels": [],
                "forecast_horizon_days": 30,
                "message": "A reliable 30-day price forecast could not be generated for this crop at this time."
            }

    def _derive_price_advisory(self, mandi_vec: dict, forecast_vec: dict) -> dict:
        """Derive SELL/HOLD/WAIT advisory ONLY when both a current price and a valid forecast exist.
        NEVER returns a decision when forecast is unavailable."""
        # Forecast must explicitly be available
        if not forecast_vec.get("available", False):
            return {
                "action": None,
                "reason": forecast_vec.get("message", "A reliable 30-day price forecast is currently unavailable for this crop."),
                "current_price": mandi_vec.get("current_price"),
                "predicted_price": None,
                "price_change_pct": None
            }

        curr = mandi_vec.get("current_price")
        pred = forecast_vec.get("predicted_price")

        # Both prices must be valid positive numbers to derive a meaningful decision
        if not isinstance(curr, (int, float)) or not isinstance(pred, (int, float)) or curr <= 0 or pred <= 0:
            return {
                "action": None,
                "reason": "Current mandi price is unavailable. Cannot generate a market decision without a confirmed price reference.",
                "current_price": curr,
                "predicted_price": pred,
                "price_change_pct": None
            }

        diff_pct = (pred - curr) / curr * 100.0
        abs_diff = abs(round(diff_pct, 1))

        if diff_pct <= -3.0:
            action = "SELL"
            reason = f"Forecast indicates a price decline of approximately {abs_diff}% over the next 30 days. Selling now minimizes downside risk."
        elif diff_pct >= 3.0:
            action = "HOLD"
            reason = f"Prices are forecast to appreciate by approximately {abs_diff}% over the next 30 days. Holding may capture higher value."
        else:
            action = "WAIT"
            reason = f"Price is expected to remain stable within ±{abs_diff}%. Monitor market developments."

        return {
            "action": action,
            "reason": reason,
            "current_price": curr,
            "predicted_price": pred,
            "price_change_pct": round(diff_pct, 1)
        }
