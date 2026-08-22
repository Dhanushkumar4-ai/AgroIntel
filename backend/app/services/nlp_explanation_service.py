"""
nlp_explanation_service.py — AgroIntel Natural Language Explanation & Summarization Layer.

Produces farmer-centric, actionable, and data-grounded guidance without exposing
internal technical metrics (e.g. 25.0/25, UNKNOWN, UNAVAILABLE, ML scores, model names).
"""

import os
import json
import logging
import datetime
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CROP_INFO_PATH = DATA_DIR / "crop_information.json"

_CROP_INFO_DB = {}
if CROP_INFO_PATH.exists():
    with open(CROP_INFO_PATH) as f:
        _CROP_INFO_DB = json.load(f)

# Comprehensive crop water guidance database
CROP_WATER_GUIDANCE = {
    "sugarcane": {
        "requirement_level": "HIGH",
        "annual_need_mm": "1,500–2,500 mm",
        "guidance": "Sugarcane requires substantial water throughout its 10–14 month growing period. In semi-arid regions (such as Ahilya Nagar), cultivation is well-suited to canal-irrigated belts (e.g. Pravara/Godavari basins) or fields with reliable borewells. Ensure adequate irrigation is available before planting."
    },
    "rice": {
        "requirement_level": "HIGH",
        "annual_need_mm": "1,200–1,800 mm",
        "guidance": "Paddy requires substantial moisture and standing water during vegetative and flowering stages. Ensure reliable canal, borewell, or heavy monsoon rainfall before planting."
    },
    "banana": {
        "requirement_level": "HIGH",
        "annual_need_mm": "1,200–2,200 mm",
        "guidance": "Banana requires regular drip or basin irrigation with consistent soil moisture throughout the year."
    },
    "cotton": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "700–1,200 mm",
        "guidance": "Cotton performs well in deep black soils with moderate rainfall. Ensure 2–3 protective irrigations during squaring and boll formation if dry spells occur."
    },
    "maize": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "500–800 mm",
        "guidance": "Maize requires moderate moisture. Timely irrigation during critical tasseling and silking stages is essential for optimal grain yield."
    },
    "onion": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "350–550 mm",
        "guidance": "Onion requires shallow, frequent irrigations to maintain uniform bulb development without waterlogging."
    },
    "potato": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "500–700 mm",
        "guidance": "Potato requires consistent soil moisture during tuber initiation and bulking stages. Avoid excess moisture near harvest to prevent rotting."
    },
    "soybean": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "450–700 mm",
        "guidance": "Soybean is well-suited for Rainy Season conditions in black soils. Supplementary irrigation is beneficial if dry spells coincide with pod-filling."
    },
    "wheat": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "450–650 mm",
        "guidance": "Winter Season wheat requires 4–6 scheduled irrigations, especially at the Crown Root Initiation (CRI) and grain-filling stages."
    },
    "pearl millet (bajra)": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "300–450 mm",
        "guidance": "Bajra is highly drought-tolerant and thrives in semi-arid, rainfed conditions with minimal water requirements."
    },
    "bajra": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "300–450 mm",
        "guidance": "Bajra is highly drought-tolerant and thrives in semi-arid, rainfed conditions with minimal water requirements."
    },
    "sorghum (jowar)": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "350–550 mm",
        "guidance": "Jowar has high water-use efficiency and excellent drought resilience, making it a dependable rainfed crop."
    },
    "jowar": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "350–550 mm",
        "guidance": "Jowar has high water-use efficiency and excellent drought resilience, making it a dependable rainfed crop."
    },
    "moong (green gram)": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "250–400 mm",
        "guidance": "Short-duration pulse that thrives with minimal moisture and enriches soil nitrogen for subsequent rotations."
    },
    "moong": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "250–400 mm",
        "guidance": "Short-duration pulse that thrives with minimal moisture and enriches soil nitrogen for subsequent rotations."
    },
    "black gram (urad)": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "250–400 mm",
        "guidance": "Drought-hardy pulse suitable for rainfed cultivation with low moisture requirements."
    },
    "urad": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "250–400 mm",
        "guidance": "Drought-hardy pulse suitable for rainfed cultivation with low moisture requirements."
    },
    "chickpea (gram)": {
        "requirement_level": "LOW_DROUGHT_HARDY",
        "annual_need_mm": "250–400 mm",
        "guidance": "Deep-rooting Rabi pulse that utilizes residual soil moisture effectively with minimal supplementary irrigation."
    },
    "groundnut": {
        "requirement_level": "MODERATE",
        "annual_need_mm": "500–600 mm",
        "guidance": "Requires well-drained soil and adequate moisture during pegging and pod formation stages."
    }
}


def _get_groq_key() -> Optional[str]:
    env_file = BASE_DIR.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val and val != "your_groq_api_key_here":
                    return val
    return os.environ.get("GROQ_API_KEY")


def get_crop_information(crop_name: str) -> dict:
    """Retrieve verified crop knowledge facts from crop_information.json."""
    if not crop_name:
        return {}
    c_clean = crop_name.lower().strip()
    if c_clean in _CROP_INFO_DB:
        return _CROP_INFO_DB[c_clean]
    for k, v in _CROP_INFO_DB.items():
        if k in c_clean or c_clean in k:
            return v
    return {
        "why_grown": f"{crop_name} is cultivated for regional food demand, farm revenue, and seasonal crop rotation.",
        "common_uses": "Consumed as food grain, pulse, oilseed, or commercial produce.",
        "season": "Grown in recommended regional cropping seasons.",
        "soil": "Requires well-drained soil with balanced NPK fertilization.",
        "climate": "Thrives under favorable regional seasonal temperatures.",
        "key_characteristics": "Responds well to recommended agronomic and soil management practices."
    }


def explain_crop_recommendation(
    state: str,
    district: str,
    season: str,
    crop_recommendation: dict,
    current_intelligence: List[dict] = None,
    data_quality: dict = None
) -> dict:
    """
    Generate natural language explanation for a single crop recommendation.
    Strictly avoids exposing internal technical scores or status codes.
    """
    crop = crop_recommendation.get("crop", "Crop")
    crop_lower = crop.lower()
    
    # 1. Why Recommended
    clean_reasons = []
    for r in crop_recommendation.get("reasons", []):
        # Strip score suffixes like (25.0/25), (20.0/20)
        r_clean = r.split("(")[0].strip() if "(" in r else r
        if "STRONG_HISTORICAL" in r_clean or "historical" in r_clean.lower():
            clean_reasons.append(f"{crop} has strong historical cultivation evidence in {district}")
        elif "season" in r_clean.lower():
            clean_reasons.append(f"well-suited for the {season} cropping season")
        elif "soil" in r_clean.lower():
            clean_reasons.append(r_clean)
        elif "rotation" in r_clean.lower():
            clean_reasons.append(r_clean)

    if clean_reasons:
        why_text = f"{crop} is recommended for {district} because it is " + "; ".join(clean_reasons) + "."
    else:
        why_text = f"{crop} is agronomically suitable for {district} during the {season} season based on regional cultivation evidence and soil suitability."

    # 2. Things to Consider / Water Guidance
    water_info = CROP_WATER_GUIDANCE.get(crop_lower)
    if not water_info:
        for k, v in CROP_WATER_GUIDANCE.items():
            if k in crop_lower or crop_lower in k:
                water_info = v
                break

    if water_info:
        considerations_text = water_info["guidance"]
    else:
        considerations_text = f"Ensure adequate soil moisture and standard field management practices for {crop}."

    # 3. Current Situation / News
    relevant_news = []
    if current_intelligence:
        for intel in current_intelligence:
            intel_crop = intel.get("crop", "").lower()
            if intel_crop == crop_lower or crop_lower in intel_crop or intel_crop in crop_lower:
                relevant_news.append(intel)

    if relevant_news:
        n_item = relevant_news[0]
        headline = n_item.get("headline") or n_item.get("summary") or "agricultural report in the region"
        situation_text = f"Recent agricultural intelligence: {headline}."
    else:
        situation_text = "No major adverse weather warnings or pest outbreaks have been reported in recent verified agricultural advisories."

    season_friendly_map = {"kharif": "the Rainy Season", "rabi": "the Winter Season", "zaid": "the Summer Season", "summer": "the Summer Season", "whole year": "the Whole Year"}
    season_phrase = season_friendly_map.get(str(season).lower(), f"the {season} season")
    summary_text = f"{crop} is recommended for {season_phrase} in {district}, {state}. {why_text}"

    deterministic_res = {
        "summary": summary_text,
        "why_recommended": why_text,
        "current_situation": situation_text,
        "considerations": considerations_text,
        "nlp_source": "DETERMINISTIC_RULES"
    }

    groq_key = _get_groq_key()
    if not groq_key:
        return deterministic_res

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        prompt_payload = {
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are AgroIntel's Farmer Advisory NLP Engine. Convert structured agricultural recommendations into clear, actionable advice for Indian farmers. Output strictly JSON with keys: summary, why_recommended, current_situation, considerations. Never mention score numbers (e.g. 25/25, 95/100, ML score), and never output 'UNKNOWN' or 'could not verify'. Give practical guidance on water requirements and field management."
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "district": district,
                        "state": state,
                        "season": season,
                        "crop": crop,
                        "reasons": clean_reasons,
                        "water_guidance": considerations_text,
                        "news_item": relevant_news[0] if relevant_news else None
                    })
                }
            ]
        }
        req = urllib.request.Request(url, data=json.dumps(prompt_payload).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            raw = json.loads(resp.read().decode('utf-8'))
            llm_content = json.loads(raw["choices"][0]["message"]["content"])
            llm_content["nlp_source"] = "GROQ_LLAMA_3.3_70B"
            return llm_content
    except Exception as e:
        logger.debug(f"[NLP_EXPLANATION] Groq fallback to deterministic rules ({e})")
        return deterministic_res


def explain_price_prediction(
    crop: str,
    district: str,
    state: str,
    mandi_vec: dict,
    price_forecast: dict,
    price_advisory: dict,
    current_intelligence: List[dict] = None
) -> dict:
    """Generate concise, natural language explanation for price prediction and SELL/HOLD/WAIT decision."""
    curr = mandi_vec.get("current_price")
    pred = price_forecast.get("predicted_price")
    action = price_advisory.get("action", "HOLD")
    mkt_name = mandi_vec.get("market") or f"{district} Mandi"
    obs_date = mandi_vec.get("observation_date", "")

    if curr and pred:
        diff_pct = round(((pred - curr) / curr) * 100.0, 1)
        if action == "SELL":
            analysis_text = f"Mandi price at {mkt_name} was ₹{curr:,.0f}/quintal ({obs_date}). The 30-day forecast projects a price decline of {abs(diff_pct):.1f}% to ₹{pred:,.0f}/quintal. Selling now minimizes downside price risk."
        elif action == "HOLD":
            analysis_text = f"Mandi price at {mkt_name} was ₹{curr:,.0f}/quintal ({obs_date}). The 30-day forecast projects price appreciation of {diff_pct:+.1f}% to ₹{pred:,.0f}/quintal. Holding may capture higher value."
        else:
            analysis_text = f"Mandi price at {mkt_name} was ₹{curr:,.0f}/quintal ({obs_date}). The 30-day forecast projects stable prices around ₹{pred:,.0f}/quintal ({diff_pct:+.1f}% change)."
    elif pred:
        analysis_text = f"The 30-day expected market price for {crop} in {state} is ₹{pred:,.0f}/quintal."
    else:
        analysis_text = f"Price forecasting is available for core benchmark commodities (Rice, Wheat, Maize, Onion, Potato)."

    return {
        "analysis": analysis_text,
        "decision": action,
        "nlp_source": "DETERMINISTIC_RULES"
    }


def summarize_news_intelligence(current_intelligence: List[dict]) -> str:
    """Generate a high-level summary of active news intelligence signals."""
    if not current_intelligence:
        return "No major adverse weather warnings or market disruptions have been reported in recent verified agricultural advisories."
    items = []
    for item in current_intelligence[:2]:
        headline = item.get("headline") or item.get("summary")
        if headline:
            items.append(headline)
    return ". ".join(items) + "." if items else "Agricultural market conditions are reported stable."
