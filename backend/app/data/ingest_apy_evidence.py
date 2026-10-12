"""
ingest_apy_evidence.py — APY Evidence Ingestion, Aliasing Normalization & De-duplication Pipeline

Ingests raw APY records from data.gov.in / DES (raw_apy_records.json) and supplemental verified records.
Performs:
1. Crop name and state/district canonicalization.
2. Alias de-duplication (e.g., duplicate Paddy and Rice observations for the same district/year/season).
3. Separation of seasonal statistics (Kharif, Rabi, Summer, Whole Year) from gross multi-season footprints.
4. Calculation of annual averages and extraction of latest verified observations per crop.
5. Emits cleaned district_crop_evidence.json and an apy_deduplication_audit.json audit trail.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_apy")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "backend" / "app" / "data" / "experimental"
if not EXP_DIR.exists():
    EXP_DIR = BASE_DIR / "app" / "data" / "experimental"

RAW_APY_PATH = EXP_DIR / "raw_apy_records.json"
OUTPUT_EVIDENCE_PATH = EXP_DIR / "district_crop_evidence.json"
AUDIT_LOG_PATH = EXP_DIR / "apy_deduplication_audit.json"

# Authoritative Crop Aliases & Normalization
CROP_ALIAS_MAP: Dict[str, str] = {
    "paddy": "Rice",
    "rice": "Rice",
    "atcanut (raw)": "Arecanut",
    "arcanut (processed)": "Arecanut",
    "arecanut": "Arecanut",
    "black pepper": "Black Pepper",
    "cotton(lint)": "Cotton",
    "cotton": "Cotton",
    "wheat": "Wheat",
    "bajra": "Pearl Millet (Bajra)",
    "pearl millet (bajra)": "Pearl Millet (Bajra)",
    "pearl millet": "Pearl Millet (Bajra)",
    "jowar": "Sorghum (Jowar)",
    "sorghum (jowar)": "Sorghum (Jowar)",
    "sorghum": "Sorghum (Jowar)",
    "soyabean": "Soybean",
    "soybean": "Soybean",
    "ragi": "Finger Millet (Ragi)",
    "finger millet (ragi)": "Finger Millet (Ragi)",
    "gram": "Chickpea (Gram)",
    "chickpea (gram)": "Chickpea (Gram)",
    "chickpea": "Chickpea (Gram)",
    "tur": "Pigeonpea (Arhar/Tur)",
    "arhar/tur": "Pigeonpea (Arhar/Tur)",
    "pigeonpea (arhar/tur)": "Pigeonpea (Arhar/Tur)",
    "pigeonpea": "Pigeonpea (Arhar/Tur)",
    "moong(green gram)": "Moong (Green Gram)",
    "moong (green gram)": "Moong (Green Gram)",
    "green gram": "Moong (Green Gram)",
    "urad": "Black Gram (Urad)",
    "black gram (urad)": "Black Gram (Urad)",
    "black gram": "Black Gram (Urad)",
    "masoor": "Lentil (Masur)",
    "lentil (masur)": "Lentil (Masur)",
    "lentil": "Lentil (Masur)",
    "groundnut": "Groundnut",
    "groundnut(raw)": "Groundnut",
    "sugarcane": "Sugarcane",
    "cashewnut": "Cashewnut",
    "cashewnut raw": "Cashewnut",
    "cashewnut processed": "Cashewnut",
    "coconut": "Coconut",
    "coconut ": "Coconut",
    "sunflower": "Sunflower",
    "maize": "Maize",
    "onion": "Onion",
    "potato": "Potato",
    "sweet potato": "Sweet Potato",
    "tapioca": "Tapioca (Cassava)",
    "tapioca (cassava)": "Tapioca (Cassava)",
    "banana": "Banana",
    "barley": "Barley",
    "coriander": "Coriander",
    "garlic": "Garlic",
    "ginger": "Ginger",
    "turmeric": "Turmeric",
    "tobacco": "Tobacco",
    "cardamom": "Cardamom",
    "rubber": "Rubber",
    "tea": "Tea",
    "coffee": "Coffee",
    "jute": "Jute",
    "mesta": "Mesta",
    "sesamum": "Sesame (Sesamum)",
    "sesame (sesamum)": "Sesame (Sesamum)",
    "niger seed": "Niger Seed",
    "safflower": "Safflower",
    "castor seed": "Castor Seed",
    "linseed": "Linseed",
    "rapeseed &mustard": "Rapeseed & Mustard",
    "rapeseed & mustard": "Rapeseed & Mustard",
    "mustard": "Mustard",
}

# Season Normalization
def normalize_season_name(season_raw: str) -> str:
    s = str(season_raw).strip()
    s_lower = s.lower()
    if "kharif" in s_lower or "autumn" in s_lower:
        return "Kharif"
    if "rabi" in s_lower or "winter" in s_lower:
        return "Rabi"
    if "summer" in s_lower or "zaid" in s_lower:
        return "Summer"
    if "whole" in s_lower or "all" in s_lower or "perennial" in s_lower:
        return "Whole Year"
    return s.title()

# District & State normalization mappings
STATE_NAME_FIXES = {
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
}

DISTRICT_CANONICAL_NAMES = {
    ("karnataka", "dakshin kannad"): "Dakshina Kannada",
    ("karnataka", "uttar kannad"): "Uttara Kannada",
    ("karnataka", "mysore"): "Mysuru",
    ("karnataka", "bellary"): "Ballari",
    ("karnataka", "belgaum"): "Belagavi",
    ("karnataka", "gulbarga"): "Kalaburagi",
    ("karnataka", "shimoga"): "Shivamogga",
    ("karnataka", "bijapur"): "Vijayapura",
    ("karnataka", "tumkur"): "Tumakuru",
    ("karnataka", "chikmagalur"): "Chikkamagaluru",
    ("maharashtra", "ahmednagar"): "Ahilya Nagar",
    ("maharashtra", "aurangabad"): "Chhatrapati Sambhajinagar",
    ("maharashtra", "osmanabad"): "Dharashiv",
    ("uttar pradesh", "allahabad"): "Prayagraj",
    ("uttar pradesh", "faizabad"): "Ayodhya",
    ("west bengal", "burdwan"): "Purba Bardhaman",
    ("west bengal", "bardhaman"): "Purba Bardhaman",
    ("punjab", "firozepur"): "Ferozepur",
    ("punjab", "nawanshahr"): "Shahid Bhagat Singh Nagar",
    ("andhra pradesh", "kadapa"): "YSR Kadapa",
    ("andhra pradesh", "spsr nellore"): "Sri Potti Sriramulu Nellore",
}


def run_pipeline() -> Dict[str, Any]:
    logger.info("Starting APY Evidence Ingestion & De-duplication Pipeline...")

    if not RAW_APY_PATH.exists():
        raise FileNotFoundError(f"Missing raw APY file at {RAW_APY_PATH}")

    with open(RAW_APY_PATH, "r") as f:
        raw_records = json.load(f)

    logger.info(f"Loaded {len(raw_records):,} raw APY records.")

    # 1. First Pass: Canonicalize and Group for De-duplication
    # Group key: (canonical_id, canonical_crop, year, season)
    grouped_observations = defaultdict(list)
    raw_count = len(raw_records)

    for i, r in enumerate(raw_records):
        st_raw = (r.get("state_name") or "").strip()
        dt_raw = (r.get("district_name") or "").strip()
        crop_raw = (r.get("crop") or "").strip()
        yr = r.get("crop_year")
        sn_raw = (r.get("season") or "").strip()
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                s = str(val).strip()
                if s in ("", "NA", "N/A", "null", "None", "-"):
                    return default
                return float(s)
            except (ValueError, TypeError):
                return default

        area = safe_float(r.get("area_"))
        prod = safe_float(r.get("production_"))

        if not st_raw or not dt_raw or not crop_raw or not yr:
            continue

        st_canon = STATE_NAME_FIXES.get(st_raw.lower(), st_raw.title())
        dt_key = (st_canon.lower(), dt_raw.lower())
        dt_canon = DISTRICT_CANONICAL_NAMES.get(dt_key, dt_raw.title())
        crop_canon = CROP_ALIAS_MAP.get(crop_raw.lower(), crop_raw.title())
        sn_norm = normalize_season_name(sn_raw)

        canonical_id = f"{st_canon}::{dt_canon}"
        obs_key = (canonical_id, crop_canon, yr, sn_norm)

        grouped_observations[obs_key].append({
            "record_index": i,
            "raw_state": st_raw,
            "raw_district": dt_raw,
            "raw_crop": crop_raw,
            "raw_season": sn_raw,
            "year": yr,
            "area": area,
            "production": prod,
            "canonical_id": canonical_id,
            "canonical_state": st_canon,
            "canonical_district": dt_canon,
            "canonical_crop": crop_canon,
            "canonical_season": sn_norm,
        })

    # 2. De-duplicate Observations & Build Audit Trail
    deduped_observations = []
    dropped_audit_log = []

    for obs_key, rec_list in grouped_observations.items():
        canonical_id, crop_canon, yr, sn_norm = obs_key

        if len(rec_list) == 1:
            deduped_observations.append(rec_list[0])
            continue

        # Multiple records exist for the exact same district, crop, year, and season!
        # Check for duplicate area/production values (e.g. Paddy vs Rice with identical area)
        # Sort so that primary canonical name ("Rice", "Arecanut") comes first
        def _priority_score(item):
            raw_c = item["raw_crop"].strip()
            # If raw name matches canonical name exactly, highest priority
            if raw_c.lower() == crop_canon.lower():
                return 10
            # Prefer "Rice" over "Paddy"
            if raw_c.lower() == "rice":
                return 9
            if raw_c.lower() == "arecanut":
                return 9
            return 1

        rec_list.sort(key=_priority_score, reverse=True)
        primary = rec_list[0]
        deduped_observations.append(primary)

        # Log all secondary duplicate records
        for sec in rec_list[1:]:
            dropped_audit_log.append({
                "canonical_id": canonical_id,
                "crop": crop_canon,
                "year": yr,
                "season": sn_norm,
                "retained_record": {
                    "raw_crop": primary["raw_crop"],
                    "area": primary["area"],
                    "production": primary["production"]
                },
                "excluded_duplicate": {
                    "raw_crop": sec["raw_crop"],
                    "area": sec["area"],
                    "production": sec["production"],
                    "record_index": sec["record_index"]
                },
                "deduplication_reason": "IDENTICAL_TEMPORAL_AND_SEASONAL_OBSERVATION"
            })

    logger.info(f"De-duplication complete: {len(raw_records):,} raw -> {len(deduped_observations):,} distinct observations. Excluded {len(dropped_audit_log):,} duplicate records.")

    # 3. Aggregate by District and Crop
    # Structure: district_data[canonical_id][crop_canon] -> list of observations
    district_data = defaultdict(lambda: defaultdict(list))
    district_meta = {}

    for obs in deduped_observations:
        cid = obs["canonical_id"]
        cname = obs["canonical_crop"]
        district_data[cid][cname].append(obs)
        if cid not in district_meta:
            district_meta[cid] = {
                "district_id": cid,
                "state": obs["canonical_state"],
                "district": obs["canonical_district"]
            }

    # 4. Compute Season-Specific & Gross Annual Multi-Season Statistics
    output_district_list = []

    for cid, crop_dict in district_data.items():
        d_meta = district_meta[cid]
        crops_list = []

        for crop_name, obs_list in crop_dict.items():
            years_set = sorted(list(set(o["year"] for o in obs_list)))
            seasons_set = sorted(list(set(o["canonical_season"] for o in obs_list)))
            all_raw_crops = sorted(list(set(o["raw_crop"] for o in obs_list)))

            n_years = len(years_set)
            earliest_yr = min(years_set)
            latest_yr = max(years_set)

            tot_area = sum(o["area"] for o in obs_list)
            tot_prod = sum(o["production"] for o in obs_list)
            yield_vals = [o["production"] / o["area"] for o in obs_list if o["area"] > 0 and o["production"] > 0]
            avg_yield = round(sum(yield_vals) / len(yield_vals), 4) if yield_vals else None

            # Gross Annual Multi-Season Averages (Total area/prod divided by distinct calendar years)
            gross_ann_area = round(tot_area / n_years, 1) if n_years > 0 else 0.0
            gross_ann_prod = round(tot_prod / n_years, 1) if n_years > 0 else 0.0

            # Seasonal Breakdown (Separate statistics for Kharif, Rabi, Summer, Whole Year)
            seasonal_breakdown = {}
            for sn in ["Kharif", "Rabi", "Summer", "Whole Year"]:
                sn_obs = [o for o in obs_list if o["canonical_season"] == sn]
                if not sn_obs:
                    continue
                sn_years = sorted(list(set(o["year"] for o in sn_obs)))
                sn_nyears = len(sn_years)
                sn_tot_area = sum(o["area"] for o in sn_obs)
                sn_tot_prod = sum(o["production"] for o in sn_obs)
                sn_yields = [o["production"] / o["area"] for o in sn_obs if o["area"] > 0 and o["production"] > 0]
                sn_avg_yield = round(sum(sn_yields) / len(sn_yields), 4) if sn_yields else None
                sn_latest_yr = max(sn_years)
                latest_sn_obs = next((o for o in sn_obs if o["year"] == sn_latest_yr), sn_obs[-1])

                seasonal_breakdown[sn] = {
                    "years_recorded": sn_nyears,
                    "earliest_year": min(sn_years),
                    "latest_year": sn_latest_yr,
                    "total_area_ha": round(sn_tot_area, 1),
                    "annual_average_area_ha": round(sn_tot_area / sn_nyears, 1) if sn_nyears > 0 else 0.0,
                    "total_production_tonnes": round(sn_tot_prod, 1),
                    "annual_average_production_tonnes": round(sn_tot_prod / sn_nyears, 1) if sn_nyears > 0 else 0.0,
                    "average_yield_tonnes_ha": sn_avg_yield,
                    "latest_year_area_ha": round(latest_sn_obs["area"], 1),
                    "latest_year_production_tonnes": round(latest_sn_obs["production"], 1),
                }

            # Latest Year Observations across all seasons in that latest year
            latest_year_obs = [o for o in obs_list if o["year"] == latest_yr]
            latest_year_dict = {
                o["canonical_season"]: {
                    "area_ha": round(o["area"], 1),
                    "production_tonnes": round(o["production"], 1)
                } for o in latest_year_obs
            }
            latest_year_total_area = round(sum(o["area"] for o in latest_year_obs), 1)
            latest_year_total_prod = round(sum(o["production"] for o in latest_year_obs), 1)

            crops_list.append({
                "crop": crop_name,
                "canonical_crop": crop_name,
                "original_crop_names": all_raw_crops,
                "years_present": years_set,
                "seasons_present": seasons_set,
                "production_records": len(obs_list),
                "total_production": round(tot_prod, 1),
                "area_records": len(obs_list),
                "total_area": round(tot_area, 1),
                "yield_records": len(yield_vals),
                "average_yield": avg_yield,
                "earliest_year": earliest_yr,
                "latest_year": latest_yr,
                "recent_year_presence": bool(latest_yr >= 2012),
                "historical_consistency": round(n_years / max(1, (latest_yr - earliest_yr + 1)), 2),
                "source": "data.gov.in (DES, Ministry of Agriculture)",
                # Enriched precision metrics
                "gross_annual_average_area_ha": gross_ann_area,
                "gross_annual_average_production_tonnes": gross_ann_prod,
                "cumulative_historical_area_ha": round(tot_area, 1),
                "cumulative_historical_production_tonnes": round(tot_prod, 1),
                "latest_year_total_area_ha": latest_year_total_area,
                "latest_year_total_production_tonnes": latest_year_total_prod,
                "latest_year_observations": latest_year_dict,
                "seasonal_breakdown": seasonal_breakdown
            })

        crops_list.sort(key=lambda x: x["total_area"], reverse=True)

        output_district_list.append({
            "district_id": cid,
            "state": d_meta["state"],
            "district": d_meta["district"],
            "crop_count": len(crops_list),
            "crops": crops_list
        })

    output_district_list.sort(key=lambda x: x["district_id"])

    # 5. Write Cleaned JSON Assets
    with open(OUTPUT_EVIDENCE_PATH, "w") as f:
        json.dump(output_district_list, f, indent=2)

    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump({
            "total_raw_records": raw_count,
            "total_deduped_observations": len(deduped_observations),
            "total_duplicates_dropped": len(dropped_audit_log),
            "total_districts_cataloged": len(output_district_list),
            "dropped_records": dropped_audit_log,
            "sample_dropped_duplicates": dropped_audit_log[:15]
        }, f, indent=2)

    logger.info(f"Saved cleaned evidence for {len(output_district_list)} districts to {OUTPUT_EVIDENCE_PATH}")
    logger.info(f"Saved deduplication audit log to {AUDIT_LOG_PATH}")

    return {
        "status": "SUCCESS",
        "districts_count": len(output_district_list),
        "deduped_observations": len(deduped_observations),
        "duplicates_dropped": len(dropped_audit_log)
    }


if __name__ == "__main__":
    res = run_pipeline()
    print("Pipeline result:", res)
