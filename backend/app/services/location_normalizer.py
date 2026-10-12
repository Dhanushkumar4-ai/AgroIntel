"""
location_normalizer.py — Canonical Location Normalization Layer for AgroIntel.

Standardizes all state and district names across:
- Dropdown inputs
- AGMARKNET mandi records
- APY crop cultivation evidence
- Weather geocoding lookups
- News intelligence queries

Enforces:
- Canonical display name for renamed districts (e.g., Ahilya Nagar, Chhatrapati Sambhajinagar, Dharashiv, Prayagraj, Ayodhya, Mysuru, Ballari, Belagavi, Kalaburagi, Shivamogga, Vijayapura).
- Strict non-destructive alias matching (raw_name -> canonical_name -> canonical_id).
"""

import re
import unicodedata
from typing import Optional, Dict, Any, List

# Primary canonical district overrides and alias mappings
DISTRICT_CANONICAL_ALIASES: Dict[str, Dict[str, Any]] = {
    # Maharashtra
    "ahilya nagar": {
        "canonical_name": "Ahilya Nagar",
        "canonical_id": "Maharashtra::Ahilya Nagar",
        "state": "Maharashtra",
        "aliases": [
            "ahmednagar", "ahmed nagar", "ahmednagar district",
            "ahilyanagar", "ahilya nagar", "ahmadnagar", "ahmad nagar",
            "ahilyanagar district"
        ]
    },
    "chhatrapati sambhajinagar": {
        "canonical_name": "Chhatrapati Sambhajinagar",
        "canonical_id": "Maharashtra::Chhatrapati Sambhajinagar",
        "state": "Maharashtra",
        "aliases": [
            "aurangabad", "aurangabad district", "chhatrapati sambhaji nagar",
            "chhatrapati sambhajinagar", "sambhajinagar", "aurangabad (maharashtra)"
        ]
    },
    "dharashiv": {
        "canonical_name": "Dharashiv",
        "canonical_id": "Maharashtra::Dharashiv",
        "state": "Maharashtra",
        "aliases": [
            "osmanabad", "osmanabad district", "dharashiv", "dharashiv district"
        ]
    },
    # Karnataka
    "dakshina kannada": {
        "canonical_name": "Dakshina Kannada",
        "canonical_id": "Karnataka::Dakshina Kannada",
        "state": "Karnataka",
        "aliases": [
            "dakshina kannada", "dakshin kannad", "south kanara",
            "dakshina kannada (mangalore)", "mangalore district"
        ]
    },
    "uttara kannada": {
        "canonical_name": "Uttara Kannada",
        "canonical_id": "Karnataka::Uttara Kannada",
        "state": "Karnataka",
        "aliases": [
            "uttara kannada", "uttar kannad", "north kanara",
            "uttara kannada (karwar)", "karwar district"
        ]
    },
    "mysuru": {
        "canonical_name": "Mysuru",
        "canonical_id": "Karnataka::Mysuru",
        "state": "Karnataka",
        "aliases": ["mysore", "mysuru", "mysore district", "mysuru district"]
    },
    "ballari": {
        "canonical_name": "Ballari",
        "canonical_id": "Karnataka::Ballari",
        "state": "Karnataka",
        "aliases": ["bellary", "ballari", "bellary district", "ballari district"]
    },
    "belagavi": {
        "canonical_name": "Belagavi",
        "canonical_id": "Karnataka::Belagavi",
        "state": "Karnataka",
        "aliases": ["belgaum", "belagavi", "belgaum district", "belagavi district"]
    },
    "kalaburagi": {
        "canonical_name": "Kalaburagi",
        "canonical_id": "Karnataka::Kalaburagi",
        "state": "Karnataka",
        "aliases": ["gulbarga", "kalaburagi", "gulbarga district", "kalaburagi district"]
    },
    "shivamogga": {
        "canonical_name": "Shivamogga",
        "canonical_id": "Karnataka::Shivamogga",
        "state": "Karnataka",
        "aliases": ["shimoga", "shivamogga", "shimoga district", "shivamogga district"]
    },
    "vijayapura": {
        "canonical_name": "Vijayapura",
        "canonical_id": "Karnataka::Vijayapura",
        "state": "Karnataka",
        "aliases": ["bijapur", "vijayapura", "bijapur district", "vijayapura district"]
    },
    # Uttar Pradesh
    "prayagraj": {
        "canonical_name": "Prayagraj",
        "canonical_id": "Uttar Pradesh::Prayagraj",
        "state": "Uttar Pradesh",
        "aliases": ["allahabad", "prayagraj", "allahabad district", "prayagraj district"]
    },
    "ayodhya": {
        "canonical_name": "Ayodhya",
        "canonical_id": "Uttar Pradesh::Ayodhya",
        "state": "Uttar Pradesh",
        "aliases": ["faizabad", "ayodhya", "faizabad district", "ayodhya district"]
    },
    # West Bengal
    "purba bardhaman": {
        "canonical_name": "Purba Bardhaman",
        "canonical_id": "West Bengal::Purba Bardhaman",
        "state": "West Bengal",
        "aliases": ["burdwan", "bardhaman", "barddhaman", "purba bardhaman", "burdwan district"]
    },
    # Punjab
    "sahibzada ajit singh nagar": {
        "canonical_name": "Sahibzada Ajit Singh Nagar",
        "canonical_id": "Punjab::Sahibzada Ajit Singh Nagar",
        "state": "Punjab",
        "aliases": ["mohali", "sas nagar", "s.a.s. nagar", "sahibzada ajit singh nagar"]
    },
    "shahid bhagat singh nagar": {
        "canonical_name": "Shahid Bhagat Singh Nagar",
        "canonical_id": "Punjab::Shahid Bhagat Singh Nagar",
        "state": "Punjab",
        "aliases": ["nawanshahr", "nawan shahr", "sbs nagar", "s.b.s. nagar", "shahid bhagat singh nagar"]
    },
    "malerkotla": {
        "canonical_name": "Malerkotla",
        "canonical_id": "Punjab::Malerkotla",
        "state": "Punjab",
        "aliases": ["malerkotla", "maler kotla", "malerkotla district"]
    },
    "sri muktsar sahib": {
        "canonical_name": "Sri Muktsar Sahib",
        "canonical_id": "Punjab::Sri Muktsar Sahib",
        "state": "Punjab",
        "aliases": ["muktsar", "sri muktsar sahib", "muktsar district"]
    },
    # Andhra Pradesh
    "ysr kadapa": {
        "canonical_name": "YSR Kadapa",
        "canonical_id": "Andhra Pradesh::YSR Kadapa",
        "state": "Andhra Pradesh",
        "aliases": ["kadapa", "cuddapah", "ysr", "ysr district", "kadapa district", "ysr kadapa"]
    },
    "sri potti sriramulu nellore": {
        "canonical_name": "Sri Potti Sriramulu Nellore",
        "canonical_id": "Andhra Pradesh::Sri Potti Sriramulu Nellore",
        "state": "Andhra Pradesh",
        "aliases": ["nellore", "spsr nellore", "sri potti sriramulu nellore", "nellore district"]
    },
    "dr. b.r. ambedkar konaseema": {
        "canonical_name": "Dr. B.R. Ambedkar Konaseema",
        "canonical_id": "Andhra Pradesh::Dr. B.R. Ambedkar Konaseema",
        "state": "Andhra Pradesh",
        "aliases": ["konaseema", "dr. b.r. ambedkar konaseema", "dr br ambedkar konaseema", "konaseema district"]
    },
    "ananthapuramu": {
        "canonical_name": "Ananthapuramu",
        "canonical_id": "Andhra Pradesh::Ananthapuramu",
        "state": "Andhra Pradesh",
        "aliases": ["anantapur", "ananthapur", "ananthapuramu", "anantapur district"]
    },
    "alluri sitharama raju": {
        "canonical_name": "Alluri Sitharama Raju",
        "canonical_id": "Andhra Pradesh::Alluri Sitharama Raju",
        "state": "Andhra Pradesh",
        "aliases": ["asr district", "alluri sitharama raju", "alluri seetharama raju", "alluri sitharama raju district"]
    },
    "parvathipuram manyam": {
        "canonical_name": "Parvathipuram Manyam",
        "canonical_id": "Andhra Pradesh::Parvathipuram Manyam",
        "state": "Andhra Pradesh",
        "aliases": ["manyam", "parvathipuram manyam", "parvathipuram", "manyam district"]
    },
    # Assam
    "kamrup metropolitan": {
        "canonical_name": "Kamrup Metropolitan",
        "canonical_id": "Assam::Kamrup Metropolitan",
        "state": "Assam",
        "aliases": ["guwahati", "kamrup metro", "kamrup metropolitan"]
    },
    # Union Territories
    "nicobars": {
        "canonical_name": "Nicobars",
        "canonical_id": "Andaman and Nicobar Islands::Nicobars",
        "state": "Andaman and Nicobar Islands",
        "aliases": ["nicobar", "nicobar islands", "car nicobar", "nicobars"]
    },
    "south andaman": {
        "canonical_name": "South Andaman",
        "canonical_id": "Andaman and Nicobar Islands::South Andaman",
        "state": "Andaman and Nicobar Islands",
        "aliases": ["port blair", "andaman", "south andaman"]
    },
    "leh": {
        "canonical_name": "Leh",
        "canonical_id": "Ladakh::Leh",
        "state": "Ladakh",
        "aliases": ["leh ladakh", "leh district", "leh"]
    },
    "kargil": {
        "canonical_name": "Kargil",
        "canonical_id": "Ladakh::Kargil",
        "state": "Ladakh",
        "aliases": ["kargil ladakh", "kargil district", "kargil"]
    },
    "lakshadweep": {
        "canonical_name": "Lakshadweep",
        "canonical_id": "Lakshadweep::Lakshadweep",
        "state": "Lakshadweep",
        "aliases": ["lakshadweep district", "kavaratti", "lakshadweep islands", "lakshadweep"]
    }
}

# State and Union Territory official normalization mappings
STATE_CANONICAL_ALIASES: Dict[str, str] = {
    # 8 Union Territories
    "andaman & nicobar": "Andaman and Nicobar Islands",
    "andaman and nicobar": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "andaman and nicobar islands (ut)": "Andaman and Nicobar Islands",
    "chandigarh (ut)": "Chandigarh",
    "chandigarh": "Chandigarh",
    "dadra and nagar haveli (ut)": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu (ut)": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra & nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman & diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu (ut)": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi (nct)": "Delhi",
    "nct of delhi": "Delhi",
    "national capital territory of delhi": "Delhi",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "jammu and kashmir": "Jammu and Kashmir",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir (ut)": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "ladakh (ut)": "Ladakh",
    "lakshadweep (ut)": "Lakshadweep",
    "lakshadweep": "Lakshadweep",
    "puducherry (ut)": "Puducherry",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    # 28 States (historical aliases)
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
}

# Fast alias-to-canonical lookup dictionary
_ALIAS_LOOKUP: Dict[str, Dict[str, Any]] = {}
for canon_key, entry in DISTRICT_CANONICAL_ALIASES.items():
    _ALIAS_LOOKUP[canon_key] = entry
    _ALIAS_LOOKUP[entry["canonical_name"].lower()] = entry
    _ALIAS_LOOKUP[entry["canonical_id"].lower()] = entry
    for alias in entry["aliases"]:
        _ALIAS_LOOKUP[alias.lower()] = entry
        _ALIAS_LOOKUP[f"{entry['state'].lower()}::{alias.lower()}"] = entry


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-ASCII accents."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text)).encode("ASCII", "ignore").decode("utf-8")
    return text.strip().lower()


def normalize_state_name(state_raw: Optional[str]) -> str:
    """Normalize state or UT string to official 36 entity name."""
    if not state_raw:
        return "India"
    st_clean = clean_text(state_raw)
    return STATE_CANONICAL_ALIASES.get(st_clean, state_raw.strip().title())


def normalize_location(district_raw: str, state_raw: Optional[str] = None) -> Dict[str, str]:
    """
    Resolve any raw district string (and optional state) to its canonical display name and ID.

    Returns:
        {
            "raw_name": district_raw,
            "canonical_name": "Ahilya Nagar",
            "canonical_id": "Maharashtra::Ahilya Nagar",
            "state": "Maharashtra"
        }
    """
    if not district_raw:
        return {
            "raw_name": "",
            "canonical_name": "Unknown",
            "canonical_id": "Unknown::Unknown",
            "state": state_raw or "Unknown"
        }

    raw_clean = clean_text(district_raw)
    st_clean = clean_text(state_raw) if state_raw else ""

    # 1. Direct match in alias lookup
    if st_clean and f"{st_clean}::{raw_clean}" in _ALIAS_LOOKUP:
        match = _ALIAS_LOOKUP[f"{st_clean}::{raw_clean}"]
        return {
            "raw_name": district_raw,
            "canonical_name": match["canonical_name"],
            "canonical_id": match["canonical_id"],
            "state": match["state"],
            "canonical_state": match["state"],
            "canonical_district": match["canonical_name"]
        }

    if raw_clean in _ALIAS_LOOKUP:
        match = _ALIAS_LOOKUP[raw_clean]
        if not state_raw or clean_text(match["state"]) == st_clean:
            return {
                "raw_name": district_raw,
                "canonical_name": match["canonical_name"],
                "canonical_id": match["canonical_id"],
                "state": match["state"],
                "canonical_state": match["state"],
                "canonical_district": match["canonical_name"]
            }

    # 2. General title casing if not in special alias list
    canon_district = district_raw.strip().title()
    # Normalize common abbreviations
    canon_district = re.sub(r"\bDistrict\b", "", canon_district, flags=re.IGNORECASE).strip()
    canon_state = normalize_state_name(state_raw)

    return {
        "raw_name": district_raw,
        "canonical_name": canon_district,
        "canonical_id": f"{canon_state}::{canon_district}",
        "state": canon_state,
        "canonical_state": canon_state,
        "canonical_district": canon_district
    }
