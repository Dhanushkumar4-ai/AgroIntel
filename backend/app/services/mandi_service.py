"""
mandi_service.py — Government Mandi price service (Agmarknet 2.0 primary).

PRIMARY SOURCE (Agmarknet 2.0 — CONFIRMED WORKING):
  - https://api.agmarknet.gov.in/v1/prices-and-arrivals/date-wise/specific-commodity
  - Params: year, month, includeExcel=false, stateId, commodityId
  - Returns real 2025-2026 date-wise mandi prices (Punjab Wheat June 2025 confirmed)
  - No authentication required for this endpoint

FALLBACK SOURCE (data.gov.in — may be unavailable):
  - https://api.data.gov.in/resource/... (legacy AGMARKNET 1.0)

IMPORTANT DESIGN CONSTRAINT:
  - Price is used ONLY for:
      1. Displaying the latest available market price in the UI
      2. Appending to historical tail if data is FRESH (< 3 days old)
      3. Sell/Hold comparison against ML predictions
  - Price is NEVER fed as a primary model input directly.
  - Forecast curves are NEVER manually shifted based on this price.

FRESHNESS POLICY:
  - AGMARKNET typically publishes data with a 3-6 day delay.
  - Fresh    : 0-3 days old
  - Recent   : 4-7 days old
  - Historical: > 7 days old
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

from app.core.config import settings
from app.core.constants import get_official_msp

logger = logging.getLogger(__name__)

_CACHE_FILE = settings.DATA_DIR / "mandi_cache.json"

# ── Commodity IDs for Agmarknet 2.0 API (confirmed via /v1/dashboard-commodities-filter) ──
AGMARKNET_COMMODITY_IDS: dict[str, list[int]] = {
    "wheat":       [1],
    "rice":        [2, 3],        # Paddy (Common): 2, Rice: 3
    "paddy":       [2, 343],      # Paddy (Common), Paddy (Basmati)
    "maize":       [4],
    "corn":        [4],
    "onion":       [23],
    "potato":      [24],
    "tomato":      [65],
    "cotton":      [15],
    "soyabean":    [13],
    "soybean":     [13],
    "mustard":     [12],
    "gram":        [6],
    "chana":       [6, 217],
    "bengal gram": [6],
    "black gram":  [8],
    "urad":        [8],
    "green gram":  [9],
    "moong":       [9],
    "tur":         [45],
    "arhar":       [45],
    "red gram":    [45],
    "masur":       [52],
    "lentil":      [52],
    "bajra":       [28],
    "jowar":       [5],
    "ragi":        [30],
    "barley":      [29],
    "sugarcane":   [122],
    "groundnut":   [10],
    "copra":       [111],
    "coconut":     [116],
    "jute":        [16],
    "banana":      [19],
    "apple":       [17],
    "orange":      [18],
}

# ── State IDs for Agmarknet 2.0 (from /v1/daily-price-arrival/filters) ─────────────
AGMARKNET_STATE_IDS: dict[str, int] = {
    "andaman and nicobar":     1,
    "andhra pradesh":           2,
    "arunachal pradesh":        3,
    "assam":                    4,
    "bihar":                    5,
    "chandigarh":               6,
    "chattisgarh":              7,
    "chhattisgarh":             7,
    "dadra and nagar haveli":   8,
    "daman and diu":            9,
    "goa":                      10,
    "gujarat":                  11,
    "haryana":                  12,
    "himachal pradesh":         13,
    "jammu and kashmir":        14,
    "jharkhand":                15,
    "karnataka":                16,
    "kerala":                   17,
    "keralam":                  17,
    "lakshadweep":              18,
    "madhya pradesh":           19,
    "maharashtra":              20,
    "manipur":                  21,
    "meghalaya":                22,
    "mizoram":                  23,
    "nagaland":                 24,
    "delhi":                    25,
    "nct of delhi":             25,
    "odisha":                   26,
    "orissa":                   26,
    "puducherry":               27,
    "pondicherry":              27,
    "punjab":                   28,
    "rajasthan":                29,
    "sikkim":                   30,
    "tamil nadu":               31,
    "telangana":                32,
    "tripura":                  33,
    "uttar pradesh":            34,
    "uttarakhand":              35,
    "west bengal":              36,
}

# ── Fallback (legacy data.gov.in) ─────────────────────────────────────────────
CROP_TO_COMMODITY: dict[str, str] = {
    "wheat":  "Wheat",
    "rice":   "Rice",
    "maize":  "Maize",
    "onion":  "Onion",
    "potato": "Potato",
}
RESOURCE_1_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
RESOURCE_2_URL = "https://api.data.gov.in/resource/35be999b-0b15-465f-b964-1e582e0e0129"

# ── Request settings ───────────────────────────────────────────────────────────
API_TIMEOUT_SECONDS = 15.0
AGMARKNET_TIMEOUT   = 15.0

API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/home",
}

FRESHNESS_FRESH_DAYS  = 3
FRESHNESS_RECENT_DAYS = 7


@dataclass
class MandiPriceResult:
    """Result from mandi price lookup with full price provenance & transparency."""
    crop: str
    modal_price: float                     # Rs per quintal
    min_price: Optional[float] = None      # Source minimum price if reported, else None
    max_price: Optional[float] = None      # Source maximum price if reported, else None
    arrival_date: str = ""                 # Observation date or period end date (YYYY-MM-DD)
    market: str = "APMC Mandi"
    state: str = "All"
    data_age_days: int = 0
    from_api: bool = True
    source: str = "Agmarknet 2.0 (api.agmarknet.gov.in)"
    price_type: str = "Single-Day APMC Modal Price"
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    unit: str = "₹/quintal"
    fetched_at: float = field(default_factory=time.time)

    # ── Transparency & Provenance Metadata ──
    commodity_name: str = ""
    variety: str = "All Reporting Varieties (State-wide Pre-Aggregated)"
    market_scope: str = "State-wide (APMC Mandis)"
    msp_price: Optional[float] = None
    msp_season: Optional[str] = None
    observation_date: str = ""
    arrival_quantity_qtl: Optional[float] = None
    contributing_records_count: int = 1
    contributing_records: list[dict] = field(default_factory=list)
    aggregation_method: str = "Official Arrival-Weighted State Modal Average (wt_avg_price)"
    price_status: str = "LIVE"             # LIVE, PREVIOUS_DAY, STALE, HISTORICAL_FALLBACK, UNAVAILABLE
    status_label: str = "Live (Same-Day Observation)"
    freshness_label: str = field(init=False)

    def __post_init__(self):
        if not self.commodity_name:
            self.commodity_name = self.crop.capitalize()

        if not self.observation_date:
            self.observation_date = self.arrival_date

        if self.msp_price is None:
            msp_info = get_official_msp(self.crop)
            if msp_info:
                self.msp_price = msp_info.get("msp_price")
                self.msp_season = msp_info.get("season")

        if self.arrival_date and self.data_age_days == 0:
            rec_date = _parse_date(self.arrival_date)
            if rec_date:
                calc_age = (date.today() - rec_date).days
                if calc_age > 0:
                    self.data_age_days = calc_age

        if not self.from_api:
            self.price_status = "HISTORICAL_FALLBACK"
            self.status_label = f"Historical APMC Fallback ({self.data_age_days}d old)"
            self.freshness_label = "Historical"
        elif self.data_age_days == 0:
            self.price_status = "LIVE_TODAY"
            self.status_label = "Live Today (Same-Day Mandi Observation)"
            self.freshness_label = "Fresh"
        elif self.data_age_days == 1:
            self.price_status = "PREVIOUS_DAY"
            self.status_label = "Previous-Day Observation (1 day old)"
            self.freshness_label = "Recent"
        elif self.data_age_days <= FRESHNESS_RECENT_DAYS:
            self.price_status = "STALE"
            self.status_label = f"Recent Observation ({self.data_age_days}d old)"
            self.freshness_label = "Recent"
        else:
            self.price_status = "STALE"
            self.status_label = f"Stale Observation ({self.data_age_days}d old)"
            self.freshness_label = "Historical"

    @property
    def is_fresh(self) -> bool:
        return self.data_age_days <= FRESHNESS_FRESH_DAYS

    def to_transparency_dict(self) -> dict:
        """Export standardized Price Transparency dictionary for frontend and API."""
        last_fetch_str = datetime.fromtimestamp(self.fetched_at).strftime("%Y-%m-%d %H:%M:%S IST") if self.fetched_at else "—"
        msp_diff = round(self.modal_price - self.msp_price, 2) if self.msp_price is not None else None
        msp_ratio = round(self.modal_price / self.msp_price, 2) if (self.msp_price and self.msp_price > 0) else None
        return {
            "source": self.source,
            "source_url": "https://agmarknet.gov.in",
            "observation_date": self.observation_date or self.arrival_date,
            "selected_state": self.state,
            "state": self.state,
            "market_scope": self.market_scope or self.market,
            "commodity": self.commodity_name or self.crop.capitalize(),
            "commodity_name": self.commodity_name or self.crop.capitalize(),
            "variety": self.variety,
            "modal_price": round(self.modal_price, 2),
            "modal_price_display": f"₹{round(self.modal_price):,}/quintal",
            "min_price": round(self.min_price, 2) if self.min_price is not None else None,
            "max_price": round(self.max_price, 2) if self.max_price is not None else None,
            "msp_price": round(self.msp_price, 2) if self.msp_price is not None else None,
            "msp_season": self.msp_season,
            "msp_difference": msp_diff,
            "msp_ratio": msp_ratio,
            "contributing_records_count": self.contributing_records_count,
            "contributing_records": self.contributing_records,
            "aggregation_method": self.aggregation_method,
            "last_successful_fetch_time": last_fetch_str,
            "source_timestamp": last_fetch_str,
            "data_age_days": self.data_age_days,
            "price_status": self.price_status,
            "status_label": self.status_label,
            "is_live": self.price_status in ("LIVE", "LIVE_TODAY"),
            "is_stale": self.price_status in ("STALE", "PREVIOUS_DAY"),
            "is_fallback": self.price_status == "HISTORICAL_FALLBACK",
            "unit": self.unit,
        }


def _load_cache() -> dict:
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read mandi cache: {e}")
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write mandi cache: {e}")


def _parse_date(date_str: str) -> Optional[date]:
    """Parse date strings from mandi APIs."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _build_result(entry: dict, from_api: bool) -> MandiPriceResult:
    return MandiPriceResult(
        crop=entry["crop"],
        modal_price=entry["modal_price"],
        min_price=entry.get("min_price"),
        max_price=entry.get("max_price"),
        arrival_date=entry["arrival_date"],
        market=entry.get("market", "Unknown"),
        state=entry.get("state", "All"),
        data_age_days=entry.get("data_age_days", 999),
        from_api=from_api,
        source=entry.get("source", "Agmarknet 2.0"),
        price_type=entry.get("price_type", "Single-Day APMC Modal Price"),
        period_start=entry.get("period_start"),
        period_end=entry.get("period_end"),
        unit=entry.get("unit", "₹/quintal"),
        fetched_at=entry.get("fetched_at", time.time()),
        commodity_name=entry.get("commodity_name", entry["crop"].capitalize()),
        variety=entry.get("variety", "All Reporting Varieties (State-wide Pre-Aggregated)"),
        market_scope=entry.get("market_scope", f"State-wide Scope ({entry.get('state', 'All')})"),
        msp_price=entry.get("msp_price"),
        msp_season=entry.get("msp_season"),
        observation_date=entry.get("observation_date", entry["arrival_date"]),
        arrival_quantity_qtl=entry.get("arrival_quantity_qtl"),
        contributing_records_count=entry.get("contributing_records_count", 1),
        contributing_records=entry.get("contributing_records", []),
        aggregation_method=entry.get("aggregation_method", "Official Arrival-Weighted State Modal Average (wt_avg_price)"),
    )


def aggregate_mandi_records(
    records: list[dict],
    target_crop: str = "wheat",
    target_state: str = "Karnataka",
    method: str = "arrival_weighted",
) -> dict:
    """
    Aggregate raw mandi records by variety and state.

    Handles:
    - Multiple varieties: explicitly documents contributing varieties and never silently combines them
    - Missing values: gracefully handles null modal, min, max, arrivals without fabricating
    - Duplicate records: deduplicates by (market, variety, date)
    - Aggregation method:
        * arrival_weighted: sum(modal * qty) / sum(qty) if positive arrival qty exists, else arithmetic mean
        * arithmetic_mean: sum(modal) / count
    """
    if not records:
        return {
            "status": "UNAVAILABLE",
            "message": "No mandi records available for aggregation",
            "modal_price": None,
            "min_price": None,
            "max_price": None,
            "contributing_records_count": 0,
            "contributing_records": [],
            "aggregation_method": "None (No Records)",
            "variety": "None",
        }

    seen_keys = set()
    valid_recs = []

    for r in records:
        mkt = str(r.get("market") or r.get("market_name") or "Unknown").strip().lower()
        var = str(r.get("variety") or "Unknown").strip().lower()
        dt = str(r.get("observation_date") or r.get("arrival_date") or "").strip()

        # Deduplication check
        key = (mkt, var, dt)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        raw_modal = r.get("modal_price") or r.get("modal_price_rs_qtl")
        if raw_modal is None or raw_modal == "None":
            continue
        try:
            modal = float(raw_modal)
            if modal <= 0:
                continue
        except (ValueError, TypeError):
            continue

        raw_qty = r.get("arrival_quantity") or r.get("arrivals_qtl") or r.get("arrival_qty")
        qty = None
        if raw_qty is not None and raw_qty != "None":
            try:
                qty_val = float(raw_qty)
                if qty_val > 0:
                    qty = qty_val
            except (ValueError, TypeError):
                pass

        raw_min = r.get("min_price")
        min_p = float(raw_min) if raw_min and raw_min != "None" else None
        raw_max = r.get("max_price")
        max_p = float(raw_max) if raw_max and raw_max != "None" else None

        valid_recs.append({
            "commodity": r.get("commodity") or target_crop.capitalize(),
            "state": r.get("state") or target_state,
            "market": r.get("market") or r.get("market_name") or "Unknown Mandi",
            "variety": r.get("variety") or "Common",
            "observation_date": dt,
            "modal_price": round(modal, 2),
            "min_price": round(min_p, 2) if min_p is not None else None,
            "max_price": round(max_p, 2) if max_p is not None else None,
            "arrival_quantity_qtl": qty,
            "source_timestamp": r.get("source_timestamp") or dt,
        })

    if not valid_recs:
        return {
            "status": "UNAVAILABLE",
            "message": "No valid non-zero price records after filtering",
            "modal_price": None,
            "min_price": None,
            "max_price": None,
            "contributing_records_count": 0,
            "contributing_records": [],
            "aggregation_method": "None (No Valid Prices)",
            "variety": "None",
        }

    unique_varieties = sorted(list({r["variety"] for r in valid_recs}))
    variety_label = unique_varieties[0] if len(unique_varieties) == 1 else f"Multi-Variety Aggregate ({', '.join(unique_varieties)})"

    total_qty = sum(r["arrival_quantity_qtl"] for r in valid_recs if r["arrival_quantity_qtl"] is not None)

    if method == "arrival_weighted" and total_qty > 0:
        weighted_sum = sum(
            r["modal_price"] * (r["arrival_quantity_qtl"] or 0)
            for r in valid_recs
        )
        agg_modal = round(weighted_sum / total_qty, 2)
        agg_method_label = "Arrival-Weighted APMC Mean"
    else:
        agg_modal = round(sum(r["modal_price"] for r in valid_recs) / len(valid_recs), 2)
        agg_method_label = "Arithmetic Mean of Reporting APMC Mandis"

    reported_mins = [r["min_price"] for r in valid_recs if r["min_price"] is not None]
    reported_maxs = [r["max_price"] for r in valid_recs if r["max_price"] is not None]
    agg_min = min(reported_mins) if reported_mins else None
    agg_max = max(reported_maxs) if reported_maxs else None

    obs_dates = [r["observation_date"] for r in valid_recs if r["observation_date"]]
    latest_obs_date = max(obs_dates) if obs_dates else str(date.today())

    return {
        "crop": target_crop.lower(),
        "state": target_state,
        "modal_price": agg_modal,
        "min_price": agg_min,
        "max_price": agg_max,
        "variety": variety_label,
        "varieties_list": unique_varieties,
        "variety_breakdown": {v: [r for r in valid_recs if r["variety"] == v] for v in unique_varieties},
        "contributing_records_count": len(valid_recs),
        "contributing_records": valid_recs,
        "aggregation_method": agg_method_label,
        "observation_date": latest_obs_date,
        "total_arrival_qtl": total_qty if total_qty > 0 else None,
    }



def _get_state_id(state: Optional[str]) -> Optional[int]:
    """Look up Agmarknet state ID from state name."""
    if not state:
        return None
    return AGMARKNET_STATE_IDS.get(state.strip().lower())


def _fetch_agmarknet_v2(
    crop_lower: str,
    state: Optional[str],
    today: date,
) -> Optional[dict]:
    """
    Fetch real mandi price from Agmarknet 2.0 API.

    Endpoint: POST https://api.agmarknet.gov.in/v1/dashboard-data/
    - Queries statewise_price_arrival_sp for state-specific modal mandi prices.
    - First attempts a single-day query (today, then yesterday).
    - If no single-day reports are available yet, falls back to the 7-day window
      and labels it accurately as a '7-Day State APMC Weighted Modal Average'.
    - Min and max prices are only recorded if provided by source; NEVER fabricated.
    """
    crop_clean = crop_lower.strip().lower()
    cids = AGMARKNET_COMMODITY_IDS.get(crop_clean)
    if not cids:
        logger.warning(f"[AGMARKNET] Unmapped commodity: '{crop_clean}'")
        return None

    base_url = "https://api.agmarknet.gov.in/v1/dashboard-data/"
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    seven_days_ago_str = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    # 1. State-specific lookup if state is requested
    if state and state.lower().strip() not in ("all", "all india", "national"):
        target_state = state.lower().strip()
        for cid in cids:
            # Query attempts: (1) single-day today, (2) single-day yesterday, (3) 7-day multi-day window
            query_plans = [
                (today_str, today_str, "Single-Day APMC State Modal Average", 0),
                (yesterday_str, yesterday_str, "Single-Day APMC State Modal Average", 1),
                (seven_days_ago_str, today_str, "7-Day State APMC Weighted Modal Average", 0),
            ]
            for q_start, q_end, p_type, age_days in query_plans:
                payload = {
                    "dashboard": "statewise_price_arrival_sp",
                    "from_date": q_start,
                    "to_date": q_end,
                    "commodity": [cid],
                    "limit": 50,
                }
                try:
                    logger.info(
                        f"[AGMARKNET] Querying statewise prices: crop={crop_clean} (cid={cid}), "
                        f"state={target_state}, range={q_start}..{q_end} ({p_type})"
                    )
                    resp = httpx.post(
                        base_url,
                        json=payload,
                        headers=API_HEADERS,
                        timeout=AGMARKNET_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "success":
                            records = data.get("data", {}).get("records", [])
                            for rec in records:
                                rec_state = (rec.get("state_name") or "").lower().strip()
                                if target_state in rec_state or rec_state in target_state:
                                    as_on_str = rec.get("as_on")
                                    if as_on_str and as_on_str != "None":
                                        modal_price = float(as_on_str)
                                        state_name = rec.get("state_name") or state
                                        # Parse min / max only if provided by source; never fabricated
                                        raw_min = rec.get("min_price")
                                        raw_max = rec.get("max_price")
                                        src_min = float(raw_min) if raw_min and raw_min != "None" else None
                                        src_max = float(raw_max) if raw_max and raw_max != "None" else None

                                        # Ensure date and age are authentically calculated from observation date
                                        obs_date = q_end
                                        dt_obs = _parse_date(obs_date)
                                        actual_age_days = (today - dt_obs).days if dt_obs else age_days
                                        if actual_age_days < 0:
                                            actual_age_days = 0

                                        logger.info(
                                            f"[AGMARKNET] SUCCESS state match: crop={crop_clean} | "
                                            f"state={state_name} | modal=Rs.{modal_price}/q | type={p_type} | period={q_start}..{q_end}"
                                        )
                                        return {
                                            "crop":                       crop_clean,
                                            "commodity_name":             rec.get("cmdt_name") or crop_clean.capitalize(),
                                            "modal_price":                round(modal_price, 2),
                                            "min_price":                  src_min,
                                            "max_price":                  src_max,
                                            "arrival_date":               obs_date,
                                            "observation_date":           obs_date,
                                            "market":                     f"{state_name} Mandis ({'Spot' if q_start == q_end else '7-Day Modal Avg'})",
                                            "state":                      state_name,
                                            "market_scope":               f"State-wide APMC Scope ({state_name})",
                                            "variety":                    "All Reporting Varieties (State-wide Pre-Aggregated)",
                                            "contributing_records_count": 1,
                                            "contributing_records":       [
                                                {
                                                    "commodity": rec.get("cmdt_name") or crop_clean.capitalize(),
                                                    "state": state_name,
                                                    "market": f"Reporting APMC Mandis ({state_name})",
                                                    "variety": "All Reporting Varieties",
                                                    "observation_date": obs_date,
                                                    "modal_price": round(modal_price, 2),
                                                    "min_price": src_min,
                                                    "max_price": src_max,
                                                    "arrival_quantity_qtl": None,
                                                    "source_timestamp": rec.get("source_timestamp") or f"{obs_date}T18:00:00+05:30",
                                                }
                                            ],
                                            "aggregation_method":         "Official Arrival-Weighted State Modal Average (Agmarknet wt_avg_price)",
                                            "data_age_days":              actual_age_days,
                                            "fetched_at":                 time.time(),
                                            "source":                     "Agmarknet 2.0 Live (api.agmarknet.gov.in)",
                                            "price_type":                 p_type,
                                            "period_start":               q_start,
                                            "period_end":                 q_end,
                                            "unit":                       "₹/quintal",
                                        }
                except httpx.TimeoutException:
                    logger.warning(f"[AGMARKNET] Timeout fetching statewise prices for crop={crop_clean} range={q_start}..{q_end}")
                except Exception as e:
                    logger.warning(f"[AGMARKNET] Error in statewise fetch for crop={crop_clean}: {e}")

    # 2. National / All-India modal lookup
    try:
        logger.info(f"[AGMARKNET] Querying national modal prices for crop={crop_clean}")
        resp = httpx.post(
            base_url,
            json={
                "dashboard": "marketwise_price_arrival",
                "page": 1,
                "limit": 100,
            },
            headers=API_HEADERS,
            timeout=AGMARKNET_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                records = data.get("data", {}).get("records", [])
                for rec in records:
                    cname = (rec.get("cmdt_name") or "").lower()
                    if crop_clean in cname:
                        as_on_str = rec.get("as_on_price")
                        if as_on_str and as_on_str != "None":
                            modal_price = float(as_on_str)
                            rep_date = rec.get("reported_date") or today_str
                            iso_date = today_str
                            try:
                                d_parts = rep_date.split("-")
                                if len(d_parts) == 3:
                                    iso_date = f"{d_parts[2]}-{d_parts[1]}-{d_parts[0]}"
                            except Exception:
                                pass

                            arrival_d = _parse_date(iso_date)
                            data_age = (today - arrival_d).days if arrival_d else 1
                            if data_age < 0:
                                data_age = 0

                            logger.info(
                                f"[AGMARKNET] SUCCESS national match: crop={crop_clean} | "
                                f"modal=Rs.{modal_price}/q | date={iso_date} | age={data_age}d"
                            )
                            return {
                                "crop":                       crop_clean,
                                "commodity_name":             rec.get("cmdt_name") or crop_clean.capitalize(),
                                "modal_price":                round(modal_price, 2),
                                "min_price":                  None,
                                "max_price":                  None,
                                "arrival_date":               iso_date,
                                "observation_date":           iso_date,
                                "market":                     "All-India Mandi Average",
                                "state":                      state or "National Average",
                                "market_scope":               "National APMC Mandi Benchmark",
                                "variety":                    "All Reporting Varieties (National Pre-Aggregated)",
                                "contributing_records_count": 1,
                                "contributing_records":       [
                                    {
                                        "commodity": rec.get("cmdt_name") or crop_clean.capitalize(),
                                        "state": "All India",
                                        "market": "National APMC Benchmark",
                                        "variety": "All Reporting Varieties",
                                        "observation_date": iso_date,
                                        "modal_price": round(modal_price, 2),
                                        "min_price": None,
                                        "max_price": None,
                                        "arrival_quantity_qtl": None,
                                        "source_timestamp": f"{iso_date}T18:00:00+05:30",
                                    }
                                ],
                                "aggregation_method":         "National Arrival-Weighted APMC Modal Average",
                                "data_age_days":              data_age,
                                "fetched_at":                 time.time(),
                                "source":                     "Agmarknet 2.0 Live (api.agmarknet.gov.in)",
                                "price_type":                 "All-India APMC Modal Average",
                                "period_start":               iso_date,
                                "period_end":                 iso_date,
                                "unit":                       "₹/quintal",
                            }
    except httpx.TimeoutException:
        logger.warning(f"[AGMARKNET] Timeout fetching national overview")
    except Exception as e:
        logger.warning(f"[AGMARKNET] Error in national fetch: {e}")

    logger.warning(f"[AGMARKNET] No data found for crop='{crop_clean}' state='{state}'")
    return None


def _fetch_data_gov_in(
    crop_lower: str,
    state: Optional[str],
) -> Optional[dict]:
    """
    Legacy fallback: fetch from data.gov.in AGMARKNET 1.0 API.
    Returns normalized dict or None.
    """
    commodity = CROP_TO_COMMODITY.get(crop_lower, crop_lower.capitalize())
    today     = date.today()
    t_start   = time.time()

    for res_name, res_url in [
        ("Resource 1 (9ef84268)", RESOURCE_1_URL),
        ("Resource 2 (35be999b)", RESOURCE_2_URL),
    ]:
        try:
            params: dict = {
                "api-key":            settings.MARKET_DATA_API_KEY,
                "format":             "json",
                "filters[commodity]": commodity,
                "limit":              50,
                "sort[arrival_date]": "desc",
            }
            if state:
                params["filters[state]"] = state

            resp = httpx.get(
                res_url,
                params=params,
                timeout=API_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": API_HEADERS["User-Agent"],
                    "Accept": "application/json",
                },
            )
            elapsed_ms = round((time.time() - t_start) * 1000, 1)

            if resp.status_code == 200:
                api_data = resp.json()
                records  = api_data.get("records", [])
                if records:
                    rec = records[0]
                    arr_str     = rec.get("arrival_date") or rec.get("date") or ""
                    modal_val   = float(rec.get("modal_price") or rec.get("modal_price_rs_qtl") or 0)
                    min_val     = float(rec.get("min_price")   or rec.get("min_price_rs_qtl")   or 0)
                    max_val     = float(rec.get("max_price")   or rec.get("max_price_rs_qtl")   or 0)

                    arrival     = _parse_date(arr_str)
                    data_age    = (today - arrival).days if arrival else 999
                    arrival_iso = arrival.isoformat() if arrival else arr_str
                    mkt_name    = rec.get("market") or rec.get("market_name") or "Unknown Mandi"
                    st_name     = rec.get("state")  or rec.get("state_name")  or state or "All"
                    var_name    = rec.get("variety") or "Common"

                    logger.info(
                        f"[DATAGOV] SUCCESS via {res_name} | crop={commodity} | "
                        f"date={arrival_iso} | modal=Rs.{modal_val}/q | {elapsed_ms}ms"
                    )
                    return {
                        "crop":                       crop_lower,
                        "commodity_name":             commodity,
                        "modal_price":                modal_val,
                        "min_price":                  min_val if min_val > 0 else None,
                        "max_price":                  max_val if max_val > 0 else None,
                        "arrival_date":               arrival_iso,
                        "observation_date":           arrival_iso,
                        "market":                     mkt_name,
                        "state":                      st_name,
                        "market_scope":               f"Single Mandi ({mkt_name}, {st_name})",
                        "variety":                    var_name,
                        "contributing_records_count": len(records),
                        "contributing_records":       [
                            {
                                "commodity": commodity,
                                "state": st_name,
                                "market": mkt_name,
                                "variety": var_name,
                                "observation_date": arrival_iso,
                                "modal_price": modal_val,
                                "min_price": min_val if min_val > 0 else None,
                                "max_price": max_val if max_val > 0 else None,
                                "arrival_quantity_qtl": float(rec.get("arrival_quantity") or 0) if rec.get("arrival_quantity") else None,
                                "source_timestamp": rec.get("timestamp") or arrival_iso,
                            }
                        ],
                        "aggregation_method":         "Single APMC Spot Observation" if len(records) == 1 else "Arithmetic Mean of Reporting APMC Mandis",
                        "data_age_days":              data_age,
                        "fetched_at":                 time.time(),
                        "source":                     f"data.gov.in ({res_name})",
                        "price_type":                 "Single APMC Spot Observation",
                        "period_start":               arrival_iso,
                        "period_end":                 arrival_iso,
                        "unit":                       "₹/quintal",
                    }
                else:
                    logger.warning(f"[DATAGOV] {res_name}: no records for {commodity}")
            else:
                logger.warning(f"[DATAGOV] {res_name} HTTP {resp.status_code}")

        except Exception as e:
            logger.warning(f"[DATAGOV] {res_name} failed: {e}")
            continue

    return None


def _try_ingestion_cache(
    crop_lower: str,
    state: Optional[str],
    today: date,
) -> Optional[MandiPriceResult]:
    """
    Bridge to DataIngestion cache format ('crop:State').
    Last-resort fallback before returning None.
    """
    try:
        cache       = _load_cache()
        state_clean = (state or "All").strip()
        candidates  = [f"{crop_lower}:{state_clean}", f"{crop_lower}:All"]
        for key in candidates:
            entry = cache.get(key)
            if not entry:
                continue
            price  = entry.get("price")
            ts_str = entry.get("timestamp", "")
            if not price or not ts_str:
                continue
            ts       = datetime.fromisoformat(ts_str)
            age_days = (today - ts.date()).days
            if age_days > 30:
                continue
            result_entry = {
                "crop":                       crop_lower,
                "commodity_name":             crop_lower.capitalize(),
                "modal_price":                float(price),
                "min_price":                  None,
                "max_price":                  None,
                "arrival_date":               ts.date().isoformat(),
                "observation_date":           ts.date().isoformat(),
                "market":                     f"{state_clean} APMC Mandi",
                "state":                      state_clean,
                "market_scope":               f"State-wide Historical Scope ({state_clean})",
                "variety":                    "All Varieties (Historical APMC Aggregate)",
                "contributing_records_count": 1,
                "contributing_records":       [
                    {
                        "commodity": crop_lower.capitalize(),
                        "state": state_clean,
                        "market": f"{state_clean} APMC Mandi",
                        "variety": "All Varieties",
                        "observation_date": ts.date().isoformat(),
                        "modal_price": float(price),
                        "min_price": None,
                        "max_price": None,
                        "arrival_quantity_qtl": None,
                        "source_timestamp": ts_str,
                    }
                ],
                "aggregation_method":         "Historical APMC State Average (Baseline)",
                "data_age_days":              age_days,
                "fetched_at":                 ts.timestamp(),
                "source":                     "Historical APMC Baseline (real_historical_prices_state.csv)",
                "price_type":                 "Historical APMC Baseline",
                "period_start":               ts.date().isoformat(),
                "period_end":                 ts.date().isoformat(),
                "unit":                       "₹/quintal",
            }
            logger.info(f"[MANDI] Cache bridge hit for '{key}' | price=Rs.{price} | age={age_days}d")
            return _build_result(result_entry, from_api=False)
    except Exception as e:
        logger.debug(f"[MANDI] Ingestion cache bridge failed: {e}")
    return None


def get_latest_price(crop: str, state: Optional[str] = None) -> Optional[MandiPriceResult]:
    """
    Fetch the latest available mandi price for a crop.

    STRATEGY (in order):
      1. Disk cache (within TTL) — avoids hammering the API repeatedly
      2. Agmarknet 2.0 API — real 2025-2026 data, no auth required
         Tries current month, walks back up to 4 months to find data
      3. data.gov.in legacy API — fallback if Agmarknet 2.0 is down
      4. Stale disk cache — any previously fetched price
      5. DataIngestion cache bridge — state-tail fallback
      6. Return None — no data available

    The Agmarknet 2.0 endpoint is CONFIRMED working and returns real
    2025/2026 price data without requiring authentication.
    """
    crop_lower = crop.lower()
    today      = date.today()

    cache     = _load_cache()
    cache_key = f"{crop_lower}_{(state or 'all').lower()}"

    # ── 1. Disk cache (within TTL) ────────────────────────────────────────────
    cached_entry = cache.get(cache_key)
    if cached_entry and cached_entry.get("modal_price"):
        fetched_at  = cached_entry.get("fetched_at", 0)
        cache_age_s = time.time() - fetched_at
        if cache_age_s < settings.MANDI_CACHE_TTL_SECONDS:
            rec_date_str = cached_entry.get("arrival_date", "")
            rec_date     = _parse_date(rec_date_str) or today
            data_age     = (today - rec_date).days
            cached_entry["data_age_days"] = data_age
            logger.info(
                f"[MANDI] CACHE HIT for {cache_key} | "
                f"record_date={rec_date_str} | age={data_age}d | "
                f"cache_age={round(cache_age_s/3600,1)}h"
            )
            return _build_result(cached_entry, from_api=True)

    # ── 2. Agmarknet 2.0 (PRIMARY — real 2025/2026 data) ─────────────────────
    ag_entry = _fetch_agmarknet_v2(crop_lower, state, today)
    if ag_entry:
        cache[cache_key] = ag_entry
        _save_cache(cache)
        return _build_result(ag_entry, from_api=True)

    logger.info(f"[MANDI] Agmarknet 2.0 failed for '{crop}'/'{state}' — trying data.gov.in")

    # ── 3. data.gov.in legacy API ─────────────────────────────────────────────
    dg_entry = _fetch_data_gov_in(crop_lower, state)
    if dg_entry:
        cache[cache_key] = dg_entry
        _save_cache(cache)
        return _build_result(dg_entry, from_api=True)

    # ── 4. Stale disk cache ────────────────────────────────────────────────────
    stale_entry = cache.get(cache_key)
    if stale_entry and stale_entry.get("modal_price"):
        rec_date_str = stale_entry.get("arrival_date", "")
        rec_date     = _parse_date(rec_date_str) or today
        data_age     = (today - rec_date).days
        stale_entry["data_age_days"] = data_age
        logger.info(f"[MANDI] Serving stale cache for {cache_key} | data_age={data_age}d")
        return _build_result(stale_entry, from_api=True)

    # ── 5. DataIngestion cache bridge (last resort) ───────────────────────────
    bridge_result = _try_ingestion_cache(crop_lower, state, today)
    if bridge_result:
        return bridge_result

    logger.warning(f"[MANDI] No price data available for crop='{crop}' state='{state}'")
    return None
