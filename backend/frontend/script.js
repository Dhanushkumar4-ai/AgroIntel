/**
 * AgroIntel AI — Frontend Application Script v4.1
 *
 * KEY RULES:
 *   - Crop Recommendation: State + District (district IS required)
 *   - Price Prediction:    Crop + State    (district is NOT used)
 *   - Price Prediction calls /api/predict ONLY — NOT /api/phase6/recommend
 *   - renderPredResults() uses predData ONLY (no p6Data cross-contamination)
 *   - Graph shows actual observation date + 30-day forecast series
 *   - SELL/HOLD/WAIT derived from real values + freshness + model uncertainty
 */

// ─── Global State ───────────────────────────────────────────────────────────
let activeChart = null;

/** Single source of truth: { "Karnataka": ["Bagalkot", "Ballari", ...], ... } */
let indianDistrictsMap = {};

/** Supported states list (from indianDistrictsMap keys after loading) */
let supportedStates = [];

// ─── App Initialization ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    await loadIndianDistricts();   // Must load first before populating selects
    populateStateSelects();
    setDefaultSelections();
    checkHealth();
    setupEventListeners();
}

function setupEventListeners() {
    const markPredStale = () => {
        const el = document.getElementById("predResults");
        if (el && el.innerHTML.trim()) {
            el.classList.add("results-stale");
        }
    };
    document.getElementById("predState")?.addEventListener("change", markPredStale);
    document.getElementById("predCrop")?.addEventListener("change", markPredStale);
    document.getElementById("predHorizon")?.addEventListener("change", markPredStale);
}

// ─── Load indian_districts.json (Single Load, Single Source) ─────────────────
async function loadIndianDistricts() {
    try {
        const res = await fetch("/indian_districts.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        indianDistrictsMap = {};
        if (data && Array.isArray(data.states)) {
            data.states.forEach(stateObj => {
                if (stateObj.state && Array.isArray(stateObj.districts)) {
                    indianDistrictsMap[stateObj.state] = stateObj.districts.slice().sort();
                }
            });
        }

        supportedStates = Object.keys(indianDistrictsMap).sort();
        console.log(`[AgroIntel] Loaded ${supportedStates.length} states, ${Object.values(indianDistrictsMap).reduce((a,b)=>a+b.length,0)} districts.`);

    } catch (err) {
        console.error("[AgroIntel] Failed to load indian_districts.json:", err);
        showToast("Unable to load districts. Please refresh the page.", "error");
        indianDistrictsMap = {};
        supportedStates = [];
    }
}

// ─── Populate State Selects ──────────────────────────────────────────────────
function populateStateSelects() {
    // State select dropdowns across Recommendation, Prediction, and Advisory
    const stateSelectIds = ["recState", "predState", "advState"];
    stateSelectIds.forEach(id => {
        const sel = document.getElementById(id);
        if (!sel) return;
        sel.innerHTML = '<option value="">Select State</option>';

        if (supportedStates.length === 0) {
            sel.innerHTML = '<option value="">Unable to load states</option>';
            return;
        }

        supportedStates.forEach(state => {
            const opt = document.createElement("option");
            opt.value = state;
            opt.textContent = state;
            sel.appendChild(opt);
        });
    });
}

// ─── State → District: Crop Recommendation and Advisory ONLY ─────────────────
function onStateChange(stateSelectId, districtSelectId) {
    const stateEl = document.getElementById(stateSelectId);
    const distEl  = document.getElementById(districtSelectId);
    if (!stateEl || !distEl) return;

    const selectedState = stateEl.value;
    distEl.innerHTML = '<option value="">Select District</option>';
    if (!selectedState) return;

    const districts = indianDistrictsMap[selectedState];
    if (!districts || districts.length === 0) {
        distEl.innerHTML = '<option value="">No districts found</option>';
        console.warn(`[AgroIntel] No districts found for state: "${selectedState}"`);
        return;
    }

    districts.forEach(district => {
        const opt = document.createElement("option");
        opt.value  = district;
        opt.textContent = district;
        distEl.appendChild(opt);
    });

    if (districts.length > 0) {
        distEl.value = districts[0];
    }
}

// ─── Default Selections on Load ─────────────────────────────────────────────
function setDefaultSelections() {
    const defaultState = "Maharashtra";

    // Crop Recommendation — needs state + district
    ["recState", "advState"].forEach(id => {
        const distId = id === "recState" ? "recDistrict" : "advDistrict";
        const sel = document.getElementById(id);
        if (!sel) return;
        if (supportedStates.includes(defaultState)) {
            sel.value = defaultState;
            onStateChange(id, distId);
        } else if (supportedStates.length > 0) {
            sel.value = supportedStates[0];
            onStateChange(id, distId);
        }
    });

    // Price Prediction — needs state only (NO district)
    const predStateSel = document.getElementById("predState");
    if (predStateSel) {
        predStateSel.value = supportedStates.includes(defaultState) ? defaultState : (supportedStates[0] || "");
    }
}

// ─── Page Navigation ─────────────────────────────────────────────────────────
function showPage(pageId) {
    let cleanId = pageId.replace(/^(view-|page-)/, "");
    if (cleanId === "news") {
        cleanId = "landing";
    }

    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    const target = document.getElementById(`page-${cleanId}`) || document.getElementById(`view-${cleanId}`);
    if (target) {
        target.classList.add("active");
    } else {
        console.warn(`[AgroIntel] Navigation target not found: "page-${cleanId}" or "view-${cleanId}"`);
    }

    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("nav-active"));
    const navMap = { recommendation: "navRecommend", prediction: "navPrediction", advisory: "navAdvisory" };
    if (navMap[cleanId]) document.getElementById(navMap[cleanId])?.classList.add("nav-active");

    window.scrollTo({ top: 0, behavior: "smooth" });
}

const showView   = showPage;
const navigate   = showPage;
const switchPage = showPage;
const openPage   = showPage;

// ─── Theme Toggle ────────────────────────────────────────────────────────────
function toggleTheme() {
    const html = document.documentElement;
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    document.getElementById("themeIcon").textContent = next === "dark" ? "dark_mode" : "light_mode";
}

// ─── Health Check ────────────────────────────────────────────────────────────
async function checkHealth() {
    const badge = document.getElementById("sysBadge");
    const text  = document.getElementById("sysText");
    try {
        const res = await fetch("/health");
        if (res.ok) {
            const data = await res.json();
            const ok = data.status === "healthy";
            text.textContent = ok ? "System Ready" : "System Degraded";
            badge.classList.toggle("badge-warn", !ok);
        } else {
            text.textContent = "API Unavailable";
            badge.classList.add("badge-warn");
        }
    } catch {
        text.textContent = "Offline";
        badge.classList.add("badge-warn");
    }
}

// ─── Crop Recommendation ─────────────────────────────────────────────────────
async function submitCropRec(event) {
    event.preventDefault();
    const btn   = document.getElementById("btnRec");
    const spin  = document.getElementById("recSpin");
    setLoading(btn, spin, true);

    const state    = document.getElementById("recState").value;
    const district = document.getElementById("recDistrict").value;
    const season   = document.getElementById("recSeason").value;

    if (!state || !district) {
        showToast("Please select both a State and a District.", "error");
        setLoading(btn, spin, false);
        return;
    }

    const payload = { state, district, season };

    const n        = parseFloat(document.getElementById("recN").value);
    const p        = parseFloat(document.getElementById("recP").value);
    const k        = parseFloat(document.getElementById("recK").value);
    const ph       = parseFloat(document.getElementById("recPh").value);
    const temp     = parseFloat(document.getElementById("recTemp")?.value);
    const humidity = parseFloat(document.getElementById("recHumidity")?.value);
    const rainfall = parseFloat(document.getElementById("recRainfall")?.value);
    const prev     = document.getElementById("recPrevCrop")?.value?.trim();

    if (!isNaN(n))        payload.n = n;
    if (!isNaN(p))        payload.p = p;
    if (!isNaN(k))        payload.k = k;
    if (!isNaN(ph))       payload.soil_ph = ph;
    if (!isNaN(temp))     payload.temperature = temp;
    if (!isNaN(humidity)) payload.humidity = humidity;
    if (!isNaN(rainfall)) payload.rainfall = rainfall;
    if (prev)             payload.previous_crop = prev;

    try {
        const res = await fetch("/api/phase6/recommend", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Request failed" }));
            throw new Error(err.detail || "Recommendation failed. Please try again.");
        }
        const data = await res.json();
        renderRecResults(data);
    } catch (err) {
        showToast(err.message, "error");
        document.getElementById("recResults").innerHTML = errorCard(err.message);
    } finally {
        setLoading(btn, spin, false);
    }
}

// ─── Season Display Mapping & Sanitization ──────────────────────────────────
const SEASON_DISPLAY_MAP = {
    "Kharif": "Rainy Season",
    "Rabi": "Winter Season",
    "Zaid": "Summer Season",
    "Summer": "Summer Season",
    "Rainy": "Rainy Season",
    "Winter": "Winter Season",
    "Whole Year": "Whole Year / Perennial"
};

function formatSeasonLabel(season) {
    if (!season) return "";
    return SEASON_DISPLAY_MAP[season] || season;
}

function sanitizeSeasonText(text) {
    if (!text || typeof text !== 'string') return text;
    return text
        .replace(/\bKharif\b/gi, "Rainy Season")
        .replace(/\bRabi\b/gi, "Winter Season")
        .replace(/\bZaid\b/gi, "Summer Season");
}

function getWaterNeed(cropName, rec) {
    const c = (cropName || '').toLowerCase();
    const mm = (rec.water_requirement_mm || rec.water_notes || '').toLowerCase();
    if (mm.includes('high') || mm.includes('1000') || mm.includes('1200') || mm.includes('1500') || mm.includes('2000')) {
        return 'High';
    }
    if (mm.includes('low') || mm.includes('300') || mm.includes('200') || mm.includes('400')) {
        return 'Low';
    }
    const highWaterCrops = ['rice', 'paddy', 'sugarcane', 'banana', 'jute'];
    const lowWaterCrops = ['jowar', 'bajra', 'ragi', 'millet', 'gram', 'chickpea', 'moong', 'urad', 'horse-gram', 'moth', 'mustard', 'lentil', 'castor'];
    if (highWaterCrops.some(h => c.includes(h))) return 'High';
    if (lowWaterCrops.some(l => c.includes(l))) return 'Low';
    return 'Medium';
}

function getCleanReasons(rec) {
    let reasons = [];
    if (Array.isArray(rec.reasons) && rec.reasons.length > 0) {
        reasons = rec.reasons.slice();
    } else if (rec.explanation) {
        reasons = [rec.explanation];
    } else if (rec.nlp_explanation?.why_recommended) {
        reasons = [rec.nlp_explanation.why_recommended];
    } else if (rec.crop_information?.why_grown) {
        reasons = [rec.crop_information.why_grown];
    }

    const cleaned = reasons
        .map(r => sanitizeSeasonText(r)
            .replace(/\bSignature Local Crop\b/gi, 'Historical local crop')
            .replace(/\bguaranteed\b/gi, 'favorable')
            .replace(/\bbest crop\b/gi, 'suitable crop')
            .trim()
        )
        .filter(r => r.length > 0);

    if (cleaned.length === 0) {
        cleaned.push('Agronomically well-suited to local soil texture, rainfall and temperature patterns.');
    }
    return cleaned.slice(0, 2);
}

function renderRecResults(data) {
    const loc  = data.location || {};
    const districtName = loc.district || data.district || '';
    const stateName = loc.state || data.state || '';
    const displaySeason = formatSeasonLabel(data.season);
    const rawRecs = data.recommendations || [];

    if (rawRecs.length === 0) {
        const msg = sanitizeSeasonText(data.message || "No suitable candidate crops found for the specified conditions.");
        document.getElementById("recResults").innerHTML = `
            <div class="placeholder-card glass-card">
                <span class="material-symbols-rounded ph-icon-sym">grass</span>
                <h3>No Crops Recommended</h3>
                <p>${msg}</p>
                <p style="font-size:0.78rem;opacity:0.65;margin-top:8px">Try adjusting seasonal or soil parameters.</p>
            </div>`;
        return;
    }

    // 1. Deduplicate Paddy and Rice & filter duplicate entries
    const seenCrops = new Set();
    const dedupedRecs = [];
    for (const rec of rawRecs) {
        let cropName = (rec.crop || '').trim();
        if (cropName.toLowerCase() === 'paddy') {
            cropName = 'Rice';
        }
        const key = cropName.toLowerCase();
        if (seenCrops.has(key)) {
            continue;
        }
        seenCrops.add(key);
        rec.displayCrop = cropName;
        dedupedRecs.push(rec);
    }

    const recs = dedupedRecs.slice(0, 5);
    const rankColors = ["first-rank", "second-rank", "third-rank", "", ""];

    const recHtml = recs.map((rec, i) => {
        const rank = i + 1;
        const cropTitle = rec.displayCrop || rec.crop;

        // Suitability score (calibrated final score or agronomic score)
        let suitabilityScore = null;
        if (typeof rec.final_score === 'number') {
            suitabilityScore = Math.round(rec.final_score);
        } else if (typeof rec.agronomic_suitability_score === 'number') {
            suitabilityScore = Math.round(rec.agronomic_suitability_score);
        }
        const suitabilityDisplay = suitabilityScore !== null ? `${suitabilityScore}%` : 'High Match';

        // Water requirement: Low, Medium, High
        const waterNeed = getWaterNeed(cropTitle, rec);
        const waterColor = waterNeed === 'Low' ? '#38bdf8' : waterNeed === 'Medium' ? '#a3e635' : '#34d399';

        // 1-2 Clean, evidence-based reasons
        const cleanReasons = getCleanReasons(rec);
        const reasonsHtml = cleanReasons.map(r => `
            <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:4px">
                <span style="color:#22c55e;font-weight:700;line-height:1.4">✓</span>
                <span style="color:#cbd5e1;line-height:1.4">${r}</span>
            </div>
        `).join('');

        // Local production evidence
        const dev = rec.district_cultivation_evidence;
        let evidenceHtml = '';
        if (dev && dev.geographic_level === 'DISTRICT_LEVEL' && (dev.annual_average_area_ha || dev.annual_average_production_tonnes || dev.season_reported_annual_area_ha)) {
            const area = dev.annual_average_area_ha ? `${Math.round(dev.annual_average_area_ha).toLocaleString('en-IN')} ha/yr` : '';
            const prod = dev.annual_average_production_tonnes ? `${Math.round(dev.annual_average_production_tonnes).toLocaleString('en-IN')} tonnes/yr` : '';
            const period = dev.latest_verified_year ? `DES records through ${dev.latest_verified_year}` : (dev.historical_period_label || 'Verified records');
            const parts = [area, prod].filter(Boolean).join(' • ');
            evidenceHtml = `
                <div style="font-size:0.75rem;color:#38bdf8;display:flex;align-items:center;gap:6px;margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06)">
                    <span class="material-symbols-rounded" style="font-size:1rem;color:#38bdf8">verified</span>
                    <span><strong>District Production Evidence:</strong> ${parts} (${period})</span>
                </div>`;
        } else {
            evidenceHtml = `
                <div style="font-size:0.75rem;color:#94a3b8;display:flex;align-items:center;gap:6px;margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06)">
                    <span class="material-symbols-rounded" style="font-size:1rem;color:#64748b">info</span>
                    <span>Local production data unavailable</span>
                </div>`;
        }

        return `
        <div class="rec-card glass-card ${rankColors[i] || ''}" style="margin-bottom:14px;padding:16px 20px">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap">
                <div style="display:flex;align-items:center;gap:12px">
                    <span class="rec-rank" style="font-size:1.1rem;font-weight:800;color:#a3e635;background:rgba(163,230,53,0.12);border:1px solid rgba(163,230,53,0.3);padding:3px 10px;border-radius:18px">#${rank}</span>
                    <div>
                        <h3 class="rec-crop-name" style="margin:0;font-size:1.25rem;color:#f8fafc">${cropTitle}</h3>
                        <div style="font-size:0.78rem;color:#94a3b8;margin-top:2px">
                            Season: <strong style="color:#e2e8f0">${displaySeason}</strong> • 
                            Water Need: <strong style="color:${waterColor}">${waterNeed}</strong>
                        </div>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em">Suitability</div>
                    <div style="font-size:1.25rem;font-weight:800;color:#a3e635">${suitabilityDisplay}</div>
                </div>
            </div>

            <!-- Why Recommended -->
            <div style="margin-top:12px;padding:10px 12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:6px;font-size:0.82rem">
                <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em;font-weight:700;margin-bottom:6px">Why Recommended:</div>
                ${reasonsHtml}
            </div>

            <!-- Local Production Evidence -->
            ${evidenceHtml}
        </div>`;
    }).join('');

    document.getElementById("recResults").innerHTML = `
        <div class="rec-header-row" style="margin-bottom:16px">
            <h3 style="margin:0;font-size:1.25rem">Recommended Crops for <strong>${districtName}</strong>, ${stateName}</h3>
            <p class="rec-season-tag" style="font-size:0.8rem;color:#94a3b8;margin-top:4px">Growing Season: ${displaySeason} • Ranked by agronomic and climate suitability</p>
        </div>
        ${recHtml}`;
}



// ─── Price Prediction ─────────────────────────────────────────────────────────
// IMPORTANT: This function calls ONLY /api/predict — never /api/phase6/recommend
// District is NOT used in price prediction.
let currentPredRequestId = 0;

async function submitPricePred(event) {
    if (event && event.preventDefault) event.preventDefault();
    const btn  = document.getElementById("btnPred");
    const spin = document.getElementById("predSpin");
    if (btn && spin) setLoading(btn, spin, true);

    const cropSelect    = document.getElementById("predCrop");
    const stateSelect   = document.getElementById("predState");
    const horizonSelect = document.getElementById("predHorizon");

    const crop    = cropSelect ? cropSelect.value : "wheat";
    const state   = stateSelect ? stateSelect.value : "";
    const horizon = horizonSelect ? horizonSelect.value : "30";

    if (!crop) {
        showToast("Please select a crop.", "error");
        if (btn && spin) setLoading(btn, spin, false);
        return;
    }

    const requestId = ++currentPredRequestId;
    const predResultsEl = document.getElementById("predResults");
    if (predResultsEl && predResultsEl.innerHTML.trim()) {
        predResultsEl.classList.add("results-stale");
    }

    try {
        let predUrl = `/api/predict?crop=${encodeURIComponent(crop)}&horizon_days=${horizon}`;
        if (state) predUrl += `&state=${encodeURIComponent(state)}`;

        const predRes = await fetch(predUrl);
        if (!predRes.ok) {
            const err = await predRes.json().catch(() => ({ detail: "Prediction request failed." }));
            throw new Error(err.detail || "Price forecast could not be retrieved.");
        }
        const predData = await predRes.json();

        if (requestId !== currentPredRequestId) {
            return;
        }

        renderPredResults(predData, crop, parseInt(horizon), state);
        if (predResultsEl) predResultsEl.classList.remove("results-stale");

    } catch (err) {
        if (requestId === currentPredRequestId) {
            showToast(err.message, "error");
            if (predResultsEl) {
                predResultsEl.classList.remove("results-stale");
                predResultsEl.innerHTML = errorCard(err.message);
            }
        }
    } finally {
        if (requestId === currentPredRequestId && btn && spin) {
            setLoading(btn, spin, false);
        }
    }
}

window.selectHorizon = function(newHorizon) {
    const horizonSelect = document.getElementById("predHorizon");
    if (horizonSelect) {
        horizonSelect.value = newHorizon;
    }
    submitPricePred();
};

/**
 * renderPredResults — Farmer-Friendly Clean Forecast Results
 * Preserves ONLY:
 * 1. Farmer-friendly 3-stat summary (Current Mandi Price, Expected Price, Expected Change)
 * 2. Horizon tab selector & simplified, obvious-boundary Price Chart
 * 3. Official Agmarknet Mandi & Variety Breakdown (preserved exactly)
 */
function renderPredResults(predData, crop, horizon, inputState) {
    const isPredAvailable = predData.available === true &&
                            typeof predData.predicted_price === 'number';

    const curPriceNum  = typeof predData.current_price === 'number'   ? predData.current_price   : null;
    const predPriceNum = typeof predData.predicted_price === 'number' ? predData.predicted_price : null;
    const isMandiAvailable = typeof curPriceNum === 'number';

    const obsDate = predData.observation_date || predData.price_cached_time?.split('T')[0] || '—';
    const mktName = predData.market_name || predData.market?.market_name || predData.market || inputState || '—';

    // State context resolution
    const activeState = (inputState && inputState.toLowerCase() !== 'all')
        ? inputState
        : (predData.state && predData.state.toLowerCase() !== 'all' ? predData.state : 'National Benchmark');
    const isNationalScope = !inputState || inputState.toLowerCase() === 'all' || activeState.toLowerCase().includes('national');
    const stateLabel = isNationalScope ? 'National Benchmark' : activeState;

    const pt = predData.price_transparency || predData.market?.price_transparency || {};
    const ptObsDate = pt.observation_date || obsDate || '—';

    // Multi-tier prices resolution
    const farmGatePrice = (typeof pt.farm_gate_modal_price === 'number')
        ? pt.farm_gate_modal_price
        : ((typeof predData.farm_gate_modal_price === 'number')
            ? predData.farm_gate_modal_price
            : ((typeof predData.market?.farm_gate_modal_price === 'number') ? predData.market.farm_gate_modal_price : curPriceNum));

    const displaySpotNum = farmGatePrice != null ? farmGatePrice : curPriceNum;

    const diffVal = (displaySpotNum != null && predPriceNum != null) ? (predPriceNum - displaySpotNum) : null;
    const changePct = (displaySpotNum != null && predPriceNum != null && displaySpotNum > 0)
        ? (((predPriceNum - displaySpotNum) / displaySpotNum) * 100.0)
        : null;

    // Filter district records strictly by requested state
    const rawDistrictRecs = (Array.isArray(pt.district_breakdown) && pt.district_breakdown.length > 0)
        ? pt.district_breakdown
        : ((Array.isArray(predData.district_breakdown) && predData.district_breakdown.length > 0)
            ? predData.district_breakdown
            : ((Array.isArray(pt.contributing_records) && pt.contributing_records.length > 0)
                ? pt.contributing_records
                : (Array.isArray(predData.contributing_records) ? predData.contributing_records : [])));

    const districtRecs = rawDistrictRecs.filter(r => {
        if (isNationalScope) return true;
        const rState = (r.state || '').trim().toLowerCase();
        const reqStateLower = activeState.trim().toLowerCase();
        return !rState || rState === reqStateLower || rState === 'all';
    });

    const districtRows = districtRecs.map(r => `
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
            <td style="padding:8px 10px;font-weight:600;color:#f8fafc">${r.district || r.state || mktName}</td>
            <td style="padding:8px 10px;color:#cbd5e1">${r.market || 'APMC Yard'}</td>
            <td style="padding:8px 10px;color:#38bdf8">${r.variety || 'Standard'}</td>
            <td style="padding:8px 10px;color:#34d399;font-weight:700">₹${Math.round(r.modal_price || r.price || 0).toLocaleString('en-IN')}</td>
            <td style="padding:8px 10px;color:#94a3b8">${r.arrival_tonnes != null ? r.arrival_tonnes + ' MT' : (r.arrival_quantity_qtl != null ? r.arrival_quantity_qtl + ' q' : '—')}</td>
            <td style="padding:8px 10px;color:#94a3b8">${formatDate(r.observation_date || ptObsDate)}</td>
            <td style="padding:8px 10px"><span style="font-size:0.68rem;padding:2px 6px;border-radius:4px;background:${(r.market_type || '').includes('Terminal') ? 'rgba(245,158,11,0.15);color:#fbbf24' : 'rgba(34,197,94,0.15);color:#4ade80'}">${r.market_type || 'Production APMC'}</span></td>
        </tr>
    `).join('');

    const cropDisplay = capitalize(crop);

    const recAction = predData.recommendation || predData.advisory?.decision || (isPredAvailable ? 'WAIT' : null);
    const recReason = predData.recommendation_reason || predData.advisory?.reason || 
        (recAction === 'HOLD' ? 'The forecast suggests the price may rise enough to justify waiting after estimated holding costs. Future prices are not guaranteed.' :
         recAction === 'SELL' ? 'The expected price increase may not cover the estimated cost of holding the crop, so selling now may be preferable under the configured rules.' :
         'The available price data is stale or insufficient for a reliable recommendation. Check again when updated market data is available.');

    const storageCostPct = typeof predData.storage_cost_pct === 'number' 
        ? predData.storage_cost_pct 
        : (typeof predData.decision_score?.estimated_storage_cost_percent === 'number' 
            ? predData.decision_score.estimated_storage_cost_percent 
            : null);

    const recColor = recAction === 'HOLD' ? '#4ade80' : (recAction === 'SELL' ? '#f87171' : '#fbbf24');
    const recBg = recAction === 'HOLD' ? 'rgba(34,197,94,0.18)' : (recAction === 'SELL' ? 'rgba(239,68,68,0.18)' : 'rgba(245,158,11,0.18)');
    const recBorder = recAction === 'HOLD' ? '#22c55e' : (recAction === 'SELL' ? '#ef4444' : '#f59e0b');

    // ── Assemble Simplified Farmer-Friendly HTML ─────────────────────────────
    const html = `
        <!-- 1. Farmer-Friendly Price Summary Card -->
        <div class="glass-card" style="padding:18px 20px;margin-bottom:16px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px">
                <div>
                    <h3 style="margin:0;font-size:1.3rem">${cropDisplay} Price Forecast</h3>
                    <div style="font-size:0.8rem;color:#94a3b8;margin-top:3px">
                        📍 State: <strong style="color:#f8fafc">${stateLabel}</strong>
                        ${obsDate !== '—' ? ` • Last Observed Date: <strong style="color:#f8fafc">${formatDate(obsDate)}</strong>` : ''}
                    </div>
                </div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:14px">
                <!-- Current Mandi Price -->
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:14px">
                    <div style="font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Current Mandi Price</div>
                    ${displaySpotNum != null ? `
                    <div style="font-size:1.6rem;font-weight:800;color:#34d399;margin:4px 0">₹${Math.round(displaySpotNum).toLocaleString('en-IN')} <span style="font-size:0.8rem;font-weight:500;color:#94a3b8">/ quintal</span></div>
                    <div style="font-size:0.72rem;color:#94a3b8">Observed: ${formatDate(obsDate)}</div>
                    ` : `
                    <div style="font-size:1.1rem;font-weight:700;color:#94a3b8;margin:6px 0;font-style:italic">Price Unavailable</div>
                    `}
                </div>

                <!-- Expected Price at Selected Horizon -->
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:14px">
                    <div style="font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Expected Price (${horizon} Days)</div>
                    ${isPredAvailable ? `
                    <div style="font-size:1.6rem;font-weight:800;color:#a78bfa;margin:4px 0">₹${Math.round(predPriceNum).toLocaleString('en-IN')} <span style="font-size:0.8rem;font-weight:500;color:#94a3b8">/ quintal</span></div>
                    <div style="font-size:0.72rem;color:#a78bfa">Estimated Forecast</div>
                    ` : `
                    <div style="font-size:1.1rem;font-weight:700;color:#94a3b8;margin:6px 0;font-style:italic">Forecast Unavailable</div>
                    `}
                </div>

                <!-- Expected Change -->
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:14px">
                    <div style="font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Expected Change</div>
                    ${(diffVal !== null && changePct !== null) ? `
                    <div style="font-size:1.6rem;font-weight:800;color:${diffVal >= 0 ? '#4ade80' : '#f87171'};margin:4px 0">
                        ${diffVal >= 0 ? '+' : ''}₹${Math.round(Math.abs(diffVal)).toLocaleString('en-IN')}
                        <span style="font-size:0.85rem;font-weight:700">(${diffVal >= 0 ? '+' : ''}${changePct.toFixed(1)}%)</span>
                    </div>
                    <div style="font-size:0.72rem;color:#94a3b8">Over ${horizon}-day horizon</div>
                    ` : `
                    <div style="font-size:1.1rem;font-weight:700;color:#94a3b8;margin:6px 0;font-style:italic">—</div>
                    `}
                </div>
            </div>
        </div>

        <!-- 2. Restored Farmer's Market Recommendation (SELL / HOLD / WAIT) -->
        ${recAction ? `
        <div class="glass-card" style="padding:18px 20px;margin-bottom:18px;border-left:4px solid ${recBorder};background:rgba(255,255,255,0.02)">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:10px">
                <div style="display:flex;align-items:center;gap:12px">
                    <span style="font-size:0.78rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px">Market Recommendation:</span>
                    <span style="font-size:1.1rem;font-weight:800;padding:4px 16px;border-radius:6px;background:${recBg};color:${recColor};letter-spacing:0.5px;box-shadow:0 2px 8px rgba(0,0,0,0.2)">${recAction}</span>
                </div>
                <div style="font-size:0.76rem;color:#94a3b8">
                    Forecast Horizon: <strong style="color:#f8fafc">${horizon} Days</strong>
                </div>
            </div>

            <div style="font-size:0.92rem;color:#f1f5f9;line-height:1.55;margin-bottom:14px">
                ${recReason}
            </div>

            <!-- Decision Inputs Breakdown -->
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));gap:10px;padding:12px 14px;background:rgba(0,0,0,0.25);border-radius:8px;border:1px solid rgba(255,255,255,0.06);margin-bottom:10px">
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600">Current Mandi Price</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#34d399;margin-top:2px">
                        ${displaySpotNum != null ? `₹${Math.round(displaySpotNum).toLocaleString('en-IN')} <span style="font-size:0.7rem;font-weight:500;color:#94a3b8">/ qtl</span>` : '—'}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600">Expected Price (${horizon}d)</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#a78bfa;margin-top:2px">
                        ${isPredAvailable ? `₹${Math.round(predPriceNum).toLocaleString('en-IN')} <span style="font-size:0.7rem;font-weight:500;color:#94a3b8">/ qtl</span>` : '—'}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600">Expected Change</div>
                    <div style="font-size:0.95rem;font-weight:700;color:${diffVal != null && diffVal >= 0 ? '#4ade80' : '#f87171'};margin-top:2px">
                        ${(diffVal !== null && changePct !== null) ? `${diffVal >= 0 ? '+' : ''}₹${Math.round(Math.abs(diffVal)).toLocaleString('en-IN')} (${diffVal >= 0 ? '+' : ''}${changePct.toFixed(1)}%)` : '—'}
                    </div>
                </div>
                ${storageCostPct != null ? `
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600">Est. Holding Cost</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#fbbf24;margin-top:2px">${storageCostPct.toFixed(1)}%</div>
                </div>` : ''}
            </div>

            <div style="font-size:0.72rem;color:#94a3b8">
                ℹ️ Model-based decision guidance based on configured decision rules and estimated holding decay. Future prices and profits are not guaranteed.
            </div>
        </div>` : ''}

        <!-- 2. Simplified Price Graph with Horizon Tabs -->
        ${isPredAvailable ? `
        <div class="chart-box glass-card" style="margin-bottom:18px;padding:18px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:10px">
                <div>
                    <h4 style="margin:0;font-size:0.95rem;color:#f8fafc">Price Trend & Forecast (${horizon} Days)</h4>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">Forecast begins after the last observed date.</div>
                </div>
                <!-- Horizon selector tabs -->
                <div class="horizon-btn-group">
                    <button type="button" class="horizon-btn ${horizon === 7 ? 'active' : ''}" onclick="selectHorizon(7)">7 Days</button>
                    <button type="button" class="horizon-btn ${horizon === 15 ? 'active' : ''}" onclick="selectHorizon(15)">15 Days</button>
                    <button type="button" class="horizon-btn ${horizon === 30 ? 'active' : ''}" onclick="selectHorizon(30)">30 Days</button>
                    <button type="button" class="horizon-btn ${horizon === 60 ? 'active' : ''}" onclick="selectHorizon(60)">60 Days</button>
                    <button type="button" class="horizon-btn ${horizon === 90 ? 'active' : ''}" onclick="selectHorizon(90)">90 Days</button>
                </div>
            </div>
            <div style="height:340px;position:relative">
                <canvas id="priceChart"></canvas>
            </div>
            <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px;text-align:center">ℹ️ Forecast begins after the last observed date.</div>
        </div>` : ''}

        <!-- 3. OFFICIAL AGMARKNET MANDI & VARIETY BREAKDOWN (Unchanged) -->
        <div class="glass-card mandi-breakdown-card" style="padding:18px;margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px">
                <span style="font-size:0.8rem;font-weight:700;color:#f1f5f9;text-transform:uppercase;letter-spacing:0.5px">Official Agmarknet Mandi & Variety Breakdown</span>
                <span style="font-size:0.74rem;color:#94a3b8">${districtRecs.length} Reporting Mandi Yard${districtRecs.length !== 1 ? 's' : ''}</span>
            </div>
            ${districtRows ? `
            <div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08);border-radius:6px">
                <table style="width:100%;border-collapse:collapse;font-size:0.76rem;text-align:left">
                    <thead>
                        <tr style="background:rgba(255,255,255,0.04);color:#94a3b8">
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">District</th>
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">APMC Market Yard</th>
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Variety</th>
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Modal Rate</th>
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Arrival</th>
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Date</th>
                            <th style="padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Market Scope</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${districtRows}
                    </tbody>
                </table>
            </div>` : `
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:16px;text-align:center;color:#94a3b8;font-size:0.82rem;font-style:italic">
                Individual APMC mandi breakdown unavailable for ${stateLabel} on the selected date.
            </div>`}
        </div>
    `;

    document.getElementById("predResults").innerHTML = html;

    if (isPredAvailable) {
        renderPriceChart(predData, horizon, curPriceNum, obsDate);
    }
}

/**
 * renderPriceChart — Farmer-Friendly Clean Line Chart
 * Uses the same line color, style, and marker style for historical and forecast data.
 * Forecast begins after the last observed date.
 */
function renderPriceChart(data, horizon, curPriceFallback, obsDateStr) {
    const ctx = document.getElementById("priceChart");
    if (!ctx) return;

    if (activeChart) {
        activeChart.destroy();
        activeChart = null;
    }

    const curPrice = typeof data.current_price === "number" ? data.current_price : (curPriceFallback || 2000.0);

    // 1. Process Historical Series (Last 12-14 daily verified observations)
    let rawHist = Array.isArray(data.historical_series) && data.historical_series.length > 0
        ? data.historical_series
        : (Array.isArray(data.forecast?.historical_series) ? data.forecast.historical_series : []);

    let histSeries = [];
    if (rawHist && rawHist.length > 0) {
        histSeries = rawHist.slice(-14);
    } else {
        histSeries = [{ date: obsDateStr || new Date().toISOString().split('T')[0], price: curPrice }];
    }

    // 2. Process Forecast Series
    let rawFcast = Array.isArray(data.forecast_series) && data.forecast_series.length > 0
        ? data.forecast_series
        : (Array.isArray(data.forecast?.forecast_series) ? data.forecast.forecast_series : []);

    let fcastSeries = [];
    if (rawFcast && rawFcast.length > 0) {
        fcastSeries = rawFcast.slice(0, horizon);
    } else {
        const rawPreds = Array.isArray(data.predictions) ? data.predictions : (data.forecast?.raw_predictions || []);
        const dateLabels = Array.isArray(data.date_labels) ? data.date_labels : [];
        fcastSeries = rawPreds.slice(0, horizon).map((p, idx) => ({
            date: dateLabels[idx] || `Day ${idx + 1}`,
            price: Math.round(p)
        }));
    }

    // Subsample forecast points if horizon > 15 for readable x-axis
    let displayedFcast = fcastSeries;
    if (horizon > 30) {
        displayedFcast = fcastSeries.filter((_, idx) => idx % 4 === 0 || idx === fcastSeries.length - 1);
    } else if (horizon > 15) {
        displayedFcast = fcastSeries.filter((_, idx) => idx % 2 === 0 || idx === fcastSeries.length - 1);
    }

    // 3. Assemble Continuous Timeline with Same Visual Style
    const labels = [];
    const prices = [];

    // Push Historical Points (stops at last observed date)
    histSeries.forEach(item => {
        labels.push(formatDate(item.date));
        prices.push(Math.round(item.price));
    });

    // Push Forecast Points (starts after last observed date)
    displayedFcast.forEach(item => {
        labels.push(formatDate(item.date));
        prices.push(Math.round(item.price));
    });

    activeChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Price (₹/quintal)",
                    data: prices,
                    borderColor: "#22c55e",
                    backgroundColor: "rgba(34, 197, 94, 0.08)",
                    borderWidth: 2.5,
                    fill: false,
                    tension: 0.15,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: "#22c55e",
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 1.5,
                }
            ]
        },
        options: {
            animation: { duration: 400 },
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: "top",
                    align: "end",
                    labels: {
                        color: "rgba(255,255,255,0.85)",
                        font: { size: 11.5, weight: "600" },
                        boxWidth: 14,
                        padding: 10
                    }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#94a3b8",
                    bodyColor: "#f8fafc",
                    borderColor: "rgba(255, 255, 255, 0.15)",
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        title: items => items[0].label,
                        label: item => {
                            if (item.parsed.y == null) return null;
                            return `Price: ₹${item.parsed.y.toLocaleString('en-IN')} / quintal`;
                        }
                    },
                    filter: item => item.parsed.y != null
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Date",
                        color: "#94a3b8",
                        font: { size: 11, weight: "600" }
                    },
                    ticks: { color: "#94a3b8", font: { size: 10.5 }, maxRotation: 35, autoSkip: true, maxTicksLimit: 8 },
                    grid: { color: "rgba(255,255,255,0.04)" }
                },
                y: {
                    title: {
                        display: true,
                        text: "Price (₹/quintal)",
                        color: "#94a3b8",
                        font: { size: 11, weight: "600" }
                    },
                    ticks: {
                        color: "#94a3b8",
                        font: { size: 11 },
                        callback: v => `₹${Math.round(v).toLocaleString('en-IN')}`
                    },
                    grid: { color: "rgba(255,255,255,0.05)" }
                }
            }
        }
    });
}

// ─── Farmer Advisory ─────────────────────────────────────────────────────────
let currentAdvRequestId = 0;

async function submitAdvisory(event) {
    if (event && event.preventDefault) event.preventDefault();
    const requestId = ++currentAdvRequestId;
    const btn  = document.getElementById("btnAdv");
    const spin = document.getElementById("advSpin");
    const advResultsEl = document.getElementById("advResults");
    if (advResultsEl) advResultsEl.classList.add("results-stale");
    setLoading(btn, spin, true);

    const payload = {
        state:    document.getElementById("advState").value,
        district: document.getElementById("advDistrict").value,
        season:   document.getElementById("advSeason").value,
    };
    const crop = document.getElementById("advCrop").value;
    if (crop) payload.crop = crop;

    try {
        const res = await fetch("/api/advisory", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload),
        });
        if (requestId !== currentAdvRequestId) return;
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Request failed" }));
            throw new Error(err.detail || "Advisory failed. Please try again.");
        }
        const data = await res.json();
        if (requestId !== currentAdvRequestId) return;
        renderAdvisoryResults(data);
        if (advResultsEl) advResultsEl.classList.remove("results-stale");
    } catch (err) {
        if (requestId === currentAdvRequestId) {
            showToast(err.message, "error");
            if (advResultsEl) {
                advResultsEl.classList.remove("results-stale");
                advResultsEl.innerHTML = errorCard(err.message);
            }
        }
    } finally {
        if (requestId === currentAdvRequestId && btn && spin) {
            setLoading(btn, spin, false);
        }
    }
}

function renderAdvisoryResults(data) {
    const priceData = data.price_prediction || {};
    const rawDecision  = priceData.decision;
    const forecastAvail = priceData.forecast_available === true;
    const curPrice      = priceData.current_price;
    const avgPrice      = priceData.predicted_30d_avg;
    const obsDate       = priceData.observation_date;
    const decisionReason = priceData.decision_reason || priceData.farmer_message || "";
    const rawRecs       = data.crop_recommendations || [];

    // Deduplicate Paddy/Rice in recommended crops
    const seenCrops = new Set();
    const dedupedRecs = [];
    for (const rec of rawRecs) {
        let cropName = (rec.crop || '').trim();
        if (cropName.toLowerCase() === 'paddy') cropName = 'Rice';
        const key = cropName.toLowerCase();
        if (seenCrops.has(key)) continue;
        seenCrops.add(key);
        rec.displayCrop = cropName;
        dedupedRecs.push(rec);
    }
    const recs = dedupedRecs.slice(0, 3);

    // Format display values
    const curPriceDisplay  = typeof curPrice === 'number'
        ? `₹${Math.round(curPrice).toLocaleString('en-IN')}/qtl`
        : 'Not available';
    const predPriceDisplay = forecastAvail && typeof avgPrice === 'number'
        ? `₹${Math.round(avgPrice).toLocaleString('en-IN')}/qtl`
        : 'Currently unavailable';

    // Decision display
    let decDisplay = 'Not available';
    let decColor   = '#94a3b8';
    if (rawDecision === 'SELL') { decDisplay = 'SELL'; decColor = '#f87171'; }
    else if (rawDecision === 'HOLD') { decDisplay = 'HOLD'; decColor = '#4ade80'; }
    else if (rawDecision === 'WAIT') { decDisplay = 'WAIT'; decColor = '#fbbf24'; }

    const targetCropTitle = capitalize(data.target_price_crop || '');

    const html = `
        <div class="adv-summary glass-card" style="padding:16px 20px;margin-bottom:16px">
            <p class="adv-summary-text" style="margin:0;font-size:0.95rem;line-height:1.5">${sanitizeSeasonText(data.combined_summary || "Advisory generated.")}</p>
        </div>

        ${recs.length > 0 ? `
        <div class="adv-section" style="margin-bottom:16px">
            <h4 style="font-size:0.95rem;color:#f8fafc;margin-bottom:10px">🌱 Recommended Crops for ${data.district || ''}, ${data.state || ''}</h4>
            <div style="display:flex;flex-direction:column;gap:8px">
                ${recs.map((r, i) => {
                    const waterNeed = getWaterNeed(r.displayCrop || r.crop, r);
                    const cleanReasons = getCleanReasons(r);
                    const scorePct = Math.round(r.suitability_score || r.confidence_score || r.model_confidence || 80);
                    return `
                    <div class="glass-card" style="padding:12px 16px;border-radius:8px">
                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px">
                            <div style="display:flex;align-items:center;gap:10px">
                                <span style="font-weight:800;color:#38bdf8;font-size:0.9rem">#${i+1}</span>
                                <strong style="font-size:1.05rem;color:#f8fafc">${r.displayCrop || capitalize(r.crop)}</strong>
                                <span style="font-size:0.75rem;font-weight:700;color:#38bdf8;padding:2px 8px;border-radius:4px;background:rgba(56,189,248,0.12)">${scorePct}% Suitability</span>
                            </div>
                            <div style="display:flex;gap:6px;font-size:0.75rem">
                                <span style="padding:2px 7px;border-radius:4px;background:rgba(255,255,255,0.06);color:#cbd5e1">Water: <strong>${waterNeed}</strong></span>
                            </div>
                        </div>
                        <div style="font-size:0.82rem;color:#94a3b8;line-height:1.4">
                            ${cleanReasons[0] || 'Well-suited to local soil texture and regional seasonal conditions.'}
                        </div>
                    </div>`;
                }).join('')}
            </div>
        </div>` : ''}

        <div class="adv-market glass-card" style="padding:16px;border-radius:8px">
            <h4 style="margin:0 0 12px 0;font-size:0.95rem;color:#f8fafc">📈 Market Analysis — ${targetCropTitle}</h4>
            <div class="adv-market-grid">
                <div class="adv-metric">
                    <span class="adv-m-lbl">Current Mandi Price</span>
                    <span class="adv-m-val">${curPriceDisplay}</span>
                    ${obsDate ? `<div style="font-size:0.72rem;color:#94a3b8;margin-top:2px">Observed: ${formatDate(obsDate)}</div>` : ''}
                </div>
                <div class="adv-metric">
                    <span class="adv-m-lbl">30-Day Forecast</span>
                    <span class="adv-m-val" style="color:${forecastAvail ? '#a3e635' : '#94a3b8'}">${predPriceDisplay}</span>
                    ${forecastAvail ? `<div style="font-size:0.72rem;color:#a3e635;margin-top:2px">Estimated Forecast</div>` : ''}
                </div>
                <div class="adv-metric">
                    <span class="adv-m-lbl">Market Decision</span>
                    <span class="adv-m-val" style="color:${decColor};font-weight:700">${decDisplay}</span>
                </div>
            </div>
            ${(forecastAvail && rawDecision && decisionReason) ? `
            <div style="margin-top:12px;padding:10px 14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;font-size:0.84rem;color:#cbd5e1;line-height:1.45">
                💡 <strong>Advisory:</strong> ${decisionReason}
            </div>` : ''}
            ${!forecastAvail ? `
            <div style="margin-top:12px;padding:10px 14px;background:rgba(148,163,184,0.08);border:1px solid rgba(148,163,184,0.2);border-radius:8px;font-size:0.83rem;color:#94a3b8;line-height:1.5">
                ℹ️ ${decisionReason || 'A reliable 30-day price forecast is currently unavailable for this crop.'}
            </div>` : ''}
        </div>
    `;

    document.getElementById("advResults").innerHTML = html;
}

// ─── Utility: Format Date ────────────────────────────────────────────────────
function formatDate(dateStr) {
    if (!dateStr || dateStr === '—') return dateStr;
    try {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;
        return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch {
        return dateStr;
    }
}

// ─── UI Helpers ──────────────────────────────────────────────────────────────
function setLoading(btn, spin, on) {
    if (!btn) return;
    btn.disabled = on;
    btn.style.opacity = on ? "0.7" : "1";
    if (spin) spin.classList.toggle("hidden", !on);
}

function showToast(message, type = "info") {
    const box   = document.getElementById("toastBox");
    if (!box) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icons = { success: "check_circle", error: "error", info: "info" };
    toast.innerHTML = `<span class="material-symbols-rounded toast-icon">${icons[type] || "info"}</span><span>${message}</span>`;
    box.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 350);
    }, 4500);
}

function capitalize(str) {
    return str ? str.charAt(0).toUpperCase() + str.slice(1).toLowerCase() : "";
}

function errorCard(msg) {
    return `<div class="placeholder-card glass-card error-card">
        <div class="ph-icon">⚠️</div>
        <h3>Something went wrong</h3>
        <p>${msg}</p>
    </div>`;
}

