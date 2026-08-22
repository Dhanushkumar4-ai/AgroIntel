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
    // NOTE: "predState" is intentionally in this list but has NO associated district select
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
    const cleanId = pageId.replace(/^(view-|page-)/, "");

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

    const n  = parseFloat(document.getElementById("recN").value);
    const p  = parseFloat(document.getElementById("recP").value);
    const k  = parseFloat(document.getElementById("recK").value);
    const ph = parseFloat(document.getElementById("recPh").value);
    const prev = document.getElementById("recPrevCrop")?.value?.trim();

    if (!isNaN(n))  payload.n  = n;
    if (!isNaN(p))  payload.p  = p;
    if (!isNaN(k))  payload.k  = k;
    if (!isNaN(ph)) payload.soil_ph = ph;
    if (prev)       payload.previous_crop = prev;

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

function renderRecResults(data) {
    const loc  = data.location || {};
    const recs = (data.recommendations || []).slice(0, 5);
    const displaySeason = formatSeasonLabel(data.season);

    if (recs.length === 0) {
        const msg = sanitizeSeasonText(data.message || "No suitable candidate crops found.");
        document.getElementById("recResults").innerHTML = `
            <div class="placeholder-card glass-card">
                <span class="material-symbols-rounded ph-icon-sym">grass</span>
                <h3>No Crops Recommended</h3>
                <p>${msg}</p>
                <p style="font-size:0.78rem;opacity:0.65;margin-top:8px">Try a different district or season.</p>
            </div>`;
        return;
    }

    const rankColors = ["first-rank", "second-rank", "third-rank", "", ""];

    const recHtml = recs.map((rec, i) => {
        const info = rec.crop_information || {};
        return `
        <div class="rec-card glass-card ${rankColors[i] || ''}" style="margin-bottom:16px;padding:18px">
            <!-- Header: rank + crop name + location subtitle -->
            <div class="rec-card-top" style="margin-bottom:14px;display:flex;align-items:center;gap:12px">
                <span class="rec-rank" style="flex-shrink:0;font-size:1.1rem;font-weight:800;color:#a3e635;background:rgba(163,230,53,0.12);border:1px solid rgba(163,230,53,0.3);padding:4px 12px;border-radius:20px">#${rec.rank || i + 1}</span>
                <div>
                    <h3 class="rec-crop-name" style="margin:0;font-size:1.4rem">${rec.crop}</h3>
                    <div style="font-size:0.8rem;opacity:0.75;margin-top:2px">Recommended for ${displaySeason} in ${loc.district || ''}, ${loc.state || ''}</div>
                </div>
            </div>

            <!-- About this Crop (farmer-friendly general information) -->
            <div style="background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:14px">
                <h4 style="margin:0 0 10px;font-size:0.92rem;color:#e2e8f0;display:flex;align-items:center;gap:6px">
                    <span class="material-symbols-rounded" style="font-size:1.1rem;color:#a3e635">grass</span> ABOUT ${rec.crop.toUpperCase()}
                </h4>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.82rem;line-height:1.45">
                    <div><strong style="color:#a3e635">Why grown:</strong> <span style="opacity:0.9">${sanitizeSeasonText(info.why_grown || 'Cultivated for farm revenue and local food demand.')}</span></div>
                    <div><strong style="color:#38bdf8">Common uses:</strong> <span style="opacity:0.9">${sanitizeSeasonText(info.common_uses || 'Food grain, pulse, or agricultural produce.')}</span></div>
                    <div><strong style="color:#fbbf24">Season:</strong> <span style="opacity:0.9">${sanitizeSeasonText(info.season || displaySeason || 'Standard regional season.')}</span></div>
                    <div><strong style="color:#c084fc">Soil &amp; Climate:</strong> <span style="opacity:0.9">${sanitizeSeasonText(info.soil || 'Well-drained soil.')} ${sanitizeSeasonText(info.climate || '')}</span></div>
                </div>
            </div>
        </div>`;
    }).join('');

    document.getElementById("recResults").innerHTML = `
        <div class="rec-header-row" style="margin-bottom:16px">
            <h3>Recommended Crops for <strong>${loc.district || data.district || ''}</strong>, ${loc.state || data.state || ''}</h3>
            <p class="rec-season-tag">Season: ${displaySeason}</p>
        </div>
        ${recHtml}`;
}



// ─── Price Prediction ─────────────────────────────────────────────────────────
// IMPORTANT: This function calls ONLY /api/predict — never /api/phase6/recommend
// District is NOT used in price prediction.
async function submitPricePred(event) {
    event.preventDefault();
    const btn  = document.getElementById("btnPred");
    const spin = document.getElementById("predSpin");
    setLoading(btn, spin, true);

    const crop    = document.getElementById("predCrop").value;
    const state   = document.getElementById("predState").value;
    const horizon = document.getElementById("predHorizon").value;

    if (!crop) {
        showToast("Please select a crop.", "error");
        setLoading(btn, spin, false);
        return;
    }

    try {
        // Call ONLY /api/predict — no district, no phase6 recommend
        let predUrl = `/api/predict?crop=${encodeURIComponent(crop)}&horizon_days=${horizon}`;
        if (state) predUrl += `&state=${encodeURIComponent(state)}`;

        const predRes = await fetch(predUrl);
        if (!predRes.ok) {
            const err = await predRes.json().catch(() => ({ detail: "Prediction request failed." }));
            throw new Error(err.detail || "Price forecast could not be retrieved.");
        }
        const predData = await predRes.json();
        renderPredResults(predData, crop, parseInt(horizon), state);

    } catch (err) {
        showToast(err.message, "error");
        document.getElementById("predResults").innerHTML = errorCard(err.message);
    } finally {
        setLoading(btn, spin, false);
    }
}

/**
 * renderPredResults — uses ONLY predData from /api/predict
 * No p6Data. No cross-contamination from phase6 recommend.
 */
function renderPredResults(predData, crop, horizon, inputState) {

    // ── Extract core values from predData ONLY ───────────────────────────────
    const isPredAvailable = predData.available === true &&
                            typeof predData.predicted_price === 'number';

    const curPriceNum  = typeof predData.current_price === 'number'   ? predData.current_price   : null;
    const predPriceNum = typeof predData.predicted_price === 'number' ? predData.predicted_price : null;

    const isMandiAvailable = typeof curPriceNum === 'number';

    const curPriceDisplay  = isMandiAvailable ? `₹${Math.round(curPriceNum).toLocaleString('en-IN')}` : 'Price Unavailable';
    const predPriceDisplay = isPredAvailable  ? `₹${Math.round(predPriceNum).toLocaleString('en-IN')}` : 'Forecast Unavailable';

    // ── Observation date & data freshness ────────────────────────────────────
    const obsDate     = predData.observation_date   || predData.price_cached_time?.split('T')[0] || '—';
    const mktName     = predData.market_name        || predData.market || inputState || '—';
    const dataAgeDays = typeof predData.data_age_days === 'number' ? predData.data_age_days : null;
    const dataSource  = predData.price_data_source  || 'data.gov.in';

    // Determine correct label based on whether data is truly "today"
    const today = new Date().toISOString().split('T')[0];
    const obsLabel = obsDate === today ? 'Live Price' :
                     obsDate !== '—' ? `Latest Available Mandi Price` : 'Latest Available Price';
    const obsNote  = dataAgeDays !== null && dataAgeDays > 0 ? `Data Age: ${dataAgeDays} day${dataAgeDays !== 1 ? 's' : ''}` :
                     dataAgeDays === 0 ? 'Same day observation' : '';

    // Source display logic — never call old cache "Live"
    const sourceLabel = dataSource === 'api_data_gov_in' ? 'data.gov.in (AGMARKNET)' :
                        dataSource === 'cached_api'      ? 'data.gov.in (Cached)' :
                        dataSource === 'yahoo_finance'   ? 'Yahoo Finance (Global Futures)' :
                        dataSource === 'msp_estimate'    ? 'India MSP 2024-25 (Estimated)' :
                        dataSource;

    const modelName     = predData.best_model_label || predData.best_model || 'ML Model';
    const forecastScope = predData.forecast_scope   || 'Crop-level 30-day ML forecast';

    // ── SELL / HOLD / WAIT Decision ──────────────────────────────────────────
    let advAction = "WAIT";
    let advReason = "Insufficient data to make a confident market decision.";
    let advColor  = "#94a3b8"; // grey for WAIT

    if (predData.advisory?.decision) {
        // Use backend decision if provided
        advAction = predData.advisory.decision;
        advReason = predData.advisory.reason || advReason;
    } else if (isMandiAvailable && isPredAvailable) {
        const changePct = ((predPriceNum - curPriceNum) / curPriceNum) * 100.0;
        const absChange = Math.abs(changePct);

        // Adjust threshold by model uncertainty (high MAE crops need larger swing to trigger SELL/HOLD)
        const highUncertaintyCrops = ['onion', 'potato'];
        const threshold = highUncertaintyCrops.includes(crop.toLowerCase()) ? 5.0 : 3.0;

        // Stale data → WAIT
        if (dataAgeDays !== null && dataAgeDays > 14) {
            advAction = "WAIT";
            advReason = `The latest Mandi price observation is ${dataAgeDays} days old. Market conditions may have changed. Verify current local Mandi rates before making a transaction decision.`;
            advColor  = "#94a3b8";
        } else if (changePct <= -threshold) {
            advAction = "SELL";
            advColor  = "#f97316";
            advReason = `The price forecasting model projects a decline of approximately ${absChange.toFixed(1)}% from ₹${Math.round(curPriceNum).toLocaleString('en-IN')} to approximately ₹${Math.round(predPriceNum).toLocaleString('en-IN')} over the next ${horizon} days. Selling at the current observed price may reduce downside risk.`;
        } else if (changePct >= threshold) {
            advAction = "HOLD";
            advColor  = "#22c55e";
            advReason = `The price forecasting model projects an increase of approximately ${absChange.toFixed(1)}% from ₹${Math.round(curPriceNum).toLocaleString('en-IN')} to approximately ₹${Math.round(predPriceNum).toLocaleString('en-IN')} over the next ${horizon} days. Holding may provide a better expected selling price.`;
        } else {
            advAction = "WAIT";
            advColor  = "#94a3b8";
            advReason = `The forecast indicates a small movement of approximately ${changePct > 0 ? '+' : ''}${changePct.toFixed(1)}% — within model uncertainty bounds. The market is expected to remain relatively stable.`;
        }
    } else if (!isMandiAvailable) {
        advReason = "Current Mandi price is unavailable. Cannot generate a SELL/HOLD/WAIT decision without a current price reference.";
    } else if (!isPredAvailable) {
        const reason = predData.forecast?.reason || predData.message || "Price forecast is currently unavailable for this crop.";
        advReason = reason;
    }

    if (advAction === "SELL") advColor = "#f97316";
    else if (advAction === "HOLD") advColor = "#22c55e";
    else advColor = "#94a3b8";

    // ── Black Swan Warning ────────────────────────────────────────────────────
    const bsWarning = predData.black_swan_warning;
    const bsHtml = bsWarning ? `
        <div style="background:#dc262630;border:1px solid #dc2626;border-radius:8px;padding:12px;margin-bottom:16px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <span class="material-symbols-rounded" style="color:#f87171">warning</span>
                <strong style="color:#f87171;font-size:0.88rem">Major Market Event Detected</strong>
            </div>
            <div style="font-size:0.82rem;opacity:0.9">${bsWarning.message || bsWarning.label}</div>
        </div>` : '';

    // ── NLP Explanation ───────────────────────────────────────────────────────
    const cropDisplay = capitalize(crop);
    let nlpText = '';
    if (isMandiAvailable && isPredAvailable) {
        const changePct = ((predPriceNum - curPriceNum) / curPriceNum) * 100.0;
        const direction = changePct >= 0 ? 'increase' : 'decline';
        nlpText = `${cropDisplay} is currently trading at ₹${Math.round(curPriceNum).toLocaleString('en-IN')} per quintal in ${mktName}${obsDate !== '—' ? ' (observed on ' + formatDate(obsDate) + ')' : ''}. The state-aware forecasting model projects approximately ₹${Math.round(predPriceNum).toLocaleString('en-IN')} per quintal after ${horizon} days — an expected market ${direction} of about ${Math.abs(changePct).toFixed(1)}%. Based on price trends and market uncertainty, the recommended decision is <strong>${advAction}</strong>.`;
    } else if (isMandiAvailable) {
        nlpText = `${cropDisplay} is currently trading at ₹${Math.round(curPriceNum).toLocaleString('en-IN')} per quintal in ${mktName}. A 30-day price forecast is currently unavailable.`;
    } else {
        nlpText = `${cropDisplay} market data is currently unavailable for ${mktName}. Please check back later or verify local Mandi rates.`;
    }

    // ── Assemble Farmer-Friendly HTML ─────────────────────────────────────────
    const html = `
        <!-- Header -->
        <div class="pred-summary-row" style="margin-bottom:16px">
            <div class="pred-meta">
                <h3 style="margin:0;font-size:1.4rem">${cropDisplay} Price Outlook</h3>
                <div style="font-size:0.82rem;opacity:0.75;margin-top:2px">📍 State: ${inputState || 'National'}</div>
            </div>
            <div class="pred-decision" style="background:${advColor}22;border:1px solid ${advColor};color:${advColor};padding:8px 18px">
                <span class="dec-word" style="font-size:1.1rem;font-weight:800;letter-spacing:1px">${advAction}</span>
            </div>
        </div>

        ${bsHtml}

        <!-- LATEST MANDI PRICE & EXPECTED PRICE ROW -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">
            <div class="glass-card" style="padding:18px">
                <div style="font-size:0.78rem;opacity:0.75;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">${obsLabel}</div>
                <div style="font-size:1.8rem;font-weight:800;color:#34d399;margin:8px 0">${curPriceDisplay} <span style="font-size:0.85rem;font-weight:500;opacity:0.7">${isMandiAvailable ? '/ quintal' : ''}</span></div>
                <div style="font-size:0.75rem;opacity:0.8;margin-top:4px">
                    ${obsDate !== '—' ? `<div><strong>Observed:</strong> ${formatDate(obsDate)}</div>` : ''}
                    ${obsNote ? `<div style="color:#fbbf24"><strong>${obsNote}</strong></div>` : ''}
                    <div><strong>Market:</strong> ${mktName}</div>
                    <div><strong>Source:</strong> ${sourceLabel}</div>
                </div>
            </div>
            <div class="glass-card" style="padding:18px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:0.78rem;opacity:0.75;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Expected Price in ${horizon} Days</div>
                ${isPredAvailable ? `
                <div style="font-size:1.8rem;font-weight:800;color:#a78bfa;margin:8px 0">${predPriceDisplay} <span style="font-size:0.85rem;font-weight:500;opacity:0.7">/ quintal</span></div>
                ` : `
                <div style="font-size:0.82rem;color:#fbbf24;margin-top:10px;line-height:1.4">${predData.forecast?.reason || predData.message || "Forecast unavailable for this crop."}</div>
                `}
            </div>
        </div>

        <!-- 30-DAY PRICE FORECAST GRAPH -->
        ${isPredAvailable ? `
        <div class="chart-box glass-card" style="margin-bottom:18px;padding:16px">
            <div class="chart-header" style="margin-bottom:12px">
                <h4 style="margin:0;font-size:1rem">Price Forecast — Observed + ${horizon}-Day Trend</h4>
                <span class="chart-horizon" style="font-size:0.78rem;opacity:0.75">🟡 Observed · 🟢 Forecast</span>
            </div>
            <canvas id="priceChart"></canvas>
        </div>` : ''}

        <!-- MARKET DECISION ADVISORY -->
        <div class="glass-card" style="padding:18px;margin-bottom:16px;border-left:4px solid ${advColor}">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <span style="font-size:0.82rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;opacity:0.8">MARKET DECISION</span>
                <span style="background:${advColor}22;color:${advColor};border:1px solid ${advColor};padding:2px 10px;border-radius:4px;font-weight:800;font-size:0.9rem">${advAction}</span>
            </div>
            <div style="font-size:0.9rem;line-height:1.45;opacity:0.92;margin-top:6px">${advReason}</div>
        </div>

        <!-- PRICE OUTLOOK NLP EXPLANATION -->
        <div class="glass-card" style="padding:18px;margin-bottom:16px">
            <h4 style="margin:0 0 8px;font-size:0.95rem;color:#38bdf8">Price Outlook</h4>
            <div style="font-size:0.88rem;line-height:1.5;opacity:0.92">${nlpText}</div>
        </div>`;

    document.getElementById("predResults").innerHTML = html;

    if (isPredAvailable) {
        renderPriceChart(predData, horizon, curPriceNum, obsDate);
    }

}

/**
 * renderPriceChart — Corrected graph
 *
 * Structure:
 *   [Observed — DD MMM]  →  [Forecast Day 1]  →  ...  →  [Forecast Day N]
 *
 * The current observed Mandi price is plotted at its ACTUAL observation date.
 * Forecast points start from the day after the observation date.
 */
function renderPriceChart(data, horizon, curPriceFallback, obsDateStr) {
    const ctx = document.getElementById("priceChart");
    if (!ctx) return;

    if (activeChart) {
        activeChart.destroy();
        activeChart = null;
    }

    const curPrice    = typeof data.current_price === "number" ? data.current_price : (curPriceFallback || 2000.0);
    const predsList   = Array.isArray(data.predictions) ? data.predictions : [];
    const dateLabels  = Array.isArray(data.date_labels) ? data.date_labels : [];
    const actualHorizon = Math.min(horizon, predsList.length > 0 ? predsList.length : horizon);

    // Build labels — "Observed — DD MMM" then "Day 1", "Day 7", "Day 15", "Day 30"
    const obsLabel = obsDateStr && obsDateStr !== '—' ? `Observed — ${formatDate(obsDateStr)}` : 'Current (Observed)';

    // Select which forecast points to display for readability
    let forecastIndices = [];
    if (actualHorizon <= 7) {
        forecastIndices = Array.from({length: actualHorizon}, (_, i) => i);
    } else if (actualHorizon <= 15) {
        forecastIndices = [0, 3, 6, 9, 12, actualHorizon - 1].filter((v, i, a) => a.indexOf(v) === i && v < actualHorizon);
    } else {
        forecastIndices = [0, 6, 13, 20, 29].filter(v => v < actualHorizon);
        if (!forecastIndices.includes(actualHorizon - 1)) forecastIndices.push(actualHorizon - 1);
    }

    const chartLabels = [obsLabel];
    const chartValues = [Math.round(curPrice)];
    const pointColors = ['#f59e0b'];   // amber for observed
    const pointSizes  = [9];

    forecastIndices.forEach(idx => {
        const dayLabel = dateLabels[idx] ? formatDate(dateLabels[idx]) : `Day ${idx + 1}`;
        chartLabels.push(dayLabel);
        chartValues.push(Math.round(predsList[idx] || curPrice));
        pointColors.push(idx === forecastIndices[forecastIndices.length - 1] ? '#22c55e' : '#38bdf8');
        pointSizes.push(idx === forecastIndices[forecastIndices.length - 1] ? 8 : 5);
    });

    activeChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: "Observed Mandi Price (₹/quintal)",
                    data: [chartValues[0], null, null, null, null, null],
                    borderColor: "#f59e0b",
                    backgroundColor: "transparent",
                    borderWidth: 0,
                    pointRadius: [9],
                    pointHoverRadius: [11],
                    pointBackgroundColor: ["#f59e0b"],
                    pointBorderColor: ["#fff"],
                    pointBorderWidth: 2,
                    segment: { borderDash: [4, 4] },
                    tension: 0,
                },
                {
                    label: "ML Forecast (₹/quintal)",
                    data: chartValues,
                    borderColor: "#22c55e",
                    backgroundColor: "rgba(34,197,94,0.07)",
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.3,
                    pointRadius: pointSizes,
                    pointHoverRadius: pointSizes.map(s => s + 3),
                    pointBackgroundColor: pointColors,
                    pointBorderColor: "#fff",
                    pointBorderWidth: 2,
                }
            ],
        },
        options: {
            animation: { duration: 600, easing: "easeInOutQuart" },
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: "rgba(255,255,255,0.7)", font: { size: 11 }, boxWidth: 14 }
                },
                tooltip: {
                    callbacks: {
                        title: items => items[0].label,
                        label: item => {
                            if (item.datasetIndex === 0) return item.parsed.y != null ? `Observed Mandi: ₹${item.parsed.y?.toLocaleString('en-IN')} / quintal` : null;
                            return `ML Forecast: ₹${item.parsed.y?.toLocaleString('en-IN')} / quintal`;
                        }
                    },
                    backgroundColor: "rgba(15,25,18,0.92)",
                    titleColor: "#a1a1aa",
                    bodyColor: "#f0fdf4",
                    borderColor: "rgba(34,197,94,0.3)",
                    borderWidth: 1,
                    padding: 12,
                    filter: item => item.parsed.y != null,
                }
            },
            scales: {
                x: {
                    ticks: { color: "var(--text-secondary)", font: { size: 11, weight: "500" }, maxRotation: 30 },
                    grid: { color: "rgba(255,255,255,0.04)" }
                },
                y: {
                    ticks: {
                        color: "var(--text-secondary)",
                        font: { size: 12 },
                        callback: v => `₹${Math.round(v).toLocaleString('en-IN')}`
                    },
                    grid: { color: "rgba(255,255,255,0.05)" },
                    title: {
                        display: true,
                        text: "Price (₹/quintal)",
                        color: "var(--text-muted)",
                        font: { size: 11 }
                    }
                }
            }
        }
    });
}

// ─── Farmer Advisory ─────────────────────────────────────────────────────────
async function submitAdvisory(event) {
    event.preventDefault();
    const btn  = document.getElementById("btnAdv");
    const spin = document.getElementById("advSpin");
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
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Request failed" }));
            throw new Error(err.detail || "Advisory failed. Please try again.");
        }
        const data = await res.json();
        renderAdvisoryResults(data);
    } catch (err) {
        showToast(err.message, "error");
        document.getElementById("advResults").innerHTML = errorCard(err.message);
    } finally {
        setLoading(btn, spin, false);
    }
}

function renderAdvisoryResults(data) {
    const priceData = data.price_prediction || {};
    // decision is null when no forecast model exists — NEVER default to HOLD
    const rawDecision  = priceData.decision;
    const forecastAvail = priceData.forecast_available === true;
    const curPrice      = priceData.current_price;
    const avgPrice      = priceData.predicted_30d_avg;
    const farmerMsg     = priceData.farmer_message || "A reliable 30-day price forecast is currently unavailable for this crop.";
    const recs          = data.crop_recommendations?.slice(0, 3) || [];

    // Format display values — never show ₹— for predicted when forecast unavailable
    const curPriceDisplay  = typeof curPrice === 'number'
        ? `₹${Math.round(curPrice).toLocaleString('en-IN')}/qtl`
        : 'Not available';
    const predPriceDisplay = forecastAvail && typeof avgPrice === 'number'
        ? `₹${Math.round(avgPrice).toLocaleString('en-IN')}/qtl`
        : 'Currently unavailable';

    // Decision display — null means no forecast, show "Not available" not HOLD
    let decDisplay = 'Not available';
    let decColor   = '#94a3b8';  // neutral grey
    if (rawDecision === 'SELL') { decDisplay = 'SELL'; decColor = '#ef4444'; }
    else if (rawDecision === 'HOLD') { decDisplay = 'HOLD'; decColor = '#22c55e'; }
    else if (rawDecision === 'WAIT') { decDisplay = 'WAIT'; decColor = '#f59e0b'; }

    const html = `
        <div class="adv-summary glass-card">
            <p class="adv-summary-text">${sanitizeSeasonText(data.combined_summary || "Advisory generated.")}</p>
        </div>

        ${recs.length > 0 ? `
        <div class="adv-section">
            <h4>🌱 Recommended Crops</h4>
            ${recs.map((r, i) => `
                <div class="adv-crop-row glass-card">
                    <span class="adv-rank">#${i+1}</span>
                    <span class="adv-crop-name">${capitalize(r.crop)}</span>
                </div>`).join('')}
        </div>` : ''}

        <div class="adv-market glass-card">
            <h4>📈 Market Analysis — ${capitalize(data.target_price_crop || '')}</h4>
            <div class="adv-market-grid">
                <div class="adv-metric">
                    <span class="adv-m-lbl">Current Mandi Price</span>
                    <span class="adv-m-val">${curPriceDisplay}</span>
                </div>
                <div class="adv-metric">
                    <span class="adv-m-lbl">30-Day Forecast</span>
                    <span class="adv-m-val" style="color:${forecastAvail ? '#a3e635' : '#94a3b8'}">${predPriceDisplay}</span>
                </div>
                <div class="adv-metric">
                    <span class="adv-m-lbl">Market Decision</span>
                    <span class="adv-m-val" style="color:${decColor};font-weight:700">${decDisplay}</span>
                </div>
            </div>
            ${!forecastAvail ? `
            <div style="margin-top:12px;padding:10px 14px;background:rgba(148,163,184,0.08);border:1px solid rgba(148,163,184,0.2);border-radius:8px;font-size:0.83rem;color:#94a3b8;line-height:1.5">
                ℹ️ ${farmerMsg}
            </div>` : ''}
        </div>

        ${(data.consolidated_reasons || []).length > 0 ? `
        <div class="adv-reasons glass-card">
            <h4>💬 Key Reasons</h4>
            <ul class="reasons-list">
                ${data.consolidated_reasons.slice(0, 5).map(r => `<li>${r}</li>`).join('')}
            </ul>
        </div>` : ''}`;

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
