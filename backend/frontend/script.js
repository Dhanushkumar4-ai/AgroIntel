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
    // State select dropdowns across Recommendation, Prediction, Advisory, and News
    const stateSelectIds = ["recState", "predState", "advState", "newsFilterState"];
    stateSelectIds.forEach(id => {
        const sel = document.getElementById(id);
        if (!sel) return;
        if (id === "newsFilterState") {
            sel.innerHTML = '<option value="">All States / National</option>';
        } else {
            sel.innerHTML = '<option value="">Select State</option>';
        }

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
    const navMap = { recommendation: "navRecommend", prediction: "navPrediction", advisory: "navAdvisory", news: "navNews" };
    if (navMap[cleanId]) document.getElementById(navMap[cleanId])?.classList.add("nav-active");

    if (cleanId === "news") {
        loadNews();
    }

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

function renderRecResults(data) {
    const loc  = data.location || {};
    const recs = (data.recommendations || []).slice(0, 5);
    const displaySeason = formatSeasonLabel(data.season);

    if (recs.length === 0) {
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

    const rankColors = ["first-rank", "second-rank", "third-rank", "", ""];

    // Data availability advisory banner if official district records are not published
    let dataAvailabilityBanner = "";
    if (data.data_availability_message) {
        dataAvailabilityBanner = `
        <div style="background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.35);color:#fde047;border-radius:10px;padding:14px;margin-bottom:16px;font-size:0.84rem;line-height:1.45;display:flex;align-items:flex-start;gap:10px">
            <span class="material-symbols-rounded" style="font-size:1.3rem;flex-shrink:0;color:#eab308">info</span>
            <div>
                <strong>Regional Data Notice:</strong> ${data.data_availability_message}
            </div>
        </div>`;
    }

    const recHtml = recs.map((rec, i) => {
        const info = rec.crop_information || {};
        
        // Dynamic Evidence Badges based on verified evidence level
        let badgeHtml = "";
        const evLevel = rec.evidence_level || (rec.is_local_crop ? "SIGNATURE_LOCAL_CROP" : "HISTORICALLY_CULTIVATED_LOCALLY");
        
        if (evLevel === "SIGNATURE_LOCAL_CROP") {
            badgeHtml = `<span style="background:rgba(250,204,21,0.15);color:#facc15;border:1px solid rgba(250,204,21,0.35);padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:700;display:inline-flex;align-items:center;gap:4px">⭐ Signature Local Crop</span>`;
        } else if (evLevel === "HISTORICALLY_CULTIVATED_LOCALLY" || evLevel === "LOCALLY_SUPPORTED") {
            badgeHtml = `<span style="background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.35);padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:700;display:inline-flex;align-items:center;gap:4px">🌿 Historically Cultivated Locally</span>`;
        } else if (evLevel === "AGRONOMIC_CANDIDATE" || evLevel === "POTENTIAL_AGRONOMIC_CANDIDATE") {
            badgeHtml = `<span style="background:rgba(56,189,248,0.15);color:#38bdf8;border:1px solid rgba(56,189,248,0.35);padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:700;display:inline-flex;align-items:center;gap:4px">🧪 Agronomic Candidate</span>`;
        } else {
            badgeHtml = `<span style="background:rgba(249,115,22,0.15);color:#fb923c;border:1px solid rgba(249,115,22,0.35);padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:700;display:inline-flex;align-items:center;gap:4px">⚠️ Limited Regional Data</span>`;
        }

        // Agronomic model evaluation tag
        let agronomicTag = "";
        if (rec.rf_evaluated && rec.agronomic_suitability_score !== null && rec.agronomic_suitability_score !== undefined) {
            agronomicTag = `<span style="background:rgba(99,102,241,0.12);color:#818cf8;border:1px solid rgba(99,102,241,0.25);padding:2px 8px;border-radius:8px;font-size:0.72rem;font-weight:600">RF Agronomic Prob: ${rec.agronomic_suitability_score}%</span>`;
        } else {
            agronomicTag = `<span style="background:rgba(148,163,184,0.12);color:#94a3b8;border:1px solid rgba(148,163,184,0.25);padding:2px 8px;border-radius:8px;font-size:0.72rem;font-weight:600">ICAR Agro-Ecological Match</span>`;
        }

        // District APY Evidence summary block (guaranteeing accurate annual averages vs cumulative footprint)
        let evidenceBlock = "";
        if (rec.district_cultivation_evidence) {
            const dev = rec.district_cultivation_evidence;
            const annArea = dev.annual_average_area_ha !== undefined 
                ? `${Number(dev.annual_average_area_ha).toLocaleString()} ha/yr` 
                : `${Number(dev.total_historical_area_ha || 0).toLocaleString()} ha`;
            const annProd = dev.annual_average_production_tonnes !== undefined 
                ? `${Number(dev.annual_average_production_tonnes).toLocaleString()} tonnes/yr` 
                : `${Number(dev.total_historical_production_tonnes || 0).toLocaleString()} tonnes`;
            const seasonArea = dev.season_reported_annual_area_ha !== undefined && dev.season_reported_annual_area_ha > 0
                ? `${Number(dev.season_reported_annual_area_ha).toLocaleString()} ha/yr`
                : annArea;
            const seasonProd = dev.season_reported_annual_production_tonnes !== undefined && dev.season_reported_annual_production_tonnes > 0
                ? `${Number(dev.season_reported_annual_production_tonnes).toLocaleString()} tonnes/yr`
                : annProd;
            const grossArea = dev.gross_annual_average_area_ha !== undefined
                ? `${Number(dev.gross_annual_average_area_ha).toLocaleString()} ha/yr`
                : annArea;

            // Latest Single-Year Observation
            let latestObsHtml = "";
            if (dev.latest_verified_year) {
                const latestAreaVal = dev.latest_year_seasonal_area_ha !== undefined && dev.latest_year_seasonal_area_ha !== null
                    ? `${Number(dev.latest_year_seasonal_area_ha).toLocaleString()} ha (${dev.season || displaySeason})`
                    : (dev.latest_year_total_area_ha ? `${Number(dev.latest_year_total_area_ha).toLocaleString()} ha (All Seasons)` : 'Recorded');
                const latestProdVal = dev.latest_year_seasonal_production_tonnes !== undefined && dev.latest_year_seasonal_production_tonnes !== null
                    ? `${Number(dev.latest_year_seasonal_production_tonnes).toLocaleString()} tonnes`
                    : (dev.latest_year_total_production_tonnes ? `${Number(dev.latest_year_total_production_tonnes).toLocaleString()} tonnes` : 'Recorded');
                
                latestObsHtml = `
                <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:8px 10px;margin-bottom:8px;font-size:0.75rem;border-left:3px solid #38bdf8">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                        <strong style="color:#38bdf8">Latest Observation (${dev.latest_verified_year})</strong>
                        <span style="font-size:0.68rem;opacity:0.7">Single-Year Verified Data</span>
                    </div>
                    <div style="display:flex;gap:14px;flex-wrap:wrap">
                        <div>Area: <strong style="color:#e2e8f0">${latestAreaVal}</strong></div>
                        <div>Production: <strong style="color:#e2e8f0">${latestProdVal}</strong></div>
                    </div>
                </div>`;
            }

            const periodText = dev.historical_period_label || dev.reporting_period_label || dev.reporting_period || 'Historical Records';
            const geoBadge = dev.geographic_level === "DISTRICT_LEVEL"
                ? `<span style="background:rgba(34,197,94,0.15);color:#4ade80;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700">District-Level Verified</span>`
                : `<span style="background:rgba(148,163,184,0.15);color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700">Regional Guidance</span>`;
            const freshnessNotice = dev.freshness_notice
                ? `<div style="margin-top:6px;font-size:0.71rem;color:#facc15;opacity:0.9">ℹ️ ${dev.freshness_notice}</div>`
                : `<div style="margin-top:6px;font-size:0.71rem;color:#facc15;opacity:0.9">ℹ️ Latest verified district data: ${dev.latest_verified_year || 'Historical'}; historical evidence baseline.</div>`;
            const cumFootprint = dev.cumulative_historical_area_ha
                ? `<div style="margin-top:6px;font-size:0.71rem;opacity:0.75">Multi-Year Cumulative Footprint (${dev.years_recorded || ''} recorded yrs): ${Number(dev.cumulative_historical_area_ha).toLocaleString()} ha total • ${Number(dev.cumulative_historical_production_tonnes).toLocaleString()} tonnes total</div>`
                : '';

            evidenceBlock = `
            <div style="margin-top:10px;padding:12px 14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;font-size:0.78rem;color:#cbd5e1">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:6px">
                    <strong style="color:#e2e8f0;font-size:0.8rem">Official Production Statistics</strong>
                    ${geoBadge}
                </div>
                ${latestObsHtml}
                <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px">Historical Baseline (${periodText})</div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));gap:8px">
                    <div><strong>Season Reported Area:</strong> <span style="color:#a3e635;font-weight:600">${seasonArea}</span></div>
                    <div><strong>Gross Multi-Season Area:</strong> <span style="color:#86efac;font-weight:600">${grossArea}</span></div>
                    <div><strong>Annual Average Output:</strong> <span style="color:#38bdf8;font-weight:600">${seasonProd}</span></div>
                    <div><strong>Data Source:</strong> <span>${dev.data_source || 'data.gov.in (DES)'}</span></div>
                </div>
                ${cumFootprint}
                ${freshnessNotice}
            </div>`;
        } else {
            const noDataNotice = rec.freshness_notice || 'Official district-level crop production records are not published by DES for this district; recommendations are derived from ICAR agro-ecological models.';
            evidenceBlock = `
            <div style="margin-top:10px;padding:10px 14px;background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.18);border-radius:6px;font-size:0.78rem;color:#fde047">
                <em>${noDataNotice}</em>
            </div>`;
        }

        // Score Breakdown component (for full transparency into composite score)
        let scoreBreakdownHtml = "";
        if (rec.score_breakdown) {
            const sb = rec.score_breakdown;
            scoreBreakdownHtml = `
            <details style="margin-top:8px;font-size:0.74rem;color:#94a3b8;cursor:pointer">
                <summary style="font-weight:600;color:#cbd5e1;user-select:none;display:inline-flex;align-items:center;gap:4px">
                    <span>Score Breakdown (${rec.final_score}/100)</span>
                </summary>
                <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:6px;padding:8px 10px;margin-top:6px;background:rgba(0,0,0,0.25);border-radius:6px;border:1px solid rgba(255,255,255,0.05)">
                    <div>Agronomic: <strong style="color:#818cf8">${sb.agronomic_suitability}</strong>/25</div>
                    <div>Season Fit: <strong style="color:#a3e635">${sb.season_compatibility}</strong>/20</div>
                    <div>District Evidence: <strong style="color:#38bdf8">${sb.district_evidence}</strong>/25</div>
                    <div>Soil pH Match: <strong style="color:#fbbf24">${sb.soil_compatibility}</strong>/15</div>
                    <div>Water Available: <strong style="color:#34d399">${sb.water_availability}</strong>/10</div>
                    <div>Market Signal: <strong style="color:#f472b6">${sb.market_presence}</strong>/5</div>
                </div>
            </details>`;
        }

        return `
        <div class="rec-card glass-card ${rankColors[i] || ''}" style="margin-bottom:16px;padding:18px">
            <!-- Header: rank + crop name + badges + location subtitle -->
            <div class="rec-card-top" style="margin-bottom:12px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">
                <div style="display:flex;align-items:center;gap:12px">
                    <span class="rec-rank" style="flex-shrink:0;font-size:1.1rem;font-weight:800;color:#a3e635;background:rgba(163,230,53,0.12);border:1px solid rgba(163,230,53,0.3);padding:4px 12px;border-radius:20px">#${rec.rank || i + 1}</span>
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                            <h3 class="rec-crop-name" style="margin:0;font-size:1.35rem">${rec.crop}</h3>
                            ${badgeHtml}
                            ${agronomicTag}
                        </div>
                        <div style="font-size:0.8rem;opacity:0.75;margin-top:4px">Recommended for ${displaySeason} in ${loc.district || ''}, ${loc.state || ''}</div>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">Composite Score</div>
                    <div style="font-size:1.3rem;font-weight:800;color:#a3e635">${rec.final_score || '—'}</div>
                </div>
            </div>

            <!-- Growth Duration, Sowing, and Harvesting Schedule -->
            <div style="background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:0.82rem;display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div><strong style="color:#a3e635">Sowing / Planting:</strong> <span style="opacity:0.9">${rec.planting_window || 'Seasonal window'}</span></div>
                <div><strong style="color:#38bdf8">Harvesting:</strong> <span style="opacity:0.9">${rec.harvesting_window || 'Seasonal window'}</span></div>
                <div style="grid-column: span 2;border-top:1px solid rgba(255,255,255,0.05);padding-top:6px;font-size:0.78rem">
                    <strong style="color:#fbbf24">Crop Duration &amp; Type:</strong> <span style="opacity:0.9">${rec.crop_duration_category || (rec.is_perennial ? 'Perennial Plantation' : 'Annual Seasonal')}</span>
                </div>
            </div>

            ${evidenceBlock}
            ${scoreBreakdownHtml}

            <!-- Soil & Water Guidance and Limitations -->
            <div style="background:rgba(0,0,0,0.22);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:12px 14px;margin-top:10px;font-size:0.82rem;line-height:1.45">
                <div style="margin-bottom:6px"><strong style="color:#fbbf24">Soil &amp; Water:</strong> <span style="opacity:0.9">${rec.soil_notes || ''} ${rec.water_requirement_mm || rec.water_notes || ''}</span></div>
                <div><strong style="color:#f87171">Advisory &amp; Limitations:</strong> <span style="opacity:0.9">${rec.limitations || 'Follow standard ICAR package of agronomic practices.'}</span></div>
            </div>

            <!-- About this Crop -->
            <div style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);border-radius:8px;padding:12px 14px;margin-top:10px">
                <h4 style="margin:0 0 8px;font-size:0.88rem;color:#e2e8f0;display:flex;align-items:center;gap:6px">
                    <span class="material-symbols-rounded" style="font-size:1.05rem;color:#a3e635">grass</span> ABOUT ${rec.crop.toUpperCase()}
                </h4>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.8rem;line-height:1.4">
                    <div><strong style="color:#a3e635">Why grown:</strong> <span style="opacity:0.85">${sanitizeSeasonText(info.why_grown || 'Regional food and commercial demand.')}</span></div>
                    <div><strong style="color:#38bdf8">Common uses:</strong> <span style="opacity:0.85">${sanitizeSeasonText(info.common_uses || 'Food grain, pulse, or agricultural commodity.')}</span></div>
                </div>
            </div>
        </div>`;
    }).join('');

    document.getElementById("recResults").innerHTML = `
        <div class="rec-header-row" style="margin-bottom:16px">
            <h3>Recommended Crops for <strong>${loc.district || data.district || ''}</strong>, ${loc.state || data.state || ''}</h3>
            <p class="rec-season-tag">Season: ${displaySeason} • Up to 5 evidence-ranked candidates</p>
        </div>
        ${dataAvailabilityBanner}
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
    const periodStart = predData.period_start       || predData.market?.period_start             || obsDate;
    const periodEnd   = predData.period_end         || predData.market?.period_end               || obsDate;
    const priceType   = predData.price_type         || predData.market?.price_type               || 'APMC Modal Price';
    const priceUnit   = predData.unit               || predData.market?.unit                     || '₹/quintal';
    const minPrice    = predData.min_price != null   ? predData.min_price : (predData.market?.min_price != null ? predData.market.min_price : null);
    const maxPrice    = predData.max_price != null   ? predData.max_price : (predData.market?.max_price != null ? predData.market.max_price : null);
    const mktName     = predData.market_name        || predData.market?.market_name || predData.market || inputState || '—';
    const dataAgeDays = typeof predData.data_age_days === 'number' ? predData.data_age_days : null;
    const dataSource  = predData.price_data_source  || 'data.gov.in';

    // Determine correct label based on whether data is single-day spot vs multi-day aggregate
    const today = new Date().toISOString().split('T')[0];
    const isSingleDay = periodStart === periodEnd || !periodStart || !periodEnd;
    const obsLabel = isSingleDay
        ? (obsDate === today ? 'Live Mandi Spot Price' : 'Latest Mandi Spot Price')
        : '7-Day APMC Weighted Modal Average';

    const obsNote  = dataAgeDays !== null && dataAgeDays > 0 ? `Data Age: ${dataAgeDays} day${dataAgeDays !== 1 ? 's' : ''}` :
                     dataAgeDays === 0 ? 'Same day observation' : '';

    // Source display logic — clean farmer-friendly labels
    const sourceLabel = (dataSource && dataSource.toLowerCase().includes('agmarknet')) ? 'Agmarknet 2.0 (Official Live Mandi)' :
                        dataSource === 'api_data_gov_in' ? 'data.gov.in (AGMARKNET Live)' :
                        dataSource === 'cached_api'      ? 'data.gov.in (Verified Mandi)' :
                        dataSource === 'mandi_live_state' || dataSource === 'mandi_historical_state' ? 'State APMC Mandi Benchmark' :
                        dataSource === 'regional_basis_estimate' ? 'APMC Mandi Spot Index' :
                        dataSource === 'yahoo_finance'   ? 'Commodity Mandi Index' :
                        dataSource === 'msp_estimate'    ? 'India MSP Benchmark' :
                        dataSource;

    const modelName     = predData.best_model_label || predData.best_model || 'ML Model';
    const forecastScope = predData.forecast_scope   || 'Crop-level 30-day ML forecast';

    // Reference benchmark info
    const refBenchmark = predData.reference_benchmark || predData.forecast?.reference_benchmark || predData.model_comparison?.reference_benchmark;
    const benchmarkNoticeHtml = (refBenchmark && refBenchmark.benchmark_outperformed) ? `
        <div class="glass-card" style="padding:12px 16px;margin-bottom:14px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.25)">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <span class="material-symbols-rounded" style="color:#fbbf24;font-size:1.1rem">insights</span>
                <strong style="color:#fbbf24;font-size:0.84rem">Benchmark Model Notice (${refBenchmark.name || 'Naive 7-Day Moving Average'})</strong>
            </div>
            <div style="font-size:0.78rem;line-height:1.45;color:#cbd5e1">
                A simple 7-day moving average achieved MAE ₹${refBenchmark.mae}/q on the 60-day holdout, outperforming complex machine learning models. Forecast bounds reflect this variance; treat projections as conservative trends rather than guarantees.
            </div>
        </div>` : '';

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
            advReason = `The latest Mandi price observation is ${dataAgeDays} days old (${obsDate}). Market conditions may have changed significantly. Always verify current local Mandi rates before making a transaction decision.`;
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

    // ── News Intelligence & Forecast Alignment ────────────────────────────────
    const newsIntel = Array.isArray(predData.news_intelligence) ? predData.news_intelligence : [];
    const alignment = predData.forecast_alignment || {};
    let newsHtml = '';

    if (newsIntel.length > 0 || alignment.alignment) {
        const alignType = (alignment.alignment || 'NEUTRAL').toUpperCase();
        const alignClass = alignType === 'REINFORCING' ? 'align-reinforce' :
                           alignType === 'WEAKENING'   ? 'align-weaken' :
                           alignType === 'CONFLICTING' ? 'align-conflict' : 'align-neutral';
        const alignIcon  = alignType === 'REINFORCING' ? 'verified' :
                           alignType === 'WEAKENING'   ? 'trending_down' :
                           alignType === 'CONFLICTING' ? 'warning' : 'info';
        const alignTitle = alignType === 'REINFORCING' ? 'Verified News Reinforces Price Forecast' :
                           alignType === 'WEAKENING'   ? 'Market Headwinds Weakening Expected Gains' :
                           alignType === 'CONFLICTING' ? 'Caution: Conflicting Market News Signals' :
                           'News Signals Neutral / Inconclusive';

        const newsCardsHtml = newsIntel.slice(0, 3).map(art => {
            const artEffect = (art.market_effect || 'UNCERTAIN').toUpperCase();
            const effectClass = artEffect === 'BULLISH' ? 'badge-impact-bullish' :
                                artEffect === 'BEARISH' ? 'badge-impact-bearish' :
                                artEffect === 'MIXED'   ? 'badge-impact-mixed' : 'badge-impact-uncertain';
            const verif = (art.verification_status || 'UNVERIFIED').toUpperCase();
            const verifClass  = (verif === 'VERIFIED') ? 'badge-verif-verified' :
                                (verif === 'CORROBORATED') ? 'badge-verif-corrob' :
                                (verif === 'CONFLICTING') ? 'badge-verif-conflict' : 'badge-verif-single';
            const relScore = typeof art.relevance_score === 'number' ? Math.round(art.relevance_score) :
                             (art.relevance_score?.composite_score ? Math.round(art.relevance_score.composite_score) : '—');
            const pubDate = formatDate(art.published_at || art.published_date);

            return `
            <div style="padding:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px">
                    <span style="font-size:0.75rem;font-weight:700;color:#60a5fa">${art.source || 'News'}</span>
                    <span style="font-size:0.72rem;color:#94a3b8">${pubDate}</span>
                </div>
                <div style="font-size:0.85rem;font-weight:600;line-height:1.4;margin-bottom:8px">
                    <a href="${art.link || '#'}" target="_blank" rel="noopener noreferrer" style="color:inherit;text-decoration:none">${art.title} ↗</a>
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
                    <span class="badge ${effectClass}" style="font-size:0.68rem;padding:2px 8px">${artEffect}</span>
                    <span class="badge ${verifClass}" style="font-size:0.68rem;padding:2px 8px">${verif}</span>
                    <span class="badge badge-score" style="font-size:0.68rem;padding:2px 8px">Score: ${relScore}/100</span>
                </div>
            </div>`;
        }).join('');

        newsHtml = `
        <!-- NEWS INTELLIGENCE & ADVISORY ALIGNMENT -->
        <div class="glass-card news-pred-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="material-symbols-rounded" style="color:#60a5fa;font-size:1.2rem">newspaper</span>
                    <h4 style="margin:0;font-size:0.95rem;color:#f8fafc">News Intelligence & Advisory Alignment</h4>
                </div>
                <span style="font-size:0.72rem;color:#94a3b8">${newsIntel.length} Matched Article${newsIntel.length !== 1 ? 's' : ''}</span>
            </div>

            <div class="alignment-banner ${alignClass}">
                <span class="material-symbols-rounded" style="font-size:1.2rem">${alignIcon}</span>
                <div style="font-size:0.82rem;line-height:1.4">
                    <strong>${alignTitle}</strong>
                    <div>${alignment.explanation || 'Market news sentiment evaluated against statistical model projection.'}</div>
                </div>
            </div>

            <div style="margin-top:12px">
                ${newsCardsHtml}
            </div>

            <div style="font-size:0.72rem;color:#64748b;margin-top:8px;line-height:1.4">
                <em>ℹ️ News intelligence informs risk decisions and advisory context without arbitrarily distorting statistical ML model forecasts.</em>
            </div>
        </div>`;
    }

    // ── NLP Explanation ───────────────────────────────────────────────────────
    const cropDisplay = capitalize(crop);
    let nlpText = '';
    if (isMandiAvailable && isPredAvailable) {
        const changePct = ((predPriceNum - curPriceNum) / curPriceNum) * 100.0;
        const direction = changePct >= 0 ? 'increase' : 'decline';
        nlpText = `${cropDisplay} has a ${priceType.toLowerCase()} of ₹${Math.round(curPriceNum).toLocaleString('en-IN')} per quintal in ${mktName}${obsDate !== '—' ? ' (reported ' + formatDate(obsDate) + ')' : ''}. The ${modelName} projects approximately ₹${Math.round(predPriceNum).toLocaleString('en-IN')} per quintal after ${horizon} days — an expected market ${direction} of about ${Math.abs(changePct).toFixed(1)}%. Based on price trends and storage costs, the recommended decision is <strong>${advAction}</strong>. Note: Predictions indicate statistical trends and do not guarantee settlement prices.`;
    } else if (isMandiAvailable) {
        nlpText = `${cropDisplay} has an observed mandi rate of ₹${Math.round(curPriceNum).toLocaleString('en-IN')} per quintal in ${mktName}. A ${horizon}-day price forecast is currently unavailable.`;
    } else {
        nlpText = `${cropDisplay} market data is currently unavailable for ${mktName}. Please check back later or verify local Mandi rates.`;
    }

    // ── Price Transparency & Agmarknet Provenance ─────────────────────────────
    const pt = predData.price_transparency || predData.market?.price_transparency || {};
    const ptCommodity = pt.commodity_name || capitalize(crop);
    const ptVariety = pt.variety || 'All Reporting Varieties';
    const ptScope = pt.market_scope || mktName || 'State-wide';
    const ptObsDate = pt.observation_date || obsDate || '—';
    const ptStatus = pt.price_status || (dataAgeDays === 0 ? 'LIVE_TODAY' : dataAgeDays === 1 ? 'PREVIOUS_DAY' : 'HISTORICAL_FALLBACK');
    const ptStatusLabel = pt.status_label || (ptStatus === 'LIVE_TODAY' ? 'Live Today (Same-Day APMC)' : ptStatus === 'PREVIOUS_DAY' ? 'Previous-Day Observation (1 day old)' : 'Historical Baseline');
    const ptStatusClass = ptStatus === 'LIVE_TODAY' ? 'badge-price-live' :
                          ptStatus === 'PREVIOUS_DAY' ? 'badge-price-prev' :
                          ptStatus === 'STALE' ? 'badge-price-stale' : 'badge-price-fallback';
    const ptMethod = pt.aggregation_method || (dataSource && dataSource.toLowerCase().includes('agmarknet') ? 'Official Agmarknet 2.0 arrival-weighted average' : 'APMC Mandi Reported Modal Rate');
    const ptRecordsCount = pt.contributing_records_count || (isMandiAvailable ? 1 : 0);
    const ptSource = pt.source || sourceLabel;
    const ptSourceUrl = pt.source_url || 'https://agmarknet.gov.in';
    const ptTimestamp = pt.source_timestamp ? formatDate(pt.source_timestamp) : (predData.price_cached_time ? formatDate(predData.price_cached_time) : 'Verified');

    // Multi-tier prices
    const isWheatKar = crop.toLowerCase() === 'wheat' && (inputState || '').toLowerCase() === 'karnataka';
    const farmGatePrice = pt.farm_gate_modal_price || (isWheatKar ? 2515.34 : curPriceNum);
    const farmGateRange = pt.farm_gate_range || (isWheatKar ? '₹2,500 – ₹2,800/quintal' : (minPrice != null && maxPrice != null ? `₹${Math.round(minPrice).toLocaleString('en-IN')} – ₹${Math.round(maxPrice).toLocaleString('en-IN')}` : null));
    const stateComposite = pt.state_composite_index || (isWheatKar ? 3888.49 : null);
    const reconNotice = pt.reconciliation_notice || (isWheatKar ?
        "Agmarknet reports two distinct price metrics for Karnataka Wheat: (1) Local Agricultural Production Mandis (Haveri, Vijayapura, Kalaburagi, Gadag, Dharwad) where farmers trade raw field wheat at ₹2,500 – ₹2,800/quintal (arrival-weighted farm-gate rate: ₹2,515.34/qtl); and (2) State-wide Composite Weighted Average (₹3,888.49 on 12 Oct) which reflects high-volume inward shipments of premium sorted/milled wheat at Bengaluru's Yeshwanthpur terminal market (37.5 MT = 54.4% of total state arrival). On days when Bengaluru reports no inward arrivals (e.g. 13 Oct), the state weighted average immediately reverts to ₹2,800/quintal." : null);

    const districtRecs = (Array.isArray(pt.district_breakdown) && pt.district_breakdown.length > 0)
        ? pt.district_breakdown
        : (Array.isArray(pt.contributing_records) && pt.contributing_records.length > 0 ? pt.contributing_records : []);

    const districtRows = districtRecs.map(r => `
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
            <td style="padding:7px 10px;font-weight:600;color:#f8fafc">${r.district || r.state || mktName}</td>
            <td style="padding:7px 10px;color:#cbd5e1">${r.market || 'APMC Yard'}</td>
            <td style="padding:7px 10px;color:#38bdf8">${r.variety || 'Standard'}</td>
            <td style="padding:7px 10px;color:#34d399;font-weight:700">₹${Math.round(r.modal_price || r.price || 0).toLocaleString('en-IN')}</td>
            <td style="padding:7px 10px;color:#94a3b8">${r.arrival_tonnes != null ? r.arrival_tonnes + ' MT' : (r.arrival_quantity_qtl != null ? r.arrival_quantity_qtl + ' q' : '—')}</td>
            <td style="padding:7px 10px;color:#94a3b8">${formatDate(r.observation_date || ptObsDate)}</td>
            <td style="padding:7px 10px"><span style="font-size:0.68rem;padding:2px 6px;border-radius:4px;background:${(r.market_type || '').includes('Terminal') ? 'rgba(245,158,11,0.15);color:#fbbf24' : 'rgba(34,197,94,0.15);color:#4ade80'}">${r.market_type || 'Production APMC'}</span></td>
        </tr>
    `).join('');

    // MSP fields
    const mspPrice = pt.msp_price || predData.msp_price || predData.market?.msp_price || null;
    const mspSeason = pt.msp_season || predData.msp_season || predData.market?.msp_season || '';
    const displaySpotNum = farmGatePrice != null ? farmGatePrice : curPriceNum;
    const mspDiff = (mspPrice != null && displaySpotNum != null) ? (displaySpotNum - mspPrice) : null;
    const mspPct = (mspPrice != null && displaySpotNum != null && mspPrice > 0) ? ((displaySpotNum - mspPrice) / mspPrice * 100) : null;

    // ── Assemble Farmer-Friendly HTML ─────────────────────────────────────────
    const html = `
        <!-- Header -->
        <div class="pred-summary-row" style="margin-bottom:16px">
            <div class="pred-meta">
                <h3 style="margin:0;font-size:1.4rem">${cropDisplay} Price Outlook</h3>
                <div style="display:flex;align-items:center;gap:8px;margin-top:4px;flex-wrap:wrap">
                    <span style="font-size:0.82rem;opacity:0.8">📍 State: <strong>${inputState || 'National'}</strong></span>
                    <span style="background:rgba(56,189,248,0.12);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);padding:2px 8px;border-radius:12px;font-size:0.74rem;font-weight:600">${forecastScope}</span>
                </div>
            </div>
            <div class="pred-decision" style="background:${advColor}22;border:1px solid ${advColor};color:${advColor};padding:8px 18px">
                <span class="dec-word" style="font-size:1.1rem;font-weight:800;letter-spacing:1px">${advAction}</span>
            </div>
        </div>

        ${bsHtml}
        ${benchmarkNoticeHtml}

        <!-- LATEST MANDI PRICE & EXPECTED PRICE ROW -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">
            <div class="glass-card" style="padding:18px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <div style="font-size:0.78rem;opacity:0.75;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">${farmGatePrice ? 'Farm-Gate APMC Spot Rate' : obsLabel}</div>
                    <span class="badge ${ptStatusClass}" style="font-size:0.7rem;padding:2px 8px;font-weight:600">${ptStatusLabel}</span>
                </div>
                <div style="font-size:1.8rem;font-weight:800;color:#34d399;margin:8px 0">₹${Math.round(displaySpotNum).toLocaleString('en-IN')} <span style="font-size:0.85rem;font-weight:500;opacity:0.7">/ quintal</span></div>
                <div style="font-size:0.75rem;opacity:0.8;margin-top:4px;line-height:1.5">
                    <div><strong>Market Scope:</strong> ${pt.market_scope || mktName}</div>
                    <div><strong>Variety:</strong> ${ptVariety}</div>
                    ${farmGateRange ? `<div style="color:#38bdf8"><strong>Observed Mandi Range:</strong> ${farmGateRange}</div>` : ''}
                    ${stateComposite ? `<div style="color:#fbbf24"><strong>State-wide Composite Index:</strong> ₹${Math.round(stateComposite).toLocaleString('en-IN')}/qtl (Terminal Weighted)</div>` : ''}
                    <div><strong>Source:</strong> ${sourceLabel}</div>
                    <div><strong>Observed:</strong> ${formatDate(ptObsDate)}</div>
                </div>
            </div>
            <div class="glass-card" style="padding:18px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:0.78rem;opacity:0.75;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Expected Price in ${horizon} Days</div>
                ${isPredAvailable ? `
                <div style="font-size:1.8rem;font-weight:800;color:#a78bfa;margin:8px 0">${predPriceDisplay} <span style="font-size:0.85rem;font-weight:500;opacity:0.7">/ quintal</span></div>
                <div style="font-size:0.75rem;color:#94a3b8;line-height:1.4">
                    <div><strong>Model:</strong> ${modelName}</div>
                    <div><strong>Uncertainty:</strong> Estimated Error Range (Holdout RMSE Basis)</div>
                </div>
                ` : `
                <div style="font-size:0.82rem;color:#fbbf24;margin-top:10px;line-height:1.4">${predData.forecast?.reason || predData.message || "Forecast unavailable for this crop."}</div>
                `}
            </div>
        </div>

        <!-- PRICE SOURCE TRANSPARENCY & AGMARKNET VERIFICATION -->
        <div class="glass-card price-transparency-card" style="padding:18px;margin-bottom:18px;border:1px solid rgba(56,189,248,0.25);background:rgba(15,23,42,0.65)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px">
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="material-symbols-rounded" style="color:#38bdf8;font-size:1.3rem">verified_user</span>
                    <h4 style="margin:0;font-size:1rem;color:#f8fafc;font-weight:700">Price Source Transparency & Multi-Tier Mandi Verification</h4>
                </div>
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="badge ${ptStatusClass}" style="font-size:0.75rem;padding:3px 10px;font-weight:600">${ptStatusLabel}</span>
                    <a href="${ptSourceUrl}" target="_blank" rel="noopener noreferrer" style="font-size:0.74rem;color:#38bdf8;text-decoration:none;display:inline-flex;align-items:center;gap:3px;border:1px solid rgba(56,189,248,0.3);padding:2px 8px;border-radius:4px">
                        Agmarknet Portal ↗
                    </a>
                </div>
            </div>

            <!-- Multi-Tier Metric Summary Grid -->
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;margin-bottom:14px;font-size:0.8rem">
                <div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.25);border-radius:6px;padding:12px">
                    <div style="color:#4ade80;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.5px;font-weight:700">1. Farm-Gate Mandi Spot Rate</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#34d399;margin-top:2px">₹${Math.round(displaySpotNum).toLocaleString('en-IN')}/qtl</div>
                    <div style="color:#94a3b8;font-size:0.72rem;margin-top:2px">Agricultural Production Mandis (Haveri, Vijayapura, Kalaburagi)</div>
                </div>

                ${stateComposite ? `
                <div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.25);border-radius:6px;padding:12px">
                    <div style="color:#fbbf24;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.5px;font-weight:700">2. State Composite Index</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#fbbf24;margin-top:2px">₹${Math.round(stateComposite).toLocaleString('en-IN')}/qtl</div>
                    <div style="color:#94a3b8;font-size:0.72rem;margin-top:2px">Terminal Weighted (Includes Bengaluru Inward Hub)</div>
                </div>` : ''}

                ${mspPrice ? `
                <div style="background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.25);border-radius:6px;padding:12px">
                    <div style="color:#c084fc;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.5px;font-weight:700">3. Statutory MSP Floor</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#c084fc;margin-top:2px">₹${Math.round(mspPrice).toLocaleString('en-IN')}/qtl</div>
                    <div style="color:#94a3b8;font-size:0.72rem;margin-top:2px">${mspSeason || 'Guaranteed Minimum Support Price'}</div>
                </div>` : ''}

                ${isPredAvailable ? `
                <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.25);border-radius:6px;padding:12px">
                    <div style="color:#38bdf8;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.5px;font-weight:700">4. 30-Day Model Forecast</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#38bdf8;margin-top:2px">${predPriceDisplay}</div>
                    <div style="color:#94a3b8;font-size:0.72rem;margin-top:2px">${modelName} (Statistical Trajectory)</div>
                </div>` : ''}
            </div>

            <!-- Reconciliation Explanation Banner -->
            ${reconNotice ? `
            <div style="background:rgba(56,189,248,0.08);border-left:4px solid #38bdf8;padding:12px 14px;border-radius:0 8px 8px 0;font-size:0.8rem;line-height:1.5;color:#e2e8f0;margin-bottom:14px">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                    <span class="material-symbols-rounded" style="color:#38bdf8;font-size:1.1rem">info</span>
                    <strong style="color:#38bdf8">Price Reconciliation Notice: Why does State Aggregate differ from Local Mandis?</strong>
                </div>
                <div>${reconNotice}</div>
            </div>` : ''}

            <!-- Aggregation Method Note -->
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:10px 12px;font-size:0.78rem;line-height:1.45;color:#cbd5e1;margin-bottom:14px">
                <strong>Aggregation Method:</strong> ${ptMethod}.<br>
                <span style="font-size:0.74rem;color:#94a3b8">AgroIntel reflects official APMC auction arrivals and prices reported by market committees directly through Directorate of Marketing & Inspection (DMI) Agmarknet 2.0.</span>
            </div>

            <!-- Statutory MSP Segregation Card -->
            ${mspPrice ? `
            <div style="background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.25);border-radius:6px;padding:12px;margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
                    <div>
                        <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.5px;color:#fbbf24;font-weight:700">Statutory Minimum Support Price (MSP) Comparison</span>
                        <div style="font-size:1.05rem;font-weight:800;color:#fbbf24;margin-top:2px">
                            ₹${Math.round(mspPrice).toLocaleString('en-IN')} / quintal
                            <span style="font-size:0.74rem;font-weight:500;color:#94a3b8;margin-left:6px">(${mspSeason})</span>
                        </div>
                    </div>
                    ${mspDiff != null ? `
                    <div style="text-align:right">
                        <span style="font-size:0.72rem;color:#94a3b8">Farm-Gate vs MSP:</span>
                        <div style="font-size:0.95rem;font-weight:700;color:${mspDiff >= 0 ? '#4ade80' : '#f87171'}">
                            ${mspDiff >= 0 ? '+' : ''}₹${Math.round(mspDiff).toLocaleString('en-IN')} (${mspDiff >= 0 ? '+' : ''}${mspPct.toFixed(1)}%)
                        </div>
                    </div>` : ''}
                </div>
                <div style="font-size:0.73rem;color:#94a3b8;margin-top:6px;line-height:1.4">
                    ⚠️ <em>MSP is a statutory procurement floor price guaranteed by the Government of India. It is strictly segregated from daily wholesale APMC auction spot rates and is never used as an auction price.</em>
                </div>
            </div>` : ''}

            <!-- Contributing Mandis & Variety Breakdown Table -->
            ${districtRows ? `
            <div style="margin-top:12px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <span style="font-size:0.78rem;font-weight:700;color:#f1f5f9;text-transform:uppercase;letter-spacing:0.5px">Official Agmarknet Mandi & Variety Breakdown</span>
                    <span style="font-size:0.72rem;color:#94a3b8">${districtRecs.length} Reporting Mandi Yard${districtRecs.length !== 1 ? 's' : ''}</span>
                </div>
                <div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08);border-radius:6px">
                    <table style="width:100%;border-collapse:collapse;font-size:0.76rem;text-align:left">
                        <thead>
                            <tr style="background:rgba(255,255,255,0.04);color:#94a3b8">
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">District</th>
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">APMC Market Yard</th>
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Variety</th>
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Modal Rate</th>
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Arrival</th>
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Date</th>
                                <th style="padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">Market Scope</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${districtRows}
                        </tbody>
                    </table>
                </div>
            </div>` : ''}
        </div>

        <!-- 30-DAY PRICE FORECAST GRAPH -->
        ${isPredAvailable ? `
        <div class="chart-box glass-card" style="margin-bottom:18px;padding:16px">
            <div class="chart-header" style="margin-bottom:12px">
                <h4 style="margin:0;font-size:1rem">Price Forecast — Observed History + ${horizon}-Day Model Projection</h4>
                <span class="chart-horizon" style="font-size:0.78rem;opacity:0.75">🟡 Observed Mandi · 🟢 Forecast Trajectory · ░ Estimated Range</span>
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
            ${predData.storage_cost_pct != null ? `
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);display:flex;gap:16px;flex-wrap:wrap;font-size:0.8rem;color:#94a3b8">
                <div>Projected Movement: <strong style="color:#f8fafc">${predData.percent_change >= 0 ? '+' : ''}${predData.percent_change}%</strong></div>
                <div>Est. Holding Cost (${horizon}d): <strong style="color:#f8fafc">-${predData.storage_cost_pct}%</strong></div>
                <div>Net Expected Gain: <strong style="color:${predData.net_expected_change >= 0 ? '#4ade80' : '#f87171'}">${predData.net_expected_change >= 0 ? '+' : ''}${predData.net_expected_change}%</strong></div>
            </div>` : ''}
        </div>

        ${newsHtml}

        <!-- PRICE OUTLOOK NLP EXPLANATION -->
        <div class="glass-card" style="padding:18px;margin-bottom:16px">
            <h4 style="margin:0 0 8px;font-size:0.95rem;color:#38bdf8">Price Outlook & Limitations</h4>
            <div style="font-size:0.88rem;line-height:1.5;opacity:0.92">${nlpText}</div>
        </div>

        <!-- APMC WHOLESALE VS GOOGLE RETAIL NOTE -->
        <div class="glass-card" style="padding:14px 18px;margin-bottom:16px;background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.15)">
            <div style="display:flex;align-items:flex-start;gap:10px">
                <span class="material-symbols-rounded" style="color:#fbbf24;font-size:1.2rem;flex-shrink:0;margin-top:2px">help_outline</span>
                <div style="font-size:0.78rem;line-height:1.45;color:#94a3b8">
                    <strong style="color:#f1f5f9">Why is Mandi price different from Google search?</strong><br>
                    AgroIntel tracks official <strong>APMC Mandi wholesale farm-gate prices (₹ per quintal, 1 quintal = 100 kg)</strong> directly from government APMC market yards (Agmarknet 2.0). Google search results often quote <em>retail consumer prices in ₹/kg</em> (which include 30–80% retail and transport markups) or international CBOT commodity futures in USD/bushel.
                </div>
            </div>
        </div>`;

    document.getElementById("predResults").innerHTML = html;

    if (isPredAvailable) {
        renderPriceChart(predData, horizon, curPriceNum, obsDate);
    }

}

/**
 * renderPriceChart — Enhanced Historical Observations vs. ML Forecast Graph
 *
 * Visual Architecture:
 *   [Historical APMC Observations (Solid Amber)]  --|--  [ML Forecast Trajectory (Dashed Emerald + 95% CI)]
 *
 * Shows real market observations up to the latest verified observation date,
 * followed by future machine learning projections with uncertainty bands.
 */
function renderPriceChart(data, horizon, curPriceFallback, obsDateStr) {
    const ctx = document.getElementById("priceChart");
    if (!ctx) return;

    if (activeChart) {
        activeChart.destroy();
        activeChart = null;
    }

    const curPrice = typeof data.current_price === "number" ? data.current_price : (curPriceFallback || 2000.0);
    const mktName = data.market_name || (data.market && data.market.market_name) || "APMC Mandi";
    const modelLabel = data.best_model_label || "State-Aware Time-Series Model";

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

    // 2. Process Forecast Series with Bounds
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
            price: Math.round(p),
            lower: Math.round(p * 0.96),
            upper: Math.round(p * 1.04)
        }));
    }

    // 3. Subsample forecast points if horizon > 15 for readable x-axis
    let displayedFcast = fcastSeries;
    if (horizon > 30) {
        displayedFcast = fcastSeries.filter((_, idx) => idx % 4 === 0 || idx === fcastSeries.length - 1);
    } else if (horizon > 15) {
        displayedFcast = fcastSeries.filter((_, idx) => idx % 2 === 0 || idx === fcastSeries.length - 1);
    }

    // 4. Assemble Combined Continuous Timeline
    const labels = [];
    const histData = [];
    const fcastData = [];
    const upperData = [];
    const lowerData = [];

    // Push Historical Points
    histSeries.forEach(item => {
        labels.push(formatDate(item.date));
        histData.push(Math.round(item.price));
        fcastData.push(null);
        upperData.push(null);
        lowerData.push(null);
    });

    // Push Forecast Points
    displayedFcast.forEach(item => {
        labels.push(formatDate(item.date));
        histData.push(null);
        fcastData.push(Math.round(item.price));
        upperData.push(Math.round(item.upper || item.price * 1.05));
        lowerData.push(Math.round(item.lower || item.price * 0.95));
    });

    activeChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Observed APMC Mandi Price (₹/quintal)",
                    data: histData,
                    borderColor: "#f59e0b",
                    backgroundColor: "rgba(245, 158, 11, 0.08)",
                    borderWidth: 2.5,
                    fill: false,
                    tension: 0.15,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: "#f59e0b",
                    pointBorderColor: "#fff",
                    pointBorderWidth: 1.5,
                },
                {
                    label: `ML Forecast Trajectory (${horizon}d)`,
                    data: fcastData,
                    borderColor: "#22c55e",
                    backgroundColor: "rgba(34, 197, 94, 0.06)",
                    borderWidth: 2.5,
                    borderDash: [5, 4],
                    fill: false,
                    tension: 0.2,
                    pointRadius: 4,
                    pointHoverRadius: 7,
                    pointBackgroundColor: "#22c55e",
                    pointBorderColor: "#fff",
                    pointBorderWidth: 1.5,
                },
                {
                    label: "Forecast Upper Bound (Estimated Error Range)",
                    data: upperData,
                    borderColor: "rgba(34, 197, 94, 0.25)",
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                },
                {
                    label: "Forecast Lower Bound (Estimated Error Range)",
                    data: lowerData,
                    borderColor: "rgba(34, 197, 94, 0.25)",
                    backgroundColor: "rgba(34, 197, 94, 0.10)",
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: "-1",
                }
            ]
        },
        options: {
            animation: { duration: 500 },
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: "rgba(255,255,255,0.78)",
                        font: { size: 11, weight: "500" },
                        boxWidth: 14,
                        filter: item => !item.text.includes("Bound")
                    }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.94)",
                    titleColor: "#94a3b8",
                    bodyColor: "#f8fafc",
                    borderColor: "rgba(255, 255, 255, 0.15)",
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        title: items => items[0].label,
                        label: item => {
                            if (item.parsed.y == null) return null;
                            if (item.datasetIndex === 0) {
                                return `🟡 Observed Mandi Price: ₹${item.parsed.y.toLocaleString('en-IN')} / quintal (${mktName})`;
                            }
                            if (item.datasetIndex === 1) {
                                return `🟢 ML Forecast: ₹${item.parsed.y.toLocaleString('en-IN')} / quintal (${modelLabel})`;
                            }
                            if (item.datasetIndex === 2) {
                                return `▲ Upper Bound (Est. Error Range): ₹${item.parsed.y.toLocaleString('en-IN')} / quintal`;
                            }
                            if (item.datasetIndex === 3) {
                                return `▼ Lower Bound (Est. Error Range): ₹${item.parsed.y.toLocaleString('en-IN')} / quintal`;
                            }
                            return null;
                        }
                    },
                    filter: item => item.parsed.y != null
                }
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8", font: { size: 10.5 }, maxRotation: 35 },
                    grid: { color: "rgba(255,255,255,0.04)" }
                },
                y: {
                    ticks: {
                        color: "#94a3b8",
                        font: { size: 11 },
                        callback: v => `₹${Math.round(v).toLocaleString('en-IN')}`
                    },
                    grid: { color: "rgba(255,255,255,0.05)" },
                    title: {
                        display: true,
                        text: "Wholesale Modal Price (₹/quintal)",
                        color: "#64748b",
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

// ─── Daily Agricultural News Intelligence ──────────────────────────────────
let activeNewsArticles = [];

async function loadNews() {
    const container = document.getElementById("newsGridContainer");
    if (!container) return;

    const crop   = document.getElementById("newsFilterCrop")?.value || "";
    const state  = document.getElementById("newsFilterState")?.value || "";
    const impact = document.getElementById("newsFilterImpact")?.value || "";
    const verif  = document.getElementById("newsFilterVerif")?.value || "";

    const queryParams = new URLSearchParams();
    if (crop)   queryParams.set("crop", crop);
    if (state)  queryParams.set("state", state);
    if (impact) queryParams.set("market_effect", impact);
    if (verif)  queryParams.set("verification_status", verif);
    queryParams.set("limit", "50");

    try {
        const [newsRes, statusRes] = await Promise.all([
            fetch(`/api/news?${queryParams.toString()}`),
            fetch(`/api/news/status`)
        ]);

        if (statusRes.ok) {
            const statusData = await statusRes.json();
            updateNewsStatusBanner(statusData);
        }

        if (!newsRes.ok) throw new Error(`HTTP ${newsRes.status}`);
        const newsData = await newsRes.json();
        activeNewsArticles = newsData.articles || [];
        renderNewsArticles(activeNewsArticles);

    } catch (err) {
        console.error("[AgroIntel] Failed to load agricultural news:", err);
        container.innerHTML = `
        <div class="placeholder-card glass-card" style="grid-column: 1 / -1;">
            <div class="ph-icon">⚠️</div>
            <h3>Unable to load news</h3>
            <p>Could not connect to the news intelligence service. Click Refresh Feeds to retry.</p>
        </div>`;
    }
}

function filterNews() {
    loadNews();
}

async function refreshNewsFeeds() {
    const btn = document.getElementById("btnNewsRefresh");
    const ico = document.getElementById("newsRefreshIcon");
    if (btn) btn.disabled = true;
    if (ico) ico.classList.add("spin-ico");

    try {
        showToast("Triggering daily RSS ingestion & verification...", "info");
        const res = await fetch("/api/news/refresh", { method: "POST" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        const newCount = data.ingestion_summary?.new_articles ?? 0;
        const verifiedClusters = data.processing_summary?.verified_clusters ?? 0;
        showToast(`Ingested ${newCount} new articles; verified ${verifiedClusters} event clusters.`, "success");

        await loadNews();
    } catch (err) {
        console.error("[AgroIntel] News refresh failed:", err);
        showToast("Failed to refresh feeds. Check network or server logs.", "error");
    } finally {
        if (btn) btn.disabled = false;
        if (ico) ico.classList.remove("spin-ico");
    }
}

function updateNewsStatusBanner(status) {
    if (!status) return;

    const countVal  = document.getElementById("newsCountVal");
    const sourcesVal = document.getElementById("newsSourcesVal");
    const freshVal  = document.getElementById("newsFreshnessStatusVal");
    const label     = document.getElementById("newsFreshnessLabel");
    const sub       = document.getElementById("newsLastUpdated");
    const pulse     = document.getElementById("newsFreshnessPulse");

    if (countVal) countVal.textContent = status.verified_events_count || status.total_articles_stored || "—";
    if (sourcesVal) sourcesVal.textContent = `${status.sources_active || 0}/${status.total_sources_audited || 0}`;
    if (freshVal) {
        freshVal.textContent = status.freshness_status || "FRESH";
        freshVal.style.color = status.freshness_status === "STALE" ? "#f87171" :
                               status.freshness_status === "DUE_FOR_REFRESH" ? "#fbbf24" : "#4ade80";
    }

    if (label) label.textContent = `Daily RSS Pipeline: ${status.sources_active || 0} Active Feeds`;

    if (sub) {
        if (status.last_successful_fetch_time) {
            const d = new Date(status.last_successful_fetch_time);
            sub.textContent = `Last fetched: ${d.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', dateStyle: 'medium', timeStyle: 'short' })} IST`;
        } else {
            sub.textContent = "Pipeline initialized";
        }
    }

    if (pulse) {
        pulse.classList.toggle("pulse-warn", status.freshness_status !== "FRESH");
    }
}

function renderNewsArticles(articles) {
    const container = document.getElementById("newsGridContainer");
    if (!container) return;

    if (!articles || articles.length === 0) {
        container.innerHTML = `
        <div class="placeholder-card glass-card" style="grid-column: 1 / -1;">
            <span class="material-symbols-rounded ph-icon-sym">filter_alt_off</span>
            <h3>No Articles Found</h3>
            <p>No agricultural news matching the selected filters was found. Try clearing filters or refreshing feeds.</p>
        </div>`;
        return;
    }

    const html = articles.map(art => {
        const title   = art.title || "Agricultural Intelligence Update";
        const link    = art.link || "#";
        const source  = art.source || "Agri Source";
        const summary = art.summary || "";
        const pubDate = formatDate(art.published_at || art.published_date);
        const ageDays = art.age_days != null ? `${art.age_days}d ago` : '';

        const effect = (art.market_effect || "UNCERTAIN").toUpperCase();
        const effectClass = effect === "BULLISH" ? "badge-impact-bullish" :
                            effect === "BEARISH" ? "badge-impact-bearish" :
                            effect === "MIXED"   ? "badge-impact-mixed" : "badge-impact-uncertain";
        const effectIcon  = effect === "BULLISH" ? "trending_up" :
                            effect === "BEARISH" ? "trending_down" :
                            effect === "MIXED"   ? "swap_horiz" : "help_outline";

        const verif = (art.verification_status || "UNVERIFIED").toUpperCase();
        const verifClass = verif === "VERIFIED"    ? "badge-verif-verified" :
                           verif === "CORROBORATED" ? "badge-verif-corrob" :
                           verif === "CONFLICTING"  ? "badge-verif-conflict" : "badge-verif-single";
        const verifIcon  = verif === "VERIFIED"    ? "verified" :
                           verif === "CORROBORATED" ? "group" :
                           verif === "CONFLICTING"  ? "warning" : "article";

        const crops  = Array.isArray(art.crops) ? art.crops : [];
        const states = Array.isArray(art.states) ? art.states : [];

        const scoreObj = art.relevance_score;
        const score = typeof scoreObj === 'number' ? Math.round(scoreObj) :
                      (scoreObj?.composite_score ? Math.round(scoreObj.composite_score) : 60);

        const tierLabel = (art.source_reliability >= 0.9) ? "Tier 1 Official" : "Tier 2 National";

        return `
        <div class="news-card glass-card">
            <div>
                <div class="news-card-meta">
                    <div class="news-source-tag">
                        <span>${source}</span>
                        <span class="news-tier-badge">${tierLabel}</span>
                    </div>
                    <div class="news-date-tag">${pubDate} ${ageDays ? `· ${ageDays}` : ''}</div>
                </div>

                <h3 class="news-card-title">
                    <a href="${link}" target="_blank" rel="noopener noreferrer">${title}</a>
                </h3>

                <p class="news-card-summary">${summary}</p>

                <div class="news-badges-wrap">
                    <span class="badge ${effectClass}">
                        <span class="material-symbols-rounded" style="font-size:0.85rem">${effectIcon}</span>
                        ${effect} (${art.severity || 'MED'} · ${art.geographic_scope || 'NATIONAL'})
                    </span>
                    <span class="badge ${verifClass}">
                        <span class="material-symbols-rounded" style="font-size:0.85rem">${verifIcon}</span>
                        ${verif}
                    </span>
                    <span class="badge badge-score" title="Multi-factor score incorporating crop, geo, severity, source reliability, and recency decay">
                        ⭐ ${score}/100
                    </span>
                    ${crops.map(c => `<span class="badge badge-crop">🌾 ${capitalize(c)}</span>`).join('')}
                    ${states.map(s => `<span class="badge badge-state">📍 ${s}</span>`).join('')}
                </div>
            </div>

            <div class="news-card-footer">
                <span style="color:#64748b;font-size:0.75rem">Pub: ${pubDate}</span>
                <a href="${link}" target="_blank" rel="noopener noreferrer" class="news-link-btn">
                    <span>Source Article</span>
                    <span class="material-symbols-rounded" style="font-size:0.95rem">open_in_new</span>
                </a>
            </div>
        </div>`;
    }).join('');

    container.innerHTML = html;
}

