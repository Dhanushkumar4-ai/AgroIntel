# AGROINTEL — FINAL 140 STATE-CROP PRICE FORECAST AUDIT REPORT

## Executive Summary
- **Total Combinations Evaluated**: 140 (28 States × 5 Crops)
- **Passed Inferences**: 140/140 (100%)
- **Decision Logic Compliance**: 140/140 (100% Match with Change % Thresholds)
- **Decision Breakdown**: **SELL: 35**, **HOLD: 62**, **WAIT: 43**
- **State-Specific Models (`STATE_CROP`)**: 109
- **Crop-Only Fallbacks (`CROP_ONLY_FALLBACK`)**: 31 (sparse state records < 50 rows)

## Crop Diversity & Decision Breakdown

| Crop | States Tested | STATE_CROP | CROP_ONLY Fallback | Unique Prices | SELL | HOLD | WAIT | Audit Status |
|------|---------------|------------|--------------------|---------------|------|------|------|--------------|
| Rice | 28 | 21 | 7 | 24 | **16** | **6** | **6** | **PASSED (100%)** |
| Wheat | 28 | 17 | 11 | 18 | **8** | **16** | **4** | **PASSED (100%)** |
| Maize | 28 | 19 | 9 | 21 | **9** | **5** | **14** | **PASSED (100%)** |
| Onion | 28 | 26 | 2 | 27 | **1** | **22** | **5** | **PASSED (100%)** |
| Potato | 28 | 26 | 2 | 27 | **1** | **13** | **14** | **PASSED (100%)** |

## Complete 28-State × 5-Crop Price Forecast & Decision Matrix

| State | Rice | Wheat | Maize | Onion | Potato |
|-------|------|-------|-------|-------|--------|
| Andhra Pradesh | ₹2237 (SELL) | ₹3137 (SELL) * | ₹1783 (SELL) | ₹3979 (WAIT) | ₹2814 (WAIT) |
| Arunachal Pradesh | ₹2515 (WAIT) * | ₹3137 (HOLD) * | ₹2518 (WAIT) * | ₹3200 (HOLD) * | ₹2734 (HOLD) * |
| Assam | ₹1605 (HOLD) | ₹2585 (WAIT) | ₹2051 (WAIT) | ₹3519 (WAIT) | ₹3958 (HOLD) |
| Bihar | ₹2388 (SELL) | ₹2890 (SELL) | ₹2041 (WAIT) | ₹4247 (HOLD) | ₹2280 (WAIT) |
| Chhattisgarh | ₹2314 (SELL) | ₹2797 (WAIT) | ₹2205 (SELL) | ₹3830 (HOLD) | ₹2462 (WAIT) |
| Goa | ₹2515 (WAIT) * | ₹3137 (HOLD) * | ₹2518 (WAIT) * | ₹3577 (HOLD) | ₹1999 (WAIT) |
| Gujarat | ₹2473 (HOLD) | ₹3224 (HOLD) | ₹2586 (WAIT) | ₹2856 (HOLD) | ₹2339 (WAIT) |
| Haryana | ₹3290 (SELL) | ₹2759 (SELL) | ₹1734 (HOLD) | ₹3421 (HOLD) | ₹1690 (WAIT) |
| Himachal Pradesh | ₹2515 (WAIT) * | ₹3137 (HOLD) * | ₹2518 (HOLD) * | ₹3972 (HOLD) | ₹2206 (WAIT) |
| Jharkhand | ₹2116 (HOLD) | ₹2601 (HOLD) | ₹1878 (WAIT) | ₹2085 (HOLD) | ₹1338 (WAIT) |
| Karnataka | ₹2586 (WAIT) | ₹3302 (SELL) | ₹2243 (SELL) | ₹4269 (HOLD) | ₹2983 (HOLD) |
| Kerala | ₹2407 (SELL) | ₹3196 (SELL) | ₹2518 (SELL) * | ₹6682 (HOLD) | ₹4780 (WAIT) |
| Madhya Pradesh | ₹2771 (WAIT) | ₹3398 (HOLD) | ₹2256 (SELL) | ₹2748 (HOLD) | ₹1707 (HOLD) |
| Maharashtra | ₹2549 (SELL) | ₹3658 (HOLD) | ₹2259 (SELL) | ₹2870 (HOLD) | ₹2353 (HOLD) |
| Manipur | ₹3162 (SELL) | ₹3137 (HOLD) * | ₹3717 (SELL) | ₹6631 (WAIT) | ₹4626 (HOLD) |
| Meghalaya | ₹3280 (SELL) * | ₹3137 (HOLD) * | ₹3081 (WAIT) * | ₹6071 (SELL) | ₹4677 (WAIT) |
| Mizoram | ₹2515 (WAIT) * | ₹3137 (HOLD) * | ₹2518 (HOLD) * | ₹11333 (HOLD) | ₹4583 (HOLD) |
| Nagaland | ₹3301 (SELL) * | ₹3137 (HOLD) * | ₹7586 (SELL) | ₹6460 (WAIT) | ₹4702 (HOLD) |
| Odisha | ₹2394 (SELL) | ₹2438 (WAIT) | ₹2407 (WAIT) | ₹5156 (HOLD) | ₹3970 (HOLD) |
| Punjab | ₹2398 (SELL) | ₹3157 (HOLD) | ₹3001 (HOLD) | ₹4271 (HOLD) | ₹1702 (WAIT) |
| Rajasthan | ₹3294 (HOLD) | ₹3098 (SELL) | ₹2606 (WAIT) | ₹3056 (HOLD) | ₹1887 (HOLD) |
| Sikkim | ₹2515 (HOLD) * | ₹3137 (HOLD) * | ₹2518 (WAIT) * | ₹3200 (HOLD) * | ₹2734 (HOLD) * |
| Tamil Nadu | ₹2413 (HOLD) | ₹3137 (SELL) * | ₹3023 (SELL) | ₹6506 (WAIT) | ₹4616 (SELL) |
| Telangana | ₹2388 (SELL) | ₹2575 (SELL) | ₹2404 (WAIT) | ₹3168 (HOLD) | ₹2917 (HOLD) |
| Tripura | ₹2215 (SELL) | ₹3137 (HOLD) * | ₹2518 (WAIT) * | ₹7269 (HOLD) | ₹3479 (WAIT) |
| Uttar Pradesh | ₹2373 (SELL) | ₹3016 (HOLD) | ₹2637 (WAIT) | ₹3366 (HOLD) | ₹1695 (WAIT) |
| Uttarakhand | ₹2220 (SELL) | ₹2422 (WAIT) | ₹2636 (WAIT) | ₹3242 (HOLD) | ₹1491 (HOLD) |
| West Bengal | ₹2368 (SELL) | ₹3276 (HOLD) | ₹2518 (HOLD) * | ₹3815 (HOLD) | ₹2352 (WAIT) |

*Note: Asterisk (\*) indicates legitimate CROP_ONLY fallback used due to sparse state historical market records (<50 rows).*