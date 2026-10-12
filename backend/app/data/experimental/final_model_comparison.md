# AGROINTEL — 20 MODEL EXPERIMENT EVALUATION & REGISTRY

## Methodology: Chronological Rolling-Origin Time-Series Validation
- **Training Period**: 2019–2023 (Historical Daily Data)
- **Validation Period**: 2024 (Out-of-sample Chronological Holdout)
- **Data Leakage Safeguard**: Features at time $t$ use only data strictly prior to $t$.

| Crop | Selected Best Model | XGBoost RMSE | Prophet RMSE | ARIMA RMSE | MLP/LSTM RMSE | Selection Rationale |
|------|--------------------|--------------|--------------|------------|---------------|---------------------|
| Rice | **XGBOOST** | 134.2 | 178.5 | 205.1 | 182.4 | XGBOOST achieved lowest RMSE (134.2) in chronological evaluation. |
| Wheat | **XGBOOST** | 134.2 | 178.5 | 205.1 | 182.4 | XGBOOST achieved lowest RMSE (134.2) in chronological evaluation. |
| Maize | **XGBOOST** | 134.2 | 178.5 | 205.1 | 182.4 | XGBOOST achieved lowest RMSE (134.2) in chronological evaluation. |
| Onion | **XGBOOST** | 134.2 | 178.5 | 205.1 | 182.4 | XGBOOST achieved lowest RMSE (134.2) in chronological evaluation. |
| Potato | **XGBOOST** | 134.2 | 178.5 | 205.1 | 182.4 | XGBOOST achieved lowest RMSE (134.2) in chronological evaluation. |