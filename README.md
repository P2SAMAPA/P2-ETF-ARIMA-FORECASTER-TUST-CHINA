---
title: P2-ETF-ARIMA-FORECASTER-TUST-CHINA
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# P2-ETF-ARIMA-FORECASTER-TUST-CHINA

Quantitative ETF trading strategy based on:
> *A Quantitative Trading Strategy Based on A Position Management Model*
> Xu et al., Tianjin University of Science and Technology, 2022

## Strategy Overview

- **ARIMA(p,d,q)** rolling price forecaster per ETF (auto order selection via AIC)
- **Consecutive run analysis** (Apriori-style) to detect statistically overdue reversals
- **Dynamic hold period** selection: 1d / 3d / 5d — chosen by highest expected net return after fees
- **CASH overlay**: 2-day cumulative return ≤ −15% triggers cash protection
- **5 ETFs**: TLT · TBT · VNQ · SLV · GLD
- **Benchmarks**: SPY · AGG
- **Data**: HuggingFace Dataset `P2SAMAPA/fi-etf-macro-signal-master-data` (2008→today)

## Configuration

| Parameter | Options |
|-----------|---------|
| Start year | 2008–2025 |
| Transaction cost | 0–100 bps (steps of 5) |
| Data split | 80% train / 10% val / 10% OOS |
| Auto-lookback | 30 / 45 / 60d (selected by val MAE) |
| Hold periods | 1d / 3d / 5d |

## File Structure
```
├── app.py
├── loader.py
├── arima_forecaster.py
├── run_analysis.py
├── selector.py
├── components.py
├── cache.py
├── requirements.txt
└── README.md
```

Set `HF_TOKEN` as a secret in HF Space settings.
