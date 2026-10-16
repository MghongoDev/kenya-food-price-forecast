# Kenya food price forecast

Forecasts wholesale white maize prices for five Kenyan markets one to three months ahead, with prediction intervals and a spike alert rule. Live app: <https://kenya-food-price-forecast-254.streamlit.app/>.

## Headline results

Pooled MAE in KES per kg, all periods (MASE vs. seasonal naive):

| Forecaster | h=1 | h=2 | h=3 |
|---|---|---|---|
| naive | 5.43 | 8.48 | 11.07 |
| seasonal_naive | 19.46 | 19.61 | 19.92 |
| seasonal_mean_log_change | 5.17 | 8.08 | 10.68 |
| lgbm_norain | 5.25 | 8.06 | 9.81 |
| lgbm_rain | 5.12 | 7.92 | 10.30 |

LGBM beats every baseline at every horizon. Rainfall helps at h=1 and h=2, hurts slightly at h=3.

Interval coverage (nominal 80%):

| Period | h=1 | h=2 | h=3 |
|---|---|---|---|
| all | 85% | 56% | 64% |
| drought 2022 | 77% | 37% | 38% |
| normal | 91% | 66% | 77% |

Coverage collapses in the drought year. The calm-2021 calibration does not transfer to the volatile 2022 regime.

Alert evaluation at the spike threshold (15%):

| Rule | Precision | Recall |
|---|---|---|
| model (conformal upper bound) | 0.30 | 0.75 |
| naive momentum | 0.36 | 0.26 |

The model does not dominate momentum. It trades precision for recall. The cost-optimal threshold depends on the false-alarm-to-miss ratio: 0.35 at 1:1, 0.00 at 1:5 and 1:10.

## Quick start

```
git clone https://github.com/MghongoDev/kenya-food-price-forecast.git
cd kenya-food-price-forecast
make setup
make app
```

The app serves at http://localhost:8501. It reads only committed files from `data/processed/`; nothing trains at load time.

## Reproduce

```
make setup               # uv venv (CPython 3.12) + pinned requirements
make data                # panel from WFP Kenya CSVs
make rain                # CHIRPS rainfall via ClimateSERV API
make backtest-baselines  # naive, seasonal naive, seasonal mean change
make backtest-model      # LightGBM with and without rainfall
make conformal           # split conformal intervals + coverage table
make alerts             # alert rule, PR curve, momentum comparison, costs
make dashboard-data      # precomputed files for the app
make app                 # serve the dashboard locally
make test
```

## Data sources

- **WFP Kenya food prices** (VAM Data Store, via HDX). Committed snapshot in `data/raw/`. Attribution required; the dataset may be redistributed with it.
- **CHIRPS rainfall** (UC Santa Barbara) via the ClimateSERV API. Daily rainfall averaged over a 25km-radius polygon around each market, summed to monthly totals. Public domain. Citation: Funk, C., Peterson, P., Landsfeld, M., et al. (2015). *Scientific Data* 2, 150066.

## Architecture

- Panel: one row per market and month, spliced across the 2020 series rename (ADR 0006, 0007).
- Features: lags 1, 2, 3, 6, 12; rolling mean and volatility; month of year; cross-market relative price; missing-lag indicator; rainfall observed up to the origin only.
- Model: direct per-horizon LightGBM, one model per horizon, trained on log price change (ADR 0005).
- Intervals: hand-rolled split conformal, calibrated per horizon on a 2021 validation block (ADR 0003).
- Alerts: fire when the conformal upper bound implies a rise above the spike threshold (ADR 0009).
- Deploy: Streamlit Community Cloud (ADR 0004).

Design decisions live in `docs/adr/`. Terminology in `GLOSSARY.md`. Phase checkpoints in GitHub issues (spec: #8).

## Limitations

- Conformal coverage assumes exchangeability, which time series violate. Coverage is measured, not assumed.
- The alert threshold was set from the base rate; costs are assumptions, not measurements.
- The model does not dominate naive momentum on both axes.
- Prices before 2020 are spliced from two series names.
- The latest actual price reaches about one month back (publication lag).

## What would change my mind

- New policies: import-duty changes, export bans, or subsidies that break the link between rainfall and price.
- Structural breaks: currency moves or market restructuring that makes the past a poor guide.
- Thin data: a market with long observed gaps where features and calibration cannot be estimated.
- Rainfall signal: a longer rain record showing the h=1-2 gain was noise.

## Licence

Data: WFP Kenya prices per HDX terms; CHIRPS public domain.
