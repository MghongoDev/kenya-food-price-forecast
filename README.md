# Kenya food price forecast

Forecasts of wholesale staple prices for Kenyan markets, one to three months
ahead, with prediction intervals and a spike alert rule. Status: Phases 1-2
complete (panel + baselines), Phase 3 in progress (gradient boosting + rainfall).

## Data sources and licences

- **WFP Kenya food prices** (VAM Data Store, via the Humanitarian Data Exchange):
  https://data.humdata.org/dataset/wfp-food-prices-for-kenya. Committed snapshot
  in `data/raw/`; per the HDF dataset licence terms, attribution is required and
  the dataset may be redistributed with it. The two CSVs carry the data.
- **CHIRPS rainfall** (UC Santa Barbara Climate Hazards Center) via the
  ClimateSERV API (climateserv.servirglobal.net), the official CHIRPS
  channel: daily rainfall averaged over a 25km-radius polygon around each
  market, summed to monthly totals (ADR 0008). CHIRPS is public domain and
  is not committed; `src/rainfall.py` rebuilds the cache. Citation:
  Funk, C., Peterson, P., Landsfeld, M., et al. (2015). The climate hazards
  infrared precipitation with stations - a new environmental record for
  monitoring extremes. *Scientific Data* 2, 150066.

## Reproduce

```
make setup               # uv venv (CPython 3.12) + pinned requirements
make data                # panel: data/processed/panel.parquet, coverage.csv
make rain                # CHIRPS via ClimateSERV: data/interim/rainfall.parquet
make backtest-baselines  # naive / seasonal naive / seasonal mean change
make backtest-model      # LightGBM with and without rainfall + verdict tables
make test
```

## Deploy

The app deploys to Streamlit Community Cloud (ADR 0004). The final
connect-GitHub step runs in the owner's browser:

1. Push this repo to GitHub (done — `MghongoDev/kenya-food-price-forecast`).
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Click "New app", select this repo, set the main file path to `app.py`.
4. Deploy. The app reads only committed files from `data/processed/`;
   nothing trains at load time.

Platform limits (checked at deploy time, ADR 0004): Community Cloud
sets its own resource limits. The app is light — it loads ~10 small
CSVs and one parquet, and renders a matplotlib fan chart. If limits
bite, the fallback is local running (`make app`) while a paid host is
chosen.

## Reproduce

```
make setup               # uv venv (CPython 3.12) + pinned requirements
make data                # panel: data/processed/panel.parquet, coverage.csv
make rain                # CHIRPS via ClimateSERV: data/interim/rainfall.parquet
make backtest-baselines  # naive / seasonal naive / seasonal mean change
make backtest-model      # LightGBM with and without rainfall + verdict tables
make conformal           # split conformal intervals + coverage table
make alerts             # alert rule, PR curve, momentum comparison, costs
make dashboard-data      # precomputed files for the Streamlit app
make app                 # serve the dashboard locally
make test
```

Design decisions live in `docs/adr/`, terminology in `GLOSSARY.md`, and the
phase-by-phase checkpoints in GitHub issues (spec: #8).
