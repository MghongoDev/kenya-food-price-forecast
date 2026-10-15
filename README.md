# Kenya food price forecast

Forecasts of wholesale staple prices for Kenyan markets, one to three months
ahead, with prediction intervals and a spike alert rule. Status: Phases 1-2
complete (panel + baselines), Phase 3 in progress (gradient boosting + rainfall).

## Data sources and licences

- **WFP Kenya food prices** (VAM Data Store, via the Humanitarian Data Exchange):
  https://data.humdata.org/dataset/wfp-food-prices-for-kenya. Committed snapshot
  in `data/raw/`; per the HDF dataset licence terms, attribution is required and
  the dataset may be redistributed with it. The two CSVs carry the data.
- **CHIRPS rainfall** (UC Santa Barbara Climate Hazards Center), monthly Africa
  GeoTIFFs at 0.05 degrees: https://data.chc.ucsb.edu/products/CHIRPS-2.0/.
  CHIRPS is public domain; it is not committed (≈900MB) and the download step in
  `src/rainfall.py` rebuilds the cache. Citation: Funk, C., Peterson, P.,
  Landsfeld, M., et al. (2015). The climate hazards infrared precipitation with
  stations - a new environmental record for monitoring extremes.
  *Scientific Data* 2, 150066.
- Rainfall sampling: the mean of valid CHIRPS pixels within 25km of each
  market's coordinates (ADR 0002).

## Reproduce

```
make setup               # uv venv (CPython 3.12) + pinned requirements
make data                # panel: data/processed/panel.parquet, coverage.csv
make rain                # CHIRPS download + sample: data/interim/rainfall.parquet
make backtest-baselines  # naive / seasonal naive / seasonal mean change
make backtest-model      # LightGBM with and without rainfall + verdict tables
make test
```

Design decisions live in `docs/adr/`, terminology in `GLOSSARY.md`, and the
phase-by-phase checkpoints in GitHub issues (spec: #8).
