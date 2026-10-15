# Kenya Staple-Food Price Forecasting and Shock Alerts: Implementation Guide

## Overview

A forecasting pipeline for monthly maize (then bean) prices across Kenyan markets. For each market and commodity it produces 1 to 3 month-ahead forecasts with prediction intervals, plus an early-warning flag for likely price spikes. "Done" for the MVP means:

1. A reproducible backtest that compares a gradient-boosting model against honest seasonal baselines, scored on observed prices only.
2. Prediction intervals whose real coverage you have measured and reported.
3. A spike-alert rule with an explicit false-alarm vs missed-spike cost analysis.
4. A small deployed dashboard and a README that tells the story.

The point of this project is the evaluation rigor, not the model. A modest model with a trustworthy backtest beats a fancy model with a leaky one.

**Assumptions:** solo builder, intermediate-to-experienced with Python and analytics, goal is a portfolio piece, about 3 to 4 weeks at 8 to 10 hours a week, free-tier tooling only, a laptop with no GPU. Time estimates are rough.

## Scope

- **Core loop:** pick a market and commodity, see the forecast fan (median plus interval) for the next 1 to 3 months, and see whether the alert is firing and why.
- **In the MVP:**
  - Maize first, beans second, on markets with good data coverage.
  - Seasonal baselines, then a LightGBM model with lag and rainfall features.
  - Rolling-origin backtest at horizons 1, 2 and 3 months.
  - Conformal prediction intervals with a coverage check.
  - One alert definition, a cost analysis, and a Streamlit dashboard.
- **Deferred:**
  - Other commodities (rice, sugar, wheat flour, cooking oil).
  - Exchange-rate, fuel and conflict covariates.
  - Hierarchical or spatial models that share information between neighbouring markets.
  - Automated monthly refresh.
  - Chronos is **optional** (Phase 6). Skip it if time is short.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language / env | Python 3.11+, `uv` or `venv` | Standard for this work; pin versions in `requirements.txt` for reproducibility |
| Data handling | pandas, pyarrow (Parquet) | Panels are small (tens of thousands of rows); Parquet keeps typed intermediate files |
| Baselines | Your own ~20 lines of pandas | Seasonal naive is trivial and avoids hiding logic in a library |
| Main model | LightGBM | Strong on tabular lag features, fast on a laptop, supports quantile objectives |
| Intervals | Hand-rolled split conformal | Short enough to write yourself, and you will understand its limits; the `MAPIE` library is an alternative if you prefer |
| Rainfall | CHIRPS monthly rasters via `rasterio` or `xarray` | Standard rainfall product for East Africa; sample around each market's lat/lon so you don't need county shapefiles |
| Optional foundation model | Chronos from the Hugging Face Hub | Zero-shot time-series baseline; no training needed |
| Dashboard | Streamlit + Plotly | Fastest path to a shareable app; free hosting on Streamlit Community Cloud |
| Quality | pytest, a `Makefile` | A leakage test and one-command reproducibility matter more here than usual |

Check current package versions and the Chronos install instructions and model ids on its repo or model card before you start, since these change.

## Architecture at a glance

```
WFP Kenya food prices (HDX CSV) ─┐
                                 ├─> clean panel (market x commodity x month, Parquet)
CHIRPS rainfall rasters ─────────┘            │
                                              ▼
                         features at forecast origin t (lags, seasonality, rainfall)
                                              │
              ┌───────────────────────────────┼───────────────────────────┐
              ▼                               ▼                           ▼
       baselines (naive,              LightGBM, one model         (optional) Chronos
       seasonal naive)                per horizon h = 1,2,3       zero-shot
              └───────────────┬───────────────┴───────────────────────────┘
                              ▼
                rolling-origin backtest (observed targets only)
                              ▼
            conformal intervals  ──>  spike alert rule  ──>  cost analysis
                              ▼
                      Streamlit dashboard + README
```

Suggested repo layout:

```
kenya-food-prices/
  data/raw/  data/interim/  data/processed/
  src/  load.py  panel.py  rainfall.py  features.py  backtest.py  models.py  conformal.py  alerts.py
  app/streamlit_app.py
  tests/
  notebooks/        # exploration only; anything repeatable lives in src/
  Makefile  README.md  requirements.txt
```

## Build steps

### Phase 1: A clean, trustworthy monthly panel (est. 3 to 5 days)

**Goal:** a Parquet file with one row per market, commodity and month, where every price is flagged as observed or not, and you've chosen which series to model.

- [ ] Create the repo, virtual environment and folder layout above. Commit early.
- [ ] **Check the licence and terms of use** for the WFP dataset (and later CHIRPS) and note them in the README.
- [ ] Download the WFP Kenya food-prices CSV from the Humanitarian Data Exchange (https://data.humdata.org/dataset/wfp-food-prices-for-kenya) into `data/raw/`. Don't commit the raw file if the licence discourages redistribution; commit a download script instead.
- [ ] Load it with pandas. HDX files often have a second row of HXL tags (for example `#date`) under the header. Inspect the first rows and skip that row if present.
- [ ] Inspect the columns you need: date, market, county or admin region, latitude and longitude, commodity, unit, price type (retail vs wholesale), the price flag column, and price. Read the dataset's documentation on what the price flag values mean. If some rows are marked as aggregated or forecast rather than actual, exclude them from scoring.
- [ ] Filter to **retail maize (white maize or maize, whichever has the best coverage)** and one consistent unit (typically KES per kg). Don't mix wholesale and retail or different units in one series.
- [ ] Reindex each market's series to a complete monthly grid. Add a boolean `observed` column. **Do not interpolate prices that you'll later score against**; missing stays missing.
- [ ] Build a coverage table: per market, the first and last observed month, the share of months observed, and the longest gap. Select markets with, say, at least 75% coverage across the last 8 to 10 years and no huge recent gap. Treat that cutoff as adjustable and write down what you chose.
- [ ] Plot the selected series (small multiples) and look at them. Note obvious events (drought years, sharp spikes, unit changes, suspicious outliers) in a notebook, not in your head.
- [ ] Optional cross-check: look at the World Bank Real Time Food Prices series for Kenya (https://microdata.worldbank.org/index.php/catalog/6167/data-dictionary). Part of it is machine-learning-estimated to fill gaps, so if you use it at all, keep estimated values out of your scoring and say so.
- [ ] **Checkpoint:** `make data` runs from a clean clone and produces `data/processed/panel.parquet`. A test asserts no duplicate (market, commodity, month) rows and that the `observed` flag matches the raw data. You can state in one sentence how many markets and months you're modelling.

### Phase 2: Baselines and a leak-proof backtest harness (est. 4 to 5 days)

**Goal:** a function that takes any forecaster, runs a rolling-origin backtest, and returns a scored table. This is the riskiest and most important piece, so it comes before any fancy model.

- [ ] Define the forecasting problem precisely: at forecast origin month `t`, using only data available at `t` (be honest about publication lag; if WFP data typically lands a month or more late, consider excluding the most recent month from features), predict the price at `t+h` for `h` in {1, 2, 3}.
- [ ] Choose a target. Recommended: log price change `log(price[t+h]) - log(price[t])`. It makes markets comparable and stabilises variance. Convert back to KES for reporting.
- [ ] Implement baselines:
  - **Naive:** forecast = last observed price.
  - **Seasonal naive:** forecast = price from the same month last year (adjusted by the last observed level if you like, but keep it simple and documented).
  - **Seasonal mean change:** historical average log change for this calendar month, estimated on training data only.
- [ ] Implement the rolling-origin loop: expanding training window, origins stepping forward month by month (or quarterly if it's slow), final 3 to 4 years reserved for testing. Skeleton:

```python
def rolling_backtest(panel, fit_predict, origins, horizons=(1, 2, 3)):
    rows = []
    for t in origins:
        train = panel[panel.date <= t]          # nothing after the origin
        preds = fit_predict(train, t, horizons) # returns market, h, point forecast
        rows.append(preds.assign(origin=t))
    out = pd.concat(rows)
    truth = panel[["market", "date", "price", "observed"]]
    out = out.merge(truth, left_on=["market", "target_date"],
                    right_on=["market", "date"], how="left")
    return out[out.observed]                    # score observed targets only
```

- [ ] Pick metrics and write them down before you see results: **MASE** (scaled against the seasonal naive error), **WAPE or MAE in KES** for readability, and per-horizon breakdowns. Also report results by period (for example drought years vs normal years) so a good average can't hide a bad tail.
- [ ] Write a **leakage test**: shuffle or corrupt data after origin `t` and assert the forecasts at `t` don't change.
- [ ] **Checkpoint:** a table of baseline errors by horizon, produced by one command (`make backtest-baselines`), and the leakage test passes. You should see errors grow with horizon; if they don't, something is wrong.

### Phase 3: Gradient boosting with lag and rainfall features (est. 5 to 7 days)

**Goal:** a LightGBM model that is fairly compared with the baselines, with an honest answer to "does it actually help, and where?"

- [ ] Build features **using only information at or before the origin**: recent log prices and changes at lags 1, 2, 3, 6 and 12; rolling mean and volatility over 3 and 6 months; month of year (cyclical encoding or plain integer for trees); the market's price relative to the cross-market average at the origin; and a missing-lag indicator where data was absent.
- [ ] Add rainfall (`rainfall.py`):
  - Download CHIRPS monthly data for Kenya for the years you need. Check its terms and cite it properly.
  - For each market, average rainfall within a small radius of its coordinates (start with roughly 20 to 30 km) per month.
  - Make rainfall features: 3-month and 6-month cumulative rainfall, and an anomaly versus that calendar month's long-run mean (computed from training years only, or from a fixed pre-test climatology, to avoid subtle leakage).
  - Use rainfall observed up to the origin only. You don't have rainfall forecasts, so don't pretend you do.
- [ ] Train a **global model across markets** (market identity as a categorical or via market-level features), one model per horizon (the "direct" strategy), on the log-change target. Direct models avoid error compounding and keep the leakage story simple.
- [ ] Keep tuning light: a handful of settings (learning rate, number of leaves, min child samples, number of trees) chosen with a time-ordered validation block inside the training window. Never tune on the test period.
- [ ] Run the identical backtest harness and compare against the baselines by horizon and by period.
- [ ] Ablation: run with and without rainfall features. This is one of the most informative things in the whole project, because "rainfall helped at h=3 but not h=1" is a finding, and "rainfall didn't help" is also a legitimate finding.
- [ ] Inspect errors: the worst 20 forecasts, which markets are hardest, and whether errors cluster in spike periods.
- [ ] **Checkpoint:** one table with baselines vs LightGBM (with and without rainfall) by horizon, plus a short written verdict. If LightGBM does not beat seasonal naive at some horizon, say so plainly; that is credible evidence of rigor.

### Phase 4: Prediction intervals you can defend (est. 3 to 4 days)

**Goal:** 80% (or 90%) intervals around each forecast, with a measured coverage figure that you can compare to the claim.

- [ ] Implement **split conformal prediction per horizon**: take residuals from a calibration set of recent backtest forecasts (made on origins before the test window), compute the absolute-error quantile at the desired level, and add and subtract it from the point forecast on the log scale. Consider scaling by a per-market volatility measure so volatile markets get wider intervals.

```python
def conformal_halfwidth(cal_resid, alpha=0.2):
    n = len(cal_resid)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(np.abs(cal_resid), min(q, 1.0))
```

- [ ] Time series violate the exchangeability assumption behind conformal methods, so treat it as an approximation. Use a calibration window of recent origins, then **check coverage on a later, untouched test window**.
- [ ] Report empirical coverage by horizon, by market group, and by period (drought vs normal), plus average interval width. Coverage that's right on average but collapses in volatile periods is a real finding.
- [ ] Compare with a second approach, for example LightGBM quantile objectives, if time allows. Pick one for the dashboard.
- [ ] **Checkpoint:** a coverage table where nominal 80% is next to measured coverage per horizon, and a fan chart for two or three example markets over the test period.

### Phase 5: Spike alerts, cost analysis and a deployed dashboard (est. 5 to 6 days)

**Goal:** a defined, evaluated alert, and a live app a stranger can open.

- [ ] **Define "spike" precisely and before looking at alert results.** A starting definition: the price at `t+h` is at least 20% higher than the price at `t`. Check the base rate (how often this happens). If spikes are under a few percent of observations, lower the threshold or pool horizons so there are enough events to evaluate; if they're very common, raise it. Record the final choice and why.
- [ ] Define the alert rule from the forecast: for example, fire when the interval's upper bound at `t+h` implies a rise above the spike threshold, or when the median does. Sweep the threshold to get a precision-recall curve and a lead-time distribution (how many months of warning did you give before the spike?).
- [ ] Compare against a **naive alert rule**, such as "price already rose more than 10% last month". If your model's alert can't beat simple momentum, that's the honest headline.
- [ ] **Cost analysis:** assign relative costs to a false alarm (wasted response effort) and a missed spike (households or agencies caught unprepared). Because real costs are unknown, show the expected cost across a range of cost ratios (for example 1:1, 1:5, 1:10) and pick the threshold that minimises it for each. Say clearly that the costs are assumptions, not measurements, and who would need to supply real ones.
- [ ] Build the Streamlit app:
  - A market and commodity selector.
  - A fan chart: history, the forecast, and the interval.
  - An alert status panel showing the horizon, the rule, and its historical precision.
  - A backtest results page.
  - A short "how to read this" and "limitations" section.
- [ ] Precompute forecasts and backtest results into small files committed to the repo so the app doesn't train anything at load time.
- [ ] Deploy to Streamlit Community Cloud (or similar free hosting). Check requirements and platform limits at deploy time.
- [ ] **Checkpoint:** a public URL works end to end from a private browser window, and the alert page shows at least one past spike that the model warned about and one it missed.

### Phase 6 (optional): Chronos as a zero-shot comparison (est. 2 to 3 days)

**Goal:** know whether a pretrained time-series model is worth the complexity for this problem.

- [ ] Install the Chronos package per its current README and load a small checkpoint from the Hugging Face Hub (verify the current model ids and licence on the model card).
- [ ] Run it zero-shot on each market's history inside your existing backtest harness, giving it only data up to each origin. Many series have gaps, so decide how to feed them (for example, shorter contiguous history or light, flagged filling of **features only**) and document it. Still score on observed targets only.
- [ ] Compare MASE and interval coverage to LightGBM and the seasonal baselines.
- [ ] **Checkpoint:** one extra row in the results table and a paragraph on whether it was worth it. A "no, and here's why" is a good outcome.

### Phase 7: Write-up and polish (est. 2 to 3 days)

**Goal:** someone can understand, trust and reproduce the project in ten minutes.

- [ ] README structure: the question, data sources (with licences and the caveat on estimated prices), method, headline results table, coverage table, alert evaluation, limitations, how to reproduce.
- [ ] Add a "what would change my mind" section: the conditions under which the model would fail (new policies, structural breaks, a thin data period).
- [ ] Clean up: remove dead notebooks, pin dependencies, make sure `make all` reproduces every table and figure.
- [ ] Write a 600 to 900 word post or case study built around the single most interesting finding (for example "rainfall helps at 3 months but not at 1" or "the alert barely beats momentum").
- [ ] **Checkpoint:** a friend or colleague clones the repo and reproduces the main results table with no help from you.

## Risks and gotchas

1. **Sparse, gappy data and mixed quality.** Some markets have long gaps, and estimated values in the World Bank series can silently inflate your scores. *Mitigation:* an `observed` flag from Phase 1 onward, scoring on observed targets only, and a coverage-based market selection you document.
2. **Leakage.** Using rainfall, lags or climatology computed with future information, tuning on the test period, or ignoring publication lag will make results look better than they are. *Mitigation:* origin-based feature construction, the leakage test, a time-ordered validation block, and a test window you touch once.
3. **Too few spike events.** Rare events make precision, recall and cost curves noisy. *Mitigation:* check the base rate early, report event counts next to every metric, use bootstrap or simple intervals over events, and avoid claiming more than the counts support.
4. **Structural breaks.** Droughts, policy changes (for example import-duty changes), currency moves and the pandemic mean the past is an imperfect guide, and conformal coverage can degrade exactly when you need it. *Mitigation:* report results by period and state the limitation prominently.
5. **Scope creep.** Extra commodities, covariates and models can consume the schedule. *Mitigation:* finish maize end to end, then repeat for beans using the same pipeline; treat everything in "Deferred" as a separate project.

## After the MVP

1. Add beans, then other staples, reusing the same pipeline.
2. Add covariates: exchange rate, fuel prices, and neighbouring-market prices.
3. Try a hierarchical or spatial model that borrows strength across markets.
4. Evaluate probabilistic forecasts with a proper scoring rule such as CRPS or pinball loss.
5. Automate a monthly refresh with a scheduled job and keep a log of live forecast vs outcome.
6. Talk to someone in food-security or market monitoring about what alert threshold and cost ratio would really be useful.

## Suggested first action

In the next 30 minutes: create the repo, download the WFP Kenya CSV, and load it in a notebook. Print the first five rows and the unique values of the commodity, unit, price type and price flag columns. Those four lists decide which series you'll model and what "observed" means, and everything else builds on them.
