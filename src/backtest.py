"""Phase 2: the rolling-origin backtest (issue #2, part of spec #8).

The forecasting problem, fixed in writing before any model sees results:

- Origin t: the month a forecast is made from. Every feature and every
  training row is at or before t. The last actual price in WFP Kenya data
  lands about one month back, so the panel's newest months are checked at
  origin time rather than silently trusted; the harness itself only ever
  hands a forecaster rows with month <= t.
- Horizons: h in {1, 2, 3} months ahead (glossary: horizon, forecast origin).
- Target: the price at t+h. The modeling target is the log price change
  log(p[t+h]) - log(p[t]); metrics are reported in KES per kg.
- Scoring: only observed months are scored (glossary: observed). A
  forecaster that abstains for a market-month is dropped from that score.
- Leakage: corrupting or shuffling data after t cannot change forecasts at
  t, because the harness never passes them. The leakage test pins this.

Test window: 2022-01 .. 2026-08 (ADR 0006). Metrics: MAE and WAPE in KES,
and MASE expressed as the ratio to the seasonal naive's error on the same
scored rows (glossary-consistent, no hidden scale).
"""
from __future__ import annotations

import pandas as pd

SCORE_COLUMNS = [
    "series", "market_id", "market", "origin", "horizon", "target_month",
    "price_true", "price_pred",
]

_TRUTH_COLUMNS = ["series", "market_id", "market", "month", "price", "observed"]


def month_value(m: str) -> int:
    """'YYYY-MM' -> months since year zero, zero-based within the year
    (January is month 0); order-preserving and invertible with months_ahead."""
    return int(m[:4]) * 12 + int(m[5:]) - 1


def months_ahead(origin: str, h: int) -> str:
    """Move a 'YYYY-MM' origin h months forward."""
    v = month_value(origin) + int(h)
    return f"{v // 12:04d}-{v % 12 + 1:02d}"


def make_origins(panel: pd.DataFrame, start: str, end: str) -> list[str]:
    """Every month in [start, end] that exists in the panel's grid."""
    months = sorted(panel["month"].unique())
    return [m for m in months if start <= m <= end]


def rolling_backtest(
    panel: pd.DataFrame,
    fit_predict,
    origins: list[str],
    horizons: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Run a rolling-origin backtest at the evaluation seam.

    fit_predict(train, origin, horizons) receives only rows with month <=
    origin and returns a frame with market_id, horizon, target_month and
    price_pred; missing price_pred rows are read as an abstention and are
    not scored. The output carries one row per scored (origin, horizon,
    market) target, observed months only.
    """
    truth = panel[_TRUTH_COLUMNS]
    rows = []
    for t in origins:
        train = panel[panel["month"] <= t]
        preds = fit_predict(train, t, horizons)
        if preds is None or len(preds) == 0:
            continue
        merged = preds.merge(
            truth,
            left_on=["market_id", "target_month"],
            right_on=["market_id", "month"],
            how="left",
        )
        scored = merged[merged["observed"] == True]  # noqa: E712 - observed is a flag column
        scored = scored[scored["price_pred"].notna()]
        scored = scored.rename(columns={"price": "price_true"}).assign(origin=t)
        rows.append(scored.drop(columns=["month"]))
    if not rows:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    return out[SCORE_COLUMNS].sort_values(
        ["origin", "horizon", "market_id"], kind="mergesort"
    ).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    from src.baselines import naive, seasonal_mean_log_change, seasonal_naive
    from src.metrics import metric_table

    processed = Path("data/processed")
    panel = pd.read_parquet(processed / "panel.parquet")
    # origins run to 2026-05 so every horizon has a target inside the grid
    origins = make_origins(panel, "2022-01", "2026-05")
    forecasters = {
        "naive": naive,
        "seasonal_naive": seasonal_naive,
        "seasonal_mean_log_change": seasonal_mean_log_change,
    }

    scores = []
    for name, forecaster in forecasters.items():
        s = rolling_backtest(panel, forecaster, origins, horizons=(1, 2, 3))
        scores.append(s.assign(forecaster=name))
        print(f"{name}: {len(s)} scored rows")
    score_all = pd.concat(scores, ignore_index=True)
    score_all.to_csv(processed / "backtest_baselines_scores.csv", index=False)

    reference = score_all[score_all["forecaster"] == "seasonal_naive"]
    tables = []
    for name in forecasters:
        s = score_all[score_all["forecaster"] == name]
        tables.append(metric_table(s, reference).assign(forecaster=name))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(processed / "baseline_metrics.csv", index=False)

    print("\n== pooled metric table (all periods)")
    pooled = table[
        (table["market"] == "ALL") & (table["period"] == "all")
    ][["forecaster", "horizon", "n", "mae_kes", "wape", "mase"]]
    print(pooled.to_string(index=False))

    print("\n== drought-year check: does a good average hide a bad tail?")
    tails = table[(table["market"] == "ALL") & (table["period"] == "drought")]
    print(tails[["forecaster", "horizon", "n", "mae_kes", "mase"]].to_string(index=False))
