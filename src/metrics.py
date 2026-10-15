"""Phase 2 metrics (issue #2). Written before any results were seen.

- MAE and WAPE in KES per kg, at market x horizon level and pooled.
- MASE for this project = MAE of the forecaster divided by the MAE of a
  reference forecaster (the seasonal naive) on the same scored rows.
- Period split by the TARGET month's year: 'drought' for the drought-year
  set, 'normal' otherwise. Phase 2 shakes the map down to {2022}, the only
  drought year fully inside the 2022-2026 test window; Phase 3 replaces
  the calendar label with a rainfall-based split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DROUGHT_YEARS = {2022}
PERIODS = ("all", "normal", "drought")

METRIC_ROWS = [
    "period", "market_id", "market", "series", "horizon",
    "n", "mae_kes", "wape", "mase",
]


def period_of(month: str, drought_years: set[int] = DROUGHT_YEARS) -> str:
    return "drought" if int(month[:4]) in drought_years else "normal"


def _with_errors(score: pd.DataFrame, drought_years: set[int]) -> pd.DataFrame:
    df = score.assign(_err=(score["price_true"] - score["price_pred"]).abs())
    df = df.assign(_period=df["target_month"].map(lambda m: period_of(m, drought_years)))
    return df


def _aggregate(rows: pd.DataFrame) -> dict:
    err = rows["_err"]
    truth = rows["price_true"].abs()
    return {
        "n": int(len(rows)),
        "mae_kes": float(err.mean()) if len(rows) else np.nan,
        "wape": float(err.sum() / truth.sum()) if truth.sum() else np.nan,
    }


def _reference_mae(
    reference_score: pd.DataFrame, drought_years: set[int]
) -> tuple[dict, dict]:
    """MASE denominators: mean absolute error per (period, market, horizon)
    and per (period, horizon) pooled across markets. 'all' means every
    reference row, not a bucket label the reference frame carries."""
    ref = _with_errors(reference_score, drought_years)
    subsets = {
        "all": ref,
        "normal": ref[ref["_period"] == "normal"],
        "drought": ref[ref["_period"] == "drought"],
    }
    per_market: dict = {}
    pooled: dict = {}
    for period, sub in subsets.items():
        if sub.empty:
            continue
        per_market.update(
            {(period, market_id, horizon): v for (market_id, horizon), v
             in sub.groupby(["market_id", "horizon"])["_err"].mean().items()}
        )
        pooled.update(
            {(period, horizon): v for horizon, v
             in sub.groupby("horizon")["_err"].mean().items()}
        )
    return per_market, pooled


def metric_table(
    score: pd.DataFrame,
    reference_score: pd.DataFrame | None = None,
    drought_years: set[int] = DROUGHT_YEARS,
) -> pd.DataFrame:
    """The fixed metric table: one row per period x market x horizon plus a
    pooled 'ALL' row per period x horizon. mase compares against the
    reference forecaster on the same subgroup; NaN without a reference."""
    model = _with_errors(score, drought_years)
    ref_mae_market = ref_mae_pooled = None
    if reference_score is not None:
        ref_mae_market, ref_mae_pooled = _reference_mae(reference_score, drought_years)

    rows = []
    for period in PERIODS:
        subset = model if period == "all" else model[model["_period"] == period]
        if subset.empty:
            continue
        # market-level rows
        for (market_id, market, series, horizon), g in subset.groupby(
            ["market_id", "market", "series", "horizon"]
        ):
            agg = _aggregate(g)
            mase = (
                agg["mae_kes"] / ref_mae_market[(period, market_id, horizon)]
                if ref_mae_market is not None
                and (period, market_id, horizon) in ref_mae_market
                else np.nan
            )
            rows.append(
                {
                    "period": period, "market_id": int(market_id), "market": market,
                    "series": series, "horizon": int(horizon),
                    **agg, "mase": mase,
                }
            )
        # pooled rows across markets
        for horizon, g in subset.groupby("horizon"):
            agg = _aggregate(g)
            mase = (
                agg["mae_kes"] / ref_mae_pooled[(period, horizon)]
                if ref_mae_pooled is not None and (period, horizon) in ref_mae_pooled
                else np.nan
            )
            rows.append(
                {
                    "period": period, "market_id": np.nan, "market": "ALL",
                    "series": "pooled", "horizon": int(horizon),
                    **agg, "mase": mase,
                }
            )
    table = pd.DataFrame(rows, columns=METRIC_ROWS)
    return table.sort_values(
        ["period", "market_id", "horizon"], kind="mergesort"
    ).reset_index(drop=True)
