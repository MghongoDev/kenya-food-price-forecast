"""Phase 3 features (issue #3).

All feature rows use only data at or before their own month (glossary:
forecast origin) - nothing here looks forward, so the leakage test at the
harness seam guards every consumer.

Rain features (ADR 0002): cumulative sums over 3 and 6 calendar months and
the deviation of the month's rainfall from that calendar month's mean over
a fixed pre-test climatology (CLIMATOLOGY_YEARS). Gaps stay missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLIMATOLOGY_YEARS = (2010, 2019)

RAIN_FEATURE_COLUMNS = ["rain_mm", "rain_3m", "rain_6m", "rain_anom"]

LAGS = (1, 2, 3, 6, 12)

PRICE_FEATURE_COLUMNS = [
    "price", *[f"lag_{k}" for k in LAGS],
    "missing_lag_1", "missing_lag_2",
    "roll_mean_3", "roll_vol_3", "roll_mean_6", "roll_vol_6",
    "month_of_year", "rel_price",
]


def feature_table(
    panel: pd.DataFrame,
    rain_features: pd.DataFrame | None = None,
    lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    """Origin-honest feature rows, one row per (market, month) of the panel.

    Lags are calendar shifts on the monthly grid, so a hole in the history
    leaves the shifted feature missing and the missing_lag flags record it
    (missing_lag_1 = price a month ago unobserved, missing_lag_2 = two months
    ago). Rolling statistics use whole calendar windows only. rel_price is
    the market's price over the cross-market average of the same month.
    """
    df = panel.sort_values(["market_id", "month"], kind="mergesort")
    out = df[["series", "market_id", "market", "month", "price", "observed"]].copy()
    g = df.groupby("market_id", sort=False)["price"]
    for k in lags:
        out[f"lag_{k}"] = g.shift(k)
    out["missing_lag_1"] = out["lag_1"].isna().astype(int)
    out["missing_lag_2"] = out["lag_2"].isna().astype(int)
    roll3 = g.rolling(3, min_periods=3)
    roll6 = g.rolling(6, min_periods=6)
    out["roll_mean_3"] = roll3.mean().reset_index(level=0, drop=True)
    out["roll_vol_3"] = roll3.std().reset_index(level=0, drop=True)
    out["roll_mean_6"] = roll6.mean().reset_index(level=0, drop=True)
    out["roll_vol_6"] = roll6.std().reset_index(level=0, drop=True)
    out["month_of_year"] = out["month"].str[5:].astype(int)
    cross_mean = df.groupby("month")["price"].transform("mean")
    out["rel_price"] = df["price"] / cross_mean

    if rain_features is not None:
        rain_cols = ["market_id", "month", *RAIN_FEATURE_COLUMNS]
        out = out.merge(rain_features[rain_cols], on=["market_id", "month"], how="left")
    else:
        for c in RAIN_FEATURE_COLUMNS:
            out[c] = np.nan
    return out


def add_rain_features(
    rain: pd.DataFrame,
    climatology_years: tuple[int, int] = CLIMATOLOGY_YEARS,
) -> pd.DataFrame:
    """Add rain_3m, rain_6m and rain_anom to a (market_id, month, rain_mm) table.

    A cumulative sum exists only when its whole calendar window is observed,
    so gaps keep the feature missing rather than quietly shortening the
    window. The anomaly compares against the calendar-month mean computed
    inside the climatology years only - data after them never leaks in.
    """
    df = rain.sort_values(["market_id", "month"], kind="mergesort").copy()
    df["rain_3m"] = (
        df.groupby("market_id")["rain_mm"].rolling(3, min_periods=3).sum().reset_index(level=0, drop=True)
    )
    df["rain_6m"] = (
        df.groupby("market_id")["rain_mm"].rolling(6, min_periods=6).sum().reset_index(level=0, drop=True)
    )
    lo, hi = climatology_years
    clim = df[
        df["rain_mm"].notna()
        & (df["month"].str[:4].astype(int) >= lo)
        & (df["month"].str[:4].astype(int) <= hi)
    ]
    clim_mean = (
        clim.assign(calendar_month=clim["month"].str[5:].astype(int))
        .groupby(["market_id", "calendar_month"])["rain_mm"]
        .mean()
    )
    df["rain_anom"] = df["rain_mm"] - np.array(
        [clim_mean.get((i, int(str(m)[5:])), np.nan) for i, m in zip(df["market_id"], df["month"])],
        dtype="float64",
    )
    return df.reset_index(drop=True)
