"""Phase 2 baselines (issue #2). Three simple rules any model must beat.

Naive: carry the last observed price forward (glossary: baseline).
Seasonal naive: the price from the same calendar month one, two or three
years back, nearest first.
Seasonal mean log change: the origin price grown by the historical mean
log change into the target calendar month, estimated on training rows only.

Every forecaster follows the harness contract: fit(train, origin,
horizons) in, a frame of (market_id, horizon, target_month, price_pred)
out. An absent row means the rule abstains, and the harness drops it.
"""
from __future__ import annotations

import math

import pandas as pd

from src.backtest import month_value, months_ahead

def _consecutive_log_changes(obs: pd.DataFrame) -> pd.DataFrame:
    """Log changes between consecutive observed months, per market, from the
    training frame only. A transition counts when the price is observed in
    the month and in the month before it (no gap filling)."""
    out = []
    for market_id, g in obs.groupby("market_id"):
        g = g.sort_values("month", kind="mergesort")
        months = g["month"].to_numpy()
        prices = g["price"].to_numpy()
        for i in range(1, len(months)):
            if month_value(months[i]) - month_value(months[i - 1]) == 1:
                out.append(
                    {
                        "market_id": market_id,
                        "calendar_month": int(str(months[i])[5:]),
                        "log_change": math.log(float(prices[i]) / float(prices[i - 1])),
                    }
                )
    return pd.DataFrame(out, columns=["market_id", "calendar_month", "log_change"])



def _last_observed(train: pd.DataFrame) -> pd.Series:
    """Last observed price at or before the origin, per market, from the
    training frame only."""
    obs = train.loc[train["observed"]].sort_values("month", kind="mergesort")
    return obs.groupby("market_id")["price"].last()


def naive(train: pd.DataFrame, origin: str, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Forecast = last observed price."""
    last = _last_observed(train)
    rows = [
        {
            "market_id": market_id,
            "horizon": h,
            "target_month": months_ahead(origin, h),
            "price_pred": price,
        }
        for (market_id, price) in last.items()
        for h in horizons
    ]
    return pd.DataFrame(
        rows,
        columns=["market_id", "horizon", "target_month", "price_pred"],
    )


def seasonal_naive(
    train: pd.DataFrame,
    origin: str,
    horizons: tuple[int, ...],
    max_lookback: int = 36,
) -> pd.DataFrame:
    """Forecast = the price of the same calendar month one, two or three
    years back, nearest first. Abstains when no same-month price exists
    inside the lookback."""
    obs = train.loc[train["observed"], ["market_id", "month", "price"]]
    prices = {
        market_id: dict(zip(g["month"], g["price"]))
        for market_id, g in obs.groupby("market_id")
    }
    rows = []
    for market_id, month_prices in prices.items():
        for h in horizons:
            target = months_ahead(origin, h)
            pred = None
            for k in range(12, max_lookback + 1, 12):
                look = month_value(target) - k
                look_month = f"{look // 12:04d}-{look % 12 + 1:02d}"
                if look_month in month_prices:
                    pred = month_prices[look_month]
                    break
            if pred is not None:
                rows.append(
                    {
                        "market_id": market_id,
                        "horizon": h,
                        "target_month": target,
                        "price_pred": pred,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["market_id", "horizon", "target_month", "price_pred"],
    )


def seasonal_mean_log_change(
    train: pd.DataFrame,
    origin: str,
    horizons: tuple[int, ...],
    min_transitions: int = 3,
) -> pd.DataFrame:
    """Forecast = the origin month's price grown by the historical mean log
    change into the target calendar month, estimated on consecutive observed
    transitions in the training frame only (glossary: seasonal mean change).
    Abstains under min_transitions observations for that calendar month."""
    obs = train.loc[train["observed"], ["market_id", "month", "price"]]
    changes = _consecutive_log_changes(obs)
    if changes.empty:
        return pd.DataFrame(columns=["market_id", "horizon", "target_month", "price_pred"])
    last = _last_observed(train)
    stats = changes.groupby(["market_id", "calendar_month"])["log_change"].agg(["mean", "size"])
    rows = []
    for (market_id, cal), agg in stats.iterrows():
        if int(agg["size"]) < min_transitions:
            continue
        base = last.get(market_id)
        if base is None or pd.isna(base):
            continue
        for h in horizons:
            target = months_ahead(origin, h)
            if int(target[5:]) != cal:
                continue
            rows.append(
                {
                    "market_id": market_id,
                    "horizon": h,
                    "target_month": target,
                    "price_pred": float(base) * math.exp(float(agg["mean"])),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["market_id", "horizon", "target_month", "price_pred"],
    )
