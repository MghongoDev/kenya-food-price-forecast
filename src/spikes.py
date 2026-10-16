"""Phase 5 spike definition (issue #9).

A spike is a target-month price at least THRESHOLD * 100% above the
origin price. The threshold is set from the base rate: if spikes are
under a few percent of observations, lower the threshold or pool
horizons; if very common, raise it.

Threshold decision (recorded before any alert evaluation, per the
guide): at 20% the pooled base rate is 5.2% at h=1, which is under a
few percent of observations, so the guide's rule says lower the
threshold. At 15% the rates are 8.8% / 16.8% / 23.5% - every horizon
is out of the "few percent" zone with enough events to evaluate
(84/157/216 spikes). Pooling horizons was the alternative remedy; it
was not taken because the alert rule is per-horizon and pooled
statistics would blur the per-horizon evaluation. Final threshold:
15%.
"""
from __future__ import annotations

import pandas as pd

from src.backtest import months_ahead

THRESHOLD = 0.15  # 15% rise = spike (set from the base rate; see docstring)
HORIZONS = (1, 2, 3)


def spike_events(
    panel: pd.DataFrame,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """One row per (market, origin, horizon) where both the origin and
    the target month are observed. spike = price[t+h] >= (1+threshold) * price[t]."""
    rows = []
    for (mid, mkt), g in panel.groupby(["market_id", "market"]):
        obs = g[g["observed"]].set_index("month")["price"]
        for origin in obs.index:
            p0 = obs[origin]
            for h in HORIZONS:
                target = months_ahead(origin, h)
                if target not in obs.index:
                    continue
                p1 = obs[target]
                rows.append({
                    "market_id": int(mid),
                    "market": mkt,
                    "origin": origin,
                    "horizon": h,
                    "target_month": target,
                    "price_origin": p0,
                    "price_target": p1,
                    "spike": bool(p1 >= (1 + threshold) * p0),
                })
    return pd.DataFrame(rows, columns=[
        "market_id", "market", "origin", "horizon", "target_month",
        "price_origin", "price_target", "spike",
    ])


def base_rate(events: pd.DataFrame) -> pd.DataFrame:
    """Spike counts and share by horizon, per market and pooled (ALL)."""
    rows = []
    groups = [
        (mkt, g) for (h, mkt), g in events.groupby(["horizon", "market"])
    ] + [("ALL", g) for h, g in events.groupby("horizon")]
    for mkt, g in groups:
        rows.append({
            "horizon": int(g["horizon"].iloc[0]),
            "market": mkt,
            "n": int(len(g)),
            "spikes": int(g["spike"].sum()),
            "base_rate": float(g["spike"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["horizon", "market"], kind="mergesort"
    ).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    processed = Path("data/processed")
    panel = pd.read_parquet(processed / "panel.parquet")
    events = spike_events(panel)
    rate = base_rate(events)
    rate.to_csv(processed / "spike_base_rate.csv", index=False)
    print(rate.to_string(index=False))
    pooled = rate[rate["market"] == "ALL"]
    print("\n== pooled base rate at threshold", THRESHOLD)
    print(pooled.to_string(index=False))
