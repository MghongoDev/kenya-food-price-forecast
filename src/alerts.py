"""Phase 5 alert rule (issue #10).

The model alert: fire when the conformal interval upper bound at t+h
implies a rise above the alert threshold. The alert threshold is swept
to produce a precision-recall curve; the lead-time distribution
(months of warning) is measured on true positives.

Built on the Phase 4 conformal intervals and the Phase 3 lgbm_rain
forecasts. The spike definition (src/spikes.py) sets the target; the
alert threshold is a separate, stricter knob on the implied rise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.spikes import THRESHOLD as SPIKE_THRESHOLD

ALERT_THRESHOLDS = np.round(np.arange(0.0, 0.51, 0.05), 2)  # implied-rise sweep


def classify_alerts(
    scored: pd.DataFrame,
    spikes: pd.DataFrame,
    alert_threshold: float,
) -> pd.DataFrame:
    """Merge scored forecasts (with intervals) with spike events on
    (market_id, origin, horizon); classify each row TP/FP/FN/TN at the
    given alert threshold. alert_fired = upper >= (1+threshold) * price_origin."""
    m = scored.merge(
        spikes[["market_id", "origin", "horizon", "price_origin", "spike"]],
        on=["market_id", "origin", "horizon"],
        how="inner",
    )
    m["alert_fired"] = m["upper"] >= (1 + alert_threshold) * m["price_origin"]
    m["outcome"] = np.select(
        [m["alert_fired"] & m["spike"], m["alert_fired"] & ~m["spike"],
         ~m["alert_fired"] & m["spike"]],
        ["TP", "FP", "FN"],
        default="TN",
    )
    return m


def pr_curve(outcomes: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Precision and recall at each alert threshold."""
    rows = []
    for t in thresholds:
        sub = outcomes[outcomes["alert_threshold"] == t]
        tp = int((sub["outcome"] == "TP").sum())
        fp = int((sub["outcome"] == "FP").sum())
        fn = int((sub["outcome"] == "FN").sum())
        rows.append({
            "alert_threshold": float(t),
            "n": int(len(sub)),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
        })
    return pd.DataFrame(rows)


def lead_time_distribution(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Horizon distribution among true positives (months of warning)."""
    tp = outcomes[outcomes["outcome"] == "TP"]
    if tp.empty:
        return pd.DataFrame(columns=["horizon", "n_tp", "share"])
    rows = []
    for h, g in tp.groupby("horizon"):
        rows.append({
            "horizon": int(h),
            "n_tp": int(len(g)),
            "share": float(len(g) / len(tp)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json

    from src.backtest import make_origins, rolling_backtest
    from src.conformal import apply_conformal, fit_conformal
    from src.features import add_rain_features
    from src.model import choose_hyperparams, make_fit_predict
    from src.spikes import spike_events

    processed = Path("data/processed")
    panel = pd.read_parquet(processed / "panel.parquet")

    rain_features = None
    rain_path = Path("data/interim/rainfall.parquet")
    if rain_path.exists():
        rain_features = add_rain_features(pd.read_parquet(rain_path))

    # reuse the Phase 4 pipeline: tune, calibrate, score the test window
    from src.conformal import CAL_START, CAL_END, TEST_START, TEST_END

    hp = choose_hyperparams(panel, rain_features, use_rain=True)
    print(f"hyperparams: {json.dumps(hp)}", flush=True)
    fp = make_fit_predict(rain_features, use_rain=True, params_by_horizon=hp)

    cal = rolling_backtest(panel, fp, make_origins(panel, CAL_START, CAL_END))
    halfwidths = fit_conformal(cal)
    test = rolling_backtest(panel, fp, make_origins(panel, TEST_START, TEST_END))
    covered = apply_conformal(test, halfwidths)
    print(f"scored rows: {len(covered)}", flush=True)

    spikes = spike_events(panel)
    print(f"spike events: {len(spikes)} rows, {int(spikes['spike'].sum())} spikes", flush=True)

    frames = []
    for t in ALERT_THRESHOLDS:
        frames.append(
            classify_alerts(covered, spikes, alert_threshold=round(float(t), 2)).assign(
                alert_threshold=round(float(t), 2)
            )
        )
    outcomes = pd.concat(frames, ignore_index=True)

    curve = pr_curve(outcomes, ALERT_THRESHOLDS)
    curve.to_csv(processed / "alert_pr_curve.csv", index=False)
    print("\n== precision-recall curve (spike threshold = "
          f"{SPIKE_THRESHOLD:.0%})")
    print(curve.to_string(index=False))

    # lead time per threshold: the sweep ties both products together
    lt_rows = []
    for t in ALERT_THRESHOLDS:
        sub = outcomes[outcomes["alert_threshold"] == round(float(t), 2)]
        for _, row in lead_time_distribution(sub).iterrows():
            lt_rows.append({"alert_threshold": round(float(t), 2), **row.to_dict()})
    lt = pd.DataFrame(lt_rows, columns=["alert_threshold", "horizon", "n_tp", "share"])
    lt.to_csv(processed / "alert_lead_time.csv", index=False)
    at_spike = lt[lt["alert_threshold"] == round(SPIKE_THRESHOLD, 2)]
    print("\n== lead-time distribution at the spike threshold")
    print(at_spike.to_string(index=False))
