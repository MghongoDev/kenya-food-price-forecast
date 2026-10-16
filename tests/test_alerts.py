"""Phase 5 alert-rule tests (issue #10).

The seam is the alert outcome on hand-built frames: scored forecasts
(with intervals) merged with spike events, classified TP/FP/FN/TN at a
given alert threshold. All literals are hand-computed.
"""
import numpy as np
import pandas as pd
import pytest

from src.alerts import classify_alerts, lead_time_distribution, pr_curve


def _scored(rows):
    return pd.DataFrame(rows, columns=[
        "market_id", "origin", "horizon", "target_month",
        "price_true", "price_pred", "lower", "upper",
    ])


def _spikes(rows):
    return pd.DataFrame(rows, columns=[
        "market_id", "origin", "horizon", "price_origin", "spike",
    ])


# three (origin, horizon) pairs for market 1; price_origin = 100 throughout
SCORED = _scored([
    [1, "2022-01", 1, "2022-02", 110.0, 105.0, 90.0, 120.0],
    [1, "2022-01", 2, "2022-03", 100.0, 100.0, 90.0, 110.0],
    [1, "2022-02", 1, "2022-03", 120.0, 115.0, 100.0, 130.0],
])
SPIKES = _spikes([
    [1, "2022-01", 1, 100.0, True],
    [1, "2022-01", 2, 100.0, False],
    [1, "2022-02", 1, 100.0, True],
])


def test_alert_fires_when_upper_bound_crosses_the_threshold():
    # threshold 0.15: fire when upper >= 115
    #  h=1 origin 2022-01: upper 120 >= 115, spike True  -> TP
    #  h=2 origin 2022-01: upper 110 < 115,  spike False -> TN
    #  h=1 origin 2022-02: upper 130 >= 115, spike True  -> TP
    out = classify_alerts(SCORED, SPIKES, alert_threshold=0.15)
    counts = out["outcome"].value_counts()
    assert counts["TP"] == 2 and counts["TN"] == 1
    assert "FP" not in counts and "FN" not in counts


def test_higher_threshold_trades_recall_for_precision():
    # threshold 0.25: fire when upper >= 125
    #  h=1 origin 2022-01: upper 120 < 125, spike True  -> FN
    #  h=2 origin 2022-01: upper 110 < 125, spike False -> TN
    #  h=1 origin 2022-02: upper 130 >= 125, spike True -> TP
    out = classify_alerts(SCORED, SPIKES, alert_threshold=0.25)
    counts = out["outcome"].value_counts()
    assert counts["TP"] == 1 and counts["FN"] == 1 and counts["TN"] == 1


def test_lower_threshold_adds_false_alarms():
    # threshold 0.05: fire when upper >= 105
    #  h=2 origin 2022-01: upper 110 >= 105, spike False -> FP
    out = classify_alerts(SCORED, SPIKES, alert_threshold=0.05)
    counts = out["outcome"].value_counts()
    assert counts["TP"] == 2 and counts["FP"] == 1


def test_alert_fires_at_the_exact_boundary():
    # upper exactly 115 at threshold 0.15: >= means it fires
    scored = _scored([[1, "2022-01", 1, "2022-02", 115.0, 110.0, 100.0, 115.0]])
    spikes = _spikes([[1, "2022-01", 1, 100.0, True]])
    out = classify_alerts(scored, spikes, alert_threshold=0.15)
    assert out.iloc[0]["outcome"] == "TP"


def test_pr_curve_computes_precision_and_recall():
    thresholds = np.array([0.05, 0.15, 0.25])
    frames = []
    for t in thresholds:
        frames.append(classify_alerts(SCORED, SPIKES, alert_threshold=t).assign(alert_threshold=t))
    curve = pr_curve(pd.concat(frames, ignore_index=True), thresholds)
    # 0.05: TP=2 FP=1 -> precision 2/3, recall 1.0
    # 0.15: TP=2 FP=0 -> precision 1.0,  recall 1.0
    # 0.25: TP=1 FP=0 FN=1 -> precision 1.0, recall 0.5
    assert np.isclose(curve.loc[0, "precision"], 2 / 3)
    assert np.isclose(curve.loc[1, "precision"], 1.0)
    assert np.isclose(curve.loc[2, "recall"], 0.5)


def test_lead_time_distribution_counts_true_positive_horizons():
    # add a third TP at h=3: origin 2021-12, upper 120, spike True
    scored = pd.concat([SCORED, _scored([
        [1, "2021-12", 3, "2022-03", 110.0, 105.0, 90.0, 120.0],
    ])], ignore_index=True)
    spikes = pd.concat([SPIKES, _spikes([
        [1, "2021-12", 3, 100.0, True],
    ])], ignore_index=True)
    out = classify_alerts(scored, spikes, alert_threshold=0.15)
    lt = lead_time_distribution(out)
    # three TPs: two at h=1, one at h=3
    assert lt["n_tp"].sum() == 3
    h1 = lt[lt["horizon"] == 1].iloc[0]
    h3 = lt[lt["horizon"] == 3].iloc[0]
    assert h1["n_tp"] == 2 and np.isclose(h1["share"], 2 / 3)
    assert h3["n_tp"] == 1 and np.isclose(h3["share"], 1 / 3)
