"""Phase 5 cost-analysis tests (issue #12).

The seam is the expected-cost model on a hand-built PR curve: all
literals are hand-computed. Costs are assumptions, not measurements.
"""
import numpy as np
import pandas as pd
import pytest

from src.alerts import expected_costs, pick_threshold


# Synthetic PR curve for unit tests. The real sweep (alert_pr_curve.csv)
# has 11 thresholds; the 1:1 optimum there is 0.35, not 0.30. These
# values are chosen so the hand-computed costs are exact fractions.
PR_CURVE = pd.DataFrame([
    {"alert_threshold": 0.0, "n": 329, "tp": 77, "fp": 223, "fn": 3},
    {"alert_threshold": 0.15, "n": 329, "tp": 60, "fp": 142, "fn": 20},
    {"alert_threshold": 0.30, "n": 329, "tp": 19, "fp": 24, "fn": 61},
])


def test_expected_cost_at_ratio_1_to_1():
    # cost = (fp * 1 + fn * 1) / n
    #  0.00: (223 + 3) / 329 = 226/329
    #  0.15: (142 + 20) / 329 = 162/329
    #  0.30: (24 + 61) / 329 = 85/329
    costs = expected_costs(PR_CURVE, ratios=(1.0,))
    c0 = costs[costs["alert_threshold"] == 0.0].iloc[0]
    c15 = costs[costs["alert_threshold"] == 0.15].iloc[0]
    c30 = costs[costs["alert_threshold"] == 0.30].iloc[0]
    assert np.isclose(c0["expected_cost"], 226 / 329)
    assert np.isclose(c15["expected_cost"], 162 / 329)
    assert np.isclose(c30["expected_cost"], 85 / 329)


def test_expected_cost_at_ratio_1_to_5():
    # cost = (fp * 1 + fn * 5) / n
    #  0.00: (223 + 15) / 329 = 238/329
    #  0.15: (142 + 100) / 329 = 242/329
    #  0.30: (24 + 305) / 329 = 329/329 = 1.0
    costs = expected_costs(PR_CURVE, ratios=(5.0,))
    c0 = costs[costs["alert_threshold"] == 0.0].iloc[0]
    c15 = costs[costs["alert_threshold"] == 0.15].iloc[0]
    c30 = costs[costs["alert_threshold"] == 0.30].iloc[0]
    assert np.isclose(c0["expected_cost"], 238 / 329)
    assert np.isclose(c15["expected_cost"], 242 / 329)
    assert np.isclose(c30["expected_cost"], 1.0)


def test_pick_threshold_minimises_expected_cost():
    # ratio 1:1 -> min at 0.30 (85/329); ratio 1:5 -> min at 0.0 (238/329)
    costs = expected_costs(PR_CURVE, ratios=(1.0, 5.0))
    t1 = pick_threshold(costs, ratio=1.0)
    t5 = pick_threshold(costs, ratio=5.0)
    assert t1 == 0.30
    assert t5 == 0.0


def test_pick_threshold_returns_nan_when_no_rows():
    costs = pd.DataFrame(columns=["alert_threshold", "expected_cost", "cost_ratio"])
    assert np.isnan(pick_threshold(costs, ratio=1.0))


def test_costs_are_labelled_as_assumptions():
    costs = expected_costs(PR_CURVE, ratios=(1.0,))
    assert "assumption" in costs.columns
    assert costs["assumption"].iloc[0] == (
        "costs are assumptions, not measurements; real values would need "
        "to come from a response agency (e.g. WFP Kenya or the National "
        "Drought Management Authority) holding the response-cost data"
    )
