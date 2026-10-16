"""Phase 4 conformal tests at the calibration seam (issue #4, ADR 0003).

Synthetic frames carry hand-worked expectations: the halfwidth literals
come from the quantile formula computed by hand, never from the code's
own re-computation. The exchangeable-coverage test uses a fixed seed.
"""
import numpy as np
import pandas as pd
import pytest

from src.conformal import (
    ALPHA,
    apply_conformal,
    conformal_halfwidth,
    coverage_table,
    fit_conformal,
)


def _scored(resid, horizon, market_id=1, market="M", origin="2022-01"):
    """One scored row per (horizon, residual) with price_pred fixed at 100:
    price_true = 100 * exp(resid), so the log residual is exactly resid."""
    n = len(resid)
    return pd.DataFrame({
        "market_id": [market_id] * n,
        "market": [market] * n,
        "series": ["s"] * n,
        "origin": [origin] * n,
        "horizon": [horizon] * n,
        "target_month": [f"2022-0{h + 1}" for h in [horizon] * n],
        "price_true": [100.0 * np.exp(r) for r in resid],
        "price_pred": [100.0] * n,
    })


def test_conformal_halfwidth_matches_the_guides_formula():
    # n=4, alpha=0.2: q = ceil(5*0.8)/4 = 1.0 -> the max of |resid|
    assert conformal_halfwidth([1.0, 2.0, 3.0, 4.0], alpha=0.2) == 4.0
    # n=5: q = ceil(6*0.8)/5 = 1.0 -> max
    assert conformal_halfwidth([3.0, 1.0, 4.0, 1.0, 5.0], alpha=0.2) == 5.0
    # n=8, alpha=0.25: q = ceil(9*0.75)/8 = 7/8 = 0.875; sorted |r| is
    # [1,1,2,3,4,5,6,9], position 0.875*7 = 6.125 -> 6 + 0.125*(9-6)
    assert np.isclose(
        conformal_halfwidth([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], alpha=0.25),
        6.375,
    )


def test_halfwidth_scales_with_the_residuals():
    base = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    hw_small = conformal_halfwidth(base, alpha=0.2)
    hw_large = conformal_halfwidth([4 * r for r in base], alpha=0.2)
    assert np.isclose(hw_large, 4 * hw_small)


def test_fit_conformal_calibrates_per_horizon():
    cal = pd.concat([
        _scored([0.1, -0.2, 0.3, -0.4, 0.5], horizon=1),
        _scored([1.0, -2.0, 3.0, -4.0, 5.0], horizon=3),
    ])
    hw = fit_conformal(cal, alpha=0.2)
    assert set(hw.keys()) == {1, 3}
    # horizon 3 residuals are 10x horizon 1: q = ceil(6*0.8)/5 = 1.0,
    # so each halfwidth is the max |resid| of its own horizon
    assert np.isclose(hw[1], 0.5)
    assert np.isclose(hw[3], 5.0)


def test_apply_conformal_builds_log_scale_intervals_and_flags_coverage():
    scored = _scored([0.05, -0.3, 0.6], horizon=1, origin="2022-03")
    out = apply_conformal(scored, {1: 0.4})
    # pred=100, hw=0.4: lower = 100*e^-0.4, upper = 100*e^0.4 (hand literals)
    assert np.isclose(out.loc[0, "lower"], 100.0 * np.exp(-0.4))
    assert np.isclose(out.loc[0, "upper"], 100.0 * np.exp(0.4))
    # covered: |log(true/pred)| <= 0.4 -> 0.05 yes, 0.3 yes, 0.6 no
    assert out["covered"].tolist() == [True, True, False]
    assert np.isclose(out.loc[0, "width_kes"], 100.0 * (np.exp(0.4) - np.exp(-0.4)))


def test_calibration_is_blind_to_test_window_residuals():
    """Extreme errors after the calibration window must not move the
    halfwidth: fit sees calibration rows only."""
    cal = _scored([0.1, -0.1, 0.2, -0.2, 0.15], horizon=1, origin="2021-07")
    hw = fit_conformal(cal, alpha=0.2)
    assert np.isclose(hw[1], 0.2)  # max |resid| at q = 1.0
    test = _scored([10.0, -10.0], horizon=1, origin="2022-01")
    out = apply_conformal(test, hw)
    assert out["covered"].tolist() == [False, False]


def test_exchangeable_coverage_lands_near_nominal():
    rng = np.random.default_rng(0)
    resid = rng.normal(0.0, 1.0, 4000)
    cal, test = resid[:2000], resid[2000:]
    hw = conformal_halfwidth(cal, alpha=ALPHA)
    covered = np.abs(test) <= hw
    assert 0.78 <= covered.mean() <= 0.82  # 80% +- 2SE at n=2000


def test_coverage_table_reports_by_market_and_period():
    rows = []
    for market_id, market in ((1, "A"), (2, "B")):
        for horizon in (1, 2):
            for target, resid in (("2022-04", 0.01), ("2022-08", 5.0),
                                   ("2023-04", 0.02), ("2023-08", 6.0)):
                rows.append({
                    "market_id": market_id, "market": market, "series": "s",
                    "origin": "2022-01", "horizon": horizon,
                    "target_month": target,
                    "price_true": 100.0 * np.exp(resid),
                    "price_pred": 100.0,
                    "lower": 90.0, "upper": 110.0,
                })
    table = coverage_table(pd.DataFrame(rows))
    # market A horizon 1: 4 target months, the small-residue ones (2022-04,
    # 2023-04) sit inside [90,110], the e^5/e^6 ones do not -> coverage 0.5
    a1 = table[(table["market"] == "A") & (table["horizon"] == 1)].iloc[0]
    assert a1["n"] == 4 and np.isclose(a1["coverage"], 0.5)
    # drought rows are the 2022 targets; the pooled drought row for
    # horizon 1 spans both markets: 4 rows, 2 covered (the 2022-04 ones)
    d = table[(table["market"] == "ALL") & (table["horizon"] == 1)
              & (table["period"] == "drought")].iloc[0]
    assert d["n"] == 4 and np.isclose(d["coverage"], 0.5)
