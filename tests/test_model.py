"""Phase 3 model tests (issue #3): the direct per-horizon wrapper at the
harness seam. Synthetics carry hand-worked expectations; nothing here
claims model quality, only contract correctness.
"""
import numpy as np
import pandas as pd

from src.backtest import rolling_backtest
from src.features import PRICE_FEATURE_COLUMNS, RAIN_FEATURE_COLUMNS
from src.model import model_feature_columns, make_fit_predict

DEFAULT_PARAMS = {"n_estimators": 60, "learning_rate": 0.5, "num_leaves": 31,
                  "min_child_samples": 1, "verbose": -1}


def _growth_panel(markets=((1, "A", 100.0), (2, "B", 40.0)), growth=1.1):
    """Prices grow a fixed factor every month: the model can learn the
    log change exactly from any origin's features."""
    rows = []
    months = [f"{y}-{m:02d}" for y in range(2013, 2024) for m in range(1, 13)]
    for mid, mkt, base in markets:
        p = base
        for m in months:
            rows.append({"series": "s", "market_id": mid, "market": mkt,
                         "month": m, "price": round(p, 4), "observed": True})
            p *= growth
    return pd.DataFrame(rows)


def _rain_all_zero(panel):
    return pd.DataFrame({
        "market_id": panel["market_id"],
        "month": panel["month"],
        "rain_mm": 0.0, "rain_3m": 0.0, "rain_6m": 0.0, "rain_anom": 0.0,
    })


# --- slice D ------------------------------------------------------------------

def test_model_feature_columns_switch_rain_in_and_out():
    base = model_feature_columns(use_rain=False)
    with_rain = model_feature_columns(use_rain=True)
    assert "rain_3m" not in base and "rain_3m" in with_rain
    assert base == list(PRICE_FEATURE_COLUMNS)
    assert with_rain == list(PRICE_FEATURE_COLUMNS) + list(RAIN_FEATURE_COLUMNS)


def test_direct_model_learns_a_constant_log_change_per_horizon():
    panel = _growth_panel()
    rain = _rain_all_zero(panel)
    fp = make_fit_predict(rain_features=rain, use_rain=True, params=DEFAULT_PARAMS)
    score = rolling_backtest(panel, fp, origins=["2023-05"], horizons=[1, 2, 3])
    # the series grows exactly 10% per month, so a correct model recovers
    # the target: price_pred == price_true. A broken delta prediction shows
    # up as price_true / 1.1**h, far outside the 2% tolerance.
    for h in (1, 2, 3):
        rows = score[score["horizon"] == h]
        assert len(rows) == 2
        assert np.allclose(rows["price_pred"], rows["price_true"], rtol=0.02)


def test_direct_model_abstains_when_the_origin_price_is_unobserved():
    panel = _growth_panel()
    mask = (panel["month"] == "2023-05") & (panel["market_id"] == 1)
    panel.loc[mask, "price"] = np.nan
    panel.loc[mask, "observed"] = False
    fp = make_fit_predict(use_rain=False, params=DEFAULT_PARAMS)
    preds = fp(panel[panel["month"] <= "2023-05"], "2023-05", horizons=[1])
    assert 1 not in preds["market_id"].values
    assert 2 in preds["market_id"].values


# --- slice E: light tuning on a time-ordered validation block -------------------
from src.model import choose_hyperparams

GOOD = {"n_estimators": 60, "num_leaves": 31, "min_child_samples": 1}
BAD = {"n_estimators": 1, "num_leaves": 2, "min_child_samples": 1}


def _parity_panel():
    """y = log change flips by (month_of_year + market parity): a split a
    two-leaf tree cannot express, trees with depth can."""
    months = [f"{y}-{m:02d}" for y in range(2013, 2022) for m in range(1, 13)]
    rows = []
    for mid, base in ((1, 100.0), (2, 200.0)):
        p = base
        for m in months:
            up = (int(m[5:]) + mid) % 2 == 0
            rows.append({"series": "s", "market_id": mid, "market": f"M{mid}",
                         "month": m, "price": round(p, 2), "observed": True})
            p *= 1.0513 if up else 1 / 1.0513
    return pd.DataFrame(rows)


def test_choose_hyperparams_picks_the_combo_the_validation_block_prefers():
    panel = _parity_panel()
    picked = choose_hyperparams(
        panel, use_rain=False, grid=[GOOD, BAD],
        val_start="2020-01", val_end="2021-03",
    )
    assert set(picked.keys()) == {1, 2, 3}
    for h, params in picked.items():
        assert params["n_estimators"] == 60   # the 1-tree combo cannot express parity
