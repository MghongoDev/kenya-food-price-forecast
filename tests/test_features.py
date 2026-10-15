"""Phase 3 feature tests (issue #3).

Synthetic rain tables and panels with hand-computed expectations.
"""
import numpy as np
import pandas as pd

from src.features import CLIMATOLOGY_YEARS, add_rain_features


def _rain_rows(rows):
    """rows: (market_id, month, rain_mm) with None for an unobserved month."""
    return pd.DataFrame(
        rows, columns=["market_id", "month", "rain_mm"]
    ).assign(rain_mm=lambda d: d["rain_mm"].astype("float64"))


# --- slice B: rain features ----------------------------------------------------

def test_rain_cumulative_sums_need_full_windows():
    rain = _rain_rows([
        (1, "2022-01", 10.0), (1, "2022-02", 20.0), (1, "2022-03", 0.0),
        (1, "2022-04", None),  # gap: the 6-month window stays honest
        (1, "2022-05", 2.0), (1, "2022-06", 4.0), (1, "2022-07", 6.0),
    ])
    out = add_rain_features(rain)
    out = out.set_index("month")
    assert np.isclose(out.loc["2022-03", "rain_3m"], 30.0)
    assert np.isnan(out.loc["2022-04", "rain_3m"])       # partial window: missing stays missing
    assert np.isclose(out.loc["2022-07", "rain_3m"], 12.0)
    assert np.isnan(out.loc["2022-07", "rain_6m"])       # only 7 months of history


def test_rain_anomaly_uses_fixed_climatology_years():
    rows = []
    for y in range(2010, 2020):
        rows.append((1, f"{y}-02", 20.0))                # every Feb in the climatology decade
    rows += [(1, "2023-02", 50.0), (1, "2024-02", 5.0)]
    out = add_rain_features(_rain_rows(rows)).set_index("month")
    # climatology Feb mean = 20.0; anomaly is the deviation from it
    assert np.isclose(out.loc["2023-02", "rain_anom"], 30.0)
    assert np.isclose(out.loc["2024-02", "rain_anom"], -15.0)
    assert int(CLIMATOLOGY_YEARS[0]) == 2010 and int(CLIMATOLOGY_YEARS[1]) == 2019


# --- slice C: the origin-honest feature table -----------------------------------
from src.features import feature_table


def _mini_panel():
    """Two markets; market 1 has a hole in its price history."""
    rows = []
    seq = {1: {**{f"2022-{m:02d}": 10.0 + m for m in range(1, 13) if m != 5},
               "2023-01": 22.0, "2023-02": 24.0},
           2: {**{f"2022-{m:02d}": 40.0 - m for m in range(1, 13)},
               "2023-01": 30.0, "2023-02": 28.0}}
    cols = ["series", "market_id", "market", "month", "price", "observed"]
    rows = [
        {"series": "s", "market_id": mid, "market": f"M{mid}", "month": m,
         "price": p, "observed": True}
        for mid, ms in seq.items() for m, p in ms.items()
    ]
    # the missing month for market 1 exists as an unobserved grid row
    rows.append({"series": "s", "market_id": 1, "market": "M1",
                 "month": "2022-05", "price": np.nan, "observed": False})
    return pd.DataFrame(rows, columns=cols)


def test_feature_table_lags_season_and_relative_price():
    feat = feature_table(_mini_panel()).set_index(["market_id", "month"])
    row = feat.loc[(1, "2022-07")]
    assert row["price"] == 17.0
    assert row["lag_1"] == 16.0
    assert np.isnan(row["lag_2"])              # calendar lag: May hole stays missing
    assert row["lag_3"] == 14.0                # April
    assert row["lag_6"] == 11.0                # January 2022
    assert np.isnan(row["lag_12"])             # July 2021 is outside the panel
    assert row["month_of_year"] == 7
    assert np.isnan(row["roll_mean_3"])        # the May hole opens the window
    assert np.isclose(row["rel_price"], 17.0 / np.mean([17.0, 33.0]))


def test_feature_table_flags_missing_lag_and_keeps_gaps_missing():
    feat = feature_table(_mini_panel()).set_index(["market_id", "month"])
    assert feat.loc[(1, "2022-06"), "missing_lag_1"] == 1   # May price unobserved
    assert feat.loc[(1, "2022-07"), "missing_lag_1"] == 0
    assert feat.loc[(1, "2022-07"), "missing_lag_2"] == 1
    assert feat.loc[(2, "2022-06"), "missing_lag_1"] == 0
    assert np.isnan(feat.loc[(1, "2022-05"), "roll_mean_3"])
    assert np.isclose(feat.loc[(2, "2022-03"), "roll_mean_3"], 38.0)  # 39,38,37
    assert np.isclose(feat.loc[(2, "2022-03"), "roll_vol_3"], np.std([39.0, 38.0, 37.0], ddof=1))


def test_feature_table_joins_rain_columns_when_given():
    rain = pd.DataFrame([
        {"market_id": 1, "month": "2023-02", "rain_mm": 12.0,
         "rain_3m": 30.0, "rain_6m": 60.0, "rain_anom": -8.0},
    ])
    feat = feature_table(_mini_panel(), rain_features=rain).set_index(["market_id", "month"])
    row = feat.loc[(1, "2023-02")]
    assert row["rain_3m"] == 30.0 and row["rain_anom"] == -8.0
    assert np.isnan(feat.loc[(2, "2023-02")].get("rain_3m", np.nan))
