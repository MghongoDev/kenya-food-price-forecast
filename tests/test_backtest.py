"""Phase 2 tests at the backtest seam (issue #2).

A synthetic panel replaces the parquet: small, hand-checkable, and it
keeps expectations coming from worked examples rather than the code.
"""
import numpy as np
import pandas as pd

PANEL_COLUMNS = ["series", "market_id", "market", "month", "price", "observed"]


def _panel_rows(rows, series="white_maize_wholesale"):
    """rows: (market_id, market, month, price) with price None for unobserved months."""
    return pd.DataFrame(
        [
            {
                "series": series,
                "market_id": mid,
                "market": mkt,
                "month": m,
                "price": np.nan if p is None else float(p),
                "observed": p is not None,
            }
            for mid, mkt, m, p in rows
        ],
        columns=PANEL_COLUMNS,
    )


def _seq(months, base, step):
    return base + step * np.arange(len(months), dtype=float)


# --- slice 1: month arithmetic + the harness contract -------------------------

def test_months_ahead_steps_the_origin():
    from src.backtest import months_ahead
    assert months_ahead("2022-01", 1) == "2022-02"
    assert months_ahead("2022-11", 3) == "2023-02"
    assert months_ahead("2022-12", 12) == "2023-12"


def test_harness_passes_only_data_at_or_before_the_origin():
    from src.backtest import rolling_backtest, months_ahead

    panel = _panel_rows([
        (1, "Kangemi (Nairobi)", m, p)
        for m, p in zip(
            ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06"],
            [10.0, 11.0, 12.0, None, 14.0, 15.0],
        )
    ])

    seen_maxes = []
    def fit_predict(train, origin, horizons):
        seen_maxes.append(train["month"].max())
        assert (train["month"] <= origin).all()          # nothing after the origin
        return pd.DataFrame([
            {"market_id": 1, "horizon": h, "target_month": months_ahead(origin, h), "price_pred": 99.0}
            for h in horizons
        ])

    score = rolling_backtest(panel, fit_predict, origins=["2022-01", "2022-02"], horizons=[1, 2])
    assert seen_maxes == ["2022-01", "2022-02"]          # expanding window honours the origin
    assert list(score.columns) == [
        "series", "market_id", "market", "origin", "horizon", "target_month",
        "price_true", "price_pred",
    ]


def test_harness_scores_observed_targets_only():
    from src.backtest import rolling_backtest, months_ahead

    panel = _panel_rows([
        (1, "Kangemi (Nairobi)", "2022-01", 10.0),
        (1, "Kangemi (Nairobi)", "2022-02", None),   # unobserved month: must never be scored
        (1, "Kangemi (Nairobi)", "2022-03", 12.0),
    ])

    def fit_predict(train, origin, horizons):
        rows = []
        for m_id, g in train[train["observed"]].groupby("market_id"):
            for h in horizons:
                rows.append({"market_id": m_id, "horizon": h,
                             "target_month": months_ahead(origin, h), "price_pred": float(g["price"].iloc[-1])})
        return pd.DataFrame(rows)

    score = rolling_backtest(panel, fit_predict, origins=["2022-01"], horizons=[1, 2])
    # h=1 targets 2022-02 (unobserved: dropped); h=2 targets 2022-03 (kept)
    assert len(score) == 1
    row = score.iloc[0]
    assert row["horizon"] == 2 and row["target_month"] == "2022-03"
    assert row["price_true"] == 12.0 and row["price_pred"] == 10.0


# --- slice 2: naive baseline ---------------------------------------------------
from src.baselines import naive
from src.backtest import months_ahead as _months_ahead  # noqa: F401 (used in fixtures)


def test_naive_forecasts_the_last_observed_price_for_every_horizon():
    panel = _panel_rows([
        (1, "Kangemi (Nairobi)", "2022-01", 10.0),
        (1, "Kangemi (Nairobi)", "2022-02", None),
        (1, "Kangemi (Nairobi)", "2022-03", 12.0),
        (2, "Kibuye (Kisumu)", "2022-01", 40.0),
        (2, "Kibuye (Kisumu)", "2022-02", 41.0),
    ])
    train = panel[panel["month"] <= "2022-02"]
    preds = naive(train, "2022-02", horizons=[1, 2])
    by = preds.set_index(["market_id", "target_month", "horizon"])["price_pred"]
    assert by.loc[(1, "2022-03", 1)] == 10.0   # last observed at or before the origin
    assert by.loc[(1, "2022-04", 2)] == 10.0
    assert by.loc[(2, "2022-03", 1)] == 41.0
    assert by.loc[(2, "2022-04", 2)] == 41.0


def test_naive_abstains_for_markets_without_observed_history():
    panel = _panel_rows([(3, "New market", "2022-03", 9.0)])
    train = panel[panel["month"] <= "2022-02"]   # market 3 has nothing yet
    preds = naive(train, "2022-02", horizons=[1])
    assert (preds["market_id"] == 3).sum() == 0 if len(preds) else True
    assert len(preds) == 0


# --- slice 3: seasonal naive ---------------------------------------------------
from src.baselines import seasonal_naive


def test_seasonal_naive_uses_the_nearest_same_calendar_month():
    panel = _panel_rows([
        (1, "Kangemi (Nairobi)", "2020-03", 30.0),
        (1, "Kangemi (Nairobi)", "2021-03", 20.0),
        (1, "Kangemi (Nairobi)", "2022-02", 41.0),
    ])
    train = panel[panel["month"] <= "2022-02"]
    preds = seasonal_naive(train, "2022-02", horizons=[1, 2])
    by = preds.set_index(["target_month", "horizon"])["price_pred"]
    assert by.loc[("2022-03", 1)] == 20.0      # March target: 2021-03 beats 2020-03
    assert ("2022-04", 2) not in by.index      # April target with no April history: abstain


def test_seasonal_naive_falls_back_twenty_four_months_and_abstains():
    panel = _panel_rows([
        (2, "Kibuye (Kisumu)", "2020-03", 30.0),
        (2, "Kibuye (Kisumu)", "2022-02", 41.0),
        (3, "No history", "2022-02", 41.0),
    ])
    train = panel[panel["month"] <= "2022-02"]
    preds = seasonal_naive(train, "2022-02", horizons=[1])
    by = preds.set_index(["market_id"])["price_pred"]
    assert by.loc[2] == 30.0                    # 24 months back fills where 12 has nothing
    assert 3 not in preds["market_id"].values   # no same-month history at all: abstain


# --- slice 4: seasonal mean log change -----------------------------------------
from src.baselines import seasonal_mean_log_change
from src.backtest import months_ahead as ma4


def test_seasonal_mean_log_change_grows_the_last_price_by_the_mean_log_change():
    # the worked example: three Februaries, each fed by an observed January
    panel = _panel_rows([
        (1, "Kangemi (Nairobi)", "2021-01", 10.0), (1, "Kangemi (Nairobi)", "2021-02", 11.0),
        (1, "Kangemi (Nairobi)", "2022-01", 20.0), (1, "Kangemi (Nairobi)", "2022-02", 24.0),
        (1, "Kangemi (Nairobi)", "2023-01", 40.0), (1, "Kangemi (Nairobi)", "2023-02", 36.0),
        (1, "Kangemi (Nairobi)", "2024-01", 50.0),
    ])
    train = panel[panel["month"] <= "2024-01"]
    preds = seasonal_mean_log_change(train, "2024-01", horizons=[1])
    by = preds.set_index(["target_month", "horizon"])["price_pred"]
    # mu = mean(ln(11/10), ln(24/20), ln(36/40)) = 0.0574237; pred = 50 * exp(mu)
    import math
    mu = (math.log(11 / 10) + math.log(24 / 20) + math.log(36 / 40)) / 3
    assert np.isclose(by.loc[("2024-02", 1)], 50.0 * math.exp(mu), rtol=1e-9)


def test_seasonal_mean_log_change_abstains_without_enough_transitions():
    panel = _panel_rows([
        (1, "Kangemi (Nairobi)", "2021-01", 10.0), (1, "Kangemi (Nairobi)", "2021-02", 11.0),
        (1, "Kangemi (Nairobi)", "2022-01", 20.0), (1, "Kangemi (Nairobi)", "2022-02", 24.0),
        (1, "Kangemi (Nairobi)", "2024-01", 50.0),
    ])
    train = panel[panel["month"] <= "2024-01"]
    preds = seasonal_mean_log_change(train, "2024-01", horizons=[1])
    assert len(preds) == 0   # two February transitions: below the threshold


# --- slice 5: metrics ----------------------------------------------------------
from src.metrics import metric_table


def _score_frames():
    """Three scored rows for two markets, plus a seasonal-naive reference with
    unit errors so MASE arithmetic stays hand-checkable."""
    model = pd.DataFrame(
        [
            # m1: target in 2022 -> drought year; target in 2023 -> normal
            {"series": "s", "market_id": 1, "market": "A", "origin": "2022-01",
             "horizon": 1, "target_month": "2022-02", "price_true": 11.0, "price_pred": 12.0},
            {"series": "s", "market_id": 1, "market": "A", "origin": "2023-01",
             "horizon": 1, "target_month": "2023-02", "price_true": 14.0, "price_pred": 10.0},
            {"series": "s", "market_id": 2, "market": "B", "origin": "2022-01",
             "horizon": 1, "target_month": "2022-02", "price_true": 41.0, "price_pred": 39.0},
        ]
    )
    reference = pd.DataFrame(model.assign(price_pred=[10.0, 15.0, 40.0]))
    return model, reference


def test_metric_table_computes_mae_wape_and_mase_by_period_and_market():
    model, reference = _score_frames()
    table = metric_table(model, reference)
    rows = table.set_index(["period", "market", "horizon"])

    all_a = rows.loc[("all", "A", 1)]
    assert all_a["n"] == 2 and np.isclose(all_a["mae_kes"], 2.5)
    assert np.isclose(all_a["wape"], (1 + 4) / (11 + 14))
    assert np.isclose(all_a["mase"], 2.5 / 1.0)   # reference errors: |1| and |1|

    drought_b = rows.loc[("drought", "B", 1)]
    assert drought_b["n"] == 1 and np.isclose(drought_b["mase"], 2.0)

    assert ("normal", "A", 1) in rows.index and rows.loc[("normal", "A", 1)]["mae_kes"] == 4.0


def test_metric_table_pools_markets_into_an_all_row():
    model, reference = _score_frames()
    table = metric_table(model, reference)
    pooled = table[table["market"] == "ALL"].set_index(["period", "horizon"])
    all_rows = pooled.loc["all", 1]
    assert all_rows["n"] == 3
    assert np.isclose(all_rows["mae_kes"], (1 + 4 + 2) / 3)
    assert np.isclose(all_rows["wape"], 7 / 66)
    assert np.isclose(all_rows["mase"], (7 / 3) / 1.0)
    assert pooled.loc["normal", 1]["n"] == 1   # 2023 target only


# --- slice 6: leakage test at the harness seam ----------------------------------

def test_forecasts_at_t_unchanged_when_data_after_t_is_corrupted():
    from src.backtest import rolling_backtest
    from src.baselines import naive, seasonal_naive

    months = [f"{y}-{m:02d}" for y in (2021, 2022) for m in range(1, 13)]
    months = [m for m in months if "2021-01" <= m <= "2022-10" and m != "2022-05"]
    panel = _panel_rows(
        (mid, mkt, m, 30 + i + mid)
        for mid, mkt in [(1, "A"), (2, "B")]
        for i, m in enumerate(sorted(months))
    )

    for forecaster in (naive, seasonal_naive):
        origins = ["2021-06", "2022-01", "2022-04"]
        for t in origins:
            clean = rolling_backtest(panel, forecaster, [t], horizons=[1, 2])
            dirty = panel.copy()
            dirty.loc[dirty["month"] > t, "price"] = 999.0
            dirty_score = rolling_backtest(dirty, forecaster, [t], horizons=[1, 2])
            key = ["origin", "horizon", "market_id", "target_month"]
            a = clean.set_index(key)["price_pred"].sort_index()
            b = dirty_score.set_index(key)["price_pred"].sort_index()
            pd.testing.assert_series_equal(a, b)
            # the truth DOES change: the corruption is visible to scoring only
            # (when this origin produced rows at all; abstentions stay empty)
            if len(dirty_score) and len(clean):
                assert (dirty_score["price_true"] == 999.0).any()


# --- slice 7: real-panel smoke test ----------------------------------------------
from pathlib import Path

import pytest


def test_naive_backtest_runs_on_the_real_panel():
    panel_path = Path("data/processed/panel.parquet")
    if not panel_path.exists():
        pytest.skip("run make data first: the processed panel is missing")
    from src.backtest import make_origins, rolling_backtest
    from src.baselines import naive

    real = pd.read_parquet(panel_path)
    origins = make_origins(real, "2022-06", "2022-08")
    score = rolling_backtest(real, naive, origins, horizons=[1, 2, 3])
    assert len(score) > 0
    assert score["price_true"].notna().all() and score["price_pred"].notna().all()
    assert set(score["horizon"]) <= {1, 2, 3}
