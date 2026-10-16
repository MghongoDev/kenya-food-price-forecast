"""Phase 5 dashboard-artifact tests (issue #13).

The seam is the file writer: given the pipeline's covered forecasts
and alert outcomes, write the small files the Streamlit app loads.
"""
import pandas as pd
import pytest

from src.dashboard_data import write_alert_history, write_forecasts


def _covered():
    return pd.DataFrame([
        {"market_id": 1, "market": "A", "origin": "2022-01", "horizon": 1,
         "target_month": "2022-02", "price_true": 110.0, "price_pred": 105.0,
         "lower": 90.0, "upper": 120.0, "covered": True, "width_kes": 30.0},
        {"market_id": 1, "market": "A", "origin": "2022-01", "horizon": 2,
         "target_month": "2022-03", "price_true": 100.0, "price_pred": 100.0,
         "lower": 85.0, "upper": 115.0, "covered": False, "width_kes": 30.0},
    ])


def _outcomes():
    return pd.DataFrame([
        {"market_id": 1, "origin": "2022-01", "horizon": 1,
         "price_origin": 100.0, "spike": True, "alert_fired": True,
         "outcome": "TP"},
        {"market_id": 1, "origin": "2022-01", "horizon": 2,
         "price_origin": 100.0, "spike": False, "alert_fired": False,
         "outcome": "TN"},
    ])


def test_write_forecasts_produces_the_app_columns(tmp_path):
    out = tmp_path / "forecasts.csv"
    write_forecasts(_covered(), out)
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "market_id", "market", "origin", "horizon", "target_month",
        "price_true", "price_pred", "lower", "upper",
    ]
    assert len(df) == 2
    assert df.iloc[0]["lower"] == 90.0
    assert df.iloc[1]["upper"] == 115.0


def test_write_alert_history_produces_the_app_columns(tmp_path):
    out = tmp_path / "alert_history.csv"
    write_alert_history(_outcomes(), out)
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "market_id", "origin", "horizon", "price_origin", "spike",
        "alert_fired", "outcome", "lead_time",
    ]
    assert len(df) == 2
    assert df.iloc[0]["outcome"] == "TP"
    assert df.iloc[1]["alert_fired"] == False  # noqa: E712


def test_write_alert_history_computes_lead_time(tmp_path):
    out = tmp_path / "alert_history.csv"
    write_alert_history(_outcomes(), out)
    df = pd.read_csv(out)
    # lead time = horizon for true positives; NaN otherwise
    assert df.iloc[0]["lead_time"] == 1.0
    assert pd.isna(df.iloc[1]["lead_time"])
