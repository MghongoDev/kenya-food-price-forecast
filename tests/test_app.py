"""Phase 5 dashboard-app tests (issue #14).

The seam is the fan-chart builder: given a panel and forecasts frame,
produce a matplotlib figure with the observed line, forecast points,
and interval bands per horizon.
"""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app import make_fan_chart


def _panel():
    return pd.DataFrame([
        {"market_id": 1, "month": "2022-01", "price": 100.0},
        {"market_id": 1, "month": "2022-02", "price": 110.0},
        {"market_id": 1, "month": "2022-03", "price": 105.0},
        {"market_id": 2, "month": "2022-01", "price": 50.0},
    ])


def _forecasts():
    return pd.DataFrame([
        {"market_id": 1, "origin": "2022-01", "horizon": 1,
         "target_month": "2022-02", "price_pred": 108.0,
         "lower": 95.0, "upper": 120.0},
        {"market_id": 1, "origin": "2022-01", "horizon": 2,
         "target_month": "2022-03", "price_pred": 106.0,
         "lower": 92.0, "upper": 118.0},
        {"market_id": 2, "origin": "2022-01", "horizon": 1,
         "target_month": "2022-02", "price_pred": 52.0,
         "lower": 48.0, "upper": 56.0},
    ])


def test_fan_chart_returns_a_figure():
    fig = make_fan_chart(_panel(), _forecasts(), market_id=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_fan_chart_filters_to_the_selected_market():
    fig = make_fan_chart(_panel(), _forecasts(), market_id=1)
    ax = fig.axes[0]
    # one observed line + two horizons x (band + pred line) = 5 artists
    # market 2's forecast must not appear
    assert len(ax.lines) == 3  # observed + 2 pred lines
    assert len(ax.collections) == 2  # 2 interval bands
    plt.close(fig)


def test_fan_chart_handles_a_market_with_no_forecasts():
    fig = make_fan_chart(_panel(), _forecasts(), market_id=99)
    ax = fig.axes[0]
    assert len(ax.lines) == 1  # observed only
    plt.close(fig)
