"""Phase 3 rainfall tests at the ClimateSERV seam (issue #3, ADR 0008).

The API is mocked at requests.get, so the suite stays offline. Expected
monthly values are hand-computed literals from the daily rows, never the
code's own re-computation.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

import src.rainfall as rainfall_mod
from src.rainfall import (
    build_rain_table,
    circle_polygon,
    fetch_daily,
    job_result,
    submit_job,
    _get_json,
)


def test_circle_polygon_is_a_closed_ring_at_the_radius():
    ring = circle_polygon(36.74, -1.27, 25.0)["coordinates"][0]
    assert ring[0] == ring[-1]  # closed ring
    cos_lat = math.cos(math.radians(-1.27))
    for lon, lat in ring[:-1]:
        dx_km = (lon - 36.74) * 111.0 * cos_lat
        dy_km = (lat - (-1.27)) * 111.0
        assert abs(math.hypot(dx_km, dy_km) - 25.0) < 0.01


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_submit_job_posts_chirps_daily_average_over_the_polygon(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResponse(["job-abc"])

    monkeypatch.setattr(rainfall_mod.requests, "get", fake_get)
    job = submit_job(circle_polygon(36.74, -1.27), "01/01/2022", "12/31/2022")
    assert job == "job-abc"
    url, params = calls[0]
    assert url.endswith("/api/submitDataRequest/")
    assert params["datatype"] == "0"           # UCSB CHIRPS daily rainfall
    assert params["operationtype"] == "5"      # average over the polygon
    assert params["intervaltype"] == "0"       # daily
    assert params["begintime"] == "01/01/2022" and params["endtime"] == "12/31/2022"
    assert json.loads(params["geometry"])["type"] == "Polygon"


def test_job_result_polls_past_errmsg_until_data_arrives(monkeypatch):
    responses = [
        {"errMsg": "not ready"},
        {"errMsg": "still running"},
        {"data": [{"isodate": "01/01/2022", "value": {"avg": 1.5}}]},
    ]
    monkeypatch.setattr(rainfall_mod, "_get_json", lambda url, params: responses.pop(0))
    monkeypatch.setattr(rainfall_mod.time, "sleep", lambda s: None)
    rows = job_result("job-abc", max_polls=3)
    assert rows == [{"isodate": "01/01/2022", "value": {"avg": 1.5}}]


def test_job_result_raises_when_the_job_never_completes(monkeypatch):
    monkeypatch.setattr(rainfall_mod, "_get_json", lambda url, params: {"errMsg": "nope"})
    monkeypatch.setattr(rainfall_mod.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        job_result("job-abc", max_polls=2)


def test_get_json_retries_transient_failures(monkeypatch):
    attempts = []

    def flaky(url, params=None, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise rainfall_mod.requests.ConnectionError("boom")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(rainfall_mod.requests, "get", flaky)
    monkeypatch.setattr(rainfall_mod.time, "sleep", lambda s: None)
    assert _get_json("http://x", {}) == {"ok": True}
    assert len(attempts) == 3


def test_fetch_daily_chunks_a_multi_year_span_by_calendar_year(monkeypatch):
    submits = []
    chunks = {"n": 0}

    def fake_get_json(url, params):
        if "submitDataRequest" in url:
            submits.append((params["begintime"], params["endtime"]))
            return ["job-1"]
        chunks["n"] += 1
        return {"data": [{"isodate": f"01/01/{2020 + chunks['n']}", "value": {"avg": 2.0}}]}

    monkeypatch.setattr(rainfall_mod, "_get_json", fake_get_json)
    monkeypatch.setattr(rainfall_mod.time, "sleep", lambda s: None)
    daily = fetch_daily(36.74, -1.27, "06/01/2021", "03/31/2022")
    assert submits == [("06/01/2021", "12/31/2021"), ("01/01/2022", "03/31/2022")]
    assert daily["date"].dt.year.tolist() == [2021, 2022]
    assert daily["rain_mm"].tolist() == [2.0, 2.0]


# --- slice F: the market rain table -------------------------------------------

def test_build_rain_table_sums_daily_rows_into_months(monkeypatch):
    def fake_fetch(lon, lat, begin, end, radius_km=rainfall_mod.RADIUS_KM):
        return pd.DataFrame({
            "date": pd.to_datetime(["2022-01-15", "2022-01-20", "2022-02-10", "2022-02-11"]),
            "rain_mm": [10.0, 5.0, 3.0, float("nan")],
        })

    monkeypatch.setattr(rainfall_mod, "fetch_daily", fake_fetch)
    coords = pd.DataFrame([{"market_id": 4626, "longitude": 36.74, "latitude": -1.27}])
    table = build_rain_table(coords, start="01/01/2022", end="02/28/2022")
    assert table.to_dict("records") == [
        {"market_id": 4626, "month": "2022-01", "rain_mm": 15.0},
        {"market_id": 4626, "month": "2022-02", "rain_mm": 3.0},
    ]


def test_build_rain_table_keeps_a_fully_nan_month_as_nan(monkeypatch):
    def fake_fetch(lon, lat, begin, end, radius_km=rainfall_mod.RADIUS_KM):
        return pd.DataFrame({
            "date": pd.to_datetime(["2022-03-01", "2022-03-02"]),
            "rain_mm": [float("nan"), float("nan")],
        })

    monkeypatch.setattr(rainfall_mod, "fetch_daily", fake_fetch)
    coords = pd.DataFrame([{"market_id": 185, "longitude": 35.28, "latitude": 0.52}])
    table = build_rain_table(coords, start="03/01/2022", end="03/31/2022")
    assert len(table) == 1
    assert np.isnan(table.loc[0, "rain_mm"])
