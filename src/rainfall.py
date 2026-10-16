"""Phase 3 rainfall (issue #3, ADR 0002, route change ADR 0008).

CHIRPS = Climate Hazards Group InfraRed Precipitation with Station data.
Access: the ClimateSERV API (climateserv.servirglobal.net), the official
CHIRPS channel listed by UCSB CHC. datatype 0 = UCSB CHIRPS daily rainfall;
operationtype 5 = average over the polygon; intervaltype 0 = daily.
Sampling: a 25 km radius circle around each market, approximated by a
24-gon — the same region the raster pixel mask selected before the move
to the API (ADR 0008). The monthly value is the sum of the polygon's
daily averages over the month; this reproduces the raster path's area
mean of the monthly total to rounding (validated on 2010-2012: e.g.
Kangemi 2012-04, 401.3 mm both routes).
Rain used by features is rain observed at or before the forecast origin
(glossary: observed); never a rainfall forecast.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

CLIMATESERV_BASE = "https://climateserv.servirglobal.net/api"
CHIRPS_DATATYPE = 0          # UCSB CHIRPS daily rainfall
OP_AVG = 5                   # average over the polygon
INTERVAL_DAILY = 0           # daily
RADIUS_KM = 25.0
KM_PER_DEGREE = 111.0
POLYGON_SIDES = 24
MAX_ATTEMPTS = 4
MAX_POLLS = 24               # 24 x 5 s = 2 min per job chunk
POLL_SECONDS = 5.0


def circle_polygon(lon: float, lat: float, radius_km: float = RADIUS_KM) -> dict:
    """GeoJSON polygon approximating a circle of radius_km around (lon, lat).
    Longitude degrees are compressed by cos(lat) so the ring is round in km."""
    cos_lat = math.cos(math.radians(lat))
    pts = []
    for i in range(POLYGON_SIDES):
        ang = 2 * math.pi * i / POLYGON_SIDES
        dx = radius_km * math.cos(ang) / (KM_PER_DEGREE * cos_lat)
        dy = radius_km * math.sin(ang) / KM_PER_DEGREE
        pts.append([round(lon + dx, 6), round(lat + dy, 6)])
    pts.append(pts[0])  # closed ring
    return {"type": "Polygon", "coordinates": [pts]}


def _get_json(url: str, params: dict) -> dict:
    """One GET with retries; the API is a shared public service."""
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError, OSError) as e:
            last_error = e
            time.sleep(2 ** attempt)  # 1, 2, 4 ..s between attempts
    raise last_error


def submit_job(geometry: dict, begin: str, end: str) -> str:
    """Submit one CHIRPS request ('MM/DD/YYYY' bounds); returns the job id."""
    data = _get_json(
        f"{CLIMATESERV_BASE}/submitDataRequest/",
        {
            "datatype": str(CHIRPS_DATATYPE),
            "begintime": begin,
            "endtime": end,
            "intervaltype": str(INTERVAL_DAILY),
            "operationtype": str(OP_AVG),
            "geometry": json.dumps(geometry),
        },
    )
    return data[0]


def job_result(job_id: str, max_polls: int = MAX_POLLS,
               poll_seconds: float = POLL_SECONDS) -> list[dict]:
    """Poll until the job's daily rows come back. The server answers
    {'errMsg': ...} while the job runs; that is a wait, not a failure."""
    for _ in range(max_polls):
        time.sleep(poll_seconds)
        data = _get_json(f"{CLIMATESERV_BASE}/getDataFromRequest/", {"id": job_id})
        if "data" in data:
            return data["data"]
    raise RuntimeError(f"ClimateSERV job {job_id} never completed")


def fetch_daily(lon: float, lat: float, begin: str, end: str,
                radius_km: float = RADIUS_KM) -> pd.DataFrame:
    """Daily rainfall rows for one market over [begin, end], chunked by
    calendar year: the server rejects very long spans in one request."""
    first_y, last_y = int(begin[-4:]), int(end[-4:])
    frames = []
    for y in range(first_y, last_y + 1):
        b = begin if y == first_y else f"01/01/{y}"
        e = end if y == last_y else f"12/31/{y}"
        job = submit_job(circle_polygon(lon, lat, radius_km), b, e)
        rows = job_result(job)
        frames.append(pd.DataFrame(rows))
    if not any(len(f) for f in frames):
        return pd.DataFrame(columns=["date", "rain_mm"])
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["isodate"], format="%m/%d/%Y")
    daily["rain_mm"] = daily["value"].map(lambda v: (v or {}).get("avg"))
    return daily[["date", "rain_mm"]]


def build_rain_table(
    coords: pd.DataFrame,
    start: str = "01/01/2010",
    end: str = "08/31/2026",
    radius_km: float = RADIUS_KM,
    workers: int = 5,
) -> pd.DataFrame:
    """Monthly rainfall per market: one (market_id, month, rain_mm) row.
    The monthly value sums the polygon's daily averages over the month,
    the API-route equivalent of the raster path's area-mean monthly total.
    A month whose days are all NaN stays NaN (glossary: observed)."""
    def one_market(c) -> pd.DataFrame:
        daily = fetch_daily(c.longitude, c.latitude, start, end, radius_km)
        daily["month"] = daily["date"].dt.strftime("%Y-%m")
        monthly = daily.groupby("month")["rain_mm"].agg(
            lambda s: s.sum() if s.notna().any() else float("nan")
        )
        return monthly.rename("rain_mm").reset_index().assign(market_id=int(c.market_id))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(one_market, coords.itertuples()))
    return pd.concat(frames, ignore_index=True)[["market_id", "month", "rain_mm"]]


if __name__ == "__main__":
    import json as _json

    processed = Path("data/interim")
    processed.mkdir(parents=True, exist_ok=True)
    out = processed / "rainfall.parquet"
    if out.exists():
        print("rainfall table already built:", out)
    else:
        markets = pd.read_csv("data/raw/markets.csv")
        coords = markets[markets["market_id"].isin([4626, 185, 5669, 5671, 5672])]
        table = build_rain_table(coords)
        table.to_parquet(out, index=False)
        print(_json.dumps({"rows": int(len(table)), "markets": int(table["market_id"].nunique()),
                          "months": int(table["month"].nunique()),
                          "nan_share": float(table["rain_mm"].isna().mean())}, indent=2))
