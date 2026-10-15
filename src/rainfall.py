"""Phase 3 rainfall (issue #3, ADR 0002).

CHIRPS = Climate Hazards Group InfraRed Precipitation with Station data.
Product: monthly Africa GeoTIFFs at https://data.chc.ucsb.edu/products/
CHIRPS-2.0/africa_monthly/tifs/ (public domain; cited in the README).
Sampling: the mean of valid pixels whose centers sit within RADIUS_KM of a
market's coordinates. Rain used by features is rain observed at or before
the forecast origin (glossary: observed); never a rainfall forecast.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CHIRPS_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/"
RADIUS_KM = 25.0
KM_PER_DEGREE = 111.0


def chirps_url(month: tuple[int, int], base: str = CHIRPS_BASE) -> str:
    """The monthly GeoTIFF, gzipped: chirps-v2.0.YYYY.MM.tif.gz."""
    year, m = month
    return f"{base}/chirps-v2.0.{year:04d}.{m:02d}.tif.gz"


def mean_rain_in_radius(ds, lon: float, lat: float, radius_km: float = RADIUS_KM) -> float:
    """Mean rainfall over valid pixels inside radius_km of (lon, lat).

    Works on an open rasterio dataset; nan when no valid pixel is inside
    the radius (Co-ordinates known in the panel may sit off the raster).
    """
    import rasterio.windows as windows

    pad_deg = radius_km / KM_PER_DEGREE
    win0 = windows.from_bounds(lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg, ds.transform)
    # window = the pad bbox clamped to the raster, opened to Integer bounds so
    # every pixel ring whose center can sit inside the radius stays readable;
    # the radius mask below prunes the extras
    col_off, row_off = max(0.0, win0.col_off), max(0.0, win0.row_off)
    col_end = min(float(ds.width), win0.col_off + win0.width)
    row_end = min(float(ds.height), win0.row_off + win0.height)
    if col_end <= col_off or row_end <= row_off:
        return float("nan")
    win = windows.Window(math.floor(col_off), math.floor(row_off),
                         math.ceil(col_end) - math.floor(col_off),
                         math.ceil(row_end) - math.floor(row_off))
    data = ds.read(1, window=win, masked=True).astype("float64")
    t = ds.transform
    rows, cols = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    xs = t.c + (win.col_off + cols + 0.5) * t.a
    ys = t.f + (win.row_off + rows + 0.5) * t.e
    cos_deg = math.cos(math.radians(lat))
    dx_km = (xs - lon) * KM_PER_DEGREE * max(cos_deg, 1e-6)
    dy_km = (ys - lat) * KM_PER_DEGREE
    inside = (dx_km**2 + dy_km**2) <= radius_km**2
    mask = np.asarray(~data.mask) if np.ma.is_masked(data) else np.ones(data.shape, bool)
    usable = inside & mask
    if not usable.any():
        return float("nan")
    return float(data[usable].mean())


def download_month(month: tuple[int, int], out_dir: str = "data/raw/chirps") -> "Path":
    """Fetch one monthly .tif.gz into out_dir; skips an existing file.
    Network IO lives here, at the edges, so tests stay offline."""
    import gzip
    import shutil
    from pathlib import Path

    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"chirps-v2.0.{month[0]:04d}.{month[1]:02d}.tif"
    if target.exists():
        return target
    gz = out_dir / (target.name + ".gz")
    if not gz.exists():
        r = requests.get(chirps_url(month), timeout=120)
        r.raise_for_status()
        gz.write_bytes(r.content)
    with gzip.open(gz, "rb") as f_in, open(target, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz.unlink()
    return target


def build_rain_table(tif_paths, coords: pd.DataFrame, radius_km: float = RADIUS_KM) -> pd.DataFrame:
    """Sample every GeoTIFF at every market: (market_id, month, rain_mm).
    The month comes from the file name, so the table's keys need no clock."""
    import rasterio
    from pathlib import Path

    rows = []
    for path in tif_paths:
        parts = Path(path).stem.split(".")
        month = f"{parts[-2]}-{parts[-1]}"
        with rasterio.open(path) as ds:
            for _, c in coords.iterrows():
                rows.append(
                    {
                        "market_id": int(c["market_id"]),
                        "month": month,
                        "rain_mm": mean_rain_in_radius(ds, float(c["lon"]), float(c["lat"]), radius_km),
                    }
                )
    return pd.DataFrame(rows, columns=["market_id", "month", "rain_mm"])


if __name__ == "__main__":
    import calendar
    import json
    from pathlib import Path

    processed = Path("data/interim")
    processed.mkdir(parents=True, exist_ok=True)
    out = processed / "rainfall.parquet"
    if out.exists():
        print("rainfall table already built:", out)
    else:
        months = [(y, m) for y in range(2010, 2027) for m in range(1, 13)]
        months = [m for m in months if m <= (2026, 8)]
        files = []
        for m in months:
            files.append(download_month(m))
            if len(files) % 36 == 0:
                print(f"downloaded/named through {files[-1].name}", flush=True)
        markets = pd.read_csv("data/raw/markets.csv")
        coords = markets[markets["market_id"].isin([4626, 185, 5669, 5671, 5672])]
        table = build_rain_table(files, coords)
        table.to_parquet(out, index=False)
        print(json.dumps({"rows": int(len(table)), "markets": int(table["market_id"].nunique()),
                          "months": int(table["month"].nunique()),
                          "nan_share": float(table["rain_mm"].isna().mean())}, indent=2))
