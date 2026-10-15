"""Phase 3 rainfall tests at the sampling seam (issue #3).

The fixture is a synthetic GeoTIFF built in-test, so the expected radius
mean is a hand-computed literal, never the code's own re-computation.
"""
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from src.rainfall import chirps_url, mean_rain_in_radius


def _fixture_tif(tmp_path, values):
    """6x6 raster: 1-degree pixels, centers at (col+0.5, 5.5-row), values given."""
    data = np.array(values, dtype="float32")
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0)
    path = tmp_path / "chirps-v2.0.2022.01.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=6, height=6, count=1,
        dtype="float32", transform=transform, nodata=-9999.0,
    ) as ds:
        ds.write(data, 1)
    return path


VALUES = [
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
    [24.0, 25.0, 25.0, 27.0, 28.0, 29.0],
    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
]


def test_chirps_url_names_the_monthly_tif():
    assert chirps_url((2022, 1)).endswith("chirps-v2.0.2022.01.tif.gz")
    assert chirps_url((2026, 12)).endswith("chirps-v2.0.2026.12.tif.gz")


def test_mean_rain_in_radius_averages_only_pixels_inside_the_radius(tmp_path):
    path = _fixture_tif(tmp_path, VALUES)
    with rasterio.open(path) as ds:
        # north-up raster: the pixel covering (2.5, 2.5) is row 3, col 2 = 20.0;
        # inside radius 1.2 deg sit (1.5,2.5)=19, (3.5,2.5)=21, (2.5,3.5)=14,
        # (2.5,1.5)=25; diagonals at sqrt(2) deg are outside.
        mean = mean_rain_in_radius(ds, lon=2.5, lat=2.5, radius_km=111.0 * 1.2)
    assert np.isclose(mean, (20.0 + 19.0 + 21.0 + 14.0 + 25.0) / 5)


def test_mean_rain_in_radius_skips_nodata_pixels(tmp_path):
    with_nodata = [row[:] for row in VALUES]
    with_nodata[3][2] = -9999.0  # the center pixel becomes unusable
    path = _fixture_tif(tmp_path, with_nodata)
    with rasterio.open(path) as ds:
        mean = mean_rain_in_radius(ds, lon=2.5, lat=2.5, radius_km=111.0 * 1.2)
    assert np.isclose(mean, (19.0 + 21.0 + 14.0 + 25.0) / 4)


def test_mean_rain_in_radius_returns_nan_when_everything_is_nodata(tmp_path):
    empty = [[-9999.0] * 6 for _ in range(6)]
    path = _fixture_tif(tmp_path, empty)
    with rasterio.open(path) as ds:
        mean = mean_rain_in_radius(ds, lon=2.5, lat=2.5, radius_km=111.0 * 1.2)
    assert np.isnan(mean)


# --- slice F: the market rain table (no network in tests) -----------------------
from src.rainfall import build_rain_table


def test_build_rain_table_means_around_each_market(tmp_path):
    path = _fixture_tif(tmp_path, VALUES)
    coords = pd.DataFrame([
        {"market_id": 31, "lon": 2.5, "lat": 2.5},
        {"market_id": 64, "lon": 5.5, "lat": 5.5},   # corner: most pixels unavailable
    ])
    table = build_rain_table([path], coords, radius_km=111.0 * 1.2)
    t = table.set_index(["market_id", "month"])
    assert len(table) == 2
    assert np.isclose(t.loc[(31, "2022-01"), "rain_mm"], (20.0 + 19.0 + 21.0 + 14.0 + 25.0) / 5)
    # corner market: center (5.5,5.5) is row 0 col 5 = 5.0 (north-up); the two
    # in-radius neighbors are 4.0 (row 0 col 4) and 11.0 (row 1 col 5)
    assert np.isclose(t.loc[(64, "2022-01"), "rain_mm"], (5.0 + 4.0 + 11.0) / 3)
