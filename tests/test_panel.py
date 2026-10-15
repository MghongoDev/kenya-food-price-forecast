"""Phase 1 tests for the panel seam: raw cleaner, series selection, spliced panel."""
import numpy as np
import pandas as pd

from src.panel import clean_raw


def _raw(**over):
    """One synthetic WFP-style row; keyword overrides change anything."""
    row = {
        "date": "2020-01-15",
        "admin1": "Nairobi",
        "admin2": "Nairobi",
        "market": "Nairobi",
        "market_id": 184,
        "latitude": -1.28,
        "longitude": 36.82,
        "category": "cereals and tubers",
        "commodity": "Maize",
        "commodity_id": 51,
        "unit": "KG",
        "priceflag": "actual",
        "pricetype": "Wholesale",
        "currency": "KES",
        "price": 40.0,
        "usdprice": 0.4,
    }
    row.update(over)
    return row


# --- slice 1: clean_raw -------------------------------------------------------

def test_clean_raw_keeps_actual_and_mixed_rows():
    raw = pd.DataFrame([
        _raw(),                               # plain actual
        _raw(market="Mombasa", market_id=191,
             priceflag="aggregate", price=39.0),  # aggregate only: out
        _raw(market="Kisumu", market_id=186,
             priceflag="actual,aggregate", price=41.0),  # mix: still actual
    ])
    out = clean_raw(raw)
    assert set(out["priceflag"]) == {"actual", "actual,aggregate"}
    assert len(out) == 2


def test_clean_raw_drops_blank_or_zero_prices():
    raw = pd.DataFrame([
        _raw(),
        _raw(market="Mombasa", market_id=191, price=np.nan),
        _raw(market="Kisumu", market_id=186, price=0.0),
    ])
    assert len(clean_raw(raw)) == 1


def test_clean_raw_keeps_only_kes():
    raw = pd.DataFrame([_raw(), _raw(market="Border", market_id=999, currency="USD", price=1.0)])
    out = clean_raw(raw)
    assert set(out["currency"]) == {"KES"}


# --- slice 2: select_series (target legs + unit normalization) ----------------
from src.panel import BAG_KG, MARKET_MAP, select_series


def test_select_series_keeps_only_target_commodities_and_wholesale_kg_units():
    raw = pd.DataFrame([
        _raw(),                                                        # old leg, KG
        _raw(commodity="Maize (white)", market_id=123),                # unrelated name
        _raw(commodity="Beans (dry)", market_id=124),                  # wrong commodity
        _raw(commodity="Maize", unit="50 KG", market_id=125),          # unsupported unit
        _raw(commodity="Maize", pricetype="Retail", market_id=126),    # wrong price type
    ])
    out = select_series(clean_raw(raw))
    assert len(out) == 1
    assert out.iloc[0]["unit_in"] == "KG"


def test_select_series_converts_90kg_bags_to_per_kg():
    raw = pd.DataFrame([
        _raw(commodity="Maize (white, dry)", unit="90 KG", market_id=4626,
             market="Kangemi (Nairobi)", price=3600.0),
    ])
    out = select_series(clean_raw(raw))
    assert len(out) == 1
    assert np.isclose(out.iloc[0]["price"], 3600.0 / BAG_KG)
    assert out.iloc[0]["unit_in"] == "90 KG"


def test_select_series_maps_old_market_ids_to_successor_markets():
    assert MARKET_MAP[184] == (4626, "Kangemi (Nairobi)")
    raw = pd.DataFrame([_raw()])  # market_id 184 = old Nairobi
    out = select_series(clean_raw(raw))
    assert out.iloc[0]["market_id"] == 4626
    assert out.iloc[0]["market"] == "Kangemi (Nairobi)"


# --- slice 3: build_panel (monthly grid, observed flag, splice) ---------------
from src.panel import build_panel


def test_panel_reindexes_complete_grid_and_flags_observed():
    raw = pd.DataFrame([
        _raw(date="2021-01-15", market_id=4626, market="Kangemi (Nairobi)", price=10.0),
        _raw(date="2021-01-20", market_id=4626, market="Kangemi (Nairobi)", price=11.0),
        _raw(date="2021-03-15", market_id=4626, market="Kangemi (Nairobi)", price=12.0),
    ])
    panel = build_panel(raw, both_eras=False)
    assert list(panel["month"]) == ["2021-01", "2021-02", "2021-03"]
    feb = panel[panel["month"] == "2021-02"].iloc[0]
    assert not feb["observed"] and np.isnan(feb["price"])
    jan = panel[panel["month"] == "2021-01"].iloc[0]
    assert jan["observed"] and jan["price"] == 11.0  # latest actual of the month wins


def test_panel_splice_successor_wins_on_overlap_months():
    raw = pd.DataFrame([
        # old leg ('Maize') covers Jan-Jun; new leg ('Maize (white, dry)') from Feb
        _raw(date="2021-01-15", market_id=4626, market="Nairobi", price=10.0),
        _raw(date="2021-02-15", market_id=4626, market="Nairobi", price=10.5),
        _raw(date="2021-03-15", market_id=4626, market="Nairobi", price=11.0),
        _raw(date="2021-04-15", market_id=4626, market="Nairobi", price=11.5),
        _raw(date="2021-02-20", market_id=4626, market="Kangemi (Nairobi)",
             commodity="Maize (white, dry)", unit="90 KG", price=1080.0),
        _raw(date="2021-03-20", market_id=4626, market="Kangemi (Nairobi)",
             commodity="Maize (white, dry)", unit="90 KG", price=1170.0),
        _raw(date="2021-04-60".replace("60", "20"), market_id=4626, market="Kangemi (Nairobi)",
             commodity="Maize (white, dry)", unit="90 KG", price=1260.0),
    ])
    panel = build_panel(raw)
    sel = panel.set_index("month")["price"]
    assert sel["2021-01"] == 10.0                 # old leg only
    assert sel["2021-02"] == 1080.0 / BAG_KG      # overlap: successor wins
    assert sel["2021-03"] == 1170.0 / BAG_KG
    assert sel["2021-04"] == 1260.0 / BAG_KG
    assert panel["market"].nunique() == 1         # one canonical market


def test_panel_has_no_duplicate_keys():
    raw = pd.DataFrame([
        _raw(date="2021-01-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-02-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-02-18", market_id=4626, market="Nairobi", price=9.0),
    ])
    panel = build_panel(raw, both_eras=False)
    keys = panel[["market_id", "month"]]
    assert not keys.duplicated().any()


# --- slice 4: coverage_table --------------------------------------------------
from src.panel import coverage_table


def test_coverage_table_counts_share_and_longest_interior_gap():
    raw = pd.DataFrame([
        # observed Jan, Mar, Apr; Feb missing inside the window
        _raw(date="2021-01-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-03-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-04-18", market_id=4626, market="Nairobi"),
    ])
    panel = build_panel(raw, both_eras=False)
    cov = coverage_table(panel, start="2021-01", end="2021-06")
    row = cov.iloc[0]
    assert row["n_observed"] == 3 and np.isclose(row["share"], 0.75)
    assert row["first_obs"] == "2021-01" and row["last_obs"] == "2021-04"
    assert row["longest_gap_months"] == 1  # Feb, interior only
    # Design note: the panel grid ends at the dataset's last observed month
    # (Apr here), so the window shrinks to what exists; months with no data
    # anywhere (May, Jun) neither count as observed nor as gaps.


def test_coverage_table_respects_window_edges():
    raw = pd.DataFrame([
        _raw(date="2020-12-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-02-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-03-15", market_id=4626, market="Nairobi"),
    ])
    panel = build_panel(raw, both_eras=False)
    cov = coverage_table(panel, start="2021-01", end="2021-03")
    row = cov.iloc[0]
    assert row["n_observed"] == 2 and np.isclose(row["share"], 2 / 3)
    assert row["first_obs"] == "2021-02"
    assert row["longest_gap_months"] == 0  # Dec obs is outside the window


# --- slice 5: both-eras market selection --------------------------------------

def test_panel_keeps_only_markets_with_both_naming_eras():
    raw = pd.DataFrame([
        # market (4626): old leg Jan, new leg Feb -> kept
        _raw(date="2021-01-15", market_id=4626, market="Nairobi"),
        _raw(date="2021-02-15", market_id=4626, market="Kangemi (Nairobi)",
             commodity="Maize (white, dry)", unit="90 KG", price=900.0),
        # new-era-only market -> dropped
        _raw(date="2021-03-15", market_id=777, market="Somewhere",
             commodity="Maize (white, dry)", unit="90 KG", price=900.0),
        # old-era-only market -> dropped
        _raw(date="2021-03-15", market_id=888, market="Elsewhere"),
    ])
    panel = build_panel(raw)
    assert set(panel["market_id"]) == {4626}
