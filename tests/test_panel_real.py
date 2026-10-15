"""Phase 1 integration checks against the committed raw CSV (issue #1).

The expectations here are computed from the raw file with plain pandas in
the test, so the observed-flag and duplicate checks are anchored in
independent truth rather than in the functions under test.
"""
from pathlib import Path

import pandas as pd

from src.panel import NEW_COMMODITY, OLD_COMMODITY, MARKET_MAP, build_panel, clean_raw, select_series

RAW = Path("data/raw/food_prices.csv")


def _truth_months(raw: pd.DataFrame) -> dict[int, set[str]]:
    """Months with an actual, priced, in-scope row per canonical market id."""
    ok_within = (
        raw["priceflag"].str.contains("actual", na=False)
        & (raw["currency"] == "KES")
        & (raw["price"] > 0)
        & (raw["pricetype"] == "Wholesale")
        & raw["unit"].isin(["KG", "90 KG"])
        & raw["commodity"].isin([OLD_COMMODITY, NEW_COMMODITY])
    )
    subset = raw.loc[ok_within, ["market_id", "date", "price", "commodity"]].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    subset["market_id"] = (
        subset["market_id"].astype(int).map(lambda i: MARKET_MAP.get(i, (i,))[0])
    )
    subset = subset.assign(month=subset["date"].dt.to_period("M").astype(str))
    return subset.groupby("market_id")["month"].agg(set)


def test_panel_keys_are_unique_across_the_real_data():
    panel = build_panel(pd.read_csv(RAW))
    keys = panel[["series", "market_id", "month"]]
    assert not keys.duplicated().any()


def test_observed_flag_matches_the_raw_file_per_market():
    raw = pd.read_csv(RAW)
    panel = build_panel(raw)
    truth = _truth_months(raw)
    for market_id, g in panel.groupby("market_id"):
        observed = set(g.loc[g["observed"], "month"])
        assert observed == truth[int(market_id)], f"market {market_id} disagrees with raw"


def test_splice_overlap_months_come_from_the_successor_name():
    raw = pd.read_csv(RAW)
    panel = build_panel(raw)
    sel = select_series(clean_raw(raw))
    sel = sel.assign(date=pd.to_datetime(sel["date"]))
    sel = sel.assign(month=sel["date"].dt.to_period("M").astype(str))
    for market_id, g in sel.groupby("market_id"):
        overlap = g.groupby("month")["commodity"].apply(
            lambda s: OLD_COMMODITY in set(s) and NEW_COMMODITY in set(s)
        )
        overlap_months = set(overlap.index[overlap])
        panel_rows = (
            panel[panel["market_id"] == market_id]
            .set_index("month")["source_name"]
            .reindex(sorted(overlap_months))
        )
        assert panel_rows.notna().all(), f"overlap rows missing for {market_id}"
        assert (panel_rows == NEW_COMMODITY).all(), f"overlap not successor-won for {market_id}"


def test_panel_shape_matches_the_adr_0006_scope():
    panel = build_panel(pd.read_csv(RAW))
    assert panel["market_id"].nunique() == 5
    assert panel["month"].min() == "2006-01"
    assert panel["month"].max() == "2026-08"
