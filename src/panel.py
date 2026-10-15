"""Phase 1: build the spliced wholesale white maize panel.

Public seam for tests: clean_raw, select_series, build_panel, coverage_table.
Design follows ADR 0006 (target series, splice rule) and ADR 0007
(unit normalization 90 KG -> KG, old-to-new market-id map).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date", "market", "market_id", "commodity", "unit",
    "pricetype", "currency", "price", "priceflag",
]

# --- target-series constants (ADR 0006/0007) --------------------------------

OLD_COMMODITY = "Maize"                # pre-rename per-kg wholesale series
NEW_COMMODITY = "Maize (white, dry)"   # successor name, priced in 90 KG bags
BAG_KG = 90.0                          # a '90 KG' unit row reports the bag price

# Old wholesale markets were renamed (and re-id'd) in the successor era.
# The faucet-market name maps to the successor market that kept it.
MARKET_MAP = {
    184: (4626, "Kangemi (Nairobi)"),
    186: (5671, "Kibuye (Kisumu)"),
    1850: (5669, "Wakulima (Nakuru)"),
    191: (5672, "Kongowea (Mombasa)"),
    # 185 (Eldoret town) kept its id and name; no remap needed.
}

PRICE_TYPE = "Wholesale"
SERIES_NAME = "white_maize_wholesale"

SUPPORTED_UNITS = {"KG": 1.0, "90 KG": BAG_KG}   # unit -> kg divisor


def select_series(
    clean: pd.DataFrame,
    old_commodity: str = OLD_COMMODITY,
    new_commodity: str = NEW_COMMODITY,
    price_type: str = PRICE_TYPE,
    market_map: dict | None = None,
) -> pd.DataFrame:
    """Keep only the two target legs, normalize units to KES per kg, and
    remap old market ids to their successor markets. Rows carry unit_in so the
    splice inspection can still tell which naming era a month came from."""
    market_map = MARKET_MAP if market_map is None else market_map
    df = clean.loc[
        clean["commodity"].isin([old_commodity, new_commodity])
        & (clean["pricetype"] == price_type)
    ].copy()
    divisors = df["unit"].map(SUPPORTED_UNITS)
    df = df.assign(unit_in=df["unit"], price=df["price"] / divisors)
    df = df[df["price"].notna()]
    remapped = df["market_id"].map(market_map)
    new_ids = remapped.str[0]
    new_names = remapped.str[1]
    df["market_id"] = new_ids.fillna(df["market_id"])
    df["market"] = new_names.fillna(df["market"])
    return df.reset_index(drop=True)


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep actual, priced, KES rows. An aggregate-only row is not an observed
    price; 'actual,aggregate' still reports an actual price and stays."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    keep = (
        df["priceflag"].str.contains("actual", na=False)
        & (df["currency"] == "KES")
        & df["price"].notna()
        & (df["price"] > 0)
    )
    df = df.loc[keep, REQUIRED_COLUMNS].reset_index(drop=True)
    return df


def build_panel(
    raw: pd.DataFrame,
    market_map: dict | None = None,
    old_commodity: str = OLD_COMMODITY,
    new_commodity: str = NEW_COMMODITY,
    price_type: str = PRICE_TYPE,
    series_name: str = SERIES_NAME,
    both_eras: bool = True,
) -> pd.DataFrame:
    """Build the target panel from raw rows: monthly, spliced, complete grid.

    Rules (ADR 0006/0007):
    - one series per market; the successor commodity name wins any overlapping
      market-month, so the naming switch turns into one continuous series;
    - within a month, the latest market-day observation wins;
    - every market gets the full global monthly range, with observed False and
      a missing price where no actual row exists. Scored months stay missing;
    - with both_eras (default), only markets carrying both naming eras are
      kept: a half-era fragment is not a spliceable series. Set it False to
      build a single-leg panel, which is what the unit-level seam tests do.
    """
    sel = select_series(
        clean_raw(raw),
        old_commodity=old_commodity,
        new_commodity=new_commodity,
        price_type=price_type,
        market_map=market_map,
    )
    empty = pd.DataFrame(
        columns=["series", "market_id", "market", "month", "price", "observed", "source_name"]
    )
    if sel.empty:
        return empty

    sel = sel.assign(month=pd.PeriodIndex(sel["date"], freq="M"))
    rank = sel["commodity"].map({new_commodity: 0, old_commodity: 1})
    sel = sel.assign(_rank=rank)

    # canonical market names: successor name wins if a market carries both eras
    names = (
        sel.sort_values("_rank", kind="mergesort")
        .drop_duplicates("market_id")
        .set_index("market_id")["market"]
    )
    # latest market-day observation of each month per leg
    monthly = sel.sort_values("date", kind="mergesort").drop_duplicates(
        ["market_id", "commodity", "month"], keep="last"
    )
    # splice: one row per market-month; the lower rank (successor) survives
    combined = (
        monthly.sort_values(["_rank"], kind="mergesort")
        .drop_duplicates(["market_id", "month"], keep="first")
        .sort_values(["market_id", "month"])
    )
    # keep only markets that hold both naming eras (ADR 0006): one market
    # means one spliceable series, not a half-era fragment
    if both_eras:
        old_have = set(combined.loc[combined["commodity"] == old_commodity, "market_id"])
        new_have = set(combined.loc[combined["commodity"] == new_commodity, "market_id"])
        combined = combined[combined["market_id"].isin(old_have & new_have)]
        if combined.empty:
            return empty

    market_ids = sorted(combined["market_id"].unique())
    months = pd.period_range(combined["month"].min(), combined["month"].max(), freq="M")
    grid = pd.MultiIndex.from_product([market_ids, months], names=["market_id", "month"])
    panel = combined.set_index(["market_id", "month"]).reindex(grid).reset_index()

    panel["market"] = panel["market_id"].map(names)
    panel["observed"] = panel["price"].notna()
    panel["series"] = series_name
    panel["source_name"] = panel["commodity"]
    panel["month"] = panel["month"].astype(str)
    return panel[
        ["series", "market_id", "market", "month", "price", "observed", "source_name"]
    ]


def coverage_table(
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Data coverage per market over a closed month window.

    Longest gap counts unobserved months strictly between the first and last
    observed month inside the window. Months before the market's first or
    after its last observation do not count as gaps.
    """
    grid = panel.copy()
    if start is not None:
        grid = grid[grid["month"] >= start]
    if end is not None:
        grid = grid[grid["month"] <= end]

    rows = []
    window = sorted(grid["month"].unique())
    window_ord = {}
    for m in window:
        window_ord[m] = int(m[:4]) * 12 + int(m[5:])
    for market_id, g in grid.groupby("market_id"):
        obs = g.loc[g["observed"], "month"]
        if obs.empty:
            continue
        first, last = obs.min(), obs.max()
        obs_ords = sorted(window_ord[m] for m in obs)
        gaps = np.diff(obs_ords) - 1 if len(obs_ords) > 1 else np.array([])
        rows.append({
            "market_id": int(market_id),
            "market": g["market"].iloc[0],
            "first_obs": first,
            "last_obs": last,
            "n_observed": int(len(obs)),
            "share": len(obs) / len(window),
            "longest_gap_months": int(gaps.max()) if len(gaps) else 0,
        })
    return pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)


def _write_artifacts() -> dict:
    """Run the Phase 1 pipeline end to end and write the artifacts.

    Writes the interim normalized rows, the processed panel and coverage
    table, and a splice-inspection plot covering 2019-2023 so the 2020/21
    switch is visible in the middle, not at the edge.
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = pd.read_csv(Path("data/raw/food_prices.csv"))
    panel = build_panel(raw)
    sel = select_series(clean_raw(raw))
    cov = coverage_table(panel, start="2016-01")

    interim = Path("data/interim")
    processed = Path("data/processed")
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(interim / "selected_rows.parquet", index=False)
    panel.to_parquet(processed / "panel.parquet", index=False)
    cov.to_csv(processed / "coverage.csv", index=False)

    # splice inspection: one line per market, colored by naming era
    window = panel[(panel["month"] >= "2019-01") & (panel["month"] <= "2023-12")]
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (market_id, g) in enumerate(window.groupby("market_id")):
        g = g.sort_values("month")
        is_new_name = g["source_name"].eq(NEW_COMMODITY)
        color = palette[i % len(palette)]
        label = g["market"].iloc[0]
        ax.plot(g.loc[~is_new_name, "month"], g.loc[~is_new_name, "price"],
                color=color, label=f"{label} (old name)")
        ax.plot(g.loc[is_new_name, "month"], g.loc[is_new_name, "price"],
                color=color, linestyle="--", label=f"{label} (new name)")
    ax.set_title("Splice inspection: wholesale white maize KES/kg, 2019-2023")
    ax.set_ylabel("KES per kg")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path("notebooks/splice_inspection.png"))
    plt.close(fig)

    splice_window = panel[(panel["month"] >= "2020-07") & (panel["month"] <= "2021-06")]
    return {
        "panel_rows": int(len(panel)),
        "markets": int(panel["market_id"].nunique()),
        "months": int(panel["month"].nunique()),
        "first_month": panel["month"].min(),
        "last_month": panel["month"].max(),
        "observed_share": float(panel["observed"].mean()),
        "interim": str(interim / "selected_rows.parquet"),
        "panel": str(processed / "panel.parquet"),
        "coverage": str(processed / "coverage.csv"),
        "splice_window_rows": int(len(splice_window)),
    }


if __name__ == "__main__":
    import json

    raw_csv = pd.read_csv("data/raw/food_prices.csv")
    summary = _write_artifacts()
    print(json.dumps(summary, indent=2))

    print("\n== coverage table (2016-01 onward)")
    print(pd.read_csv("data/processed/coverage.csv").to_string(index=False))

    print("\n== splice window 2020-07 .. 2021-06 (observed rows only)")
    spliced = build_panel(raw_csv)
    win = spliced[
        (spliced["month"] >= "2020-07")
        & (spliced["month"] <= "2021-06")
        & spliced["observed"]
    ]
    print(win["month"].min(), "..", win["month"].max())
    print(win.to_string(index=False))
