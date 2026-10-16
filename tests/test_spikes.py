"""Phase 5 spike-definition tests (issue #9).

The seam is the spike flag on a synthetic panel: hand-computed literals,
never the code's own re-computation. The panel fixture mirrors the real
one's columns (market_id, market, month, price, observed).
"""
import pandas as pd

from src.spikes import THRESHOLD, base_rate, spike_events


def _panel(rows):
    return pd.DataFrame(rows, columns=[
        "series", "market_id", "market", "month", "price", "observed",
    ])


def _flat_market(mid, mkt, prices, start="2022-01"):
    """One market, one price per month from start; observed throughout."""
    months = pd.period_range(start, periods=len(prices), freq="M").strftime("%Y-%m")
    return _panel([
        ["s", mid, mkt, m, p, True] for m, p in zip(months, prices)
    ])


def test_default_threshold_is_the_recorded_decision():
    assert THRESHOLD == 0.15


def test_spike_flag_at_the_threshold_boundary():
    # 100 -> 115 at h=1 is exactly +15%: a spike at the default threshold
    panel = _flat_market(1, "A", [100.0, 115.0, 100.0, 100.0])
    ev = spike_events(panel)
    rows = ev[ev["horizon"] == 1].set_index("origin")
    assert rows.loc["2022-01", "spike"] == True  # noqa: E712
    assert rows.loc["2022-02", "spike"] == False  # noqa: E712


def test_just_below_the_threshold_is_not_a_spike():
    panel = _flat_market(1, "A", [100.0, 114.9, 100.0, 100.0])
    ev = spike_events(panel)
    h1 = ev[ev["horizon"] == 1].set_index("origin")
    assert h1.loc["2022-01", "spike"] == False  # noqa: E712


def test_unobserved_origin_or_target_is_not_scored():
    panel = _panel([
        ["s", 1, "A", "2022-01", 100.0, True],
        ["s", 1, "A", "2022-02", 200.0, False],   # target unobserved
        ["s", 1, "A", "2022-03", 100.0, True],
        ["s", 1, "A", "2022-04", 100.0, True],
    ])
    ev = spike_events(panel)
    # 2022-01 -> 2022-02 skipped (target unobserved);
    # only the 2022-03 -> 2022-04 pair has both ends observed at h=1
    h1 = ev[ev["horizon"] == 1]
    assert len(h1) == 1
    assert h1.iloc[0]["origin"] == "2022-03"


def test_horizon_two_compares_origin_to_two_months_ahead():
    panel = _flat_market(1, "A", [100.0, 100.0, 120.0, 100.0])
    ev = spike_events(panel)
    rows = ev[ev["horizon"] == 2].set_index("origin")
    assert rows.loc["2022-01", "spike"] == True  # noqa: E712 - 100 -> 120
    assert rows.loc["2022-02", "spike"] == False  # noqa: E712 - 100 -> 100


def test_base_rate_is_the_spike_share_over_scored_rows():
    # h=1: three scored pairs, one spike (100 -> 120); h=2: two scored
    # pairs, no spikes (100 -> 100, 120 -> 100)
    panel = _flat_market(1, "A", [100.0, 120.0, 100.0, 100.0])
    ev = spike_events(panel)
    rate = base_rate(ev)
    h1 = rate[rate["horizon"] == 1].iloc[0]
    h2 = rate[rate["horizon"] == 2].iloc[0]
    assert h1["n"] == 3 and h1["spikes"] == 1
    assert h2["n"] == 2 and h2["spikes"] == 0


def test_base_rate_pools_markets_and_reports_per_market():
    panel = pd.concat([
        _flat_market(1, "A", [100.0, 120.0, 100.0, 100.0]),
        _flat_market(2, "B", [100.0, 100.0, 100.0, 100.0]),
    ], ignore_index=True)
    ev = spike_events(panel)
    rate = base_rate(ev)
    h1 = rate[rate["horizon"] == 1]
    per_market = h1[h1["market"] != "ALL"].set_index("market")
    pooled = h1[h1["market"] == "ALL"].iloc[0]
    assert per_market.loc["A", "spikes"] == 1 and per_market.loc["A", "n"] == 3
    assert per_market.loc["B", "spikes"] == 0 and per_market.loc["B", "n"] == 3
    assert pooled["spikes"] == 1 and pooled["n"] == 6
