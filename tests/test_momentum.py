"""Phase 5 naive-momentum tests (issue #11).

The seam is the naive rule on hand-built frames: fire when the price
already rose more than 10% last month (price[t-1] >= 1.1 * price[t-2]).
All literals are hand-computed.
"""
import pandas as pd

from src.alerts import naive_momentum_outcomes, naive_momentum_pr


def _panel(rows):
    return pd.DataFrame(rows, columns=[
        "market_id", "market", "month", "price", "observed",
    ])


def _spikes(rows):
    return pd.DataFrame(rows, columns=[
        "market_id", "origin", "horizon", "price_origin", "spike",
    ])


# panel: 2021-11=100, 2021-12=110, 2022-01=100, 2022-02=120, 2022-03=140,
#   2022-04=100, 2022-05=160
# naive rule fires at origin t when price[t-1] >= 1.1 * price[t-2]:
#   origin 2022-01: Dec 110 >= 1.1*Nov 100 -> fires
#   origin 2022-02: Jan 100 <  1.1*Dec 110 -> no
#   origin 2022-03: Feb 120 >= 1.1*Jan 100 -> fires
#   origin 2022-04: Mar 140 >= 1.1*Feb 120 -> fires
PANEL = _panel([
    [1, "A", "2021-11", 100.0, True],
    [1, "A", "2021-12", 110.0, True],
    [1, "A", "2022-01", 100.0, True],
    [1, "A", "2022-02", 120.0, True],
    [1, "A", "2022-03", 140.0, True],
    [1, "A", "2022-04", 100.0, True],
    [1, "A", "2022-05", 160.0, True],
])

# spike events (15% threshold):
#   origin 2022-01 h=1: 110->120 spike=False (120 < 126.5); h=2: 110->140 spike=True
#   origin 2022-02 h=1: 100->140 spike=True (140 >= 115)
#   origin 2022-03 h=1: 140->100 spike=False
#   origin 2022-04 h=1: 100->160 spike=True (160 >= 115)
SPIKES = _spikes([
    [1, "2022-01", 1, 110.0, False],
    [1, "2022-01", 2, 110.0, True],
    [1, "2022-02", 1, 100.0, True],
    [1, "2022-03", 1, 140.0, False],
    [1, "2022-04", 1, 100.0, True],
])


def test_naive_rule_fires_at_the_momentum_boundary():
    # origin 2022-01: last month (Dec) 110 vs prior (Nov) 100 -> exactly +10% -> fires
    out = naive_momentum_outcomes(SPIKES, PANEL, momentum_threshold=0.10)
    rows = out.set_index(["origin", "horizon"])
    assert rows.loc[("2022-01", 1), "naive_fired"] == True  # noqa: E712


def test_naive_rule_does_not_fire_below_the_boundary():
    # origin 2022-02: last month (Jan) 100 vs prior (Dec) 110 -> no rise -> does not fire
    out = naive_momentum_outcomes(SPIKES, PANEL, momentum_threshold=0.10)
    rows = out.set_index(["origin", "horizon"])
    assert rows.loc[("2022-02", 1), "naive_fired"] == False  # noqa: E712


def test_naive_outcomes_classified_against_spike_targets():
    # origin 2022-01: fired; h=1 no spike -> FP, h=2 spike -> TP
    # origin 2022-02: not fired, spike at h=1 -> FN
    # origin 2022-03: fired, no spike at h=1 -> FP
    # origin 2022-04: fired, spike at h=1 -> TP
    out = naive_momentum_outcomes(SPIKES, PANEL, momentum_threshold=0.10)
    rows = out.set_index(["origin", "horizon"])
    assert rows.loc[("2022-01", 1), "naive_outcome"] == "FP"
    assert rows.loc[("2022-01", 2), "naive_outcome"] == "TP"
    assert rows.loc[("2022-02", 1), "naive_outcome"] == "FN"
    assert rows.loc[("2022-03", 1), "naive_outcome"] == "FP"
    assert rows.loc[("2022-04", 1), "naive_outcome"] == "TP"


def test_naive_pr_computes_precision_and_recall():
    out = naive_momentum_outcomes(SPIKES, PANEL, momentum_threshold=0.10)
    pr = naive_momentum_pr(out)
    # TP=2 (2022-01 h=2, 2022-04), FP=2 (2022-01 h=1, 2022-03), FN=1 (2022-02)
    assert pr["tp"] == 2 and pr["fp"] == 2 and pr["fn"] == 1
    assert pr["n"] == 5
    assert abs(pr["precision"] - 2 / 4) < 1e-9
    assert abs(pr["recall"] - 2 / 3) < 1e-9


def test_naive_rule_needs_observed_last_and_prior_month():
    # if last month is unobserved, the naive rule cannot fire
    panel = _panel([
        [1, "A", "2021-11", 100.0, True],
        [1, "A", "2021-12", 110.0, False],   # last month unobserved
        [1, "A", "2022-01", 100.0, True],
        [1, "A", "2022-02", 120.0, True],
    ])
    spikes = _spikes([[1, "2022-01", 1, 100.0, True]])
    out = naive_momentum_outcomes(spikes, panel, momentum_threshold=0.10)
    assert out.iloc[0]["naive_fired"] == False  # noqa: E712