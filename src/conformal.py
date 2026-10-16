"""Phase 4 conformal intervals (issue #4, ADR 0003).

Split conformal per horizon, hand-rolled: log-scale residuals from a
calibration window of recent origins (2021-07..2021-12 - choose_hyperparams'
default validation block, fixed before any test-window inspection), the
ceil((n+1)(1-alpha))/n quantile of their absolute values per horizon,
added and subtracted from the point forecast on the log scale. Time
series violate the exchangeability assumption behind conformal methods,
so coverage is measured on the untouched test window (ADR 0006), not
assumed; the approximation and its limits are part of the report.

Per-market volatility scaling (ADR 0003's option) is not applied: with
five markets the per-market residual counts are too thin to calibrate
against. Named here so the choice is on the record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import DROUGHT_YEARS, period_of

CAL_START, CAL_END = "2021-07", "2021-12"   # calibration: recent, pre-test
TEST_START, TEST_END = "2022-01", "2026-05"  # test window (ADR 0006)
ALPHA = 0.2                                # nominal 80%
FAN_MARKETS = (4626, 5671, 185)           # Nairobi, Kisumu, Eldoret


def conformal_halfwidth(cal_resid, alpha: float = ALPHA) -> float:
    """The absolute-error quantile: ceil((n+1)(1-alpha))/n, capped at 1."""
    n = len(cal_resid)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(np.abs(cal_resid), min(q, 1.0)))


def fit_conformal(scored: pd.DataFrame, alpha: float = ALPHA) -> dict[int, float]:
    """Calibrate one halfwidth per horizon on scored calibration rows.
    The log residual is log(price_true) - log(price_pred)."""
    cal = scored.assign(
        _resid=np.log(scored["price_true"]) - np.log(scored["price_pred"])
    )
    return {
        int(h): conformal_halfwidth(g["_resid"].to_numpy(), alpha)
        for h, g in cal.groupby("horizon")
    }


def _with_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive covered/width_kes from the bounds when the caller has not."""
    if "covered" not in df.columns:
        df = df.assign(covered=(df["price_true"] >= df["lower"]) & (df["price_true"] <= df["upper"]))
    if "width_kes" not in df.columns:
        df = df.assign(width_kes=df["upper"] - df["lower"])
    return df


def apply_conformal(scored: pd.DataFrame, halfwidths: dict[int, float]) -> pd.DataFrame:
    """Attach log-scale intervals: [pred * e^-hw, pred * e^+hw] per horizon."""
    out = scored.copy()
    hw = out["horizon"].map(halfwidths)
    out["lower"] = out["price_pred"] * np.exp(-hw)
    out["upper"] = out["price_pred"] * np.exp(hw)
    return _with_coverage_flags(out)


def coverage_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Measured coverage and average width by period x market x horizon,
    plus a pooled 'ALL' row per period x horizon - the same shape as
    metric_table, so the two sit side by side in the write-up. The
    guide's 'market group' is read as individual market plus the pooled
    row: with five markets there is no coarser grouping defined."""
    df = scored.assign(_period=scored["target_month"].map(lambda m: period_of(m, DROUGHT_YEARS)))
    df = _with_coverage_flags(df)
    rows = []
    for period in ("all", "normal", "drought"):
        sub = df if period == "all" else df[df["_period"] == period]
        if sub.empty:
            continue
        for (market_id, market, horizon), g in (
            list(sub.groupby(["market_id", "market", "horizon"]))
            + [(("ALL", "ALL", int(h)), g) for h, g in sub.groupby("horizon")]
        ):
            rows.append({
                "period": period,
                "market_id": market_id,
                "market": market,
                "horizon": horizon,
                "n": int(len(g)),
                "coverage": float(g["covered"].mean()),
                "width_kes": float(g["width_kes"].mean()),
                "nominal": 1 - ALPHA,
            })
    return pd.DataFrame(rows).sort_values(
        ["period", "market_id", "horizon"], kind="mergesort"
    ).reset_index(drop=True)


def fan_charts(
    panel: pd.DataFrame,
    covered: pd.DataFrame,
    out_dir: Path,
    markets: tuple[int, ...] = FAN_MARKETS,
) -> None:
    """One PNG per market over the test window: observed price, point
    forecast, and the conformal band per horizon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(markets), 1, figsize=(11, 3.2 * len(markets)), sharex=True)
    for ax, market_id in zip(np.atleast_1d(axes), markets):
        hist = panel[panel["market_id"] == market_id].sort_values("month")
        ax.plot(pd.to_datetime(hist["month"]), hist["price"], "k-", lw=1.2, label="observed")
        sub = covered[covered["market_id"] == market_id]
        for h, color in ((1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")):
            g = sub[sub["horizon"] == h].sort_values("target_month")
            if g.empty:
                continue
            x = pd.to_datetime(g["target_month"])
            ax.fill_between(x, g["lower"], g["upper"], color=color, alpha=0.25,
                            label=f"h={h} interval")
            ax.plot(x, g["price_pred"], "o", color=color, ms=3, label=f"h={h} pred")
        name = sub["market"].iloc[0] if len(sub) else str(market_id)
        ax.set_title(name)
        ax.legend(fontsize=7, ncol=3)
        ax.set_ylabel("KES / kg")
    fig.suptitle("lgbm_rain conformal intervals, test window 2022-01..2026-05 (nominal 80%)")
    fig.tight_layout()
    fig.savefig(out_dir / "fan_charts.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    import json

    from src.backtest import make_origins, rolling_backtest
    from src.features import add_rain_features
    from src.model import choose_hyperparams, make_fit_predict

    processed = Path("data/processed")
    panel = pd.read_parquet(processed / "panel.parquet")

    rain_features = None
    rain_path = Path("data/interim/rainfall.parquet")
    if rain_path.exists():
        rain_features = add_rain_features(pd.read_parquet(rain_path))
        print(f"rain features: {len(rain_features)} rows")

    # the calibration window is the tuning validation block: recent origins,
    # fixed before any test-window inspection (exchangeability is an
    # approximation here, and the report says so - ADR 0003)
    hp = choose_hyperparams(panel, rain_features, use_rain=True)
    print(f"hyperparams: {json.dumps(hp)}", flush=True)
    fp = make_fit_predict(rain_features, use_rain=True, params_by_horizon=hp)

    cal_origins = make_origins(panel, CAL_START, CAL_END)
    cal = rolling_backtest(panel, fp, cal_origins)
    print(f"calibration rows: {len(cal)} over {len(cal_origins)} origins", flush=True)
    halfwidths = fit_conformal(cal)
    print("halfwidths by horizon:", {h: round(v, 4) for h, v in halfwidths.items()})

    test_origins = make_origins(panel, TEST_START, TEST_END)
    test = rolling_backtest(panel, fp, test_origins)
    covered = apply_conformal(test, halfwidths)
    print(f"test rows: {len(covered)} over {len(test_origins)} origins", flush=True)

    table = coverage_table(covered)
    table.to_csv(processed / "conformal_coverage.csv", index=False)
    pooled = table[table["market"] == "ALL"]
    print("\n== pooled coverage (nominal 80%)")
    print(pooled.to_string(index=False))
    print("\nreport notes: per-market volatility scaling NOT applied (ADR 0003's"
          " option; five markets is too few to calibrate per market);"
          " exchangeability is an approximation - coverage is measured,"
          " not assumed.")

    fan_charts(panel, covered, processed)
    print("\nfan chart: data/processed/fan_charts.png")
