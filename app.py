"""Streamlit dashboard for the Kenya food price forecast (issue #14, ADR 0004).

Reads only precomputed files from data/processed/. Nothing trains at
load time. Organised as two pages: Forecast (market/commodity selector,
fan chart, alert panel) and Backtest (metrics and coverage tables).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROCESSED = Path("data/processed")
ALERT_THRESHOLD = 0.15  # operating point for the displayed alert history
MOMENTUM_THRESHOLD = 0.10


def make_fan_chart(
    hist: pd.DataFrame,
    forecasts: pd.DataFrame,
    market_id: int,
):
    """Observed history, point forecast, and conformal interval for one market."""
    h = hist[hist["market_id"] == market_id].sort_values("month")
    f = forecasts[forecasts["market_id"] == market_id]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pd.to_datetime(h["month"]), h["price"], "k-", lw=1.5, label="observed")
    for horizon, color in ((1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")):
        g = f[f["horizon"] == horizon].sort_values("target_month")
        if g.empty:
            continue
        x = pd.to_datetime(g["target_month"])
        ax.fill_between(x, g["lower"], g["upper"], color=color, alpha=0.25,
                        label=f"h={horizon} interval")
        ax.plot(x, g["price_pred"], "o", color=color, ms=3, label=f"h={horizon} pred")
    ax.set_title(f"Market {market_id} — conformal intervals (nominal 80%)")
    ax.set_ylabel("KES / kg")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    return fig


@st.cache_data
def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name)


def _pooled(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["market"] == "ALL") & (df["period"] == "all")]


def main() -> None:
    st.set_page_config(page_title="Kenya Food Price Forecast", layout="wide")
    st.title("Kenya Food Price Forecast")
    st.caption("Wholesale white maize, 1–3 months ahead. Precomputed forecasts; "
               "nothing trains at load time.")

    forecasts = _load("dashboard_forecasts.csv")
    panel = pd.read_parquet(PROCESSED / "panel.parquet")

    # selectors
    commodities = sorted(panel["series"].unique())
    commodity = st.selectbox("Commodity", commodities)
    markets = forecasts[["market_id", "market"]].drop_duplicates().sort_values("market")
    selected = st.selectbox("Market", markets["market"])
    market_id = markets[markets["market"] == selected]["market_id"].iloc[0]

    tab_forecast, tab_backtest = st.tabs(["Forecast", "Backtest"])

    with tab_forecast:
        st.pyplot(make_fan_chart(panel, forecasts, market_id))

        # alert status panel
        alerts = _load("dashboard_alert_history.csv")
        alerts_m = alerts[alerts["market_id"] == market_id].copy()
        latest = alerts_m.sort_values("origin").iloc[-1] if len(alerts_m) else None
        st.subheader("Alert status")
        if latest is None:
            st.info("No alert history for this market.")
        else:
            tp = (alerts_m["outcome"] == "TP").sum()
            fp = (alerts_m["outcome"] == "FP").sum()
            precision = tp / (tp + fp) if tp + fp else float("nan")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current alert", "FIRED" if latest["alert_fired"] else "clear")
            c2.metric("Horizon", f"{latest['horizon']:.0f}")
            c3.metric("Rule", f"upper ≥ {ALERT_THRESHOLD:.0%} rise")
            c4.metric("Historical precision", f"{precision:.0%}")
            st.caption(
                f"Last origin: {latest['origin']}, outcome: {latest['outcome']} "
                f"(TP = alert fired and spike followed, FP = false alarm, "
                f"FN = missed spike, TN = no alert no spike)"
            )

    with tab_backtest:
        st.subheader("Backtest results")
        metrics = _load("model_metrics.csv")
        coverage = _load("conformal_coverage.csv")
        st.write("Pooled metrics (all periods)")
        st.dataframe(_pooled(metrics)[["forecaster", "horizon", "n", "mae_kes", "wape", "mase"]])
        st.write("Pooled coverage (nominal 80%)")
        st.dataframe(_pooled(coverage)[["horizon", "n", "coverage", "width_kes"]])

    with st.expander("How to read this"):
        st.markdown(
            "- The black line is the observed wholesale price.\n"
            "- Coloured dots are the point forecast; the shaded band is the conformal "
            "interval at that horizon.\n"
            f"- The alert fires when the interval upper bound implies a rise above "
            f"{ALERT_THRESHOLD:.0%} from the origin price.\n"
            f"- The naive momentum comparison: fire when the price already rose "
            f"more than {MOMENTUM_THRESHOLD:.0%} last month."
        )
    with st.expander("Limitations"):
        st.markdown(
            "- Conformal coverage assumes exchangeability, which time series violate; "
            "coverage is measured, not assumed. It collapsed in the 2022 drought.\n"
            "- The alert threshold was set from the base rate; costs are assumptions, "
            "not measurements.\n"
            "- The model does not dominate naive momentum on both axes — it trades "
            "precision for recall. The right operating point depends on the cost ratio.\n"
            "- Prices before 2020 are spliced from two series names; the splice is "
            "documented in ADR 0006/0007."
        )


if __name__ == "__main__":
    main()
