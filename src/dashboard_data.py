"""Phase 5 dashboard artifacts (issue #13).

Small committed files the Streamlit app loads so nothing trains at
load time: point forecasts with conformal intervals, alert history
with outcomes and lead time. The make target regenerates them from
the pipeline (tune -> calibrate -> score -> write).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FORECAST_COLUMNS = [
    "market_id", "market", "origin", "horizon", "target_month",
    "price_true", "price_pred", "lower", "upper",
]
ALERT_COLUMNS = [
    "market_id", "origin", "horizon", "price_origin", "spike",
    "alert_fired", "outcome", "lead_time",
]


def write_forecasts(covered: pd.DataFrame, out_path: Path) -> None:
    """Write the forecasts + intervals file the fan chart loads."""
    covered[FORECAST_COLUMNS].to_csv(out_path, index=False)


def write_alert_history(outcomes: pd.DataFrame, out_path: Path) -> None:
    """Write the alert history with lead time (horizon for TPs)."""
    df = outcomes.copy()
    df["lead_time"] = df["horizon"].where(df["outcome"] == "TP")
    df[ALERT_COLUMNS].to_csv(out_path, index=False)


if __name__ == "__main__":
    import json

    from src.alerts import ALERT_THRESHOLDS, classify_alerts
    from src.backtest import make_origins, rolling_backtest
    from src.conformal import (
        CAL_END,
        CAL_START,
        TEST_END,
        TEST_START,
        apply_conformal,
        fit_conformal,
    )
    from src.features import add_rain_features
    from src.model import choose_hyperparams, make_fit_predict
    from src.spikes import THRESHOLD as SPIKE_THRESHOLD, spike_events

    processed = Path("data/processed")
    panel = pd.read_parquet(processed / "panel.parquet")

    rain_features = None
    rain_path = Path("data/interim/rainfall.parquet")
    if rain_path.exists():
        rain_features = add_rain_features(pd.read_parquet(rain_path))

    hp = choose_hyperparams(panel, rain_features, use_rain=True)
    print(f"hyperparams: {json.dumps(hp)}", flush=True)
    fp = make_fit_predict(rain_features, use_rain=True, params_by_horizon=hp)

    cal = rolling_backtest(panel, fp, make_origins(panel, CAL_START, CAL_END))
    halfwidths = fit_conformal(cal)
    test = rolling_backtest(panel, fp, make_origins(panel, TEST_START, TEST_END))
    covered = apply_conformal(test, halfwidths)
    print(f"scored rows: {len(covered)}", flush=True)

    write_forecasts(covered, processed / "dashboard_forecasts.csv")
    print(f"forecasts: {processed / 'dashboard_forecasts.csv'}")

    # alert history at the spike threshold (the operating point)
    outcomes = classify_alerts(
        covered,
        spike_events(panel),
        alert_threshold=round(SPIKE_THRESHOLD, 2),
    )
    write_alert_history(outcomes, processed / "dashboard_alert_history.csv")
    print(f"alert history: {processed / 'dashboard_alert_history.csv'} "
          f"(at spike threshold {SPIKE_THRESHOLD:.0%})")
