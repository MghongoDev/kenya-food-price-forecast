"""Phase 3 model (issue #3, ADR 0005).

A direct, global LightGBM per horizon: every horizon has its own model
trained to predict the log change from the origin's features, so error
never compounds and the leakage story is one sentence long - the feature
table is origin-honest, and the training pairs are assembled from the
harness's train frame (rows at or before the origin) only.

The public seam stays the harness contract: make_fit_predict(...) returns
a fit_predict(train, origin, horizons) callable.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.backtest import months_ahead, month_value
from src.features import (
    PRICE_FEATURE_COLUMNS,
    RAIN_FEATURE_COLUMNS,
    feature_table,
)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "verbose": -1,
    "seed": 0,
}

FIRST_ORIGIN = "2012-01"   # training pairs start here: lag_12 needs a year
MIN_TRAIN_ROWS = 40        # a horizon model refuses to train below this


def model_feature_columns(use_rain: bool) -> list[str]:
    return list(PRICE_FEATURE_COLUMNS) + (list(RAIN_FEATURE_COLUMNS) if use_rain else [])


def _assemble_training(
    feat: pd.DataFrame, horizon: int, columns: list[str], first_origin: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Training pairs for one horizon: origin month s with observed price,
    target month s+h with observed price, y = log(p[s+h]/p[s])."""
    tgt = feat.loc[feat["observed"], ["market_id", "month", "price"]].rename(
        columns={"month": "target_month", "price": "p_target"}
    )
    tgt["origin_month"] = tgt["target_month"].map(lambda m: months_ahead(m, -horizon))
    org = feat.loc[feat["observed"], ["market_id", "month", *columns]].rename(
        columns={"month": "origin_month"}
    )
    org["p_origin"] = org["price"]   # the level itself also travels as a feature
    data = tgt.merge(
        org, on=["market_id", "origin_month"], how="inner"
    )
    data = data[data["origin_month"] >= first_origin]
    ok = (
        data["p_origin"].notna() & (data["p_origin"] > 0) & (data["p_target"] > 0)
    )
    data = data[ok]
    y = np.log(data["p_target"].to_numpy() / data["p_origin"].to_numpy())
    return data[columns], pd.Series(y, index=data.index)


def make_fit_predict(
    rain_features: pd.DataFrame | None = None,
    use_rain: bool = True,
    params: dict | None = None,
    params_by_horizon: dict[int, dict] | None = None,
    first_origin: str = FIRST_ORIGIN,
    min_train_rows: int = MIN_TRAIN_ROWS,
):
    """Harness contract: fit_predict(train, origin, horizons). The rain frame
    is sliced to months at or before each origin before use. Tuning output
    can arrive per horizon via params_by_horizon."""
    settings = dict(DEFAULT_PARAMS)
    if params:
        settings.update(params)
    columns = model_feature_columns(use_rain)

    import lightgbm as lgb

    def fit_predict(train: pd.DataFrame, origin: str, horizons: tuple[int, ...]) -> pd.DataFrame:
        rain = None
        if rain_features is not None and use_rain:
            rain = rain_features[rain_features["month"] <= origin]
        feat = feature_table(train, rain_features=rain)
        rows = []
        for h in horizons:
            X, y = _assemble_training(feat, h, columns, first_origin)
            if len(X) < min_train_rows:
                continue
            h_settings = dict(settings)
            if params_by_horizon:
                h_settings.update(params_by_horizon.get(h, {}))
            model = lgb.LGBMRegressor(**h_settings)
            model.fit(X, y)
            at_origin = feat[(feat["month"] == origin) & feat["observed"]]
            if at_origin.empty:
                continue
            deltas = model.predict(at_origin[columns])
            tgt_month = months_ahead(origin, h)
            for market_id, price, d in zip(
                at_origin["market_id"], at_origin["price"], deltas
            ):
                if not np.isfinite(d):
                    continue
                rows.append(
                    {
                        "market_id": int(market_id),
                        "horizon": h,
                        "target_month": tgt_month,
                        "price_pred": float(price) * math.exp(float(d)),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["market_id", "horizon", "target_month", "price_pred"],
        )

    return fit_predict


DEFAULT_GRID = (
    {"learning_rate": 0.05, "num_leaves": 31},
    {"learning_rate": 0.1, "num_leaves": 31},
    {"learning_rate": 0.05, "num_leaves": 63},
    {"learning_rate": 0.2, "num_leaves": 15},
)


def choose_hyperparams(
    panel: pd.DataFrame,
    rain_features: pd.DataFrame | None = None,
    use_rain: bool = True,
    val_start: str = "2021-07",
    val_end: str = "2021-12",
    grid: tuple[dict, ...] = DEFAULT_GRID,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict[int, dict]:
    """Light tuning per horizon (issue #3): a few settings, scored on a
    time-ordered validation block INSIDE the training window, never the
    test period. Criterion: mean MAE in KES on the validation block, run
    through the same rolling-origin harness."""
    from src.backtest import make_origins, rolling_backtest

    val_origins = make_origins(panel, val_start, val_end)
    picked = {}
    for h in horizons:
        best = None
        for combo in grid:
            fp = make_fit_predict(
                rain_features=rain_features, use_rain=use_rain,
                params={**combo, "seed": 0},
            )
            score = rolling_backtest(panel, fp, val_origins, horizons=(h,))
            if score.empty:
                continue
            mae = float((score["price_true"] - score["price_pred"]).abs().mean())
            if best is None or mae < best[0]:
                best = (mae, dict(combo))
        if best is not None:
            picked[h] = best[1]
    return picked


if __name__ == "__main__":
    import json
    from pathlib import Path

    from src.backtest import make_origins, rolling_backtest
    from src.baselines import naive, seasonal_mean_log_change, seasonal_naive
    from src.features import add_rain_features
    from src.metrics import metric_table

    processed = Path("data/processed")
    panel = pd.read_parquet(processed / "panel.parquet")
    origins = make_origins(panel, "2022-01", "2026-05")

    rain_features = None
    rain_path = Path("data/interim/rainfall.parquet")
    if rain_path.exists():
        rain_features = add_rain_features(pd.read_parquet(rain_path))
        print(f"rain features: {len(rain_features)} rows")
    else:
        print("no rainfall table: the with-rainfall variant is skipped (ADR 0002 fallback)")

    scored = []
    for name, fn in (
        ("naive", naive),
        ("seasonal_naive", seasonal_naive),
        ("seasonal_mean_log_change", seasonal_mean_log_change),
    ):
        s = rolling_backtest(panel, fn, origins)
        scored.append(s.assign(forecaster=name))
        print(f"{name}: {len(s)} scored rows", flush=True)

    variants = {}
    for use_rain in (False, True):
        if use_rain and rain_features is None:
            continue
        label = f"lgbm_{'rain' if use_rain else 'norain'}"
        hp = choose_hyperparams(panel, rain_features, use_rain)
        print(f"{label} hyperparams: {json.dumps(hp)}", flush=True)
        fp = make_fit_predict(rain_features, use_rain, params_by_horizon=hp)
        s = rolling_backtest(panel, fp, origins)
        scored.append(s.assign(forecaster=label))
        variants[label] = s
        print(f"{label}: {len(s)} scored rows", flush=True)

    score_all = pd.concat(scored, ignore_index=True)
    score_all.to_csv(processed / "backtest_model_scores.csv", index=False)

    reference = score_all[score_all["forecaster"] == "seasonal_naive"]
    tables = []
    for name in score_all["forecaster"].unique():
        s = score_all[score_all["forecaster"] == name]
        tables.append(metric_table(s, reference).assign(forecaster=name))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(processed / "model_metrics.csv", index=False)

    pooled = table[(table["market"] == "ALL") & (table["period"] == "all")][
        ["forecaster", "horizon", "n", "mae_kes", "wape", "mase"]
    ]
    print("\n== pooled results, all periods")
    print(pooled.to_string(index=False))

    drought = table[(table["market"] == "ALL") & (table["period"] == "drought")]
    print("\n== drought-year (2022) pooled rows")
    print(drought[["forecaster", "horizon", "n", "mae_kes", "mase"]].to_string(index=False))

    best_key = "lgbm_rain" if "lgbm_rain" in variants else "lgbm_norain"
    if best_key in variants:
        s = variants[best_key].copy()
        s["abs_err"] = (s["price_true"] - s["price_pred"]).abs()
        s["observed_flag"] = s["price_true"].notna()
        worst = s.nlargest(20, "abs_err")
        worst.to_csv(processed / "worst_forecasts.csv", index=False)
        print(f"\n== worst 20 forecasts, {best_key} (write to worst_forecasts.csv)")
        cols = ["market", "origin", "horizon", "target_month", "price_true", "price_pred", "abs_err"]
        print(worst[cols].to_string(index=False))
        print("\nerror clustering: counts by market and horizon in the worst 20")
        print(worst.groupby(["market", "horizon"]).size().to_string())
