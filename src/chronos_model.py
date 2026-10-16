"""Phase 6 Chronos zero-shot comparison (issue #6, ticket #17).

Chronos is a pretrained time-series foundation model: zero-shot
forecasting with no training on the target series. This module
verifies the install (package, model id, licence) and loads a small
checkpoint. The forecaster wrapper that fits the harness contract
lives in ticket #18; the backtest and verdict in ticket #19.

Model card: https://huggingface.co/amazon/chronos-bolt-tiny
Licence: Apache-2.0 (verified on the model card, 2026-10-16).
"""
from __future__ import annotations

import pandas as pd

MODEL_ID = "amazon/chronos-bolt-tiny"   # smallest checkpoint; fast to load
LICENCE = "Apache-2.0"


def load_pipeline(loader=None):
    """Load the Chronos pipeline from the Hugging Face Hub. Network
    call: the checkpoint downloads on first use, then caches. The
    loader parameter exists so tests can inject a fake."""
    if loader is not None:
        return loader(MODEL_ID)
    from chronos import BaseChronosPipeline

    # device_map="cpu": the only safe default for verification; a GPU
    # run would pass device_map="cuda" or a specific accelerator
    return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu")


def forecast(context: pd.DataFrame, prediction_length: int = 3, loader=None) -> pd.DataFrame:
    """Zero-shot forecast via the pandas API: returns a frame with
    quantile columns 0.1, 0.5 (median), 0.9. The context is a
    DataFrame with a timestamp column and a value column. An id
    column is added if absent (one series per market)."""
    pipe = load_pipeline(loader=loader)
    df = context.rename(columns={"value": "target"})
    if "item_id" not in df.columns:
        df = df.assign(item_id="0")
    out = pipe.predict_df(
        df,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    return out[["0.1", "0.5", "0.9"]].reset_index(drop=True)


if __name__ == "__main__":
    context = pd.DataFrame({
        "timestamp": pd.date_range("2022-01", periods=24, freq="MS"),
        "value": [100.0 + i for i in range(24)],
    })
    out = forecast(context, prediction_length=3)
    print(f"model: {MODEL_ID}  licence: {LICENCE}")
    print(out.to_string(index=False))
