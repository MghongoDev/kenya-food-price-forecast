"""Phase 6 Chronos install tests (issue #17, ticket #17).

The seam is the model load: from_pretrained with a verified model id.
Tests mock from_pretrained so the suite stays offline; the real load
is exercised once in __main__ against the Hugging Face Hub.
"""
import pandas as pd
import pytest

from src.chronos_model import MODEL_ID, LICENCE, load_pipeline, forecast


class _FakePipeline:
    """Stands in for BaseChronosPipeline: records the model id, returns
    a fixed forecast."""

    def __init__(self, model_id):
        self.model_id = model_id

    def predict_df(self, df, prediction_length, quantile_levels=None, **kwargs):
        return pd.DataFrame({
            "timestamp": pd.date_range("2022-03", periods=prediction_length, freq="MS"),
            "item_id": ["0"] * prediction_length,
            "0.1": [100.0] * prediction_length,
            "0.5": [110.0] * prediction_length,
            "0.9": [120.0] * prediction_length,
        })


def test_model_id_is_a_chronos_checkpoint():
    assert MODEL_ID.startswith("amazon/chronos-")


def test_licence_is_apache_2():
    assert LICENCE == "Apache-2.0"


def test_load_pipeline_calls_from_pretrained_with_the_model_id():
    calls = {}

    def fake_loader(model_id):
        calls["model_id"] = model_id
        return _FakePipeline(model_id)

    pipe = load_pipeline(loader=fake_loader)
    assert calls["model_id"] == MODEL_ID
    assert pipe.model_id == MODEL_ID


def test_forecast_returns_a_frame_with_median_and_quantiles():
    context = pd.DataFrame({"timestamp": pd.date_range("2022-01", periods=24, freq="MS"),
                            "value": range(24)})
    out = forecast(context, prediction_length=3,
                   loader=lambda model_id: _FakePipeline(model_id))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["0.1", "0.5", "0.9"]
    assert len(out) == 3
    assert (out["0.5"] == 110.0).all()
