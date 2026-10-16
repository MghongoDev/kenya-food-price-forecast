PY := .venv/bin/python

.PHONY: setup data rain backtest-baselines backtest-model test all

# Bootstrap the environment. Needed once on a clean clone: uv download the
# pinned CPython from .python-version and installs exact pins.
setup:
	uv venv --python 3.12
	uv pip install -r requirements.txt

data:
	$(PY) -m src.panel

rain:
	$(PY) -m src.rainfall

backtest-baselines:
	$(PY) -m src.backtest

backtest-model:
	$(PY) -m src.model

test:
	$(PY) -m pytest -q

all: data rain backtest-baselines backtest-model test
