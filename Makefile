PY := .venv/bin/python

.PHONY: setup data test all

# Bootstrap the environment. Needed once on a clean clone: uv download the
# pinned CPython from .python-version and installs exact pins.
setup:
	uv venv --python 3.12
	uv pip install -r requirements.txt

data:
	$(PY) -m src.panel

test:
	$(PY) -m pytest -q

all: data test
