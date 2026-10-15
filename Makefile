PY := .venv/bin/python

.PHONY: data test all

data:
	$(PY) -m src.panel

test:
	$(PY) -m pytest -q

all: data test
