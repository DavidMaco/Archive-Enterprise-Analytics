.PHONY: install dev build train app test lint clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

build:
	$(PYTHON) -m archive_analytics build

train:
	$(PYTHON) -m archive_analytics train

app:
	$(PYTHON) -m archive_analytics serve

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(Path(p), ignore_errors=True) for p in ('data/processed', 'data/models', 'reports', '__pycache__', '.pytest_cache', '.ruff_cache')]; [shutil.rmtree(path, ignore_errors=True) for path in Path('.').rglob('__pycache__')]"
