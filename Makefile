.PHONY: install dev build train app test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

build:
	python -m archive_analytics build

train:
	python -m archive_analytics train

app:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	python -m py_compile src/archive_analytics/data.py
	python -m py_compile src/archive_analytics/modeling.py
	python -m py_compile src/archive_analytics/retrieval.py

clean:
	rm -rf data/processed data/models reports __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
