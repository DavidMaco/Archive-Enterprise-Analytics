# Archive Enterprise Analytics

End-to-end data-engineering and analytics portfolio built over the archive
dataset.  Every component — from raw ingestion to dashboard — is designed for
maintainability, correctness, and reproducibility.

| Capability | Description |
|------------|-------------|
| **Data Pipeline** | Parquet + SQLite ingestion → cleaned fact / dimension tables |
| **Risk Scoring** | Calibrated HGBC + Logistic models for delay, complaint, and credit risk |
| **Retrieval Assistant** | Cached TF-IDF with MMR diversity and entity-level dedup |
| **Dashboard** | Six-page Streamlit app with error boundaries and safe page sections |
| **Quality Audit** | Automated data-quality report (JSON) comparing raw vs. processed |

## Source data

Set the `ARCHIVE_ANALYTICS_RAW_DIR` environment variable to the folder
containing the raw archive parquet files and SQLite database.  If unset,
the project looks for `data/raw/` under the project root and falls back
to the legacy path.

```bash
# Example
export ARCHIVE_ANALYTICS_RAW_DIR=/path/to/archive
```

## Project structure

```
src/archive_analytics/
├── __init__.py          # Public API re-exports
├── __main__.py          # CLI entry-point (build / train / app)
├── constants.py         # Single source of truth for magic numbers
├── _util.py             # Low-level stateless helpers (hashing, ID normalisation)
├── settings.py          # AppConfig dataclass, logging, env-based config
├── ingestion.py         # Raw I/O (parquet + SQLite)
├── transforms.py        # Vectorised fact / dimension builders (~880 LOC)
├── quality.py           # Data-quality reporting
├── data.py              # Thin façade: build + load processed assets
├── modeling.py          # TimeSeriesSplit CV, calibration, threshold tuning
├── retrieval.py         # Cached TF-IDF, MMR re-ranking, entity dedup
└── dashboard.py         # Streamlit helpers (safe_page_section, cached getters)

app.py                   # Streamlit entry-point
pages/                   # Six dashboard pages
scripts/                 # build_data.py, train_models.py
tests/                   # Comprehensive pytest suite (35+ tests)
```

## Quick start

```bash
# 1. Create a virtual environment and install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -c constraints-dev.txt -e ".[dev]"

# 2. Build processed data marts
python -m archive_analytics build

# 3. Train risk models
python -m archive_analytics train

# 4. Launch the dashboard
python -m archive_analytics serve
# …or: streamlit run app.py
```

## Processed outputs

Generated into `data/processed/`:

| Table | Format | Description |
|-------|--------|-------------|
| `fact_email` | `.parquet` | Enriched email fact with scope, complaint, spam flags |
| `fact_document` | `.parquet` | Supporting + business documents with financial splits |
| `fact_order` | `.parquet` | Order-level aggregations with boolean targets |
| `dim_customer` | `.parquet` | Customer dimension with issue rates |
| `fact_event_timeline` | `.parquet` | Unified timeline across orders, emails, documents |
| `fact_customer_daily` | `.parquet` | Per-customer daily event counts |
| `fact_order_risk_features` | `.parquet` | Backward-looking risk features (no leakage) |
| `retrieval_corpus` | `.parquet` | TF-IDF-ready corpus for the assistant |
| `data_quality_report` | `.json` | Raw vs. processed quality metrics |
| `build_manifest` | `.json` | Build provenance (fingerprint, paths) |

Generated into `data/models/`:

| Artefact | Description |
|----------|-------------|
| `*.joblib` | Trained model pipelines |
| `model_metrics.json` | CV metrics, thresholds, feature importance |
| `order_risk_scores.parquet` | Predicted probabilities per order |

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=archive_analytics --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type-check
mypy src/archive_analytics/
```

## Operational model

- The dashboard is read-only by default. It does not build processed data or
  train models on page load.
- Build data explicitly with `python -m archive_analytics build`.
- Train models explicitly with `python -m archive_analytics train`.
- To enable the home page's admin buttons in a private environment, set
  `ARCHIVE_ANALYTICS_ENABLE_UI_MUTATIONS=true`.

## Design notes

- **No `sys.path` hacks**: all imports use the installed package namespace.
- **Vectorised transforms**: `DataFrame.apply(axis=1)` replaced with
  `str.extract`, `vectorized_sha1`, `np.select` etc.
- **Error boundaries**: every dashboard page section is wrapped in
  `safe_page_section()` so one failing chart doesn't crash the page.
- **Calibrated models**: `CalibratedClassifierCV` ensures predicted
  probabilities are reliable for threshold-based decisions.
- **Honest retrieval**: the assistant returns ranked evidence with
  citations — no pretence of generative answering.
- **Immutable config**: `AppConfig` is a frozen dataclass; directories
  are created lazily via `ensure_directories()`.
