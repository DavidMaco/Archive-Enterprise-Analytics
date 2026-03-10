# Archive Enterprise Analytics — Project Walkthrough

> End-to-end record of every phase, commit, and engineering decision from first scaffold to live Streamlit Cloud deployment.

---

## What Was Built

An end-to-end enterprise analytics platform that ingests archive data (raw parquet files + SQLite), runs it through a vectorised data pipeline, trains calibrated ML risk models, and surfaces everything in a six-page Streamlit dashboard with a companion Power BI data model.

---

## Phase 1 — Project Scaffolding (`c0f14db`)

The foundation was laid:

- `pyproject.toml` — package manifest, dependency declarations, Ruff + mypy config
- `Makefile` — `make lint`, `make test`, `make serve` shortcuts
- `.github/workflows/ci.yml` — CI pipeline (Ruff → mypy → pytest)
- `.gitignore` — initial ignore rules for `.venv/`, `__pycache__/`, etc.

---

## Phase 2 — Core Package (`8d8e893`)

The `src/archive_analytics/` package was initialised with:

**`constants.py`** — single source of truth for every magic value:

| Constant | Purpose |
|----------|---------|
| `ORDER_PATTERN` | 10-digit order ID regex |
| `SPAM_KEYWORDS` / `COMPLAINT_KEYWORDS` | Email classification word lists |
| `PROCESSED_TABLES` | Logical name → filename registry for all 10 output tables |
| `TARGETS` | `["will_generate_complaint", "will_be_delayed", "will_generate_credit_memo"]` |
| `PREDICTIVE_NUMERIC_FEATURES` / `PREDICTIVE_CATEGORICAL_FEATURES` | ML feature lists |

**`settings.py`** — immutable `AppConfig` dataclass; resolves `raw_data_dir` from the `ARCHIVE_ANALYTICS_RAW_DIR` environment variable or falls back to `data/raw/`.

**`_util.py`** — shared helpers: `sha1_key`, `vectorized_sha1`, `normalize_id`, `safe_date`, `contains_keywords`, `extract_order_references`, `classify_message_scope`.

---

## Phase 3 — Data Pipeline (`a3b2391`)

Four modules implement the full ETL:

### `ingestion.py`

Reads 7 raw sources:

| Source | Format |
|--------|--------|
| `all_communications.parquet` | Emails |
| `erp_transactions.parquet` | SAP-style order line items |
| `supporting_documents.parquet` | Document financials per order |
| `business_documents.parquet` | Unlinked enterprise documents |
| `sales_documents.parquet` | Stub (schema placeholder) |
| `sales_items.parquet` | Stub (schema placeholder) |
| `uberjugaad_email.db` | SQLite — tables `emails`, `contacts` |

SQLite table names are validated against an `ALLOWED_SQLITE_TABLES` whitelist to prevent SQL injection.

### `transforms.py`

Builds the star schema in sequence:

1. **`build_email_fact`** — vectorised order-reference extraction, spam/complaint flag derivation, canonical SHA-1 IDs. Replaces the former row-wise `apply` call for a ~10x speed-up.
2. **`build_document_fact`** — merges supporting + business docs; splits financials into `revenue_amount`, `cost_amount`, `refund_amount` (replaces old meaningless `document_amount` sum).
3. **`build_order_fact`** — groups ERP line items; joins document and email aggregations; derives boolean targets (`will_be_delayed`, `will_generate_complaint`, `will_generate_credit_memo`) from observable evidence only.
4. **`assign_complaints_to_orders`** — bi-directional `np.searchsorted` matching within a 60-day window (no row-wise `apply`).
5. **`build_customer_dim`** — outer-joins order/email/document aggregations; derives `issue_rate`.
6. **`build_event_timeline`** — unified `order_created | email | document` event log with a shared `keep_cols` schema.
7. **`build_customer_daily`** — per-customer daily event counts by type.
8. **`build_order_risk_features`** — adds strictly backward-looking cumulative features (no data leakage): `customer_prior_*_rate`, `plant_prior_delay_rate`, log-transformed order counts.
9. **`build_retrieval_corpus`** — TF-IDF-ready text rows from orders, customers, emails, and documents; complaint emails sorted by relevance before being capped at `EMAIL_CORPUS_CAP`.

### `quality.py`

Generates `data_quality_report.json` comparing raw vs. processed counts, duplicate source-ID rates, null percentages, and date coverage.

### `data.py`

Public façade. `build_processed_assets()` orchestrates all builders and writes 10 files to `data/processed/`. `processed_assets_ready()` and `missing_processed_assets()` let the dashboard guard itself without crashing.

### Output — Processed Star Schema

| Table | Grain | Description |
|-------|-------|-------------|
| `fact_order` | 1 row / order | Order aggregations with boolean targets |
| `fact_order_risk_features` | 1 row / order | Backward-looking ML feature vector |
| `dim_customer` | 1 row / customer | Customer dimension with issue rates |
| `fact_email` | 1 row / email | Enriched with scope, complaint, and spam flags |
| `fact_document` | 1 row / document | Business and supporting docs with financial splits |
| `fact_event_timeline` | 1 row / event | Unified cross-entity timeline |
| `fact_customer_daily` | 1 row / customer / day | Daily event aggregates |
| `retrieval_corpus` | 1 row / document | TF-IDF-ready text corpus |
| `data_quality_report` | JSON | Raw vs. processed completeness metrics |
| `build_manifest` | JSON | Build provenance (SHA-256 fingerprints, timestamps) |

---

## Phase 4 — ML Risk Scoring (`bc98e85`)

**`modeling.py`** trains three binary classifiers, one per target:

- Numeric features pass through `SimpleImputer → StandardScaler`; categoricals through `OrdinalEncoder`.
- Model selection: `HistGradientBoostingClassifier` vs `LogisticRegression`; winner chosen by ROC-AUC.
- Calibration with `CalibratedClassifierCV` (isotonic regression, 5-fold) so $\hat{p} \in [0, 1]$ is a true probability.
- Threshold $\tau^*$ tuned to maximise $F_1$ on the hold-out split via the precision-recall curve.
- Time-series cross-validation produces mean ± std metric estimates.
- SHA-256 hashes of model artefacts are stored for integrity verification.

### Model Artefacts (`data/models/`)

| File | Description |
|------|-------------|
| `*.joblib` | Trained and calibrated scikit-learn pipelines |
| `model_metrics.json` | CV metrics, per-target thresholds, feature importance, run ID |
| `order_risk_scores.parquet` | Predicted probabilities and binary predictions per order |
| `model_history.json` | Rolling record of metrics from all previous runs |

---

## Phase 5 — TF-IDF Retrieval Assistant (`0019b24`)

**`retrieval.py`** — `summarise_evidence()`:

1. Builds a sparse TF-IDF matrix over the retrieval corpus.
2. Retrieves top-k results by cosine similarity.
3. Applies **MMR diversity re-ranking** to avoid returning redundant results.
4. Filters optionally by `customer_id` or `order_id`.
5. Returns a structured evidence object with summary, entity counts, and keyword themes.

---

## Phase 6 — Dashboard and CLI (`f9a82d7`)

### `app.py` — Streamlit Home Page

Admin controls (Build / Train buttons) gated by `ARCHIVE_ANALYTICS_ENABLE_UI_MUTATIONS`. KPI metrics and pipeline status displayed on load.

### Six Dashboard Pages

| File | Page | Content |
|------|------|---------|
| `pages/1_Executive_Overview.py` | Executive Overview | KPI metrics, monthly trend lines, customer-order Sankey, complaint heatmap |
| `pages/2_Customer_360.py` | Customer 360 | Issue rates, order history, communication mix, timeline scatter |
| `pages/3_Order_Timeline.py` | Order Timeline | Unified events with risk score columns per order |
| `pages/4_Risk_Scoring.py` | Risk Scoring | Probability distributions, F1/ROC metrics, feature importance, model governance |
| `pages/5_Assistant.py` | Evidence Assistant | TF-IDF Q&A with optional customer/order filter |
| `pages/6_Data_Quality.py` | Data Quality | Raw vs. processed counts, duplicate rates, null audit |

### `dashboard.py` — Caching and Guards

- `@st.cache_data` wrappers around all data loaders (load once per session).
- `require_dashboard_assets()` — calls `st.stop()` with a clear error if processed files are missing.
- `safe_page_section()` — context manager that catches and displays exceptions without crashing the entire page.
- `ensure_project_assets()` — orchestrates build + optional model training from the UI.

### CLI

`python -m archive_analytics build | train | serve` — wired via `pyproject.toml` entry points.

---

## Phase 7 — Test Suite (`a9a61db`)

38 tests under `tests/` covering every module:

- Ingestion column contracts
- All nine transform builders
- Quality report key completeness
- Modeling utilities and threshold tuning
- Retrieval and MMR logic
- Dashboard guard functions
- Settings and config resolution

All 38 pass with Ruff (linting) and mypy (strict type-checking) clean.

---

## Phase 8 — Security Hardening (`4bc5f89`)

| Hardening | Detail |
|-----------|--------|
| Read-only dashboard by default | Build/Train buttons hidden unless `ARCHIVE_ANALYTICS_ENABLE_UI_MUTATIONS=true` |
| SQLite injection prevention | Table names validated against `ALLOWED_SQLITE_TABLES` whitelist before query |
| Secrets excluded from git | `.streamlit/secrets.toml` added to `.gitignore` |

---

## Phase 9 — Streamlit Cloud Deployment Prep (`93e3866`)

Everything needed for a one-click Streamlit Cloud deploy was created:

| File | Purpose |
|------|---------|
| `requirements.txt` | Flat pinned deps (`pandas==2.3.3`, `streamlit==1.54.0`, etc.) for Streamlit Cloud's `uv` installer |
| `runtime.txt` | `python-3.11` |
| `.streamlit/config.toml` | Dark theme (`#0E1117` background, `#4F8BF9` primary) |
| `.streamlit/secrets.toml.example` | Operator template documenting all required secrets |
| `powerbi/README.md` | Full DAX measure library, relationship diagram, 5-page dashboard blueprint |
| `powerbi/theme.json` | Colour-matched Power BI theme |

`README.md` was completely overhauled with:
- CI / Python / Streamlit badges
- Mermaid architecture diagram
- KaTeX risk model formulas
- Dashboard pages reference table
- Streamlit Cloud step-by-step deployment guide
- Power BI connection and DAX section

---

## Phase 10 — Three Deployment Bug Fixes

Three successive errors surfaced from Streamlit Cloud logs and were diagnosed and fixed one by one.

### Fix 1 — `ModuleNotFoundError: No module named 'archive_analytics'` (`60f6267`)

**Root cause:** Streamlit Cloud uses `uv` and reads only `requirements.txt`; it ignores `pyproject.toml`, so the local package was never installed.

**Fix:** Added `-e .` at the bottom of `requirements.txt` so `uv` installs the local `archive_analytics` package in editable mode.

### Fix 2 — `ImportError: cannot import name 'processed_assets_ready' from 'archive_analytics.dashboard'` (`7ce1995`)

**Root cause:** `app.py` was importing `processed_assets_ready` from `archive_analytics.dashboard` but the function lives in `archive_analytics.data`.

**Fix:** Corrected the import line in `app.py`:

```python
# Before
from archive_analytics.dashboard import processed_assets_ready  # wrong module

# After
from archive_analytics.data import processed_assets_ready
```

### Fix 3 — "Processed assets are not available" on live dashboard (`5d67576`)

**Root cause:** `data/processed/` was listed in `.gitignore`, so Streamlit Cloud cloned the repo with no data files. `processed_assets_ready()` returned `False` and every page called `st.stop()`.

**Fix — three-part solution:**

1. **`scripts/seed_demo.py`** — new script that:
   - Generates synthetic raw parquet files and an SQLite stub in `data/raw/`
   - Runs the real `build_processed_assets()` pipeline on that synthetic data
   - Derives plausible risk scores from observable order features and writes them to `data/models/order_risk_scores.parquet`
   - Writes a realistic `data/models/model_metrics.json`

2. **`.gitignore` updated:**
   - `data/processed/` and `data/models/` removed from ignore → now tracked in git
   - `data/raw/` added to ignore → private archive files stay off GitHub

3. **15 artefact files committed** — 10 processed tables, 5 model files — so Streamlit Cloud finds fully-populated data on every deploy.

**Generated files:**

| Directory | Files | Total size |
|-----------|-------|------------|
| `data/processed/` | 10 parquet + JSON | ~405 KB |
| `data/models/` | 2 parquet/JSON + 3 `.joblib` pipelines | ~400 KB |

---

## Commit History

| SHA | Message |
|-----|---------|
| `5d67576` | feat: add synthetic demo data for Streamlit Cloud |
| `7ce1995` | fix: import processed_assets_ready from data module, not dashboard |
| `60f6267` | fix: install archive_analytics package via -e . in requirements.txt for Streamlit Cloud |
| `5d713bb` | docs: fix KaTeX underscore errors and remove all em dashes from README |
| `93e3866` | deploy: Streamlit Cloud prep, Power BI guide, and README overhaul |
| `4bc5f89` | hardening: make dashboard read-only by default |
| `81be8a1` | chore: align docs, lint, and dev tooling |
| `2a2414c` | chore: add src namespace package marker |
| `34866c8` | docs: add README and architecture documentation |
| `a9a61db` | test: comprehensive test suite — 35 tests, 100% pass rate |
| `5943399` | feat: add build and train pipeline scripts |
| `f9a82d7` | feat: add dashboard helpers, CLI entry-point, and all pages |
| `0019b24` | feat: implement TF-IDF retrieval with MMR diversity |
| `bc98e85` | feat: implement ML risk scoring module |
| `a3b2391` | feat: implement data pipeline — ingestion, transforms, quality, facade |
| `8d8e893` | feat: add constants, utilities, and settings modules |
| `c0f14db` | build: project scaffolding — pyproject.toml, Makefile, CI, .gitignore |

---

## Current State

| Item | Status |
|------|--------|
| Repo | `DavidMaco/Archive-Enterprise-Analytics` — branch `main`, HEAD `5d67576` |
| CI | Ruff + mypy + pytest 38/38 passing |
| Streamlit Cloud | All three deployment errors resolved; all 6 pages load with synthetic demo data |
| Power BI | Parquet tables ready to connect; DAX measures and theme documented in `powerbi/` |

---

## Operations Guide

### Regenerate demo data (after transform changes)

```bash
python scripts/seed_demo.py
git add data/processed/ data/models/
git commit -m "chore: refresh synthetic demo data"
git push
```

### Connect real archive data

```bash
# Set the raw data directory
export ARCHIVE_ANALYTICS_RAW_DIR=/path/to/archive   # macOS / Linux
set ARCHIVE_ANALYTICS_RAW_DIR=C:\path\to\archive    # Windows

# Build the processed tables
python -m archive_analytics build

# Train the risk models
python -m archive_analytics train

# Launch the dashboard
python -m archive_analytics serve
```

### Streamlit Cloud secrets

```toml
ARCHIVE_ANALYTICS_RAW_DIR = "/mount/src/archive-enterprise-analytics/data/raw"
ARCHIVE_ANALYTICS_ENABLE_UI_MUTATIONS = "false"
```

Set `ARCHIVE_ANALYTICS_ENABLE_UI_MUTATIONS = "true"` to expose the Build and Train admin buttons in the dashboard UI.
