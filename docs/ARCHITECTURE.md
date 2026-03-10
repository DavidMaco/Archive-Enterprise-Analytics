# Architecture

## System overview

```
┌─────────────┐   ┌────────────┐   ┌──────────────┐   ┌───────────┐
│  Raw Data   │──▶│ Ingestion  │──▶│  Transforms  │──▶│ Processed │
│ (parquet +  │   │ ingestion  │   │  transforms  │   │  tables   │
│  SQLite)    │   │     .py    │   │     .py      │   │ (parquet) │
└─────────────┘   └────────────┘   └──────────────┘   └─────┬─────┘
                                                            │
                                   ┌────────────────────────┼───────────────┐
                                   │                        │               │
                              ┌────▼─────┐          ┌───────▼───┐   ┌───────▼────┐
                              │ Modeling  │          │ Retrieval │   │ Dashboard  │
                              │modeling.py│          │retrieval  │   │  app.py +  │
                              │          │          │   .py     │   │  pages/    │
                              └──────────┘          └───────────┘   └────────────┘
```

## Layer details

### 1. Configuration (`constants.py`, `settings.py`)

- **`constants.py`** — single source of truth for all magic numbers, keyword
  lists, regex patterns, model hyper-parameters, and feature column names.
- **`settings.py`** — frozen `AppConfig` dataclass resolved from environment
  variables with sensible fallbacks.  `configure_logging()` sets up structured
  stderr logging once at process start-up.

### 2. Utilities (`_util.py`)

Stateless, type-annotated helper functions used across the pipeline:

| Function | Purpose |
|----------|---------|
| `sha1_key` / `vectorized_sha1` | Deterministic hashing for canonical IDs |
| `normalize_id` | Strip whitespace + replace sentinels with `pd.NA` |
| `classify_message_scope` | Domain-based internal/external/inbound/outbound |
| `extract_order_references` | Vectorised 10-digit ID extraction via `str.extract` |
| `contains_keywords` | Case-insensitive regex keyword matching |
| `safe_date` | `pd.to_datetime` with `errors="coerce"` |

### 3. Ingestion (`ingestion.py`)

- Reads 5 parquet files and 1 SQLite table from the raw data directory.
- Table names validated against `ALLOWED_SQLITE_TABLES` allowlist.
- Returns `dict[str, pd.DataFrame]` keyed by logical dataset name.

### 4. Transforms (`transforms.py`, ~880 LOC)

Vectorised fact and dimension table builders:

| Builder | Inputs | Key outputs |
|---------|--------|-------------|
| `build_email_fact` | communications | scope, complaint/spam flags, linked orders |
| `build_document_fact` | supporting + business docs | financial splits, delivery lag |
| `build_order_fact` | ERP + emails + docs | boolean targets, aggregated metrics |
| `build_customer_dim` | orders + emails + docs | issue rate, customer label |
| `build_event_timeline` | orders + emails + docs | unified chronological timeline |
| `build_customer_daily` | orders + emails + docs | daily event-count aggregation |
| `build_order_risk_features` | fact_order | backward-looking cumulative features |
| `build_retrieval_corpus` | all facts + customer dim | TF-IDF-ready text corpus |
| `assign_complaints_to_orders` | orders + emails | bi-directional nearest-order match |

All `DataFrame.apply(axis=1)` hot paths have been replaced with vectorised
alternatives (`str.extract`, `np.select`, `vectorized_sha1`, pre-computed
boolean columns).

### 5. Data façade (`data.py`)

Thin public API:

- `build_processed_assets(force, config)` — orchestrates the full pipeline.
- `load_processed_table(name, config)` — loads a parquet table by logical name.
- `load_json_asset(name, config)` — loads `build_manifest` or `data_quality_report`.

Unknown table names raise `KeyError` (implicit path-traversal protection).

### 6. Quality reporting (`quality.py`)

`build_quality_report(raw_tables, processed_tables)` produces a JSON-serialisable
dict comparing raw vs. processed record counts, duplicate rates, date coverage,
and integrity notes.

### 7. Modeling (`modeling.py`)

- **TimeSeriesSplit CV** avoids temporal leakage.
- **CalibratedClassifierCV** wraps both HistGradientBoostingClassifier and
  LogisticRegression so probabilities are reliable.
- **Threshold tuning** via F1-maximising search on the CV fold.
- **SHA-256 integrity** hash embedded in model artefacts.
- Outputs: `*.joblib`, `model_metrics.json`, `order_risk_scores.parquet`.

### 8. Retrieval (`retrieval.py`)

- **Cached TF-IDF**: module-level `_INDEX_CACHE` keyed by corpus fingerprint.
- **MMR re-ranking**: Maximal Marginal Relevance balances relevance with diversity.
- **Entity-level deduplication**: at most one result per (entity_type, order_id).
- `retrieve_evidence(query)` returns scored `DataFrame`.
- `summarise_evidence(query)` returns structured extractive summary dict.

### 9. Dashboard (`dashboard.py`, `app.py`, `pages/`)

- `safe_page_section(title)` context manager wraps each section in a
  try/except so one failing chart doesn't crash the page.
- Cached getters (`get_fact_email`, `get_fact_order`, …) use
  `@st.cache_data` for performance.
- No `sys.path` hacks — all imports via the installed package namespace.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Single `constants.py` | Eliminates magic numbers scattered across modules |
| Frozen `AppConfig` dataclass | Immutable config prevents accidental mutation |
| Vectorised transforms | 10× speed-up over `apply(axis=1)` on large datasets |
| Calibrated classifiers | Honest probability estimates for threshold decisions |
| MMR retrieval | Prevents near-duplicate evidence flooding the top-k |
| Error boundaries in dashboard | Graceful degradation — one failing section ≠ crashed page |
| No `sys.path` manipulation | Clean packaging via `pip install -e .` |

## Key assumptions

- `customer_id` can be null for valid internal or unlinked communications.
- Supporting-document financial values (`billed_amount`, `total_amount`,
  `credit_amount`) are the best available proxy for value-based analytics.
- Duplicate source IDs are preserved for lineage while canonical IDs
  support stable downstream joins.
- Risk models are retrospective and intended for portfolio demonstration.
- The retrieval assistant returns ranked evidence — it does not generate text.
