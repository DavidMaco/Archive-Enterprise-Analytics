"""Centralized constants for archive-enterprise-analytics.

All magic numbers, keyword lists, regex patterns, and configuration defaults
are declared here to provide a single source of truth.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

ORDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b\d{10}\b")

# ---------------------------------------------------------------------------
# Keyword lists for email classification
# ---------------------------------------------------------------------------

SPAM_KEYWORDS: Final[list[str]] = [
    "unsubscribe",
    "blockchain",
    "quantum",
    "free assessment",
    "mainframe called",
    "click here",
    "special offer",
]

COMPLAINT_KEYWORDS: Final[list[str]] = [
    "urgent",
    "complaint",
    "delivery issue",
    "delivery failure",
    "delay",
    "delayed",
    "damaged",
    "shortage",
    "credit",
    "replacement",
    "expedited",
    "escalation",
    "escalate",
    "issue",
    "problem",
    "late",
    "action plan",
    "root cause",
    "unacceptable",
    "reconsider",
    "immediate resolution",
]

# ---------------------------------------------------------------------------
# Processed-table registry  (logical name → filename)
# ---------------------------------------------------------------------------

PROCESSED_TABLES: Final[dict[str, str]] = {
    "fact_email": "fact_email.parquet",
    "fact_document": "fact_document.parquet",
    "fact_order": "fact_order.parquet",
    "dim_customer": "dim_customer.parquet",
    "fact_event_timeline": "fact_event_timeline.parquet",
    "fact_customer_daily": "fact_customer_daily.parquet",
    "fact_order_risk_features": "fact_order_risk_features.parquet",
    "retrieval_corpus": "retrieval_corpus.parquet",
    "build_manifest": "build_manifest.json",
    "data_quality_report": "data_quality_report.json",
}

# ---------------------------------------------------------------------------
# SQLite whitelist (prevents SQL-injection on table names)
# ---------------------------------------------------------------------------

ALLOWED_SQLITE_TABLES: Final[frozenset[str]] = frozenset({"emails", "contacts"})

# ---------------------------------------------------------------------------
# Model targets and feature sets
# ---------------------------------------------------------------------------

TARGETS: Final[list[str]] = [
    "will_generate_complaint",
    "will_be_delayed",
    "will_generate_credit_memo",
]

METRICS_FILE: Final[str] = "model_metrics.json"
SCORES_FILE: Final[str] = "order_risk_scores.parquet"
MODEL_HISTORY_FILE: Final[str] = "model_history.json"

PREDICTIVE_NUMERIC_FEATURES: Final[list[str]] = [
    "order_line_count",
    "product_nunique",
    "plant_nunique",
    "shipping_point_nunique",
    "order_year",
    "order_month_num",
    "order_quarter",
    "customer_prior_orders_log",
    "customer_prior_will_generate_complaint_rate",
    "customer_prior_will_be_delayed_rate",
    "customer_prior_will_generate_credit_memo_rate",
    "plant_prior_orders",
    "plant_prior_delay_rate",
]

PREDICTIVE_CATEGORICAL_FEATURES: Final[list[str]] = [
    "primary_plant",
    "primary_product_family",
]

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

TRAIN_TEST_TIME_QUANTILE: Final[float] = 0.8
STRATIFIED_TEST_SIZE: Final[float] = 0.2
LOGISTIC_MAX_ITER: Final[int] = 600
BOOSTING_LEARNING_RATE: Final[float] = 0.05
BOOSTING_MAX_ITER: Final[int] = 250
BOOSTING_MAX_DEPTH: Final[int] = 6
CV_N_SPLITS: Final[int] = 5

# ---------------------------------------------------------------------------
# Data-pipeline tuning
# ---------------------------------------------------------------------------

COMPLAINT_TOLERANCE_DAYS: Final[int] = 60
BODY_PREVIEW_LENGTH: Final[int] = 280
SNIPPET_LENGTH: Final[int] = 280
EMAIL_CORPUS_CAP: Final[int] = 25_000

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: Final[int] = 5
RETRIEVAL_TOP_K: Final[int] = DEFAULT_TOP_K
TFIDF_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
MMR_LAMBDA: Final[float] = 0.7
