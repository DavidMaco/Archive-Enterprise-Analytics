"""Data quality reporting.

Produces a JSON-serialisable audit of raw vs. processed record counts,
duplicate rates, date coverage, and integrity notes.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_quality_report(
    raw_tables: dict[str, pd.DataFrame],
    processed_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Generate a data-quality report comparing raw and processed tables.

    Returns
    -------
    dict
        JSON-serialisable quality metrics.
    """
    emails = processed_tables["fact_email"]
    documents = processed_tables["fact_document"]
    orders = processed_tables["fact_order"]

    report: dict[str, Any] = {
        "raw_counts": {
            name: int(len(frame)) for name, frame in raw_tables.items()
        },
        "processed_counts": {
            name: int(len(frame))
            for name, frame in processed_tables.items()
            if isinstance(frame, pd.DataFrame)
        },
        "email_duplicate_source_ids": int(emails["is_duplicate_source_id"].sum()),
        "document_duplicate_source_ids": int(
            documents["is_duplicate_source_id"].sum()
        ),
        "null_customer_ids_in_email_pct": round(
            float(emails["customer_id_clean"].isna().mean() * 100), 2
        ),
        "email_date_range": [
            str(emails["timestamp"].min()),
            str(emails["timestamp"].max()),
        ],
        "order_date_range": [
            str(orders["order_created_at"].min()),
            str(orders["order_created_at"].max()),
        ],
        "document_date_range": [
            str(documents["event_timestamp"].min()),
            str(documents["event_timestamp"].max()),
        ],
        "sqlite_vs_parquet_email_gap": int(
            len(raw_tables["communications"]) - len(raw_tables["sqlite_emails"])
        ),
        "notes": [
            "Raw documentation differs from the actual files; the build trusts actual file contents.",
            "Canonical IDs are deterministic hashes added alongside original IDs.",
            "Communications with null customer IDs are retained as internal or unlinked events.",
        ],
    }

    logger.info(
        "Quality report: %d raw datasets, %d processed tables",
        len(report["raw_counts"]),
        len(report["processed_counts"]),
    )
    return report
