"""Data pipeline facade.

This module is the public API for building and loading processed tables.
Implementation is delegated to sub-modules:

- :mod:`.ingestion` — raw-file I/O
- :mod:`.transforms` — fact / dimension builders
- :mod:`.quality` — data-quality reporting

Downstream code should continue to ``from archive_analytics.data import …``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ._util import sha1_key  # noqa: F401 – re-exported
from .constants import ORDER_PATTERN, PROCESSED_TABLES  # noqa: F401 – re-exported
from .ingestion import load_raw_tables, raw_paths  # noqa: F401 – re-exported
from .quality import build_quality_report
from .settings import AppConfig, get_config
from .transforms import (
    build_customer_daily,
    build_customer_dim,
    build_document_fact,
    build_email_fact,
    build_event_timeline,
    build_order_fact,
    build_order_risk_features,
    build_retrieval_corpus,
)

logger = logging.getLogger(__name__)


# ── I/O helpers ─────────────────────────────────────────────────────────────

def _write_parquet(name: str, frame: pd.DataFrame, config: AppConfig) -> None:
    """Write a processed parquet table to disk."""
    frame.to_parquet(config.processed_dir / PROCESSED_TABLES[name], index=False)


def _write_json(name: str, payload: dict[str, Any], config: AppConfig) -> None:
    """Write a JSON asset to disk."""
    path = config.processed_dir / PROCESSED_TABLES[name]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _raw_data_fingerprint(config: AppConfig) -> str:
    """Compute a deterministic fingerprint of the raw data files.

    Used in the build manifest so that stale processed data can be
    detected when the raw data changes.
    """
    hasher = hashlib.sha256()
    for name, path in sorted(raw_paths(config).items()):
        if path.exists():
            stat = path.stat()
            hasher.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()


# ── Build / load entry-points ──────────────────────────────────────────────

def build_processed_assets(
    force: bool = False,
    config: AppConfig | None = None,
) -> dict[str, Path]:
    """Build every processed table and write them to ``config.processed_dir``.

    Parameters
    ----------
    force:
        Rebuild even if all output files already exist.
    config:
        Optional project configuration (defaults to :func:`get_config`).

    Returns
    -------
    dict[str, Path]
        Mapping of table name → output path.
    """
    cfg = config or get_config()
    cfg.ensure_directories()
    existing = {
        name: cfg.processed_dir / fname
        for name, fname in PROCESSED_TABLES.items()
    }
    if not force and all(p.exists() for p in existing.values()):
        return existing

    logger.info("Building processed assets (force=%s)", force)
    raw = load_raw_tables(cfg)

    fact_email = build_email_fact(raw["communications"])
    fact_document = build_document_fact(
        raw["supporting_documents"], raw["business_documents"]
    )
    fact_order = build_order_fact(
        raw["erp_transactions"], fact_email, fact_document
    )
    dim_customer = build_customer_dim(fact_order, fact_email, fact_document)
    fact_event_timeline = build_event_timeline(
        fact_order, fact_email, fact_document
    )
    fact_customer_daily = build_customer_daily(
        fact_order, fact_email, fact_document
    )
    fact_order_risk = build_order_risk_features(fact_order)
    retrieval_corpus = build_retrieval_corpus(
        fact_order, dim_customer, fact_email, fact_document
    )

    processed: dict[str, pd.DataFrame] = {
        "fact_email": fact_email,
        "fact_document": fact_document,
        "fact_order": fact_order,
        "dim_customer": dim_customer,
        "fact_event_timeline": fact_event_timeline,
        "fact_customer_daily": fact_customer_daily,
        "fact_order_risk_features": fact_order_risk,
        "retrieval_corpus": retrieval_corpus,
    }

    for name, frame in processed.items():
        _write_parquet(name, frame, cfg)

    quality = build_quality_report(raw, processed)
    manifest: dict[str, Any] = {
        "raw_data_dir": str(cfg.raw_data_dir),
        "processed_dir": str(cfg.processed_dir),
        "raw_data_fingerprint": _raw_data_fingerprint(cfg),
        "tables": {
            name: str(path)
            for name, path in existing.items()
            if name not in {"build_manifest", "data_quality_report"}
        },
    }
    _write_json("data_quality_report", quality, cfg)
    _write_json("build_manifest", manifest, cfg)

    logger.info("Processed assets written to %s", cfg.processed_dir)
    return existing


def load_processed_table(
    name: str,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Load a processed parquet table by logical name.

    Triggers a build if the table does not yet exist on disk.

    Raises
    ------
    KeyError
        If *name* is not a known parquet table.
    """
    cfg = config or get_config()
    build_processed_assets(force=False, config=cfg)
    if name not in PROCESSED_TABLES or not PROCESSED_TABLES[name].endswith(".parquet"):
        raise KeyError(f"Unknown parquet table: {name}")
    return pd.read_parquet(cfg.processed_dir / PROCESSED_TABLES[name])


def load_json_asset(
    name: str,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Load a processed JSON asset by logical name.

    Raises
    ------
    KeyError
        If *name* is not ``build_manifest`` or ``data_quality_report``.
    """
    cfg = config or get_config()
    build_processed_assets(force=False, config=cfg)
    if name not in {"build_manifest", "data_quality_report"}:
        raise KeyError(f"Unknown JSON asset: {name}")
    path = cfg.processed_dir / PROCESSED_TABLES[name]
    return json.loads(path.read_text(encoding="utf-8"))
