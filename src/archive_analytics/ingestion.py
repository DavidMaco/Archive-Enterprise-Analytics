"""Raw data ingestion from parquet files and SQLite.

This module is the single entry-point for reading raw archive data.
All downstream code should call :func:`load_raw_tables` rather than
reading files directly.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .constants import ALLOWED_SQLITE_TABLES
from .settings import AppConfig, get_config

logger = logging.getLogger(__name__)


def load_sqlite_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Load a table from the SQLite database.

    Parameters
    ----------
    db_path:
        Path to the ``.db`` file.
    table_name:
        Must be in :pydata:`constants.ALLOWED_SQLITE_TABLES`.

    Raises
    ------
    ValueError
        If *table_name* is not in the whitelist (prevents SQL injection).
    """
    if table_name not in ALLOWED_SQLITE_TABLES:
        raise ValueError(
            f"Table {table_name!r} is not whitelisted. "
            f"Allowed: {sorted(ALLOWED_SQLITE_TABLES)}"
        )
    logger.info("Loading SQLite table %s from %s", table_name, db_path)
    with sqlite3.connect(str(db_path)) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)  # noqa: S608


def raw_paths(config: AppConfig | None = None) -> dict[str, Path]:
    """Return a mapping of dataset name → raw-file path."""
    cfg = config or get_config()
    return {
        "communications": cfg.raw_data_dir / "all_communications.parquet",
        "erp_transactions": cfg.raw_data_dir / "erp_transactions.parquet",
        "sales_documents": cfg.raw_data_dir / "sales_documents.parquet",
        "sales_items": cfg.raw_data_dir / "sales_items.parquet",
        "supporting_documents": cfg.raw_data_dir / "supporting_documents.parquet",
        "business_documents": cfg.raw_data_dir / "business_documents.parquet",
        "sqlite": cfg.raw_data_dir / "uberjugaad_email.db",
    }


def load_raw_tables(config: AppConfig | None = None) -> dict[str, pd.DataFrame]:
    """Load every raw dataset from parquet and SQLite.

    Returns
    -------
    dict[str, DataFrame]
        Keyed by dataset name (``communications``, ``erp_transactions``, etc.).
    """
    cfg = config or get_config()
    paths = raw_paths(cfg)
    logger.info("Loading raw tables from %s", cfg.raw_data_dir)

    tables: dict[str, pd.DataFrame] = {
        "communications": pd.read_parquet(paths["communications"]),
        "erp_transactions": pd.read_parquet(
            paths["erp_transactions"],
            columns=[
                "SALESDOCUMENT", "CREATIONDATE", "PLANT", "PRODUCT",
                "SOLDTOPARTY", "SHIPPINGPOINT", "SALESDOCUMENTITEM",
            ],
        ),
        "sales_documents": pd.read_parquet(paths["sales_documents"]),
        "sales_items": pd.read_parquet(paths["sales_items"]),
        "supporting_documents": pd.read_parquet(paths["supporting_documents"]),
        "business_documents": pd.read_parquet(paths["business_documents"]),
        "sqlite_emails": load_sqlite_table(paths["sqlite"], "emails"),
        "sqlite_contacts": load_sqlite_table(paths["sqlite"], "contacts"),
    }

    for name, df in tables.items():
        logger.info("  %-25s %d rows × %d cols", name, len(df), len(df.columns))
    return tables
