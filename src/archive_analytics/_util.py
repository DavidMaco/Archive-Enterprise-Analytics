"""Shared utility functions used across the data pipeline.

These are low-level, stateless helpers for ID normalisation, hashing,
date coercion, keyword matching, and vectorised text extraction.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha1_key(*parts: Any) -> str:
    """Generate a deterministic 16-char hex key from the given parts."""
    joined = "||".join("" if pd.isna(part) else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def vectorized_sha1(prefix: str, *columns: pd.Series) -> pd.Series:
    """Compute SHA-1 keys over aligned Series — faster than ``DataFrame.apply(axis=1)``."""
    combined = pd.Series(prefix, index=columns[0].index)
    for col in columns:
        combined = combined + "||" + col.fillna("").astype(str)
    return combined.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])


# ---------------------------------------------------------------------------
# ID normalisation
# ---------------------------------------------------------------------------

def normalize_id(series: pd.Series) -> pd.Series:
    """Strip whitespace and convert sentinel strings to ``pd.NA``."""
    cleaned = series.astype("string").str.strip()
    return cleaned.replace(["<NA>", "nan", "None", ""], pd.NA)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def mode_or_first(series: pd.Series) -> str | None:
    """Return the statistical mode of *series*, falling back to the first value."""
    series = series.dropna()
    if series.empty:
        return None
    mode = series.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(series.iloc[0])


# ---------------------------------------------------------------------------
# Email classification
# ---------------------------------------------------------------------------

def classify_message_scope(from_email: pd.Series, to_email: pd.Series) -> pd.Series:
    """Label each email as *internal*, *external_inbound*, *external_outbound*, or *mixed*."""
    from_internal = from_email.fillna("").str.endswith("@uberjugaad.com")
    to_internal = to_email.fillna("").str.endswith("@uberjugaad.com")
    return pd.Series(
        np.select(
            [from_internal & to_internal, ~from_internal & to_internal, from_internal & ~to_internal],
            ["internal", "external_inbound", "external_outbound"],
            default="mixed",
        ),
        index=from_email.index,
    )


def extract_order_references(subject: pd.Series, body: pd.Series) -> pd.Series:
    """Vectorised extraction of 10-digit order IDs (replaces row-wise ``apply``)."""
    combined = subject.fillna("") + " " + body.fillna("")
    return combined.str.extract(r"(\b\d{10}\b)", expand=False).astype("string")


def contains_keywords(series: pd.Series, keywords: list[str]) -> pd.Series:
    """Check whether each element contains any of *keywords* (case-insensitive)."""
    pattern = "|".join(re.escape(kw) for kw in keywords)
    return series.fillna("").str.lower().str.contains(pattern, regex=True)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def safe_date(series: pd.Series) -> pd.Series:
    """Coerce *series* to ``datetime64[ns]``, returning ``NaT`` for unparseable values."""
    return pd.to_datetime(series, errors="coerce")
