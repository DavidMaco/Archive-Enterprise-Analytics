"""Streamlit caching helpers and shared dashboard utilities.

All Streamlit pages should import from this module rather than calling
``data``, ``modeling``, or ``retrieval`` directly.  Cached functions ensure
that data is loaded only once per session.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from .data import (
    build_processed_assets,
    load_json_asset,
    load_processed_table,
    missing_processed_assets,
)
from .modeling import (
    load_model_metrics,
    load_risk_scores,
    model_artifacts_ready,
    train_all_targets,
)
from .settings import AppConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading table …")
def get_table(name: str) -> pd.DataFrame:
    """Load a processed parquet table (cached across the Streamlit session)."""
    return load_processed_table(name, config=get_config())


@st.cache_data(show_spinner="Loading JSON asset …")
def get_json_asset(name: str) -> dict[str, Any]:
    """Load a JSON data asset (cached across the Streamlit session)."""
    return load_json_asset(name, config=get_config())


@st.cache_data(show_spinner="Loading risk scores …")
def get_risk_scores() -> pd.DataFrame:
    """Load model risk scores (cached across the Streamlit session)."""
    return load_risk_scores(config=get_config())


@st.cache_data(show_spinner="Loading model metrics …")
def get_model_metrics_cached() -> dict[str, Any]:
    """Load model training metrics (cached across the Streamlit session)."""
    return load_model_metrics(config=get_config())


# ---------------------------------------------------------------------------
# Asset build orchestration
# ---------------------------------------------------------------------------


def ensure_project_assets(
    *,
    train_models: bool = False,
    force_rebuild: bool = False,
    config: AppConfig | None = None,
) -> None:
    """Build processed assets (and optionally train models).

    Parameters
    ----------
    train_models:
        If ``True``, also invoke ``train_all_targets``.
    force_rebuild:
        If ``True``, rebuild even if assets already exist.
    config:
        Override the default ``AppConfig``.
    """
    cfg = config or get_config()
    if not cfg.ui_mutations_enabled:
        raise PermissionError(
            "UI-triggered build and train actions are disabled. "
            "Run the CLI commands instead."
        )
    logger.info(
        "ensure_project_assets(train_models=%s, force=%s)", train_models, force_rebuild
    )
    build_processed_assets(force=force_rebuild, config=cfg)
    if train_models:
        train_all_targets(force=force_rebuild, config=cfg)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def metric_delta(current: float, baseline: float) -> str:
    """Format a KPI delta as a human-readable percentage change string."""
    if baseline == 0:
        return "n/a"
    return f"{((current - baseline) / baseline):.1%}"


def require_dashboard_assets(
    *,
    require_models: bool = False,
    config: AppConfig | None = None,
) -> None:
    """Stop the page with operator guidance when required assets are missing."""
    cfg = config or get_config()
    missing = missing_processed_assets(cfg)
    if missing:
        st.warning(
            "Processed assets are not available. "
            "Run `python -m archive_analytics build` before opening the dashboard."
        )
        st.caption("Missing assets: " + ", ".join(sorted(missing)))
        if cfg.ui_mutations_enabled:
            st.info("UI admin actions are enabled. Use the home page to build assets explicitly.")
        st.stop()

    if require_models and not model_artifacts_ready(cfg):
        st.warning(
            "Model artifacts are not available. "
            "Run `python -m archive_analytics train` before opening this page."
        )
        if cfg.ui_mutations_enabled:
            st.info("UI admin actions are enabled. Use the home page to train models explicitly.")
        st.stop()


def safe_page_section(title: str):
    """Context manager that wraps a dashboard section in an error boundary.

    Usage::

        with safe_page_section("Model metrics"):
            metrics = get_model_metrics_cached()
            st.json(metrics)

    If the block raises, a friendly warning is shown instead of crashing
    the entire page.
    """
    import contextlib

    @contextlib.contextmanager
    def _section():
        try:
            if title:
                st.subheader(title)
            yield
        except Exception as exc:
            logger.exception("Error in section '%s'", title)
            st.warning(f"⚠️ Section *{title}* could not render: {exc}")

    return _section()
