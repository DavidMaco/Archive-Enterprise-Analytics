"""Archive Enterprise Analytics — Streamlit home page.

Builds processed assets on first visit and displays a project overview with
data-quality KPIs and model status.
"""

from __future__ import annotations

import streamlit as st

from archive_analytics.dashboard import (
    ensure_project_assets,
    get_json_asset,
    get_model_metrics_cached,
    safe_page_section,
)
from archive_analytics.settings import configure_logging, get_config

configure_logging()

st.set_page_config(page_title="Archive Enterprise Analytics", layout="wide", page_icon="📊")

st.title("Archive Enterprise Analytics")
st.caption("Customer 360, risk scoring, and evidence-based retrieval over the archive dataset.")

config = get_config()

# ------------------------------------------------------------------
# Data build / model train controls
# ------------------------------------------------------------------

left, right = st.columns([1, 1])
with left:
    if st.button("Build cleaned marts", use_container_width=True):
        ensure_project_assets(train_models=False, force_rebuild=True, config=config)
        st.cache_data.clear()
        st.success("Cleaned marts rebuilt.")
with right:
    if st.button("Train risk models", use_container_width=True):
        ensure_project_assets(train_models=True, force_rebuild=True, config=config)
        st.cache_data.clear()
        st.success("Models retrained.")

ensure_project_assets(train_models=False, force_rebuild=False, config=config)

# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------

st.subheader("What is included")
st.markdown(
    """
- Customer 360 dashboard pages
- Order-to-cash event timeline
- Complaint, delay, and credit risk scores (calibrated + threshold-tuned)
- Evidence-based retrieval assistant with MMR diversity
- Data quality audit of the raw archive
    """
)

# ------------------------------------------------------------------
# Data quality KPIs
# ------------------------------------------------------------------

with safe_page_section("Data quality snapshot"):
    quality = get_json_asset("data_quality_report")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Raw data directory", str(config.raw_data_dir))
    kpi2.metric("Processed tables", len(quality.get("processed_counts", {})))
    kpi3.metric("Known email ID duplicates", f"{quality.get('email_duplicate_source_ids', 0):,}")
    st.json(quality)

# ------------------------------------------------------------------
# Model status
# ------------------------------------------------------------------

with safe_page_section("Model status"):
    metrics = get_model_metrics_cached()
    st.json(metrics)

# ------------------------------------------------------------------
# Navigation guide
# ------------------------------------------------------------------

st.subheader("Suggested navigation")
st.markdown(
    """
1. **Executive Overview** — topline trends and Sankey lifecycle.
2. **Customer 360** — account-level deep-dive.
3. **Order Timeline** — case investigation with linked docs + emails.
4. **Risk Scoring** — calibrated model outputs and feature importance.
5. **Assistant** — evidence-backed retrieval Q&A.
6. **Data Quality** — audit the raw archive before sharing outputs.
    """
)
