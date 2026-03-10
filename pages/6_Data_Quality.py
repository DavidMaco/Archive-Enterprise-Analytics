"""Data Quality — raw-data audit, duplicate detection, and coverage analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from archive_analytics.dashboard import (
    ensure_project_assets,
    get_json_asset,
    get_table,
    safe_page_section,
)

st.set_page_config(page_title="Data Quality", layout="wide", page_icon="🧪")
st.title("Data Quality")

ensure_project_assets(train_models=False)

# ------------------------------------------------------------------
# Audit summary
# ------------------------------------------------------------------

with safe_page_section("Audit summary"):
    quality = get_json_asset("data_quality_report")
    st.json(quality)

# ------------------------------------------------------------------
# Duplicates
# ------------------------------------------------------------------

with safe_page_section("Duplicate detection"):
    emails = get_table("fact_email")
    documents = get_table("fact_document")

    email_dups = emails[emails["is_duplicate_source_id"]].head(100)
    doc_dups = documents[documents["is_duplicate_source_id"]].head(100)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duplicate email source IDs", f"{quality.get('email_duplicate_source_ids', 0):,}")
        dup_email_cols = [c for c in ("source_message_id", "timestamp", "from_email", "to_email", "subject") if c in email_dups.columns]
        if not email_dups.empty:
            st.dataframe(email_dups[dup_email_cols], use_container_width=True)
    with col2:
        st.metric("Duplicate document source IDs", f"{quality.get('document_duplicate_source_ids', 0):,}")
        dup_doc_cols = [c for c in ("source_document_id", "document_type", "order_id", "customer_id_clean") if c in doc_dups.columns]
        if not doc_dups.empty:
            st.dataframe(doc_dups[dup_doc_cols], use_container_width=True)

# ------------------------------------------------------------------
# Email scope mix
# ------------------------------------------------------------------

with safe_page_section("Email scope mix"):
    scope_mix = emails.groupby("message_scope").size().rename("count").reset_index()
    fig_scope = px.bar(scope_mix, x="message_scope", y="count", title="Message scope distribution")
    st.plotly_chart(fig_scope, use_container_width=True)

# ------------------------------------------------------------------
# Date coverage
# ------------------------------------------------------------------

with safe_page_section("Date coverage"):
    date_summary = pd.DataFrame(
        {
            "dataset": ["emails", "documents"],
            "start": [
                emails["timestamp"].min(),
                pd.to_datetime(documents["event_timestamp"], errors="coerce").min(),
            ],
            "end": [
                emails["timestamp"].max(),
                pd.to_datetime(documents["event_timestamp"], errors="coerce").max(),
            ],
        }
    )
    st.dataframe(date_summary, use_container_width=True)
