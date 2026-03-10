"""Evidence-Based Assistant — retrieval-backed Q&A with MMR diversity."""

from __future__ import annotations

import streamlit as st

from archive_analytics.dashboard import ensure_project_assets, get_table, safe_page_section
from archive_analytics.retrieval import summarise_evidence

st.set_page_config(page_title="Assistant", layout="wide", page_icon="💬")
st.title("Evidence-Based Assistant")

ensure_project_assets(train_models=False)

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

customers = get_table("dim_customer")
orders = get_table("fact_order")

customer_options = [""] + customers["customer_id_clean"].dropna().astype(str).sort_values().tolist()[:5000]
order_options = [""] + orders["order_id"].dropna().astype(str).sort_values().tolist()[:5000]

col1, col2 = st.columns(2)
with col1:
    filter_customer = st.selectbox("Optional customer filter", customer_options)
with col2:
    filter_order = st.selectbox("Optional order filter", order_options)

query = st.text_area(
    "Question",
    value="Which customers show the strongest signals of delivery issues and credit risk?",
    height=120,
)

# ------------------------------------------------------------------
# Run retrieval
# ------------------------------------------------------------------

if st.button("Run assistant", use_container_width=True):
    filters = {
        "customer_id": filter_customer or None,
        "order_id": filter_order or None,
    }
    with st.spinner("Retrieving evidence …"):
        result = summarise_evidence(query=query, top_k=8, filters=filters)

    with safe_page_section("Summary"):
        st.write(result["summary"])

    # Meta sidebar
    meta = result.get("meta", {})
    if meta:
        entity_counts = meta.get("entity_counts", {})
        keyword_themes = meta.get("keyword_themes", {})
        if entity_counts:
            st.caption("Entity mix: " + ", ".join(f"{k}={v}" for k, v in entity_counts.items()))
        if keyword_themes:
            st.caption("Key themes: " + ", ".join(f"{k} ({v}×)" for k, v in keyword_themes.items()))

    with safe_page_section("Evidence"):
        for item in result["evidence"]:
            with st.expander(f"{item['citation']} | {item['title']} | score={item['score']:.3f}"):
                st.write(item["snippet"])
                st.json(
                    {
                        "entity_type": item["entity_type"],
                        "customer_id": item["customer_id"],
                        "order_id": item["order_id"],
                    }
                )
else:
    st.info("Run a query to retrieve evidence-backed results.")
