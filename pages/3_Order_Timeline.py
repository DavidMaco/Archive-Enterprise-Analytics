"""Order Timeline — case-level investigation with linked documents and emails."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from archive_analytics.dashboard import (
    ensure_project_assets,
    get_risk_scores,
    get_table,
    safe_page_section,
)

st.set_page_config(page_title="Order Timeline", layout="wide", page_icon="🗂️")
st.title("Order Timeline")

ensure_project_assets(train_models=False)  # never force model training from a page

orders = get_table("fact_order")
timeline = get_table("fact_event_timeline")
documents = get_table("fact_document")
emails = get_table("fact_email")

# ------------------------------------------------------------------
# Order selector (vectorised label)
# ------------------------------------------------------------------

interesting_orders = orders[(orders["document_count"] > 0) | (orders["linked_email_count"] > 0)].copy()
interesting_orders["label"] = (
    interesting_orders["order_id"].astype(str)
    + " | customer "
    + interesting_orders["customer_id_clean"].astype(str)
    + " | docs "
    + interesting_orders["document_count"].astype(int).astype(str)
    + " | emails "
    + interesting_orders["linked_email_count"].astype(int).astype(str)
)
selected_label = st.selectbox("Order", interesting_orders["label"].tolist())
selected_order = interesting_orders.loc[interesting_orders["label"] == selected_label].iloc[0]
order_id = selected_order["order_id"]

order_timeline = timeline[timeline["order_id"].astype("string") == str(order_id)].copy()
order_docs = documents[documents["order_id"].astype("string") == str(order_id)].copy()
order_emails = emails[emails["linked_order_id"].astype("string") == str(order_id)].copy()

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("Lines", int(selected_order["order_line_count"]))
m2.metric("Max delivery lag", f"{selected_order['max_delivery_lag_days']:.0f} days")
m3.metric("Complaint flag", str(bool(selected_order["will_generate_complaint"])))
m4.metric("Credit flag", str(bool(selected_order["will_generate_credit_memo"])))

# ------------------------------------------------------------------
# Risk scores (if models have been trained)
# ------------------------------------------------------------------

with safe_page_section(""):
    risk_scores = get_risk_scores()
    order_score = risk_scores[risk_scores["order_id"].astype("string") == str(order_id)]
    if not order_score.empty:
        s1, s2, s3 = st.columns(3)
        for col, label, container in [
            ("will_generate_complaint_score", "Complaint risk", s1),
            ("will_be_delayed_score", "Delay risk", s2),
            ("will_generate_credit_memo_score", "Credit risk", s3),
        ]:
            if col in order_score.columns:
                container.metric(label, f"{float(order_score[col].iloc[0]):.1%}")

# ------------------------------------------------------------------
# Timeline scatter
# ------------------------------------------------------------------

with safe_page_section("Order event timeline"):
    if not order_timeline.empty:
        order_timeline["event_timestamp"] = pd.to_datetime(order_timeline["event_timestamp"], errors="coerce")
        fig = px.scatter(
            order_timeline.sort_values("event_timestamp"),
            x="event_timestamp",
            y="event_type",
            color="source_table",
            hover_data=["title", "detail"],
            title=f"Timeline for order {order_id}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline events found for this order.")

# ------------------------------------------------------------------
# Linked docs & emails
# ------------------------------------------------------------------

left, right = st.columns(2)
with left, safe_page_section("Linked documents"):
    doc_cols = [c for c in (
        "document_type", "source_document_id", "event_timestamp",
        "document_amount", "revenue_amount", "cost_amount", "refund_amount",
        "delivery_lag_days", "content_summary",
    ) if c in order_docs.columns]
    st.dataframe(
        order_docs[doc_cols].sort_values("event_timestamp") if not order_docs.empty else pd.DataFrame(),
        use_container_width=True,
    )
with right, safe_page_section("Linked emails"):
    email_cols = [c for c in (
        "timestamp", "from_email", "to_email", "subject",
        "message_scope", "is_complaint_like", "body_preview",
    ) if c in order_emails.columns]
    st.dataframe(
        order_emails[email_cols].sort_values("timestamp") if not order_emails.empty else pd.DataFrame(),
        use_container_width=True,
    )
