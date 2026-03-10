"""Customer 360 — deep-dive into a single customer's orders, emails, and timeline."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from archive_analytics.dashboard import ensure_project_assets, get_table, safe_page_section

st.set_page_config(page_title="Customer 360", layout="wide", page_icon="🧭")
st.title("Customer 360")

ensure_project_assets(train_models=False)
customers = get_table("dim_customer")
orders = get_table("fact_order")
timeline = get_table("fact_event_timeline")
emails = get_table("fact_email")

# ------------------------------------------------------------------
# Customer selector (vectorised label construction)
# ------------------------------------------------------------------

customer_options = customers.dropna(subset=["customer_id_clean"]).copy()
customer_options["label"] = (
    customer_options["customer_label"].astype(str)
    + " ("
    + customer_options["customer_id_clean"].astype(str)
    + ")"
)
selected_label = st.selectbox("Customer", customer_options["label"].tolist())
selected_customer = customer_options.loc[customer_options["label"] == selected_label].iloc[0]
customer_id = selected_customer["customer_id_clean"]

customer_orders = orders[orders["customer_id_clean"].astype("string") == str(customer_id)].copy()
customer_emails = emails[emails["customer_id_clean"].astype("string") == str(customer_id)].copy()
customer_timeline = timeline[timeline["customer_id_clean"].astype("string") == str(customer_id)].copy()
customer_orders["order_created_at"] = pd.to_datetime(customer_orders["order_created_at"], errors="coerce")
customer_orders["month"] = customer_orders["order_created_at"].dt.to_period("M").astype(str)

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("Orders", f"{int(selected_customer['order_count']):,}")
m2.metric("Emails", f"{int(selected_customer['email_count']):,}")
m3.metric("Issue rate", f"{selected_customer['issue_rate']:.1%}")
credit_col = "credit_total" if "credit_total" in selected_customer.index else "refund_total"
m4.metric("Credit total", f"{selected_customer.get(credit_col, 0):.2f}")

# ------------------------------------------------------------------
# Monthly activity
# ------------------------------------------------------------------

with safe_page_section("Customer monthly activity"):
    trend = customer_orders.groupby("month").agg(
        orders=("order_id", "count"),
        delayed=("will_be_delayed", "sum"),
        complaint=("will_generate_complaint", "sum"),
    ).reset_index()
    if not trend.empty:
        fig_trend = px.bar(
            trend, x="month", y=["orders", "delayed", "complaint"],
            barmode="group",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------------------------------
# Orders table
# ------------------------------------------------------------------

with safe_page_section("Top orders"):
    order_cols = [
        "order_id", "order_created_at", "order_line_count", "document_count",
        "linked_email_count", "max_delivery_lag_days",
        "will_generate_complaint", "will_generate_credit_memo",
    ]
    order_cols = [c for c in order_cols if c in customer_orders.columns]
    st.dataframe(
        customer_orders[order_cols].sort_values("order_created_at", ascending=False).head(50),
        use_container_width=True,
    )

# ------------------------------------------------------------------
# Communication mix
# ------------------------------------------------------------------

with safe_page_section("Communication mix"):
    email_mix = customer_emails.groupby("message_scope").size().rename("count").reset_index()
    if not email_mix.empty:
        fig_mix = px.pie(email_mix, values="count", names="message_scope")
        st.plotly_chart(fig_mix, use_container_width=True)

# ------------------------------------------------------------------
# Timeline
# ------------------------------------------------------------------

with safe_page_section("Customer timeline"):
    if not customer_timeline.empty:
        customer_timeline["event_timestamp"] = pd.to_datetime(customer_timeline["event_timestamp"], errors="coerce")
        fig_timeline = px.scatter(
            customer_timeline.sort_values("event_timestamp"),
            x="event_timestamp",
            y="event_type",
            color="source_table",
            hover_data=["order_id", "title", "detail"],
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No timeline events were linked to this customer.")
