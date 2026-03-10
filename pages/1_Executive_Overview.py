"""Executive Overview — topline KPIs, monthly trends, heatmap, and Sankey."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from archive_analytics.dashboard import (
    get_risk_scores,
    get_table,
    model_artifacts_ready,
    require_dashboard_assets,
    safe_page_section,
)

st.set_page_config(page_title="Executive Overview", layout="wide", page_icon="📈")
st.title("Executive Overview")

require_dashboard_assets()

orders = get_table("fact_order")
customers = get_table("dim_customer")

orders["order_created_at"] = pd.to_datetime(orders["order_created_at"], errors="coerce")
orders["order_month"] = orders["order_created_at"].dt.to_period("M").astype(str)

# ------------------------------------------------------------------
# Top KPIs
# ------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders", f"{len(orders):,}")
k2.metric("Customers", f"{customers['customer_id_clean'].nunique():,}")
k3.metric("Delayed order rate", f"{orders['will_be_delayed'].mean():.1%}")
k4.metric("Complaint order rate", f"{orders['will_generate_complaint'].mean():.1%}")

# ------------------------------------------------------------------
# Monthly trends
# ------------------------------------------------------------------

with safe_page_section("Monthly order and issue trends"):
    agg_cols = {
        "orders": ("order_id", "count"),
        "delayed": ("will_be_delayed", "sum"),
        "complaint": ("will_generate_complaint", "sum"),
        "credit": ("will_generate_credit_memo", "sum"),
    }
    for candidate in ("revenue_total", "invoice_total"):
        if candidate in orders.columns:
            agg_cols["revenue"] = (candidate, "sum")
            break

    summary = orders.groupby("order_month").agg(**agg_cols).reset_index()
    fig_trend = px.line(
        summary,
        x="order_month",
        y=["orders", "delayed", "complaint", "credit"],
        markers=True,
    )
    fig_trend.update_layout(legend_title_text="Metric", xaxis_title="Month", yaxis_title="Count")
    st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------------------------------
# Heatmap — normalised per column to avoid mixed-scale distortion
# ------------------------------------------------------------------

with safe_page_section("Top issue customers — normalised heatmap"):
    heat_cols = ["issue_rate", "order_count", "email_count"]
    for c in ("credit_total", "refund_total"):
        if c in customers.columns:
            heat_cols.append(c)
            break

    issue_customers = customers.nlargest(20, "issue_rate").copy()
    heatmap_raw = issue_customers[["customer_label"] + heat_cols].set_index("customer_label")
    heatmap_norm = (heatmap_raw - heatmap_raw.mean()) / (heatmap_raw.std() + 1e-9)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_norm.values,
            x=list(heatmap_norm.columns),
            y=list(heatmap_norm.index),
            colorscale="RdYlBu_r",
            text=heatmap_raw.values.round(2),
            texttemplate="%{text}",
            hovertemplate="Customer: %{y}<br>Metric: %{x}<br>Raw: %{text}<br>Z: %{z:.2f}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(title="Top-20 issue customers (z-score normalised)")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------------------------------------------------------
# Sankey — data-driven
# ------------------------------------------------------------------

with safe_page_section("Order lifecycle Sankey"):
    n_total = len(orders)
    n_with_docs = int((orders["document_count"] > 0).sum())
    n_with_email = int(orders["has_linked_email"].sum())
    n_delayed = int(orders["will_be_delayed"].sum())
    n_complaint = int(orders["will_generate_complaint"].sum())
    n_credit = int(orders["will_generate_credit_memo"].sum())
    n_clean = max(n_total - n_delayed - n_complaint, 0)

    # Node index: 0=All orders, 1=With docs, 2=With emails, 3=Delayed,
    #             4=Complaint,  5=Credit memo, 6=No issue
    labels = ["All orders", "With docs", "With emails", "Delayed", "Complaint", "Credit memo", "No issue"]
    node_colors = [
        "#4F8BF9",  # 0 All orders    – blue
        "#4F8BF9",  # 1 With docs     – blue
        "#F9A84F",  # 2 With emails   – amber
        "#E05C5C",  # 3 Delayed       – red
        "#E05C5C",  # 4 Complaint     – red
        "#50C878",  # 5 Credit memo   – green
        "#6BCB77",  # 6 No issue      – light green
    ]
    link_colors = [
        "rgba(79,139,249,0.35)",   # 0→1 All → With docs
        "rgba(249,168,79,0.35)",   # 0→2 All → With emails
        "rgba(224,92,92,0.45)",    # 0→3 All → Delayed
        "rgba(107,203,119,0.35)",  # 0→6 All → No issue
        "rgba(80,200,120,0.45)",   # 3→5 Delayed → Credit memo
        "rgba(80,200,120,0.45)",   # 4→5 Complaint → Credit memo
    ]
    fig_sankey = go.Figure(
        go.Sankey(
            node=dict(
                label=labels,
                color=node_colors,
                pad=15,
                thickness=20,
                line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            ),
            link=dict(
                source=[0, 0, 0, 0, 3, 4],
                target=[1, 2, 3, 6, 5, 5],
                value=[n_with_docs, n_with_email, n_delayed, n_clean, n_credit, n_credit],
                color=link_colors,
            ),
        )
    )
    fig_sankey.update_layout(title_text="Order lifecycle (data-driven)", font_color="#E0E0E0")
    st.plotly_chart(fig_sankey, use_container_width=True)

# ------------------------------------------------------------------
# Risk scatter
# ------------------------------------------------------------------

with safe_page_section("Top modeled risk orders"):
    if model_artifacts_ready():
        risk_scores = get_risk_scores()
        merged = orders.merge(risk_scores, on=["order_id", "customer_id_clean", "order_created_at"], how="left")
        score_cols = [c for c in merged.columns if c.endswith("_score")]
        if score_cols:
            merged["composite_risk"] = merged[score_cols].fillna(0).mean(axis=1)
            fig_scatter = px.scatter(
                merged.nlargest(1000, "composite_risk"),
                x="linked_email_count",
                y="max_delivery_lag_days",
                color="composite_risk",
                hover_data=["order_id", "customer_id_clean"],
                title="Top modeled risk orders",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Train models to see risk scores.")
    else:
        st.info("Train models to see risk scores.")

# ------------------------------------------------------------------
# Table
# ------------------------------------------------------------------

st.subheader("Top customers by issue rate")
display_cols = ["customer_label", "order_count", "email_count", "issue_rate"]
for c in ("invoice_total", "revenue_total", "credit_total", "refund_total"):
    if c in customers.columns:
        display_cols.append(c)
st.dataframe(customers[display_cols].head(25), use_container_width=True)
