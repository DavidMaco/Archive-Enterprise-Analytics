"""Risk Scoring — calibrated model outputs, feature importance, and model governance."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from archive_analytics.dashboard import (
    get_model_metrics_cached,
    get_risk_scores,
    get_table,
    require_dashboard_assets,
    safe_page_section,
)

st.set_page_config(page_title="Risk Scoring", layout="wide", page_icon="⚠️")
st.title("Risk Scoring")

require_dashboard_assets(require_models=True)

# ------------------------------------------------------------------
# Model metrics and governance
# ------------------------------------------------------------------

with safe_page_section("Model metrics"):
    metrics = get_model_metrics_cached()

    # Show key governance info
    governance_cols = st.columns(3)
    with governance_cols[0]:
        st.caption("Run ID")
        st.code(metrics.get("run_id", "—"), language=None)
    governance_cols[1].metric("Train rows", f"{metrics.get('train_rows', 0):,}")
    governance_cols[2].metric(
        "Targets trained",
        sum(1 for t in metrics.get("targets", {}).values() if t.get("status") == "trained"),
    )

    # Per-target details
    targets = metrics.get("targets", {})
    if targets:
        selected_target = st.selectbox("Target", list(targets.keys()))
        tp = targets.get(selected_target, {})

        # Metrics summary
        test_metrics = tp.get("calibrated", {})
        cv_metrics = tp.get("cross_validation", {})
        if test_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AUC-ROC", f"{(test_metrics.get('roc_auc') or 0):.3f}")
            m2.metric("Average Precision", f"{(test_metrics.get('average_precision') or 0):.3f}")
            m3.metric("F1 score", f"{test_metrics.get('f1', 0):.3f}")
            m4.metric("Optimal threshold", f"{tp.get('optimal_threshold', 0.5):.3f}")

        if cv_metrics and cv_metrics.get("n_folds", 0) > 0:
            cv_mean = cv_metrics.get("mean", 0)
            cv_std = cv_metrics.get("std", 0)
            # mean / std may be a scalar float or a dict keyed by metric name
            if isinstance(cv_mean, dict):
                cv_mean = cv_mean.get("roc_auc", 0)
            if isinstance(cv_std, dict):
                cv_std = cv_std.get("roc_auc", 0)
            st.caption(
                f"CV ({cv_metrics['n_folds']} folds): "
                f"AUC {float(cv_mean):.3f} ± {float(cv_std):.3f}"
            )

        # Feature importance
        features = pd.DataFrame(tp.get("top_features", []))
        if not features.empty:
            fig_features = px.bar(
                features.head(15).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Feature importance — {selected_target} ({tp.get('selected_model', '?')})",
            )
            st.plotly_chart(fig_features, use_container_width=True)

        # Model details expander
        with st.expander("Full model details"):
            st.json(tp)
    else:
        st.info("No trained models found. Run `python -m archive_analytics train`.")

# ------------------------------------------------------------------
# Risk landscape
# ------------------------------------------------------------------

with safe_page_section("Composite risk landscape"):
    risk_scores = get_risk_scores()
    orders = get_table("fact_order")
    scored = orders.merge(risk_scores, on=["order_id", "customer_id_clean", "order_created_at"], how="inner")

    score_cols = [c for c in scored.columns if c.endswith("_score")]
    if score_cols:
        scored["composite_risk"] = scored[score_cols].mean(axis=1)
        # Plotly requires non-negative marker sizes; clip early deliveries to 0
        scored["_lag_size"] = scored["max_delivery_lag_days"].clip(lower=0)
        fig_risk = px.scatter(
            scored.nlargest(2000, "composite_risk"),
            x="linked_email_count",
            y="document_count",
            color="composite_risk",
            size="_lag_size",
            hover_data=["order_id", "customer_id_clean"],
            title="Composite risk landscape",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.subheader("Top risk orders")
        display_cols = [
            "order_id", "customer_id_clean",
            "linked_email_count", "document_count", "max_delivery_lag_days",
        ] + score_cols + ["composite_risk"]
        display_cols = [c for c in display_cols if c in scored.columns]
        st.dataframe(
            scored[display_cols].sort_values("composite_risk", ascending=False).head(100),
            use_container_width=True,
        )
    else:
        st.info("Train models to see risk scores.")
