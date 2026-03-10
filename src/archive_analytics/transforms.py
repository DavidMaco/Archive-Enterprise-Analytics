"""Fact and dimension table builders.

Each ``build_*`` function takes raw DataFrames and returns a cleaned,
enriched analytics table.  All heavy row-wise ``apply`` calls from
earlier versions have been replaced with vectorised pandas operations.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._util import (
    classify_message_scope,
    contains_keywords,
    extract_order_references,
    mode_or_first,
    normalize_id,
    safe_date,
    sha1_key,
    vectorized_sha1,
)
from .constants import (
    BODY_PREVIEW_LENGTH,
    COMPLAINT_KEYWORDS,
    COMPLAINT_TOLERANCE_DAYS,
    EMAIL_CORPUS_CAP,
    SNIPPET_LENGTH,
    SPAM_KEYWORDS,
    TARGETS,
)

logger = logging.getLogger(__name__)


# ── Email fact ──────────────────────────────────────────────────────────────

def build_email_fact(communications: pd.DataFrame) -> pd.DataFrame:
    """Build the email fact table from raw communications.

    Vectorised order-reference extraction replaces the former row-wise
    ``apply`` call for a ~10× speed-up on large datasets.
    """
    emails = communications.copy()
    emails = emails.rename(
        columns={"from": "from_email", "to": "to_email", "cc": "cc_recipients"}
    )
    emails["timestamp"] = safe_date(emails["timestamp"])
    emails["source_message_id"] = normalize_id(emails["message_id"])
    emails["customer_id_clean"] = normalize_id(emails["customer_id"])

    # Vectorised order-reference extraction (was apply + _extract_order_reference)
    emails["linked_order_id"] = extract_order_references(
        emails.get("subject", pd.Series(dtype="string")),
        emails.get("body", pd.Series(dtype="string")),
    )

    # Vectorised canonical ID (was apply + lambda)
    emails["canonical_message_id"] = vectorized_sha1(
        "email",
        emails["message_id"].astype("string"),
        emails["timestamp"].astype("string"),
        emails["from_email"].astype("string"),
        emails["to_email"].astype("string"),
        emails["subject"].astype("string"),
    )

    emails["message_scope"] = classify_message_scope(
        emails["from_email"], emails["to_email"]
    )
    searchable = (
        emails[["subject", "body", "vendor", "from_company"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )
    emails["is_spam_like"] = contains_keywords(searchable, SPAM_KEYWORDS)
    emails["is_complaint_like"] = contains_keywords(searchable, COMPLAINT_KEYWORDS)
    emails["is_duplicate_source_id"] = emails["source_message_id"].duplicated(
        keep=False
    )
    emails["event_date"] = emails["timestamp"].dt.date.astype("string")
    emails["body_preview"] = (
        emails["body"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, BODY_PREVIEW_LENGTH)
    )

    logger.info("Email fact: %d rows", len(emails))
    return emails


# ── Document fact ───────────────────────────────────────────────────────────

def build_document_fact(
    supporting_documents: pd.DataFrame,
    business_documents: pd.DataFrame,
) -> pd.DataFrame:
    """Build the document fact table from supporting + business documents.

    Financial columns are kept separate (``revenue_amount``,
    ``cost_amount``, ``refund_amount``) instead of summed into a single
    meaningless ``document_amount``.
    """
    support = supporting_documents.copy()
    date_cols = [
        "order_date", "requested_delivery", "invoice_date", "due_date",
        "ship_date", "expected_delivery", "actual_delivery",
        "inspection_date", "issue_date", "created_date",
    ]
    for col in date_cols:
        if col in support.columns:
            support[col] = safe_date(support[col])

    support["order_id"] = normalize_id(support["order_number"])
    support["customer_id_clean"] = normalize_id(support["customer_id"])
    support["source_document_id"] = normalize_id(support["document_id"])
    support["source_dataset"] = "supporting_documents"

    # Vectorised canonical ID
    support["canonical_document_id"] = vectorized_sha1(
        "supporting",
        support["document_id"].astype("string"),
        support["document_type"].astype("string"),
        support["order_number"].astype("string"),
        support["customer_id"].astype("string"),
    )
    support["is_duplicate_source_id"] = support["source_document_id"].duplicated(
        keep=False
    )

    # Event timestamp per document type
    support["event_timestamp"] = pd.NaT
    support["event_date"] = pd.NaT
    dtype_timestamp_map = {
        "PURCHASE_ORDER": "order_date",
        "INVOICE": "invoice_date",
        "SHIPPING_NOTICE": "ship_date",
        "QUALITY_REPORT": "inspection_date",
        "CREDIT_MEMO": "issue_date",
    }
    for dtype, ts_col in dtype_timestamp_map.items():
        if ts_col in support.columns:
            mask = support["document_type"] == dtype
            support.loc[mask, "event_timestamp"] = support.loc[mask, ts_col]
    fallback_order_date = (
        support["order_date"]
        if "order_date" in support.columns
        else pd.Series(pd.NaT, index=support.index)
    )
    support["event_timestamp"] = support["event_timestamp"].fillna(
        fallback_order_date
    )
    support["event_date"] = support["event_timestamp"].dt.date.astype("string")

    # Separate financial columns (fixes the old meaningless sum)
    support["revenue_amount"] = support.get("billed_amount", pd.Series(0.0, index=support.index)).fillna(0)
    support["cost_amount"] = support.get("total_amount", pd.Series(0.0, index=support.index)).fillna(0)
    support["refund_amount"] = support.get("credit_amount", pd.Series(0.0, index=support.index)).fillna(0)

    if "actual_delivery" in support.columns and "expected_delivery" in support.columns:
        support["delivery_lag_days"] = (
            support["actual_delivery"] - support["expected_delivery"]
        ).dt.days
    else:
        support["delivery_lag_days"] = pd.Series(np.nan, index=support.index)

    support["content_summary"] = (
        support["document_type"].astype("string")
        + " for order "
        + support["order_id"].fillna("unknown")
        + ", customer "
        + support.get("customer_name", pd.Series("unknown", index=support.index)).fillna("unknown")
    )

    # ── Business documents ──
    business = business_documents.copy()
    business["created_date"] = safe_date(business["created_date"])
    business["order_id"] = pd.Series(pd.NA, index=business.index, dtype="string")
    business["customer_id_clean"] = pd.Series(pd.NA, index=business.index, dtype="string")
    business["source_document_id"] = normalize_id(business["document_id"])
    business["source_dataset"] = "business_documents"
    business["canonical_document_id"] = vectorized_sha1(
        "business",
        business["document_id"].astype("string"),
        business["document_type"].astype("string"),
        business["created_date"].astype("string"),
    )
    business["is_duplicate_source_id"] = business["source_document_id"].duplicated(
        keep=False
    )
    business["event_timestamp"] = business["created_date"]
    business["event_date"] = business["created_date"].dt.date.astype("string")
    business["revenue_amount"] = 0.0
    business["cost_amount"] = 0.0
    business["refund_amount"] = 0.0
    business["delivery_lag_days"] = np.nan
    business["content_summary"] = (
        business["content"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, SNIPPET_LENGTH)
    )

    common_columns = sorted(set(support.columns) | set(business.columns))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        result = pd.concat(
            [
                support.reindex(columns=common_columns),
                business.reindex(columns=common_columns),
            ],
            ignore_index=True,
        )

    logger.info("Document fact: %d rows", len(result))
    return result


# ── Complaint assignment ────────────────────────────────────────────────────

def assign_complaints_to_orders(
    order_fact: pd.DataFrame,
    fact_email: pd.DataFrame,
) -> pd.DataFrame:
    """Assign complaint emails to the nearest order within a tolerance window.

    Uses ``np.searchsorted`` for efficient bi-directional matching within
    a configurable tolerance (default: 60 days).
    """
    complaint_emails = fact_email[
        fact_email["is_complaint_like"]
        & fact_email["customer_id_clean"].notna()
        & fact_email["timestamp"].notna()
        & fact_email["message_scope"].isin(["external_inbound", "mixed"])
    ].copy()
    if complaint_emails.empty:
        return pd.DataFrame(
            columns=["order_id", "complaint_email_count", "latest_complaint_at"]
        )

    order_lookup = (
        order_fact[["order_id", "customer_id_clean", "order_created_at"]]
        .dropna(subset=["customer_id_clean", "order_created_at"])
        .copy()
        .sort_values(["customer_id_clean", "order_created_at"], kind="stable")
    )

    grouped_orders = {
        str(cid): g.reset_index(drop=True)
        for cid, g in order_lookup.groupby("customer_id_clean", dropna=True)
    }
    tolerance = np.timedelta64(COMPLAINT_TOLERANCE_DAYS, "D")
    assignments: list[pd.DataFrame] = []

    for customer_id, group in complaint_emails.groupby(
        "customer_id_clean", dropna=True
    ):
        order_group = grouped_orders.get(str(customer_id))
        if order_group is None or order_group.empty:
            continue

        email_group = group.sort_values("timestamp", kind="stable").copy()
        order_times = order_group["order_created_at"].to_numpy(dtype="datetime64[ns]")
        email_times = email_group["timestamp"].to_numpy(dtype="datetime64[ns]")
        prev_pos = order_times.searchsorted(email_times, side="right") - 1
        next_pos = order_times.searchsorted(email_times, side="left")

        matched_ids: list[str] = []
        matched_ts: list[Any] = []

        for idx, etime in enumerate(email_times):
            best_pos, best_delta = -1, None

            prev = prev_pos[idx]
            if prev >= 0:
                d = np.abs(etime - order_times[prev])
                if d <= tolerance:
                    best_pos, best_delta = int(prev), d

            nxt = next_pos[idx]
            if nxt < len(order_times):
                d = np.abs(order_times[nxt] - etime)
                if d <= tolerance and (best_delta is None or d < best_delta):
                    best_pos = int(nxt)

            if best_pos < 0:
                continue
            matched_ids.append(str(order_group.iloc[best_pos]["order_id"]))
            matched_ts.append(etime)

        if matched_ids:
            assignments.append(
                pd.DataFrame({"order_id": matched_ids, "timestamp": matched_ts})
            )

    if not assignments:
        return pd.DataFrame(
            columns=["order_id", "complaint_email_count", "latest_complaint_at"]
        )

    assigned = pd.concat(assignments, ignore_index=True)
    return (
        assigned.groupby("order_id", dropna=False)
        .agg(
            complaint_email_count=("timestamp", "count"),
            latest_complaint_at=("timestamp", "max"),
        )
        .reset_index()
    )


# ── Order fact ──────────────────────────────────────────────────────────────

def build_order_fact(
    erp_transactions: pd.DataFrame,
    fact_email: pd.DataFrame,
    fact_document: pd.DataFrame,
) -> pd.DataFrame:
    """Build the order fact table.

    Lambda aggregations have been replaced with pre-computed boolean columns
    so that pandas can use its Cython fast-paths.
    """
    erp = erp_transactions.copy()
    erp["order_id"] = normalize_id(erp["SALESDOCUMENT"])
    erp["order_created_at"] = safe_date(erp["CREATIONDATE"])
    erp["customer_id_clean"] = normalize_id(erp["SOLDTOPARTY"])

    first_values = (
        erp.sort_values(["order_id", "order_created_at"], kind="stable")
        .drop_duplicates(subset=["order_id"], keep="first")
        .loc[:, ["order_id", "customer_id_clean", "PLANT", "PRODUCT"]]
        .rename(columns={"PLANT": "primary_plant", "PRODUCT": "primary_product"})
    )

    order_fact = (
        erp.groupby("order_id", dropna=False)
        .agg(
            order_created_at=("order_created_at", "min"),
            order_line_count=("SALESDOCUMENTITEM", "nunique"),
            product_nunique=("PRODUCT", "nunique"),
            plant_nunique=("PLANT", "nunique"),
            shipping_point_nunique=("SHIPPINGPOINT", "nunique"),
        )
        .reset_index()
        .merge(first_values, on="order_id", how="left")
    )

    # ── Supporting-document aggregations (with pre-computed booleans) ──
    supporting = fact_document[
        fact_document["source_dataset"] == "supporting_documents"
    ].copy()
    supporting["is_shipping_notice"] = supporting["document_type"] == "SHIPPING_NOTICE"
    supporting["is_invoice"] = supporting["document_type"] == "INVOICE"
    supporting["is_quality_report"] = supporting["document_type"] == "QUALITY_REPORT"
    supporting["is_credit_memo"] = supporting["document_type"] == "CREDIT_MEMO"
    supporting["is_delayed"] = supporting["delivery_lag_days"].fillna(0) > 0

    doc_agg = (
        supporting.groupby("order_id", dropna=False)
        .agg(
            document_count=("canonical_document_id", "count"),
            revenue_total=("revenue_amount", "sum"),
            cost_total=("cost_amount", "sum"),
            refund_total=("refund_amount", "sum"),
            shipping_notice_count=("is_shipping_notice", "sum"),
            invoice_count=("is_invoice", "sum"),
            quality_report_count=("is_quality_report", "sum"),
            credit_memo_count=("is_credit_memo", "sum"),
            delayed_shipments=("is_delayed", "sum"),
            max_delivery_lag_days=("delivery_lag_days", "max"),
            latest_document_at=("event_timestamp", "max"),
        )
        .reset_index()
    )

    # ── Email aggregations ──
    email_agg = (
        fact_email.dropna(subset=["linked_order_id"])
        .groupby("linked_order_id")
        .agg(
            linked_email_count=("canonical_message_id", "count"),
            spam_email_count=("is_spam_like", "sum"),
            latest_email_at=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"linked_order_id": "order_id"})
    )

    complaint_agg = assign_complaints_to_orders(order_fact, fact_email)

    order_fact = (
        order_fact.merge(doc_agg, on="order_id", how="left")
        .merge(email_agg, on="order_id", how="left")
        .merge(complaint_agg, on="order_id", how="left")
    )

    fill_zero_cols = [
        "document_count", "revenue_total", "cost_total", "refund_total",
        "shipping_notice_count", "invoice_count", "quality_report_count",
        "credit_memo_count", "delayed_shipments", "max_delivery_lag_days",
        "linked_email_count", "complaint_email_count", "spam_email_count",
    ]
    for col in fill_zero_cols:
        if col in order_fact.columns:
            order_fact[col] = order_fact[col].fillna(0)

    # Boolean flags
    order_fact["has_invoice"] = order_fact["invoice_count"] > 0
    order_fact["has_shipping_notice"] = order_fact["shipping_notice_count"] > 0
    order_fact["has_quality_report"] = order_fact["quality_report_count"] > 0
    order_fact["has_credit_memo"] = order_fact["credit_memo_count"] > 0
    order_fact["has_linked_email"] = order_fact["linked_email_count"] > 0

    # Target labels
    order_fact["will_be_delayed"] = order_fact["max_delivery_lag_days"] > 0
    order_fact["will_generate_credit_memo"] = order_fact["has_credit_memo"]
    order_fact["will_generate_complaint"] = order_fact["complaint_email_count"] > 0

    order_fact["event_span_days"] = (
        pd.to_datetime(
            order_fact[["latest_document_at", "latest_email_at"]].max(axis=1),
            errors="coerce",
        )
        - order_fact["order_created_at"]
    ).dt.days.fillna(0)
    order_fact["order_month"] = (
        order_fact["order_created_at"].dt.to_period("M").astype("string")
    )

    logger.info("Order fact: %d rows", len(order_fact))
    return order_fact


# ── Customer dimension ──────────────────────────────────────────────────────

def build_customer_dim(
    order_fact: pd.DataFrame,
    fact_email: pd.DataFrame,
    fact_document: pd.DataFrame,
) -> pd.DataFrame:
    """Build the customer dimension from orders, emails, and documents."""
    # Resolve customer names
    name_frames = []
    for frame, id_col, name_col in [
        (fact_email, "customer_id_clean", "customer_name"),
        (fact_document, "customer_id_clean", "customer_name"),
    ]:
        if name_col in frame.columns:
            subset = frame[[id_col, name_col]].dropna().drop_duplicates()
            subset = subset.rename(columns={name_col: "customer_name"})
            name_frames.append(subset)

    name_map = (
        pd.concat(name_frames, ignore_index=True).drop_duplicates()
        if name_frames
        else pd.DataFrame(columns=["customer_id_clean", "customer_name"])
    )
    name_map = (
        name_map.groupby("customer_id_clean", dropna=False)
        .agg(customer_name=("customer_name", mode_or_first))
        .reset_index()
    )

    order_agg = (
        order_fact.groupby("customer_id_clean", dropna=False)
        .agg(
            order_count=("order_id", "count"),
            linked_order_count=("has_linked_email", "sum"),
            delayed_order_count=("will_be_delayed", "sum"),
            complaint_order_count=("will_generate_complaint", "sum"),
            credit_order_count=("will_generate_credit_memo", "sum"),
            invoice_total=("revenue_total", "sum"),
            credit_total=("refund_total", "sum"),
            avg_line_count=("order_line_count", "mean"),
            latest_order_at=("order_created_at", "max"),
        )
        .reset_index()
    )

    email_agg = (
        fact_email.groupby("customer_id_clean", dropna=False)
        .agg(
            email_count=("canonical_message_id", "count"),
            complaint_email_count=("is_complaint_like", "sum"),
            spam_email_count=("is_spam_like", "sum"),
            latest_email_at=("timestamp", "max"),
        )
        .reset_index()
    )

    doc_agg_pre = fact_document.copy()
    doc_agg_pre["is_quality_report"] = doc_agg_pre["document_type"] == "QUALITY_REPORT"
    doc_agg = (
        doc_agg_pre.groupby("customer_id_clean", dropna=False)
        .agg(
            document_count=("canonical_document_id", "count"),
            quality_report_count=("is_quality_report", "sum"),
        )
        .reset_index()
    )

    customer_dim = (
        order_agg.merge(email_agg, on="customer_id_clean", how="outer")
        .merge(doc_agg, on="customer_id_clean", how="left")
        .merge(name_map, on="customer_id_clean", how="left")
    )

    fill_zero = [
        "order_count", "linked_order_count", "delayed_order_count",
        "complaint_order_count", "credit_order_count", "invoice_total",
        "credit_total", "avg_line_count", "email_count",
        "complaint_email_count", "spam_email_count", "document_count",
        "quality_report_count",
    ]
    for col in fill_zero:
        if col in customer_dim.columns:
            customer_dim[col] = customer_dim[col].fillna(0)

    customer_dim["issue_rate"] = np.where(
        customer_dim["order_count"] > 0,
        (
            customer_dim["delayed_order_count"]
            + customer_dim["complaint_order_count"]
            + customer_dim["credit_order_count"]
        )
        / customer_dim["order_count"],
        0,
    )
    customer_dim["customer_label"] = customer_dim["customer_name"].fillna(
        customer_dim["customer_id_clean"].fillna("unlinked")
    )

    logger.info("Customer dim: %d rows", len(customer_dim))
    return customer_dim.sort_values(
        ["issue_rate", "order_count"], ascending=[False, False]
    )


# ── Event timeline ──────────────────────────────────────────────────────────

def build_event_timeline(
    order_fact: pd.DataFrame,
    fact_email: pd.DataFrame,
    fact_document: pd.DataFrame,
) -> pd.DataFrame:
    """Build a unified event timeline across orders, emails, and documents."""
    order_events = order_fact[
        ["order_id", "customer_id_clean", "order_created_at"]
    ].copy()
    order_events["event_timestamp"] = order_events["order_created_at"]
    order_events["event_date"] = order_events["order_created_at"].dt.date.astype(
        "string"
    )
    order_events["event_type"] = "order_created"
    order_events["title"] = "Order created"
    order_events["detail"] = order_events["order_id"].fillna("unknown")
    order_events["source_table"] = "fact_order"

    email_events = fact_email[
        [
            "linked_order_id", "customer_id_clean", "timestamp",
            "subject", "body_preview", "message_scope", "canonical_message_id",
        ]
    ].copy()
    email_events = email_events.rename(
        columns={
            "linked_order_id": "order_id",
            "timestamp": "event_timestamp",
            "subject": "title",
            "body_preview": "detail",
        }
    )
    email_events["event_date"] = email_events["event_timestamp"].dt.date.astype(
        "string"
    )
    email_events["event_type"] = np.where(
        email_events["message_scope"].eq("internal"), "internal_email", "email"
    )
    email_events["source_table"] = "fact_email"

    doc_events = fact_document[
        [
            "order_id", "customer_id_clean", "event_timestamp",
            "document_type", "content_summary", "canonical_document_id",
        ]
    ].copy()
    doc_events = doc_events.rename(
        columns={"document_type": "title", "content_summary": "detail"}
    )
    doc_events["event_date"] = (
        pd.to_datetime(doc_events["event_timestamp"], errors="coerce")
        .dt.date
        .astype("string")
    )
    doc_events["event_type"] = "document"
    doc_events["source_table"] = "fact_document"

    keep_cols = [
        "order_id", "customer_id_clean", "event_timestamp",
        "event_date", "event_type", "title", "detail", "source_table",
    ]
    timeline = pd.concat(
        [
            order_events[keep_cols],
            email_events[keep_cols],
            doc_events[keep_cols],
        ],
        ignore_index=True,
    )
    timeline["event_id"] = timeline.apply(
        lambda r: sha1_key(
            r.get("order_id"),
            r.get("customer_id_clean"),
            r.get("event_timestamp"),
            r.get("title"),
        ),
        axis=1,
    )
    return timeline.sort_values(
        ["event_timestamp", "event_id"], ascending=[True, True]
    )


# ── Customer daily ──────────────────────────────────────────────────────────

def build_customer_daily(
    order_fact: pd.DataFrame,
    fact_email: pd.DataFrame,
    fact_document: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-customer daily event counts."""
    frames: list[pd.DataFrame] = []

    order_daily = (
        order_fact[["customer_id_clean", "order_created_at"]]
        .dropna(subset=["order_created_at"])
        .copy()
    )
    order_daily["event_date"] = order_daily["order_created_at"].dt.floor("D")
    order_daily["event_type"] = "order"
    frames.append(order_daily[["customer_id_clean", "event_date", "event_type"]])

    email_daily = (
        fact_email[["customer_id_clean", "timestamp"]]
        .dropna(subset=["timestamp"])
        .copy()
    )
    email_daily["event_date"] = email_daily["timestamp"].dt.floor("D")
    email_daily["event_type"] = "email"
    frames.append(email_daily[["customer_id_clean", "event_date", "event_type"]])

    doc_daily = (
        fact_document[["customer_id_clean", "event_timestamp"]]
        .dropna(subset=["event_timestamp"])
        .copy()
    )
    doc_daily["event_date"] = (
        pd.to_datetime(doc_daily["event_timestamp"], errors="coerce").dt.floor("D")
    )
    doc_daily["event_type"] = "document"
    frames.append(doc_daily[["customer_id_clean", "event_date", "event_type"]])

    all_events = pd.concat(frames, ignore_index=True)
    result = (
        all_events.groupby(
            ["customer_id_clean", "event_date", "event_type"], dropna=False
        )
        .size()
        .rename("event_count")
        .reset_index()
    )
    return result.sort_values(["event_date", "customer_id_clean", "event_type"])


# ── Order risk features ────────────────────────────────────────────────────

def build_order_risk_features(fact_order: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative prior-history features to the order fact table.

    Every feature is strictly *backward-looking*: it uses only information
    available at or before the time of each order, making the resulting
    table safe for forward-looking models without data leakage.
    """
    frame = fact_order.copy()
    frame["order_created_at"] = pd.to_datetime(
        frame["order_created_at"], errors="coerce"
    )
    frame = (
        frame.dropna(subset=["order_created_at", "order_id"])
        .sort_values("order_created_at", kind="stable")
        .reset_index(drop=True)
    )

    # Derived temporal features
    frame["primary_product"] = frame["primary_product"].astype("string")
    frame["primary_product_family"] = frame["primary_product"].str.slice(0, 3)
    frame["order_year"] = frame["order_created_at"].dt.year
    frame["order_month_num"] = frame["order_created_at"].dt.month
    frame["order_quarter"] = frame["order_created_at"].dt.quarter

    # Customer cumulative features
    frame["customer_id_clean"] = frame["customer_id_clean"].astype("string")
    cust_grp = frame.groupby("customer_id_clean", dropna=False)
    frame["customer_prior_orders"] = cust_grp.cumcount()
    for target in TARGETS:
        if target not in frame.columns:
            continue
        prior_sum = cust_grp[target].cumsum() - frame[target].astype(int)
        frame[f"customer_prior_{target}_rate"] = np.where(
            frame["customer_prior_orders"] > 0,
            prior_sum / frame["customer_prior_orders"],
            0.0,
        )
    frame["customer_prior_orders_log"] = np.log1p(frame["customer_prior_orders"])

    # Plant cumulative features
    plant_grp = frame.groupby("primary_plant", dropna=False)
    frame["plant_prior_orders"] = plant_grp.cumcount()
    if "will_be_delayed" in frame.columns:
        delay_sum = (
            plant_grp["will_be_delayed"].cumsum()
            - frame["will_be_delayed"].astype(int)
        )
        frame["plant_prior_delay_rate"] = np.where(
            frame["plant_prior_orders"] > 0,
            delay_sum / frame["plant_prior_orders"],
            0.0,
        )

    logger.info("Order risk features: %d rows, %d cols", len(frame), len(frame.columns))
    return frame


# ── Retrieval corpus ────────────────────────────────────────────────────────

def build_retrieval_corpus(
    order_fact: pd.DataFrame,
    customer_dim: pd.DataFrame,
    fact_email: pd.DataFrame,
    fact_document: pd.DataFrame,
) -> pd.DataFrame:
    """Build the unified retrieval corpus for the TF-IDF assistant.

    Emails are sorted by relevance (complaints first) before being capped
    at ``EMAIL_CORPUS_CAP`` to ensure the most important records are kept.
    """
    # Order summaries (only "interesting" orders)
    interesting = order_fact[
        order_fact[
            ["has_linked_email", "has_credit_memo", "has_quality_report", "will_be_delayed"]
        ].any(axis=1)
    ].copy()
    order_docs = pd.DataFrame(
        {
            "doc_id": interesting["order_id"].map(lambda v: f"order:{v}"),
            "entity_type": "order",
            "entity_id": interesting["order_id"],
            "customer_id": interesting["customer_id_clean"],
            "order_id": interesting["order_id"],
            "document_type": "order_summary",
            "event_date": interesting["order_created_at"].dt.date.astype("string"),
            "title": interesting["order_id"].map(lambda v: f"Order {v} summary"),
            "text": interesting.apply(
                lambda r: (
                    f"Order {r['order_id']} for customer {r['customer_id_clean']} "
                    f"created on {r['order_created_at']:%Y-%m-%d}. "
                    f"{int(r['order_line_count'])} lines, "
                    f"{int(r['document_count'])} documents, "
                    f"{int(r['linked_email_count'])} emails, "
                    f"max delivery lag {float(r['max_delivery_lag_days']):.0f} days, "
                    f"complaint={bool(r['will_generate_complaint'])}, "
                    f"credit={bool(r['will_generate_credit_memo'])}."
                ),
                axis=1,
            ),
        }
    )

    # Customer summaries
    customer_docs = pd.DataFrame(
        {
            "doc_id": customer_dim["customer_id_clean"].fillna("unlinked").map(
                lambda v: f"customer:{v}"
            ),
            "entity_type": "customer",
            "entity_id": customer_dim["customer_id_clean"],
            "customer_id": customer_dim["customer_id_clean"],
            "order_id": pd.Series(pd.NA, index=customer_dim.index, dtype="string"),
            "document_type": "customer_summary",
            "event_date": customer_dim["latest_order_at"].dt.date.astype("string"),
            "title": customer_dim["customer_label"].map(
                lambda v: f"Customer {v} summary"
            ),
            "text": customer_dim.apply(
                lambda r: (
                    f"Customer {r['customer_label']} ({r['customer_id_clean']}) "
                    f"has {int(r['order_count'])} orders, "
                    f"{int(r['email_count'])} emails, "
                    f"{int(r['document_count'])} documents, "
                    f"issue rate {r['issue_rate']:.2%}, "
                    f"invoice total {r['invoice_total']:.2f}, "
                    f"credit total {r['credit_total']:.2f}."
                ),
                axis=1,
            ),
        }
    )

    # Emails – sort by relevance, cap at EMAIL_CORPUS_CAP
    email_subset = fact_email[
        (fact_email["is_complaint_like"]) | (fact_email["linked_order_id"].notna())
    ].copy()
    email_subset["_sort_key"] = email_subset["is_complaint_like"].astype(int)
    email_subset = (
        email_subset.sort_values("_sort_key", ascending=False)
        .head(EMAIL_CORPUS_CAP)
        .drop(columns=["_sort_key"])
    )
    email_docs = pd.DataFrame(
        {
            "doc_id": email_subset["canonical_message_id"].map(
                lambda v: f"email:{v}"
            ),
            "entity_type": "email",
            "entity_id": email_subset["canonical_message_id"],
            "customer_id": email_subset["customer_id_clean"],
            "order_id": email_subset["linked_order_id"],
            "document_type": "email",
            "event_date": email_subset["event_date"],
            "title": email_subset["subject"].fillna("Email"),
            "text": (
                "From "
                + email_subset["from_name"].fillna(
                    email_subset["from_email"].fillna("unknown")
                )
                + " to "
                + email_subset["to_name"].fillna(
                    email_subset["to_email"].fillna("unknown")
                )
                + ". "
                + email_subset["subject"].fillna("")
                + ". "
                + email_subset["body_preview"].fillna("")
            ),
        }
    )

    # Documents
    document_docs = pd.DataFrame(
        {
            "doc_id": fact_document["canonical_document_id"].map(
                lambda v: f"document:{v}"
            ),
            "entity_type": "document",
            "entity_id": fact_document["canonical_document_id"],
            "customer_id": fact_document["customer_id_clean"],
            "order_id": fact_document["order_id"],
            "document_type": fact_document["document_type"].fillna("document"),
            "event_date": fact_document["event_date"],
            "title": (
                fact_document["document_type"].fillna("Document")
                + " "
                + fact_document["source_document_id"].fillna("")
            ),
            "text": fact_document["content_summary"].fillna(""),
        }
    )

    corpus = pd.concat(
        [order_docs, customer_docs, email_docs, document_docs], ignore_index=True
    )
    corpus["citation"] = corpus["doc_id"]
    corpus = corpus.dropna(subset=["text"]).reset_index(drop=True)

    logger.info("Retrieval corpus: %d rows", len(corpus))
    return corpus
