"""Comprehensive test suite for archive-enterprise-analytics.

Tests are organised by module:
- constants / util — pure functions, no I/O
- transforms — fact/dimension builders with synthetic DataFrames
- data — facade and I/O guards
- retrieval — scoring and deduplication
- dashboard — Streamlit helper utilities
- settings — configuration and logging
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ======================================================================
# Fixtures — synthetic DataFrames matching actual raw archive schemas
# ======================================================================

_BASE_DATE = pd.Timestamp("2023-06-01")


@pytest.fixture()
def raw_communications() -> pd.DataFrame:
    """Minimal communications table matching the real parquet schema.

    Columns expected by ``build_email_fact``:
    message_id, from, to, cc, subject, body, timestamp, customer_id,
    vendor, from_company.
    """
    rows: list[dict[str, Any]] = []
    for i in range(28):
        dt = _BASE_DATE + timedelta(days=i)
        cid = f"CUST{(i % 4):03d}"
        rows.append(
            {
                "message_id": f"<msg{i:04d}@example.com>",
                "from": f"user{i}@uberjugaad.com"
                if i % 3 != 0
                else f"vendor{i}@external.com",
                "to": f"vendor{i % 3}@external.com"
                if i % 3 != 0
                else f"user{i}@uberjugaad.com",
                "cc": None,
                "subject": f"Order 00000100{i:02d} follow-up"
                if i < 10
                else f"Status update {i}",
                "timestamp": dt.isoformat(),
                "body": "We have a delivery issue, please expedite."
                if i % 5 == 0
                else "Normal correspondence regarding the project.",
                "customer_id": cid,
                "vendor": f"Vendor{i % 3}",
                "from_company": "UberJugaad"
                if i % 3 != 0
                else f"ExtCo{i % 3}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def raw_supporting_documents() -> pd.DataFrame:
    """Minimal supporting-documents table matching the real parquet schema.

    Columns expected by ``build_document_fact`` (supporting path):
    document_id, document_type, order_number, customer_id, customer_name,
    order_date, billed_amount, total_amount, credit_amount,
    actual_delivery, expected_delivery, invoice_date, ship_date,
    inspection_date, issue_date.
    """
    doc_types = [
        "PURCHASE_ORDER",
        "INVOICE",
        "SHIPPING_NOTICE",
        "QUALITY_REPORT",
        "CREDIT_MEMO",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(30):
        dt = _BASE_DATE + timedelta(days=i % 30)
        oid = f"00000100{(i % 10):02d}"
        cid = f"CUST{(i % 4):03d}"
        rows.append(
            {
                "document_id": f"SDOC{i:05d}",
                "document_type": doc_types[i % len(doc_types)],
                "order_number": oid,
                "customer_id": cid,
                "customer_name": f"Customer {i % 4}",
                "order_date": dt.isoformat(),
                "billed_amount": round(100 + i * 7.5, 2),
                "total_amount": round(80 + i * 5.0, 2),
                "credit_amount": round(10 + i * 1.0, 2)
                if doc_types[i % len(doc_types)] == "CREDIT_MEMO"
                else 0.0,
                "actual_delivery": (dt + timedelta(days=5 + i % 3)).isoformat()
                if doc_types[i % len(doc_types)] == "SHIPPING_NOTICE"
                else None,
                "expected_delivery": (dt + timedelta(days=3)).isoformat()
                if doc_types[i % len(doc_types)] == "SHIPPING_NOTICE"
                else None,
                "invoice_date": dt.isoformat()
                if doc_types[i % len(doc_types)] == "INVOICE"
                else None,
                "ship_date": dt.isoformat()
                if doc_types[i % len(doc_types)] == "SHIPPING_NOTICE"
                else None,
                "inspection_date": dt.isoformat()
                if doc_types[i % len(doc_types)] == "QUALITY_REPORT"
                else None,
                "issue_date": dt.isoformat()
                if doc_types[i % len(doc_types)] == "CREDIT_MEMO"
                else None,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def raw_business_documents() -> pd.DataFrame:
    """Minimal business-documents table matching the real parquet schema.

    Columns expected by ``build_document_fact`` (business path):
    document_id, document_type, created_date, content.
    """
    rows: list[dict[str, Any]] = []
    for i in range(10):
        dt = _BASE_DATE + timedelta(days=i * 3)
        rows.append(
            {
                "document_id": f"BDOC{i:05d}",
                "document_type": "INTERNAL_MEMO" if i % 2 == 0 else "POLICY",
                "created_date": dt.isoformat(),
                "content": f"Internal memo content for business document {i}.",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def raw_erp_transactions() -> pd.DataFrame:
    """Minimal ERP transactions table matching the real parquet schema.

    Columns expected by ``build_order_fact``:
    SALESDOCUMENT, SALESDOCUMENTITEM, CREATIONDATE, SOLDTOPARTY,
    PRODUCT, PLANT, SHIPPINGPOINT.
    """
    rows: list[dict[str, Any]] = []
    for i in range(20):
        dt = _BASE_DATE + timedelta(days=(i // 2) * 3)
        oid = f"00000100{(i // 2):02d}"
        cid = f"CUST{(i % 4):03d}"
        rows.append(
            {
                "SALESDOCUMENT": oid,
                "SALESDOCUMENTITEM": f"{(i % 2) + 1:04d}",
                "CREATIONDATE": dt.isoformat(),
                "SOLDTOPARTY": cid,
                "PRODUCT": f"PROD{(i % 5):03d}",
                "PLANT": f"PL{(i % 3):02d}",
                "SHIPPINGPOINT": f"SP{(i % 2):02d}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def raw_tables(
    raw_communications: pd.DataFrame,
    raw_supporting_documents: pd.DataFrame,
    raw_business_documents: pd.DataFrame,
    raw_erp_transactions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Dict matching the output of ``ingestion.load_raw_tables()``."""
    return {
        "communications": raw_communications,
        "supporting_documents": raw_supporting_documents,
        "business_documents": raw_business_documents,
        "erp_transactions": raw_erp_transactions,
        # sqlite_emails key for quality report
        "sqlite_emails": raw_communications.head(10),
    }


# ======================================================================
# 1. constants
# ======================================================================


class TestConstants:
    def test_order_pattern_matches_10_digit(self) -> None:
        from archive_analytics.constants import ORDER_PATTERN

        m = ORDER_PATTERN.search("Issue with order 0002456789 in shipment")
        assert m is not None
        assert m.group(0) == "0002456789"

    def test_order_pattern_rejects_shorter(self) -> None:
        from archive_analytics.constants import ORDER_PATTERN

        assert ORDER_PATTERN.search("Short 123456789 id") is None

    def test_targets_list_nonempty(self) -> None:
        from archive_analytics.constants import TARGETS

        assert len(TARGETS) >= 3

    def test_predictive_features_no_overlap(self) -> None:
        from archive_analytics.constants import (
            PREDICTIVE_CATEGORICAL_FEATURES,
            PREDICTIVE_NUMERIC_FEATURES,
        )

        overlap = set(PREDICTIVE_NUMERIC_FEATURES) & set(PREDICTIVE_CATEGORICAL_FEATURES)
        assert overlap == set(), f"Features appear in both lists: {overlap}"


# ======================================================================
# 2. _util
# ======================================================================


class TestUtil:
    def test_sha1_key_deterministic(self) -> None:
        from archive_analytics._util import sha1_key

        assert sha1_key("a", 1, "b") == sha1_key("a", 1, "b")

    def test_sha1_key_different_inputs(self) -> None:
        from archive_analytics._util import sha1_key

        assert sha1_key("a") != sha1_key("b")

    def test_normalize_id(self) -> None:
        from archive_analytics._util import normalize_id

        s = pd.Series(["  ABC ", "  def  123 ", None])
        result = normalize_id(s)
        # normalize_id strips whitespace but does NOT lowercase
        assert result.iloc[0] == "ABC"
        assert result.iloc[1] == "def  123"
        assert pd.isna(result.iloc[2])

    def test_safe_date(self) -> None:
        from archive_analytics._util import safe_date

        s = pd.Series(["2023-01-01", "not-a-date", None])
        result = safe_date(s)
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_contains_keywords(self) -> None:
        from archive_analytics._util import contains_keywords

        s = pd.Series(["urgent complaint", "normal email", "DELAYED shipment"])
        result = contains_keywords(s, ["urgent", "delayed"])
        assert bool(result.iloc[0]) is True
        assert bool(result.iloc[1]) is False
        assert bool(result.iloc[2]) is True

    def test_classify_message_scope(self) -> None:
        from archive_analytics._util import classify_message_scope

        from_col = pd.Series([
            "alice@uberjugaad.com",   # internal → internal
            "vendor@external.com",    # external → internal
            "alice@uberjugaad.com",   # internal → external
            "vendor@external.com",    # external → external
        ])
        to_col = pd.Series([
            "bob@uberjugaad.com",     # internal
            "alice@uberjugaad.com",   # internal
            "vendor@external.com",    # external
            "other@external.com",     # external
        ])
        result = classify_message_scope(from_col, to_col)
        assert result.iloc[0] == "internal"
        assert result.iloc[1] == "external_inbound"
        assert result.iloc[2] == "external_outbound"
        assert result.iloc[3] == "mixed"

    def test_extract_order_references(self) -> None:
        from archive_analytics._util import extract_order_references

        subj = pd.Series(["Order 0002456789 issue", "No order here"])
        body = pd.Series(["See 0001234567", "Nothing"])
        result = extract_order_references(subj, body)
        # First row: subject has a 10-digit, body also has one — str.extract gets first
        assert str(result.iloc[0]) == "0002456789"
        # Second row: no 10-digit ID → pd.NA
        assert pd.isna(result.iloc[1])

    def test_mode_or_first(self) -> None:
        from archive_analytics._util import mode_or_first

        assert mode_or_first(pd.Series(["a", "b", "a"])) == "a"
        assert mode_or_first(pd.Series([])) is None


# ======================================================================
# 3. transforms — email fact
# ======================================================================


class TestEmailFact:
    def test_output_columns(self, raw_communications: pd.DataFrame) -> None:
        from archive_analytics.transforms import build_email_fact

        result = build_email_fact(raw_communications)
        required_cols = {
            "canonical_message_id",
            "source_message_id",
            "from_email",
            "to_email",
            "subject",
            "timestamp",
            "body_preview",
            "customer_id_clean",
            "message_scope",
            "is_complaint_like",
            "is_spam_like",
            "linked_order_id",
            "is_duplicate_source_id",
            "event_date",
        }
        assert required_cols.issubset(set(result.columns)), (
            f"Missing: {required_cols - set(result.columns)}"
        )

    def test_row_count_matches(self, raw_communications: pd.DataFrame) -> None:
        from archive_analytics.transforms import build_email_fact

        result = build_email_fact(raw_communications)
        assert len(result) == len(raw_communications)

    def test_deduplication_flag(self, raw_communications: pd.DataFrame) -> None:
        from archive_analytics.transforms import build_email_fact

        duped = pd.concat(
            [raw_communications, raw_communications.head(1)], ignore_index=True
        )
        result = build_email_fact(duped)
        assert result["is_duplicate_source_id"].any()


# ======================================================================
# 4. transforms — document fact
# ======================================================================


class TestDocumentFact:
    def test_output_columns(
        self,
        raw_supporting_documents: pd.DataFrame,
        raw_business_documents: pd.DataFrame,
    ) -> None:
        from archive_analytics.transforms import build_document_fact

        result = build_document_fact(raw_supporting_documents, raw_business_documents)
        required_cols = {
            "canonical_document_id",
            "source_document_id",
            "document_type",
            "order_id",
            "customer_id_clean",
            "event_timestamp",
            "revenue_amount",
            "cost_amount",
            "refund_amount",
            "content_summary",
            "source_dataset",
            "is_duplicate_source_id",
        }
        assert required_cols.issubset(set(result.columns)), (
            f"Missing: {required_cols - set(result.columns)}"
        )

    def test_row_count(
        self,
        raw_supporting_documents: pd.DataFrame,
        raw_business_documents: pd.DataFrame,
    ) -> None:
        from archive_analytics.transforms import build_document_fact

        result = build_document_fact(raw_supporting_documents, raw_business_documents)
        assert len(result) == len(raw_supporting_documents) + len(
            raw_business_documents
        )


# ======================================================================
# 5. transforms — complaint assignment
# ======================================================================


class TestComplaintAssignment:
    def test_complaint_within_tolerance(self) -> None:
        from archive_analytics.transforms import assign_complaints_to_orders

        orders = pd.DataFrame(
            {
                "customer_id_clean": ["c1"],
                "order_created_at": pd.to_datetime(["2023-06-05"]),
                "order_id": ["o1"],
            }
        )
        emails = pd.DataFrame(
            {
                "customer_id_clean": ["c1", "c1"],
                "timestamp": pd.to_datetime(["2023-06-01", "2023-06-10"]),
                "is_complaint_like": [True, False],
                "message_scope": ["external_inbound", "internal"],
                "canonical_message_id": ["e1", "e2"],
            }
        )
        result = assign_complaints_to_orders(orders, emails)
        assert not result.empty
        assert "o1" in result["order_id"].values

    def test_no_link_beyond_tolerance(self) -> None:
        from archive_analytics.transforms import assign_complaints_to_orders

        orders = pd.DataFrame(
            {
                "customer_id_clean": ["c1"],
                "order_created_at": pd.to_datetime(["2023-06-01"]),
                "order_id": ["o1"],
            }
        )
        emails = pd.DataFrame(
            {
                "customer_id_clean": ["c1"],
                "timestamp": pd.to_datetime(["2022-01-01"]),
                "is_complaint_like": [True],
                "message_scope": ["external_inbound"],
                "canonical_message_id": ["e1"],
            }
        )
        result = assign_complaints_to_orders(orders, emails)
        assert result.empty


# ======================================================================
# 6. transforms — order fact
# ======================================================================


class TestOrderFact:
    def test_order_fact_shape(
        self,
        raw_erp_transactions: pd.DataFrame,
        raw_supporting_documents: pd.DataFrame,
        raw_business_documents: pd.DataFrame,
        raw_communications: pd.DataFrame,
    ) -> None:
        from archive_analytics.transforms import (
            build_document_fact,
            build_email_fact,
            build_order_fact,
        )

        emails = build_email_fact(raw_communications)
        docs = build_document_fact(raw_supporting_documents, raw_business_documents)
        result = build_order_fact(raw_erp_transactions, emails, docs)
        assert len(result) > 0
        assert "order_id" in result.columns
        assert "customer_id_clean" in result.columns

    def test_boolean_target_columns(
        self,
        raw_erp_transactions: pd.DataFrame,
        raw_supporting_documents: pd.DataFrame,
        raw_business_documents: pd.DataFrame,
        raw_communications: pd.DataFrame,
    ) -> None:
        from archive_analytics.transforms import (
            build_document_fact,
            build_email_fact,
            build_order_fact,
        )

        emails = build_email_fact(raw_communications)
        docs = build_document_fact(raw_supporting_documents, raw_business_documents)
        result = build_order_fact(raw_erp_transactions, emails, docs)
        for col in (
            "will_be_delayed",
            "will_generate_complaint",
            "will_generate_credit_memo",
        ):
            assert col in result.columns


# ======================================================================
# 7. transforms — order risk features
# ======================================================================


class TestOrderRiskFeatures:
    def test_log_transform_added(
        self,
        raw_erp_transactions: pd.DataFrame,
        raw_supporting_documents: pd.DataFrame,
        raw_business_documents: pd.DataFrame,
        raw_communications: pd.DataFrame,
    ) -> None:
        from archive_analytics.transforms import (
            build_document_fact,
            build_email_fact,
            build_order_fact,
            build_order_risk_features,
        )

        emails = build_email_fact(raw_communications)
        docs = build_document_fact(raw_supporting_documents, raw_business_documents)
        orders = build_order_fact(raw_erp_transactions, emails, docs)
        risk = build_order_risk_features(orders)
        assert "customer_prior_orders_log" in risk.columns
        assert (risk["customer_prior_orders_log"] >= 0).all()


# ======================================================================
# 8. data — unknown-name guard
# ======================================================================


class TestDataGuards:
    def test_unknown_parquet_table_rejected(self) -> None:
        from archive_analytics.data import load_processed_table

        with (
            patch("archive_analytics.data.build_processed_assets"),
            pytest.raises(KeyError, match="Unknown parquet table"),
        ):
            load_processed_table("../etc/passwd")

    def test_unknown_json_asset_rejected(self) -> None:
        from archive_analytics.data import load_json_asset

        with (
            patch("archive_analytics.data.build_processed_assets"),
            pytest.raises(KeyError, match="Unknown JSON asset"),
        ):
            load_json_asset("../../etc/shadow")


# ======================================================================
# 9. retrieval — cached TF-IDF and dedup
# ======================================================================


class TestRetrieval:
    @pytest.fixture()
    def sample_corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "citation": [f"cit_{i}" for i in range(20)],
                "title": [f"Title {i}" for i in range(20)],
                "text": [
                    "delivery delay and complaint about order",
                    "invoice for normal order processing",
                    "credit memo issued for damaged goods",
                    "normal email about project status",
                    "urgent complaint about late delivery",
                ]
                * 4,
                "entity_type": ["email", "document", "document", "email", "email"]
                * 4,
                "order_id": [f"000000000{i % 5}" for i in range(20)],
                "customer_id": [f"C{i % 3}" for i in range(20)],
            }
        )

    def test_mmr_rerank_returns_correct_count(self) -> None:
        from scipy import sparse

        from archive_analytics.retrieval import _mmr_rerank

        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        matrix = sparse.random(5, 10, density=0.5, format="csr")
        candidates = np.array([0, 1, 2, 3, 4])
        result = _mmr_rerank(scores, matrix, candidates, top_k=3)
        assert len(result) == 3

    def test_deduplicate_entity(self) -> None:
        from archive_analytics.retrieval import _deduplicate_entity

        df = pd.DataFrame(
            {
                "entity_type": ["email", "email", "document"],
                "order_id": ["o1", "o1", "o2"],
                "score": [0.9, 0.5, 0.8],
            }
        )
        result = _deduplicate_entity(df)
        assert len(result) == 2

    def test_invalidate_cache(self) -> None:
        from archive_analytics.retrieval import _INDEX_CACHE, invalidate_cache

        _INDEX_CACHE["test"] = ("a", "b", "c")  # type: ignore[assignment]
        invalidate_cache()
        assert len(_INDEX_CACHE) == 0


# ======================================================================
# 10. quality report
# ======================================================================


class TestQualityReport:
    def test_report_structure(
        self,
        raw_tables: dict[str, pd.DataFrame],
        raw_erp_transactions: pd.DataFrame,
        raw_supporting_documents: pd.DataFrame,
        raw_business_documents: pd.DataFrame,
        raw_communications: pd.DataFrame,
    ) -> None:
        from archive_analytics.quality import build_quality_report
        from archive_analytics.transforms import (
            build_document_fact,
            build_email_fact,
            build_order_fact,
        )

        emails = build_email_fact(raw_communications)
        docs = build_document_fact(raw_supporting_documents, raw_business_documents)
        orders = build_order_fact(raw_erp_transactions, emails, docs)
        processed = {
            "fact_email": emails,
            "fact_document": docs,
            "fact_order": orders,
        }
        report = build_quality_report(raw_tables, processed)
        assert isinstance(report, dict)
        assert "raw_counts" in report
        assert "processed_counts" in report
        assert "email_duplicate_source_ids" in report


# ======================================================================
# 11. dashboard helpers (no Streamlit dependency)
# ======================================================================


class TestDashboardHelpers:
    def test_metric_delta_normal(self) -> None:
        from archive_analytics.dashboard import metric_delta

        assert metric_delta(110, 100) == "10.0%"

    def test_metric_delta_zero_baseline(self) -> None:
        from archive_analytics.dashboard import metric_delta

        assert metric_delta(50, 0) == "n/a"

    def test_metric_delta_negative(self) -> None:
        from archive_analytics.dashboard import metric_delta

        assert metric_delta(90, 100) == "-10.0%"


# ======================================================================
# 12. settings
# ======================================================================


class TestSettings:
    def test_get_config_returns_appconfig(self) -> None:
        from archive_analytics.settings import AppConfig, get_config

        cfg = get_config()
        assert isinstance(cfg, AppConfig)

    def test_config_directories_exist_after_ensure(self) -> None:
        from archive_analytics.settings import AppConfig

        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = AppConfig(
                project_root=tmp,
                raw_data_dir=tmp / "raw",
                processed_dir=tmp / "processed",
                models_dir=tmp / "models",
                reports_dir=tmp / "reports",
            )
            cfg.ensure_directories()
            assert (tmp / "processed").is_dir()
            assert (tmp / "models").is_dir()
            assert (tmp / "reports").is_dir()
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_raises_for_missing_raw(self) -> None:
        from archive_analytics.settings import AppConfig

        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = AppConfig(
                project_root=tmp,
                raw_data_dir=tmp / "nonexistent",
            )
            with pytest.raises(FileNotFoundError):
                cfg.validate()
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_configure_logging_idempotent(self) -> None:
        from archive_analytics.settings import configure_logging

        configure_logging()
        configure_logging()  # should not add duplicate handlers
