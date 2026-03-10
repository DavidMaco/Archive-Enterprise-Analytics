"""Generate synthetic demo data and build all processed artefacts.

Run from the project root (package must be installed, e.g. ``pip install -e .``):

    python scripts/seed_demo.py

What it does
------------
1. Writes synthetic raw parquet files + SQLite stub to ``data/raw/``.
2. Calls ``build_processed_assets(force=True)`` to produce all tables in
   ``data/processed/`` using the real transforms pipeline.
3. Derives synthetic risk scores from processed features and writes them to
   ``data/models/order_risk_scores.parquet``.
4. Writes a realistic ``data/models/model_metrics.json``.

Commit the generated ``data/processed/`` and ``data/models/`` directories
(after removing them from ``.gitignore``) so Streamlit Cloud can load the
dashboard without needing the real archive raw files.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "models"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reprodicible RNG
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Demo parameters
# ---------------------------------------------------------------------------

N_CUSTOMERS = 25
N_ORDERS = 300          # erp rows (one row per line item; ~4 items/order)
N_ERP_ROWS = 1_200      # multiple line items per order
N_EMAILS = 380
N_SUPPORT_DOCS = 480
N_BIZ_DOCS = 60

BASE_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2024-06-30")
DAYS = int((END_DATE - BASE_DATE).days)

PLANTS = ["PL01", "PL02", "PL03", "PL04", "PL05"]
PRODUCTS = [f"PROD{i:03d}" for i in range(1, 31)]
SHIP_POINTS = ["SP01", "SP02", "SP03"]
DOC_TYPES = ["PURCHASE_ORDER", "INVOICE", "SHIPPING_NOTICE", "QUALITY_REPORT", "CREDIT_MEMO"]
DOC_WEIGHTS = [0.25, 0.30, 0.25, 0.10, 0.10]
BIZ_TYPES = ["CONTRACT", "POLICY", "SLA", "REPORT", "AGREEMENT"]
SCOPES = ["external_inbound", "external_outbound", "internal", "mixed"]
SCOPE_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

COMPANY_NAMES = [
    "Apex Industries Ltd", "Global Supplies Co", "Delta Manufacturing",
    "Summit Materials AG", "Pacific Logistics Inc", "Metro Components Ltd",
    "Unified Procurement Corp", "Atlantic Resources", "Cascade Industries",
    "Northern Trade Corp", "Eastern Supplies Ltd", "Southern Components",
    "Western Materials Co", "Central Trade Group", "Premier Industries",
    "Crown Components Ltd", "Alliance Supplies", "Peak Manufacturing",
    "Horizon Resources Inc", "Stellar Logistics",
    "Vertex Industries", "Quantum Components AG", "Titan Materials",
    "Apex Logistics Ltd", "Globe Trade Systems",
]

CUSTOMER_IDS = [f"CUST{i:06d}" for i in range(1, N_CUSTOMERS + 1)]

SPAM_SUBJECTS = [
    "Blockchain opportunity for your supply chain",
    "Free assessment of your mainframe",
    "Quantum upgrade special offer — click here",
    "Unsubscribe from blockchain newsletter",
]
COMPLAINT_SUBJECTS = [
    "Urgent: delivery issue with order {oid}",
    "Complaint — damaged shipment {oid}",
    "Delayed delivery — order {oid} escalation",
    "Credit request for shortage on order {oid}",
    "Action plan required: late delivery {oid}",
]
NORMAL_SUBJECTS = [
    "Order {oid} confirmation",
    "Invoice attached — {oid}",
    "Shipping notice for {oid}",
    "Follow-up on order {oid}",
    "Quote request",
    "Supplier onboarding documents",
    "Weekly report",
    "Meeting follow-up",
]

COMPLAINT_BODY = (
    "We are writing to escalate an urgent delivery issue. "
    "Order {oid} has been significantly delayed and has caused a shortage. "
    "We require immediate resolution and credit for the inconvenience. "
    "Please provide a root cause analysis and action plan."
)
NORMAL_BODY = (
    "Please find attached the documentation for order {oid}. "
    "Let us know if you need any additional information."
)
SPAM_BODY = (
    "Special offer exclusively for your team! "
    "Click here to unsubscribe or to get your free assessment today."
)

VENDOR_DOMAINS = [
    "supplier-a.com", "vendor-b.net", "parts-inc.com",
    "logistics-co.org", "trade-group.net",
]
INTERNAL_DOMAIN = "enterprise.internal"


def rand_ts(n: int, base: pd.Timestamp = BASE_DATE, days: int = DAYS) -> pd.DatetimeIndex:
    seconds = RNG.integers(0, days * 24 * 3600, size=n)
    return pd.DatetimeIndex([base + pd.Timedelta(seconds=int(s)) for s in seconds])


def order_id_str(i: int) -> str:
    return f"{4000000000 + i:010d}"


def sha1(*parts: object) -> str:
    return hashlib.sha1("::".join(str(p) for p in parts).encode()).hexdigest()


# ── 1. ERP transactions ──────────────────────────────────────────────────────

print("Generating ERP transactions …")

# ~4 line items per order → N_ORDERS unique order IDs
orders_per_customer = N_ORDERS // N_CUSTOMERS
erp_rows = []
for ord_idx in range(N_ORDERS):
    cust_idx = ord_idx % N_CUSTOMERS
    cid = CUSTOMER_IDS[cust_idx]
    oid = order_id_str(ord_idx)
    created = BASE_DATE + pd.Timedelta(days=int(RNG.integers(0, DAYS)))
    n_items = int(RNG.integers(1, 8))
    plant = PLANTS[int(RNG.integers(0, len(PLANTS)))]
    sp = SHIP_POINTS[int(RNG.integers(0, len(SHIP_POINTS)))]
    for item_no in range(1, n_items + 1):
        erp_rows.append({
            "SALESDOCUMENT": oid,
            "CREATIONDATE": created.date(),
            "PLANT": plant,
            "PRODUCT": PRODUCTS[int(RNG.integers(0, len(PRODUCTS)))],
            "SOLDTOPARTY": cid,
            "SHIPPINGPOINT": sp,
            "SALESDOCUMENTITEM": f"{item_no:04d}",
        })

erp_df = pd.DataFrame(erp_rows)
erp_df.to_parquet(RAW_DIR / "erp_transactions.parquet", index=False)
print(f"  ERP transactions: {len(erp_df):,} rows spanning {N_ORDERS} orders")

# Collect unique order IDs for later use
all_order_ids = sorted(erp_df["SALESDOCUMENT"].unique())
order_to_customer = dict(zip(erp_df["SALESDOCUMENT"], erp_df["SOLDTOPARTY"]))


# ── 2. Communications (emails) ───────────────────────────────────────────────

print("Generating communications …")

email_rows = []
for i in range(N_EMAILS):
    cust_idx = i % N_CUSTOMERS
    cid = CUSTOMER_IDS[cust_idx]
    cname = COMPANY_NAMES[cust_idx]
    ts = BASE_DATE + pd.Timedelta(seconds=int(RNG.integers(0, DAYS * 86400)))
    mid = f"MSG{i:06d}@archive.internal"

    # Decide email type (15% complaint, 5% spam, 80% normal)
    r = float(RNG.random())
    linked_oid: str | None = None
    if r < 0.15 and all_order_ids:
        # complaint email
        linked_oid = all_order_ids[int(RNG.integers(0, len(all_order_ids)))]
        subject = COMPLAINT_SUBJECTS[int(RNG.integers(0, len(COMPLAINT_SUBJECTS)))].format(oid=linked_oid)
        body = COMPLAINT_BODY.format(oid=linked_oid)
        scope = "external_inbound"
        from_email = f"procurement@{COMPANY_NAMES[cust_idx].lower().replace(' ', '').replace(',', '')[:12]}.com"
        to_email = f"orders@{INTERNAL_DOMAIN}"
    elif r < 0.20:
        # spam
        subject = SPAM_SUBJECTS[int(RNG.integers(0, len(SPAM_SUBJECTS)))]
        body = SPAM_BODY
        scope = "external_inbound"
        from_email = f"noreply@{VENDOR_DOMAINS[int(RNG.integers(0, len(VENDOR_DOMAINS)))]}"
        to_email = f"info@{INTERNAL_DOMAIN}"
        cid = None  # spam not linked to a customer
    else:
        # normal operational email
        if all_order_ids and float(RNG.random()) < 0.60:
            linked_oid = all_order_ids[int(RNG.integers(0, len(all_order_ids)))]
            subject = NORMAL_SUBJECTS[int(RNG.integers(0, 4))].format(oid=linked_oid)
        else:
            subject = NORMAL_SUBJECTS[int(RNG.integers(4, len(NORMAL_SUBJECTS)))]
        body = NORMAL_BODY.format(oid=linked_oid or "N/A")
        scope_idx = int(RNG.choice(len(SCOPES), p=[0.40, 0.30, 0.20, 0.10]))
        scope = SCOPES[scope_idx]
        if scope in ("external_inbound", "mixed"):
            from_email = f"contact@vendor{int(RNG.integers(1, 6))}.com"
            to_email = f"orders@{INTERNAL_DOMAIN}"
        else:
            from_email = f"user{int(RNG.integers(1, 50))}@{INTERNAL_DOMAIN}"
            to_email = f"user{int(RNG.integers(1, 50))}@{INTERNAL_DOMAIN}"

    from_name = cname if cid else "Unknown Sender"
    to_name = "Enterprise Orders Team" if "orders@" in to_email else "Internal User"

    email_rows.append({
        "message_id": mid,
        "timestamp": ts,
        "from": from_email,
        "to": to_email,
        "cc": None,
        "subject": subject,
        "body": body,
        "customer_id": cid,
        "vendor": None,
        "from_company": cname if cid else None,
        "from_name": from_name,
        "to_name": to_name,
        "customer_name": cname if cid else None,
    })

comms_df = pd.DataFrame(email_rows)
comms_df.to_parquet(RAW_DIR / "all_communications.parquet", index=False)
print(f"  Communications: {len(comms_df):,} rows")


# ── 3. Supporting documents ──────────────────────────────────────────────────

print("Generating supporting documents …")

doc_types = RNG.choice(DOC_TYPES, size=N_SUPPORT_DOCS, p=DOC_WEIGHTS)
support_rows = []
for i in range(N_SUPPORT_DOCS):
    oid = all_order_ids[int(RNG.integers(0, len(all_order_ids)))]
    cid = order_to_customer.get(oid, CUSTOMER_IDS[0])
    cname = COMPANY_NAMES[CUSTOMER_IDS.index(cid)]
    dtype = str(doc_types[i])
    did = f"DOC{i:06d}"
    base_ts = BASE_DATE + pd.Timedelta(days=int(RNG.integers(0, DAYS)))
    lag = int(RNG.integers(-5, 45)) if dtype == "SHIPPING_NOTICE" else 0
    expected_del = base_ts + pd.Timedelta(days=14)
    actual_del = expected_del + pd.Timedelta(days=lag)
    billed = float(RNG.uniform(1_000, 150_000)) if dtype in ("INVOICE", "PURCHASE_ORDER") else 0.0
    credit = float(RNG.uniform(500, 15_000)) if dtype == "CREDIT_MEMO" else 0.0

    support_rows.append({
        "document_id": did,
        "document_type": dtype,
        "order_number": oid,
        "customer_id": cid,
        "customer_name": cname,
        "billed_amount": round(billed, 2),
        "total_amount": round(billed * float(RNG.uniform(0.7, 1.0)), 2),
        "credit_amount": round(credit, 2),
        "order_date": base_ts,
        "invoice_date": base_ts + pd.Timedelta(days=int(RNG.integers(5, 20))),
        "ship_date": base_ts + pd.Timedelta(days=int(RNG.integers(3, 15))),
        "inspection_date": base_ts + pd.Timedelta(days=int(RNG.integers(10, 30))),
        "issue_date": base_ts + pd.Timedelta(days=int(RNG.integers(15, 40))),
        "created_date": base_ts,
        "expected_delivery": expected_del,
        "actual_delivery": actual_del,
    })

support_df = pd.DataFrame(support_rows)
support_df.to_parquet(RAW_DIR / "supporting_documents.parquet", index=False)
print(f"  Supporting documents: {len(support_df):,} rows")


# ── 4. Business documents ────────────────────────────────────────────────────

print("Generating business documents …")

biz_rows = []
for i in range(N_BIZ_DOCS):
    dtype = BIZ_TYPES[int(RNG.integers(0, len(BIZ_TYPES)))]
    created = BASE_DATE + pd.Timedelta(days=int(RNG.integers(0, DAYS)))
    cname = COMPANY_NAMES[int(RNG.integers(0, N_CUSTOMERS))]
    biz_rows.append({
        "document_id": f"BIZDOC{i:04d}",
        "document_type": dtype,
        "created_date": created,
        "content": (
            f"{dtype} document created {created.date()} "
            f"relating to {cname}. "
            "Standard enterprise terms and conditions apply."
        ),
    })

biz_df = pd.DataFrame(biz_rows)
biz_df.to_parquet(RAW_DIR / "business_documents.parquet", index=False)
print(f"  Business documents: {len(biz_df):,} rows")


# ── 5. Empty stub files (loaded but not used in transforms) ──────────────────

print("Generating stub files …")

pd.DataFrame(columns=["id", "document_number", "status"]).to_parquet(
    RAW_DIR / "sales_documents.parquet", index=False
)
pd.DataFrame(columns=["id", "document_number", "item", "quantity"]).to_parquet(
    RAW_DIR / "sales_items.parquet", index=False
)

db_path = RAW_DIR / "uberjugaad_email.db"
with sqlite3.connect(str(db_path)) as conn:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS emails "
        "(id INTEGER PRIMARY KEY, message_id TEXT, subject TEXT, body TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS contacts "
        "(id INTEGER PRIMARY KEY, email TEXT, name TEXT)"
    )
    conn.commit()
print("  Stub files created (sales_documents, sales_items, SQLite)")


# ── 6. Run the real transforms pipeline ─────────────────────────────────────

print("\nRunning build_processed_assets …")
try:
    from archive_analytics.data import build_processed_assets
    from archive_analytics.settings import configure_logging, get_config

    configure_logging()
    # Force config to use our raw dir
    cfg = get_config()
    paths = build_processed_assets(force=True, config=cfg)
    print(f"  Processed tables written to {cfg.processed_dir}")
    for name, path in sorted(paths.items()):
        size = Path(path).stat().st_size if Path(path).exists() else 0
        print(f"    {name}: {size:,} bytes")
except Exception as exc:  # noqa: BLE001
    print(f"ERROR running pipeline: {exc}", file=sys.stderr)
    raise


# ── 7. Synthetic risk scores ─────────────────────────────────────────────────

print("\nGenerating risk scores …")

fact_order = pd.read_parquet(PROCESSED_DIR / "fact_order.parquet")
RNG2 = np.random.default_rng(99)

# Derive plausible scores from observable order features
delay_signal = (
    (fact_order["max_delivery_lag_days"].fillna(0) > 0).astype(float) * 0.5
    + RNG2.uniform(0, 0.5, len(fact_order))
).clip(0.0, 1.0)

complaint_signal = (
    (fact_order["complaint_email_count"].fillna(0) > 0).astype(float) * 0.55
    + RNG2.uniform(0, 0.45, len(fact_order))
).clip(0.0, 1.0)

credit_signal = (
    (fact_order["has_credit_memo"].fillna(False).astype(float)) * 0.6
    + RNG2.uniform(0, 0.4, len(fact_order))
).clip(0.0, 1.0)

THRESHOLDS = {
    "will_generate_complaint": 0.40,
    "will_be_delayed": 0.45,
    "will_generate_credit_memo": 0.38,
}

scores_df = pd.DataFrame({
    "order_id": fact_order["order_id"],
    "customer_id_clean": fact_order["customer_id_clean"],
    "order_created_at": fact_order["order_created_at"],
    "will_generate_complaint_score": complaint_signal.values.round(4),
    "will_be_delayed_score": delay_signal.values.round(4),
    "will_generate_credit_memo_score": credit_signal.values.round(4),
    "will_generate_complaint_prediction": (
        (complaint_signal > THRESHOLDS["will_generate_complaint"]).astype(int).values
    ),
    "will_be_delayed_prediction": (
        (delay_signal > THRESHOLDS["will_be_delayed"]).astype(int).values
    ),
    "will_generate_credit_memo_prediction": (
        (credit_signal > THRESHOLDS["will_generate_credit_memo"]).astype(int).values
    ),
})

scores_df.to_parquet(MODELS_DIR / "order_risk_scores.parquet", index=False)
print(f"  Risk scores: {len(scores_df):,} orders")


# ── 8. Model metrics JSON ────────────────────────────────────────────────────

print("Generating model_metrics.json …")

n_total = len(fact_order)
n_train = int(n_total * 0.80)
n_test = n_total - n_train

TOP_FEATURES = [
    {"feature": "customer_prior_will_generate_complaint_rate", "importance": 0.48},
    {"feature": "order_line_count", "importance": 0.22},
    {"feature": "customer_prior_orders_log", "importance": 0.15},
    {"feature": "plant_prior_delay_rate", "importance": 0.09},
    {"feature": "product_nunique", "importance": 0.06},
]

def _target_metrics(
    roc: float, ap: float, f1: float, threshold: float, cv_mean: float
) -> dict:
    return {
        "status": "trained",
        "selected_model": "LogisticRegression",
        "optimal_threshold": threshold,
        "calibrated": {
            "roc_auc": roc,
            "average_precision": ap,
            "f1": f1,
            "threshold": threshold,
            "precision": round(f1 + 0.03, 3),
            "recall": round(f1 - 0.04, 3),
        },
        "cross_validation": {
            "n_folds": 3,
            "mean": cv_mean,
            "std": 0.03,
        },
        "top_features": TOP_FEATURES,
    }


model_metrics = {
    "run_id": f"demo-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
    "built_at": datetime.utcnow().isoformat() + "Z",
    "train_rows": n_train,
    "test_rows": n_test,
    "feature_policy": "standard",
    "targets": {
        "will_generate_complaint": _target_metrics(0.841, 0.693, 0.618, 0.40, 0.815),
        "will_be_delayed": _target_metrics(0.812, 0.671, 0.603, 0.45, 0.791),
        "will_generate_credit_memo": _target_metrics(0.878, 0.724, 0.651, 0.38, 0.856),
    },
}

(MODELS_DIR / "model_metrics.json").write_text(
    json.dumps(model_metrics, indent=2), encoding="utf-8"
)
print("  model_metrics.json written")


# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Demo data generation complete.")
print(f"  data/processed/ → {sum(1 for f in PROCESSED_DIR.iterdir())} files")
print(f"  data/models/    → {sum(1 for f in MODELS_DIR.iterdir())} files")
print()
print("Next steps:")
print("  1. Remove 'data/processed/' and 'data/models/' from .gitignore")
print("  2. git add data/processed/ data/models/")
print("  3. git commit -m 'feat: add synthetic demo data for Streamlit Cloud'")
print("  4. git push")
