# Power BI — Archive Enterprise Analytics

This guide covers how to connect Power BI Desktop to the processed Parquet tables and build an executive dashboard aligned to the Streamlit dashboard's six pages.

---

## 1. Data Source: Processed Parquet Tables

After running `python -m archive_analytics build` and `python -m archive_analytics train`, the following Parquet files are available under `data/processed/`:

| Table | File | Grain | Key Columns |
|-------|------|-------|-------------|
| `fact_order` | `fact_order.parquet` | 1 row per order | `order_id`, `customer_id_clean`, `order_created_at`, `will_be_delayed`, `will_generate_complaint`, `will_generate_credit_memo` |
| `fact_order_risk_features` | `fact_order_risk_features.parquet` | 1 row per order | `order_id`, `order_line_count`, `customer_prior_orders_log`, `customer_prior_will_be_delayed_rate`, primary_plant |
| `dim_customer` | `dim_customer.parquet` | 1 row per customer | `customer_id_clean`, `customer_name`, `prior_order_count`, `issue_rate` |
| `fact_email` | `fact_email.parquet` | 1 row per email | `order_id`, `email_id`, `is_complaint`, `is_spam`, `scope` |
| `fact_document` | `fact_document.parquet` | 1 row per document | `order_id`, `doc_type`, `financial_amount` |
| `fact_event_timeline` | `fact_event_timeline.parquet` | 1 row per event | `order_id`, `event_date`, `event_type` |
| `fact_customer_daily` | `fact_customer_daily.parquet` | 1 row per customer per day | `customer_id_clean`, `event_date`, `event_count` |
| `order_risk_scores` | `../models/order_risk_scores.parquet` | 1 row per order | `order_id`, `delay_prob`, `complaint_prob`, `credit_memo_prob`, `delay_pred`, `complaint_pred`, `credit_memo_pred` |

---

## 2. Connecting Power BI Desktop

1. Open **Power BI Desktop → Get Data → Parquet**.
2. Navigate to `<project_root>/data/processed/` and import each table listed above.
3. Also import `<project_root>/data/models/order_risk_scores.parquet`.
4. In Power Query: ensure `order_created_at` is parsed as **Date/Time** and boolean columns (`will_be_delayed`, etc.) are set to **True/False**.

### Recommended Relationships

```
fact_order[order_id]            → fact_order_risk_features[order_id]  (1:1)
fact_order[order_id]            → order_risk_scores[order_id]          (1:1)
fact_order[order_id]            → fact_email[order_id]                 (1:many)
fact_order[order_id]            → fact_document[order_id]              (1:many)
fact_order[order_id]            → fact_event_timeline[order_id]        (1:many)
fact_order[customer_id_clean]   → dim_customer[customer_id_clean]      (many:1)
fact_customer_daily[customer_id_clean] → dim_customer[customer_id_clean] (many:1)
```

---

## 3. DAX Measures

Paste the following measures into a dedicated **Measures** table.

### KPI Measures

```dax
Total Orders =
COUNTROWS(fact_order)

Unique Customers =
DISTINCTCOUNT(fact_order[customer_id_clean])

Delay Rate =
DIVIDE(
    COUNTROWS(FILTER(fact_order, fact_order[will_be_delayed] = TRUE())),
    COUNTROWS(fact_order)
)

Complaint Rate =
DIVIDE(
    COUNTROWS(FILTER(fact_order, fact_order[will_generate_complaint] = TRUE())),
    COUNTROWS(fact_order)
)

Credit Memo Rate =
DIVIDE(
    COUNTROWS(FILTER(fact_order, fact_order[will_generate_credit_memo] = TRUE())),
    COUNTROWS(fact_order)
)
```

### Risk Score Measures

```dax
Avg Delay Probability =
AVERAGE(order_risk_scores[delay_prob])

Avg Complaint Probability =
AVERAGE(order_risk_scores[complaint_prob])

Avg Credit Memo Probability =
AVERAGE(order_risk_scores[credit_memo_prob])

High Risk Orders (Delay) =
CALCULATE(
    COUNTROWS(order_risk_scores),
    order_risk_scores[delay_prob] >= 0.7
)

High Risk Orders (Complaint) =
CALCULATE(
    COUNTROWS(order_risk_scores),
    order_risk_scores[complaint_prob] >= 0.7
)
```

### Trend Measures

```dax
Monthly Orders =
CALCULATE(
    [Total Orders],
    DATESMTD(fact_order[order_created_at])
)

Complaint Orders MoM % Change =
VAR CurrentMonth =
    CALCULATE(
        COUNTROWS(FILTER(fact_order, fact_order[will_generate_complaint] = TRUE())),
        DATESMTD(fact_order[order_created_at])
    )
VAR PreviousMonth =
    CALCULATE(
        COUNTROWS(FILTER(fact_order, fact_order[will_generate_complaint] = TRUE())),
        PREVIOUSMONTH(fact_order[order_created_at])
    )
RETURN
    DIVIDE(CurrentMonth - PreviousMonth, PreviousMonth)
```

### Customer-Level Measures

```dax
Customer Issue Rate =
DIVIDE(
    CALCULATE(
        COUNTROWS(FILTER(fact_order, fact_order[will_generate_complaint] = TRUE()))
    ),
    COUNTROWS(fact_order)
)

Top Customer Complaint Rate =
MAXX(
    VALUES(fact_order[customer_id_clean]),
    [Customer Issue Rate]
)
```

---

## 4. Recommended Dashboard Pages

### Page 1 — Executive KPIs
- Cards: Total Orders, Unique Customers, Delay Rate, Complaint Rate, Credit Memo Rate
- Line chart: Monthly order volume and complaint rate over time
- Bar chart: Top 10 plants by delay rate

### Page 2 — Customer 360
- Table: Customer name, order count, delay rate, complaint rate, credit memo rate
- Scatter plot: Prior orders (x) vs. issue rate (y), sized by order volume
- Drill-through to per-customer timeline

### Page 3 — Order Timeline
- Timeline/Gantt: `fact_event_timeline` events per order
- Slicer: filter by `event_type`

### Page 4 — Risk Scoring
- Histogram: Distribution of `delay_prob`, `complaint_prob`, `credit_memo_prob`
- Scatter: `delay_prob` vs. `complaint_prob` coloured by `delay_pred`
- Table: Top 20 highest-risk orders with order_id + all three probabilities

### Page 5 — Data Quality
- Cards: Record counts for each processed table
- KPI tiles for completeness / null rates (computed in Power Query)

---

## 5. Scheduled Refresh / Export Pipeline

For automated refresh:

1. Run the CLI pipeline on a schedule (Task Scheduler / cron / GitHub Actions):
   ```bash
   python -m archive_analytics build
   python -m archive_analytics train
   ```
2. If hosting locally, Power BI Desktop opens the updated Parquet files on next refresh.
3. For Power BI Service, export the processed Parquet files to an Azure Data Lake or SharePoint folder, then use the **Parquet connector** with the cloud path.
4. Alternatively, generate a CSV export via the CLI and use the **CSV connector** in Power BI Service for simpler gateway-free refresh:
   ```bash
   python -c "
   import pandas as pd, pathlib
   root = pathlib.Path('data/processed')
   for f in root.glob('*.parquet'):
       pd.read_parquet(f).to_csv(root / (f.stem + '.csv'), index=False)
   "
   ```

---

## 6. Theme

A Power BI theme JSON aligned to the dashboard colour palette is provided in `powerbi/theme.json`. Import it via **View → Themes → Browse for themes**.
