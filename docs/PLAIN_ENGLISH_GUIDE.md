# Plain-English Guide

## What Problem Does This Project Solve?

Imagine a company that receives thousands of orders from customers.
Every order creates a trail of information:

- order records in operational systems
- emails between the customer and the company
- invoices, shipping notices, quality reports, and credit memos
- status updates spread across multiple files and systems

In most businesses, those records live in different places and are not joined together cleanly.
That creates four practical problems:

1. Teams cannot see the full story of an order in one place.
2. Problems like delays or complaints are noticed too late.
3. Managers make decisions from scattered reports and gut feel.
4. Analysts spend too much time cleaning data instead of learning from it.

This project solves that by turning messy, disconnected records into one structured, usable system for analysis and decision-making.

---

## What This System Does In Simple Terms

You can think of the project as a factory for information.

It takes raw business records in, organises them, checks them, scores risk, and displays the results in a dashboard that a non-technical person can understand.

At a high level, the system does five things:

1. Collects raw data from different sources.
2. Cleans and joins that data into one consistent view.
3. Looks for warning signs that an order may go wrong.
4. Lets users explore customers, orders, timelines, and evidence.
5. Supports decisions about service, operations, quality, and finance.

---

## What Kinds of Questions It Answers

This project helps answer questions like:

- Which customers are having the most trouble?
- Which orders are most likely to be delayed?
- Which orders are showing early signs of complaint risk?
- Which orders may end in a credit memo or financial loss?
- Are problems increasing or decreasing month by month?
- Are we dealing with a real operational issue or just incomplete data?
- What evidence do we actually have for a claim about a customer or order?

---

## How It Solves The Problem

## Step 1: Read the raw records

The system reads raw files from two main kinds of sources:

- parquet files, which are fast storage files used for analytics
- a SQLite database, which is a lightweight database file

These raw sources contain things like:

- communications
- ERP transactions
- supporting documents
- business documents

Each source tells only part of the story. On its own, each file is incomplete.

---

## Step 2: Clean and standardise the data

Raw business data is rarely ready for decision-making.

The project cleans it by doing things like:

- converting dates into a consistent format
- normalising IDs so the same customer or order is not written in several slightly different ways
- creating stable internal IDs so records can be joined safely
- detecting whether an email looks like a complaint or spam
- extracting order references from email subject lines and message bodies

This matters because bad joins create bad insights. If records are not standardised first, the later analysis becomes unreliable.

---

## Step 3: Build business tables people can actually use

After cleaning, the system creates organised tables that describe the business from different angles.

### `fact_order`

This is the main order table.
It tells the story of each order, including:

- when it was created
- how many line items it has
- whether it has related documents
- whether emails are linked to it
- whether it was delayed
- whether it later generated a complaint
- whether it later generated a credit memo

### `dim_customer`

This is the customer summary table.
It shows:

- how many orders a customer has placed
- how many emails are linked to that customer
- how often that customer has issues
- how much value and credit activity is associated with them

### `fact_email`

This captures emails in a usable form.
It identifies:

- inbound vs outbound vs internal emails
- likely complaint emails
- likely spam emails
- which order, if any, an email is linked to

### `fact_document`

This stores business documents like invoices, shipping notices, and credit memos.
It separates different financial meanings instead of mixing them into one confusing number.

### `fact_event_timeline`

This produces one shared timeline of events across orders, emails, and documents.
That makes it possible to answer questions like, “What happened first?” and “What happened next?”

### `fact_order_risk_features`

This is the model-ready table.
It contains the features used by the machine learning models.

### `retrieval_corpus`

This is the evidence table used by the assistant.
It turns business records into searchable text so the system can retrieve relevant supporting evidence.

---

## Step 4: Measure data quality

Before trusting insights, the system checks the health of the data.

It produces a quality report that tells you things like:

- how many rows were read from each raw source
- how many rows ended up in each processed table
- whether there are duplicate IDs
- whether important fields are missing too often
- what date ranges are covered

This helps separate a real business problem from a data problem.

Example:
If complaint volume appears low, is that because service improved, or because the email records are incomplete? The data quality view helps answer that.

---

## Step 5: Score risk before problems happen

One of the main goals of the project is not just to describe the past, but to warn about possible future issues.

The system trains three prediction models:

1. Will this order be delayed?
2. Will this order generate a complaint?
3. Will this order generate a credit memo?

### How the models think

The models do not guess randomly.
They look for patterns that often appeared before previous problems.

Examples of useful signals include:

- orders with many line items
- plants with a history of delays
- customers with a history of complaints
- customers with repeated prior issues
- operational complexity, such as many products or shipping points

The model gives each order a probability score between 0 and 1.

Example:

- `0.12` means low risk
- `0.47` means moderate risk
- `0.83` means high risk

This does not mean the system is certain. It means the order looks similar to other orders that previously had problems.

### Why calibration matters

The project calibrates its models so the scores behave more like honest probabilities.

In simple language, calibration tries to make this true:

If the system marks 100 similar orders as `0.80` risk, then roughly 80 of them should really experience that issue over time.

That makes the output more useful for real decisions.

---

## Step 6: Retrieve evidence, not just charts

The assistant page is designed to answer questions using stored evidence, not free-form imagination.

If someone asks something like:

“Which customers show strong signals of delivery problems and credit risk?”

the system searches through prepared text summaries of:

- orders
- customer records
- emails
- documents

It returns the most relevant pieces of evidence and avoids showing too many near-duplicates.

That means the assistant is more like a search-and-summary tool than a chatbot making things up.

---

## What Decisions Can Be Made From This?

This project supports several practical business decisions.

## 1. Customer service decisions

Teams can decide:

- which customers need proactive outreach
- which accounts should be escalated to senior support
- which complaints need immediate investigation

Example:
If one customer has a high issue rate and several recent complaint-like emails, the team can call them before the relationship worsens.

## 2. Operations decisions

Teams can decide:

- which orders need intervention before shipment
- which plants or shipping processes need attention
- where operational bottlenecks are forming

Example:
If one plant shows rising delay-related risk across many orders, operations can investigate that location before problems spread.

## 3. Finance decisions

Teams can decide:

- where credit memo exposure is growing
- which customer/order groups are causing avoidable financial leakage
- where invoice or refund patterns suggest process failure

Example:
If high-risk orders also show frequent credit memo outcomes, finance can quantify where losses are likely and act sooner.

## 4. Quality and compliance decisions

Teams can decide:

- whether quality issues are concentrated around certain customers, plants, or order types
- where documentation quality is weak
- whether evidence exists to support an audit trail

## 5. Management decisions

Leadership can decide:

- whether risk is rising or falling over time
- whether service quality is improving
- where to invest improvement effort
- whether a problem is local or systemic

---

## The Logic Behind How Everything Works

Here is the system logic in plain language.

### Logic 1: One business event should appear in one joined story

Orders, emails, and documents are different records about the same real-world process.
The project links them together so one order can be understood as a full sequence rather than as isolated fragments.

### Logic 2: Clean input is required for trustworthy output

If dates, customer IDs, and order IDs are inconsistent, every later chart and prediction becomes weaker.
So the project cleans first, then analyses.

### Logic 3: Past patterns can help flag future risk

If delayed orders in the past tended to share certain characteristics, those characteristics can help identify current orders that deserve attention.

That is why the project builds backward-looking risk features such as prior complaint rate and prior delay rate.

### Logic 4: Evidence should support conclusions

Dashboards can make strong claims, but those claims are safer when the user can trace them back to emails, documents, and timelines.
The retrieval assistant exists to support that traceability.

### Logic 5: Not every problem is a business problem

Sometimes bad data looks like bad performance.
The data quality report helps avoid false alarms by showing whether the source data is complete enough to trust.

---

## Why The Dashboard Has Multiple Pages

Each page answers a different type of question.

### Executive Overview

This is the “management summary” page.
It answers:

- How many orders do we have?
- Are delays or complaints trending up?
- Which customers have the most issues?

### Customer 360

This is the “single customer story” page.
It answers:

- How healthy is this customer relationship?
- How many issues has this customer had?
- What events happened over time for this customer?

### Order Timeline

This is the “single order story” page.
It answers:

- What happened to this order from beginning to end?
- When did emails, documents, and other events occur?

### Risk Scoring

This is the “early warning” page.
It answers:

- Which orders are most likely to go wrong?
- How confident is the model?
- Which factors matter most?

### Evidence-Based Assistant

This is the “find the proof” page.
It answers:

- What evidence supports this conclusion?
- Which documents or emails are most relevant?

### Data Quality

This is the “can we trust the data?” page.
It answers:

- Are the source files complete?
- Are there duplicates or missing values?
- Is the pipeline producing sensible outputs?

---

## What Makes This Useful To A Non-Technical Stakeholder

You do not need to understand databases or machine learning to get value from this project.

A non-technical stakeholder can use it to:

- spot problem customers early
- prioritise risky orders
- understand whether delays are isolated or systemic
- review evidence before escalating a case
- measure whether improvement actions are working

In short, the project turns a confusing mass of records into something understandable, searchable, and actionable.

---

## What This Project Does Not Claim

It is important to be clear about the limits.

This system does not guarantee that every flagged order will fail.
It does not replace human judgement.
It does not automatically fix business problems.

What it does do is improve visibility, prioritisation, and evidence quality, so people can make better decisions faster.

---

## A Simple Mental Model

If you want the shortest possible explanation, think of the project like this:

> It is a business early-warning and evidence system.

It takes messy records from many places, turns them into one organised picture, estimates where trouble is likely, and gives people the evidence they need to act.

---

## Suggested Use In Real Life

An operations or customer-success team could use this process:

1. Check the Executive Overview to see where problems are growing.
2. Open Risk Scoring to identify the highest-risk orders.
3. Open Customer 360 to understand whether the issue is part of a larger customer pattern.
4. Open Order Timeline to inspect the event sequence.
5. Use the Evidence-Based Assistant to retrieve the most relevant emails and documents.
6. Use Data Quality to confirm the insight is based on trustworthy data.
7. Decide whether to intervene, escalate, refund, investigate, or monitor.

That is the practical logic of the entire platform.