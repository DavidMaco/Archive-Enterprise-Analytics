"""Evidence retrieval over the archive corpus using TF-IDF + cosine similarity.

Key improvements over the original implementation:
- **Cached vectorizer**: The TF-IDF index is fitted once per corpus fingerprint and
  reused across queries, eliminating redundant ``fit_transform`` calls.
- **MMR-style re-ranking**: After cosine scoring, results are diversified using
  Maximal Marginal Relevance to reduce near-duplicate evidence.
- **Honest naming**: ``retrieve_evidence`` returns scored documents;
  ``summarise_evidence`` synthesizes a structured extractive summary (no
  pretence of generative answering).
- **Entity-level deduplication**: At most one result per (entity_type, order_id)
  pair is retained so that the top-k genuinely represents *different* evidence.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .constants import (
    ORDER_PATTERN,
    RETRIEVAL_TOP_K,
    SNIPPET_LENGTH,
    TFIDF_NGRAM_RANGE,
)
from .data import load_processed_table
from .settings import AppConfig, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level TF-IDF cache  (corpus fingerprint → fitted artefacts)
# ---------------------------------------------------------------------------

_IndexCache = dict[str, tuple[TfidfVectorizer, sparse.csr_matrix, pd.DataFrame]]
_INDEX_CACHE: _IndexCache = {}


def _corpus_fingerprint(corpus: pd.DataFrame) -> str:
    """Return a fast fingerprint so we can detect stale caches."""
    n = len(corpus)
    head_id = str(corpus.iloc[0]["citation"]) if n > 0 else ""
    tail_id = str(corpus.iloc[-1]["citation"]) if n > 0 else ""
    return hashlib.md5(f"{n}|{head_id}|{tail_id}".encode()).hexdigest()


def _get_or_build_index(
    corpus: pd.DataFrame,
) -> tuple[TfidfVectorizer, sparse.csr_matrix, pd.DataFrame]:
    """Return (vectorizer, tfidf_matrix, corpus) from cache or build them."""
    fp = _corpus_fingerprint(corpus)
    if fp in _INDEX_CACHE:
        logger.debug("TF-IDF cache hit (fingerprint=%s)", fp)
        return _INDEX_CACHE[fp]

    logger.info("Building TF-IDF index for %d documents …", len(corpus))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=1,
        max_features=30_000,
    )
    matrix = vectorizer.fit_transform(corpus["text"].fillna(""))
    _INDEX_CACHE[fp] = (vectorizer, matrix, corpus)
    return vectorizer, matrix, corpus


def invalidate_cache() -> None:
    """Clear the module-level TF-IDF cache (useful after data rebuild)."""
    _INDEX_CACHE.clear()
    logger.info("TF-IDF cache cleared.")


# ---------------------------------------------------------------------------
# Corpus loading & filtering
# ---------------------------------------------------------------------------


def load_corpus(config: AppConfig | None = None) -> pd.DataFrame:
    """Load the retrieval corpus from processed assets."""
    cfg = config or get_config()
    return load_processed_table("retrieval_corpus", config=cfg)


def _apply_filters(
    corpus: pd.DataFrame, filters: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Apply user-supplied key=value equality filters to the corpus."""
    if not filters:
        return corpus
    mask = pd.Series(True, index=corpus.index)
    for key, value in filters.items():
        if value in (None, "", []):
            continue
        if key not in corpus.columns:
            continue
        mask &= corpus[key].astype("string") == str(value)
    return corpus.loc[mask]


# ---------------------------------------------------------------------------
# MMR re-ranking
# ---------------------------------------------------------------------------


def _mmr_rerank(
    scores: np.ndarray,
    matrix: sparse.csr_matrix,
    candidate_indices: np.ndarray,
    top_k: int,
    lam: float = 0.7,
) -> list[int]:
    """Select *top_k* indices via Maximal Marginal Relevance.

    Parameters
    ----------
    scores:
        Relevance scores for every document in the corpus.
    matrix:
        TF-IDF matrix (sparse, same row order as *scores*).
    candidate_indices:
        Pre-sorted indices of the *candidate_pool_size* best-scoring docs.
    top_k:
        How many to return.
    lam:
        Trade-off between relevance (1.0) and diversity (0.0).
    """
    selected: list[int] = []
    candidates = list(candidate_indices)
    while len(selected) < top_k and candidates:
        best_idx = -1
        best_mmr = -np.inf
        for idx in candidates:
            relevance = float(scores[idx])
            if selected:
                sim_to_selected = linear_kernel(
                    matrix[idx], matrix[selected]
                ).ravel()
                redundancy = float(sim_to_selected.max())
            else:
                redundancy = 0.0
            mmr = lam * relevance - (1 - lam) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Entity-level deduplication
# ---------------------------------------------------------------------------


def _deduplicate_entity(results: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-scoring row per (entity_type, order_id) pair."""
    if results.empty:
        return results
    key_cols = ["entity_type"]
    if "order_id" in results.columns:
        key_cols.append("order_id")
    return (
        results.sort_values("score", ascending=False)
        .drop_duplicates(subset=key_cols, keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------


def retrieve_evidence(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    filters: dict[str, Any] | None = None,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Retrieve the *top_k* most relevant (and diverse) evidence records.

    1. Load & filter the corpus.
    2. If the query contains a 10-digit order ID, narrow to that order first.
    3. Score via cached TF-IDF + cosine similarity.
    4. Re-rank with MMR for diversity.
    5. Deduplicate per (entity_type, order_id).
    """
    full_corpus = load_corpus(config)
    corpus = _apply_filters(full_corpus, filters)
    if corpus.empty:
        return corpus

    # Narrow to a specific order if mentioned in the query
    order_match = ORDER_PATTERN.search(query)
    if order_match:
        order_id = order_match.group(0)
        order_subset = corpus[corpus["order_id"].astype("string") == order_id]
        if not order_subset.empty:
            corpus = order_subset

    # Use cached TF-IDF index (built on the *full* corpus for consistency)
    vectorizer, full_matrix, _cached_corpus = _get_or_build_index(full_corpus)

    # Score only the filtered subset
    subset_indices = corpus.index.values
    query_vector = vectorizer.transform([query])
    all_scores = linear_kernel(query_vector, full_matrix).ravel()

    # Map subset rows to their global indices and pick top candidates
    subset_scores = all_scores[subset_indices]
    candidate_pool = min(top_k * 4, len(subset_indices))
    local_top = np.argsort(subset_scores)[::-1][:candidate_pool]
    global_top = subset_indices[local_top]

    # MMR re-rank for diversity (operates on global indices into full_matrix)
    selected_global = _mmr_rerank(
        all_scores, full_matrix, global_top, top_k=top_k
    )

    results = full_corpus.loc[selected_global].copy()
    results["score"] = all_scores[selected_global]
    results = _deduplicate_entity(results)
    return results.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Structured extractive summary
# ---------------------------------------------------------------------------


def summarise_evidence(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    filters: dict[str, Any] | None = None,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Retrieve evidence and build a structured extractive summary.

    Returns
    -------
    dict with keys:
        ``summary`` – A human-readable paragraph synthesized from the evidence.
        ``evidence`` – List of citation dicts with snippets and metadata.
        ``meta`` – Counts by entity type, unique orders, unique customers.

    .. note::
       This is purely extractive: no generative model is used.  The summary
       is assembled from templates and the retrieved snippets.
    """
    evidence = retrieve_evidence(
        query, top_k=top_k, filters=filters, config=config
    )
    if evidence.empty:
        return {
            "summary": "No evidence matched the current query and filters.",
            "evidence": [],
            "meta": {},
        }

    # Metadata aggregation
    entity_counts: dict[str, int] = dict(
        evidence["entity_type"].value_counts()
    )
    order_ids = sorted(
        {
            str(v)
            for v in evidence["order_id"].dropna().unique()
            if str(v) not in ("", "<NA>")
        }
    )
    customer_ids = sorted(
        {
            str(v)
            for v in evidence["customer_id"].dropna().unique()
            if str(v) not in ("", "<NA>")
        }
    )

    # Pattern detection  (e.g. "3/5 results involve delivery delays")
    patterns: list[str] = []
    if "entity_type" in evidence.columns:
        for etype, count in entity_counts.items():
            patterns.append(f"{count}/{len(evidence)} results are {etype} records")

    # Check for complaint / delay signals in evidence text
    text_blob = " ".join(evidence["text"].fillna("").str.lower())
    keyword_hits: Counter[str] = Counter()
    for kw in ("delay", "complaint", "credit", "damaged", "shortage", "late"):
        n = text_blob.count(kw)
        if n:
            keyword_hits[kw] = n
    if keyword_hits:
        top_kw = keyword_hits.most_common(3)
        patterns.append(
            "Key themes: "
            + ", ".join(f"'{k}' ({v}×)" for k, v in top_kw)
        )

    # Build summary paragraph
    parts = [f"Retrieved {len(evidence)} evidence records."]
    if order_ids:
        parts.append(f"Relevant orders: {', '.join(order_ids[:5])}.")
    if customer_ids:
        parts.append(f"Relevant customers: {', '.join(customer_ids[:5])}.")
    if patterns:
        parts.append(" | ".join(patterns) + ".")

    # Top highlights
    highlights = []
    for _, row in evidence.head(3).iterrows():
        highlights.append(f"[{row['entity_type']}] {row['title']}: {str(row['text'])[:180]}")
    if highlights:
        parts.append("Highlights: " + " — ".join(highlights))
    parts.append("(Extractive summary — based only on retrieved records.)")

    # Structured evidence list
    evidence_list: list[dict[str, Any]] = []
    for _, row in evidence.iterrows():
        evidence_list.append(
            {
                "citation": row["citation"],
                "title": row["title"],
                "entity_type": row["entity_type"],
                "order_id": row.get("order_id"),
                "customer_id": row.get("customer_id"),
                "score": float(row["score"]),
                "snippet": str(row["text"])[: SNIPPET_LENGTH],
            }
        )

    return {
        "summary": " ".join(parts),
        "evidence": evidence_list,
        "meta": {
            "entity_counts": entity_counts,
            "order_ids": order_ids,
            "customer_ids": customer_ids,
            "keyword_themes": dict(keyword_hits),
        },
    }


# Keep backward-compatible alias
answer_query = summarise_evidence
