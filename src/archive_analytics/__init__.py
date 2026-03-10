"""Archive Enterprise Analytics — customer 360, risk scoring, and evidence retrieval.

Public API
----------
Settings & config
    :data:`AppConfig`, :func:`get_config`
Data pipeline
    :func:`build_processed_assets`, :func:`load_processed_table`
Modeling
    :func:`train_all_targets`, :func:`load_model_metrics`, :func:`load_risk_scores`
Retrieval
    :func:`retrieve_evidence`, :func:`summarise_evidence`, :func:`load_corpus`
"""

from .data import build_processed_assets, load_processed_table
from .modeling import load_model_metrics, load_risk_scores, train_all_targets
from .retrieval import load_corpus, retrieve_evidence, summarise_evidence
from .settings import AppConfig, get_config

# Backward-compatible alias
answer_query = summarise_evidence

__all__ = [
    "AppConfig",
    "get_config",
    "build_processed_assets",
    "load_processed_table",
    "train_all_targets",
    "load_model_metrics",
    "load_risk_scores",
    "retrieve_evidence",
    "summarise_evidence",
    "answer_query",
    "load_corpus",
]
