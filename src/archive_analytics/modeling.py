"""ML risk scoring: training, calibration, threshold tuning, and evaluation.

Key improvements over earlier versions:

* Feature importance is extracted from the *selected* model (HGBC or LR).
* ``CalibratedClassifierCV`` ensures probability outputs are well-calibrated.
* Per-target thresholds are tuned to maximise F1 via the precision-recall curve.
* ``customer_prior_orders`` is log-transformed to reduce dominance.
* Time-series cross-validation provides mean ± std metric estimates.
* Model artifacts are SHA-256-hashed for integrity verification.
* A model-history JSON file preserves metrics from previous runs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .constants import (
    BOOSTING_LEARNING_RATE,
    BOOSTING_MAX_DEPTH,
    BOOSTING_MAX_ITER,
    CV_N_SPLITS,
    LOGISTIC_MAX_ITER,
    METRICS_FILE,
    MODEL_HISTORY_FILE,
    PREDICTIVE_CATEGORICAL_FEATURES,
    PREDICTIVE_NUMERIC_FEATURES,
    SCORES_FILE,
    STRATIFIED_TEST_SIZE,
    TARGETS,
    TRAIN_TEST_TIME_QUANTILE,
)
from .data import build_processed_assets, load_processed_table
from .settings import AppConfig, get_config

logger = logging.getLogger(__name__)


def model_artifacts_ready(config: AppConfig | None = None) -> bool:
    """Return ``True`` when the metrics and score artefacts exist on disk."""
    cfg = config or get_config()
    return (cfg.models_dir / METRICS_FILE).exists() and (cfg.models_dir / SCORES_FILE).exists()


# ── Frame preparation ───────────────────────────────────────────────────────

def _prepare_model_frame(order_features: pd.DataFrame) -> pd.DataFrame:
    """Load and validate the model-ready feature frame.

    The heavy cumulative / temporal feature engineering is now done in
    ``transforms.build_order_risk_features`` at data-build time; this
    function simply validates the expected columns are present.
    """
    frame = order_features.copy()
    frame["order_created_at"] = pd.to_datetime(
        frame["order_created_at"], errors="coerce"
    )
    frame = (
        frame.dropna(subset=["order_created_at", "order_id"])
        .sort_values("order_created_at")
        .reset_index(drop=True)
    )
    frame["primary_plant"] = frame["primary_plant"].astype("string")
    frame["customer_id_clean"] = frame["customer_id_clean"].astype("string")

    required = set(PREDICTIVE_NUMERIC_FEATURES + PREDICTIVE_CATEGORICAL_FEATURES)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Model frame missing columns: {sorted(missing)}")
    return frame


# ── Train / test splitting ──────────────────────────────────────────────────

def _train_test_split_by_time(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80 / 20 time-based split with a positional fallback."""
    cutoff = frame["order_created_at"].quantile(TRAIN_TEST_TIME_QUANTILE)
    train = frame[frame["order_created_at"] <= cutoff].copy()
    test = frame[frame["order_created_at"] > cutoff].copy()
    if train.empty or test.empty:
        idx = max(1, int(len(frame) * TRAIN_TEST_TIME_QUANTILE))
        train, test = frame.iloc[:idx].copy(), frame.iloc[idx:].copy()
    return train, test  # type: ignore[return-value]


# ── Preprocessor ────────────────────────────────────────────────────────────

def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the sklearn column transformer."""
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])
    return ColumnTransformer(transformers=[
        ("num", num, numeric_features),
        ("cat", cat, categorical_features),
    ])


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _evaluate(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute standard binary-classification metrics."""
    pred = (proba >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "positive_rate": float(y_true.mean()),
        "threshold": threshold,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        metrics["average_precision"] = float(average_precision_score(y_true, proba))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _find_optimal_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Find the probability threshold that maximises F1."""
    if y_true.nunique() < 2:
        return 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx])


# ── Feature importance ──────────────────────────────────────────────────────

def _extract_feature_importance(
    model: Pipeline,
    model_type: str,
) -> list[dict[str, float]]:
    """Extract feature importance from the *selected* model type.

    For logistic regression the absolute coefficient values are used; for
    histogram gradient boosting the built-in impurity-based importances
    are used.
    """
    estimator = model.named_steps["model"]
    names = list(model.named_steps["preprocessor"].get_feature_names_out())

    if model_type == "hist_gradient_boosting":
        importances = estimator.feature_importances_
        ranking = sorted(
            (
                {"feature": n, "importance": float(imp)}
                for n, imp in zip(names, importances, strict=True)
            ),
            key=lambda x: x["importance"],
            reverse=True,
        )
    else:
        coefs = estimator.coef_[0]
        ranking = sorted(
            (
                {"feature": n, "importance": float(abs(w)), "weight": float(w)}
                for n, w in zip(names, coefs, strict=True)
            ),
            key=lambda x: x["importance"],
            reverse=True,
        )
    return ranking[:20]


# ── Time-series cross-validation ────────────────────────────────────────────

def _time_series_cv(
    frame: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    n_splits: int = CV_N_SPLITS,
) -> dict[str, Any]:
    """Run expanding-window time-series CV and return mean ± std metrics."""
    frame_sorted = frame.sort_values("order_created_at").reset_index(drop=True)
    fold_size = len(frame_sorted) // (n_splits + 1)
    if fold_size < 50:
        return {"mean": {}, "std": {}, "n_folds": 0}

    fold_metrics: list[dict[str, float | None]] = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, len(frame_sorted))
        fold_train = frame_sorted.iloc[:train_end]
        fold_test = frame_sorted.iloc[train_end:test_end]
        y_tr = fold_train[target].astype(int)
        y_te = fold_test[target].astype(int)
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        prep = _build_preprocessor(numeric_features, categorical_features)
        pipe = Pipeline([
            ("preprocessor", prep),
            ("model", LogisticRegression(
                max_iter=LOGISTIC_MAX_ITER,
                class_weight="balanced",
                solver="liblinear",
            )),
        ])
        X_tr = fold_train[numeric_features + categorical_features]
        X_te = fold_test[numeric_features + categorical_features]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        fold_metrics.append(_evaluate(y_te, proba))

    if not fold_metrics:
        return {"mean": {}, "std": {}, "n_folds": 0}

    keys = [k for k in fold_metrics[0] if k != "threshold"]
    mean_m = {}
    std_m = {}
    for k in keys:
        vals: list[float] = []
        for metric in fold_metrics:
            metric_value = metric.get(k)
            if metric_value is not None:
                vals.append(float(metric_value))
        mean_m[k] = float(np.mean(vals)) if vals else None
        std_m[k] = float(np.std(vals)) if vals else None
    return {"mean": mean_m, "std": std_m, "n_folds": len(fold_metrics)}


# ── Artifact integrity ──────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Main training loop ──────────────────────────────────────────────────────

def train_all_targets(
    force: bool = False,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Train, calibrate, and evaluate risk models for all targets.

    For each target the pipeline:

    1. Trains logistic regression and histogram gradient boosting.
    2. Selects the better model by average precision.
    3. Calibrates with ``CalibratedClassifierCV``.
    4. Tunes the decision threshold to maximise F1.
    5. Runs time-series CV for robust metric estimates.
    6. Saves artefacts with SHA-256 integrity hashes.

    Returns
    -------
    dict
        ``model_metrics.json`` payload.
    """
    cfg = config or get_config()
    cfg.ensure_directories()
    metrics_path = cfg.models_dir / METRICS_FILE
    scores_path = cfg.models_dir / SCORES_FILE

    if not force and metrics_path.exists() and scores_path.exists():
        return load_model_metrics(cfg)

    run_id = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Training run %s (force=%s)", run_id, force)

    build_processed_assets(force=False, config=cfg)
    model_frame = _prepare_model_frame(
        load_processed_table("fact_order_risk_features", config=cfg)
    )
    train_frame, test_frame = _train_test_split_by_time(model_frame)

    numeric_features = PREDICTIVE_NUMERIC_FEATURES.copy()
    categorical_features = PREDICTIVE_CATEGORICAL_FEATURES.copy()

    payload: dict[str, Any] = {
        "run_id": run_id,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_policy": {
            "mode": "forward_looking",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "excluded_current_order_outcome_features": [
                "document_count", "revenue_total", "cost_total", "refund_total",
                "shipping_notice_count", "invoice_count", "quality_report_count",
                "credit_memo_count", "delayed_shipments", "max_delivery_lag_days",
                "linked_email_count", "complaint_email_count", "spam_email_count",
                "event_span_days",
            ],
        },
        "targets": {},
    }

    all_scores = test_frame[
        ["order_id", "customer_id_clean", "order_created_at"]
    ].copy()
    for t in TARGETS:
        all_scores[f"{t}_score"] = 0.0
        all_scores[f"{t}_prediction"] = 0

    for target in TARGETS:
        logger.info("── Target: %s ──", target)
        t_train = train_frame.copy()
        t_test = test_frame.copy()
        y_train = t_train[target].astype(int)
        y_test = t_test[target].astype(int)

        # Stratified fallback for rare classes
        if (y_train.nunique() < 2 or y_test.nunique() < 2) and model_frame[target].astype(int).nunique() > 1:
            logger.info("  Falling back to stratified split for %s", target)
            t_train, t_test = train_test_split(
                model_frame,
                test_size=STRATIFIED_TEST_SIZE,
                random_state=42,
                stratify=model_frame[target].astype(int),
            )
            y_train = t_train[target].astype(int)
            y_test = t_test[target].astype(int)

        if y_train.nunique() < 2 or t_test.empty:
            payload["targets"][target] = {
                "status": "skipped",
                "reason": "insufficient class variation",
            }
            continue

        X_train = t_train[numeric_features + categorical_features]
        X_test = t_test[numeric_features + categorical_features]

        # ── Train two candidate models ──
        lr_pipe = Pipeline([
            ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
            ("model", LogisticRegression(
                max_iter=LOGISTIC_MAX_ITER,
                class_weight="balanced",
                solver="liblinear",
            )),
        ])
        hgbc_pipe = Pipeline([
            ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
            ("model", HistGradientBoostingClassifier(
                max_depth=BOOSTING_MAX_DEPTH,
                learning_rate=BOOSTING_LEARNING_RATE,
                max_iter=BOOSTING_MAX_ITER,
            )),
        ])

        lr_pipe.fit(X_train, y_train)
        hgbc_pipe.fit(X_train, y_train)

        lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
        hgbc_prob = hgbc_pipe.predict_proba(X_test)[:, 1]

        lr_metrics = _evaluate(y_test, lr_prob)
        hgbc_metrics = _evaluate(y_test, hgbc_prob)

        # Select by average precision
        def _ap(m: dict) -> float:
            v = m.get("average_precision")
            return v if v is not None else -1.0

        if _ap(hgbc_metrics) >= _ap(lr_metrics):
            best_name, best_pipe = "hist_gradient_boosting", hgbc_pipe
        else:
            best_name, best_pipe = "logistic_regression", lr_pipe

        selected_metrics = lr_metrics if best_name == "logistic_regression" else hgbc_metrics
        logger.info("  Selected %s (AP=%.4f)", best_name, _ap(selected_metrics))

        # ── Calibrate ──
        calibrated = CalibratedClassifierCV(best_pipe, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)
        cal_prob = calibrated.predict_proba(X_test)[:, 1]

        # ── Threshold tuning ──
        opt_threshold = _find_optimal_threshold(y_test, cal_prob)
        cal_metrics = _evaluate(y_test, cal_prob, threshold=opt_threshold)

        logger.info("  Calibrated metrics (thr=%.3f): F1=%.4f, AP=%.4f",
                     opt_threshold, cal_metrics["f1"], cal_metrics.get("average_precision", 0))

        # ── Feature importance from the correct model ──
        feat_imp = _extract_feature_importance(best_pipe, best_name)

        # ── Time-series CV ──
        cv_results = _time_series_cv(
            model_frame, target, numeric_features, categorical_features,
        )

        # ── Save artefact ──
        artifact_path = cfg.models_dir / f"{target}_{best_name}.joblib"
        joblib.dump(calibrated, artifact_path)
        artifact_hash = _sha256_file(artifact_path)

        payload["targets"][target] = {
            "status": "trained",
            "selected_model": best_name,
            "optimal_threshold": opt_threshold,
            "artifact_sha256": artifact_hash,
            "logistic_regression": lr_metrics,
            "hist_gradient_boosting": hgbc_metrics,
            "calibrated": cal_metrics,
            "cross_validation": cv_results,
            "top_features": feat_imp,
        }

        # ── Scores ──
        target_scores = pd.DataFrame({
            "order_id": t_test["order_id"].astype("string"),
            f"{target}_score": cal_prob,
            f"{target}_prediction": (cal_prob >= opt_threshold).astype(int),
        })
        all_scores = all_scores.merge(
            target_scores, on="order_id", how="left", suffixes=("", "_new"),
        )
        for suffix_col in [f"{target}_score_new", f"{target}_prediction_new"]:
            base_col = suffix_col.replace("_new", "")
            if suffix_col in all_scores.columns:
                all_scores[base_col] = all_scores[suffix_col].fillna(
                    all_scores[base_col]
                )
                all_scores = all_scores.drop(columns=[suffix_col])

    # ── Persist ──
    all_scores.to_parquet(scores_path, index=False)
    metrics_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )

    # ── Append to history ──
    _append_to_history(payload, cfg)

    logger.info("Training complete. Artefacts in %s", cfg.models_dir)
    return payload


def _append_to_history(payload: dict[str, Any], cfg: AppConfig) -> None:
    """Append the current run's payload to the model-history file."""
    history_path = cfg.models_dir / MODEL_HISTORY_FILE
    history: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            history = []
    history.append(payload)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ── Load helpers ────────────────────────────────────────────────────────────

def load_model_metrics(config: AppConfig | None = None) -> dict[str, Any]:
    """Load model metrics from disk."""
    cfg = config or get_config()
    path = cfg.models_dir / METRICS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Model metrics are missing at {path}. "
            "Run `python -m archive_analytics train` first."
        )
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    payload.setdefault("targets", {})
    for t in TARGETS:
        payload["targets"].setdefault(
            t, {"status": "skipped", "reason": "artifact did not include this target"},
        )
    return payload


def load_risk_scores(config: AppConfig | None = None) -> pd.DataFrame:
    """Load the scored order-level risk table."""
    cfg = config or get_config()
    path = cfg.models_dir / SCORES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Risk scores are missing at {path}. "
            "Run `python -m archive_analytics train` first."
        )
    scores = pd.read_parquet(path)
    for t in TARGETS:
        if f"{t}_score" not in scores.columns:
            scores[f"{t}_score"] = 0.0
        if f"{t}_prediction" not in scores.columns:
            scores[f"{t}_prediction"] = 0
    return scores
