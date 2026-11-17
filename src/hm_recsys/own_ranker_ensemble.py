"""LightGBM/CatBoost ranker ablation on the project's saved four-week pool."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import CandidateUnion
from .data import (
    FINAL_WINDOW,
    TRAIN_WINDOW,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_recent_transactions,
    slice_window,
    truth_by_customer,
)
from .hybrid_pipeline import _pair_arrays, _tree_features
from .metrics import evaluate_rankings
from .pipeline import _select_labeled_examples
from .top_solution_reproductions import _catboost_frame, _rank_fusion


@dataclass(frozen=True)
class OwnEnsembleConfig:
    seed: int = 42
    negatives_per_positive: int = 20
    lgbm_estimators: int = 250
    catboost_iterations: int = 500
    catboost_depth: int = 8
    learning_rate: float = 0.05
    num_leaves: int = 31
    score_batch_size: int = 100_000


def _load_pool(path: Path) -> tuple[list[str], np.ndarray, CandidateUnion]:
    saved = np.load(path, allow_pickle=True)
    return (
        saved["customer_ids"].astype(str).tolist(),
        saved["article_ids"].astype(str),
        CandidateUnion(
            items=saved["candidate_item_rows"],
            source_rr=saved["source_rr"],
            source_names=tuple(saved["source_names"].astype(str).tolist()),
        ),
    )


def _labels(features, users, union, truth, user_rows, positions) -> np.ndarray:
    return np.fromiter(
        (
            features.article_ids[union.items[user_row, position]] in truth[users[user_row]]
            for user_row, position in zip(user_rows, positions, strict=True)
        ),
        dtype=np.int32,
    )


def _score_models(
    lightgbm,
    catboost,
    categorical_columns: int,
    features,
    users: list[str],
    union: CandidateUnion,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    width = union.items.shape[1]
    lightgbm_scores = np.empty((len(users), width), dtype=np.float32)
    catboost_scores = np.empty((len(users), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(users), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(users))
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        cats, dense = _pair_arrays(
            features,
            users,
            union,
            flat_users[start:stop],
            flat_positions[start:stop],
        )
        values = _tree_features(cats, dense)
        lightgbm_scores.reshape(-1)[start:stop] = lightgbm.predict(values)
        catboost_scores.reshape(-1)[start:stop] = catboost.predict_proba(
            _catboost_frame(values, categorical_columns)
        )[:, 1]
    return lightgbm_scores, catboost_scores


def _ranked(scores: np.ndarray, items: np.ndarray) -> np.ndarray:
    return np.take_along_axis(
        items, np.argsort(-scores, axis=1, kind="stable"), axis=1
    )


def _metrics(users, rows, features, truth, active) -> dict:
    rankings = {
        user: features.article_ids[rows[row]].tolist()
        for row, user in enumerate(users)
    }
    return evaluate_rankings(rankings, truth, active, rows.shape[1])


def run_own_ranker_ensemble(root: Path, config: OwnEnsembleConfig) -> Path:
    started = time.perf_counter()
    source_dir = root / "artifacts" / "hybrid_expanded_repeat_experiment"
    train_users, train_articles, train_union = _load_pool(
        source_dir / "train_candidate_pool.npz"
    )
    final_users, final_articles, final_union = _load_pool(
        source_dir / "final_candidate_pool.npz"
    )
    recent = load_recent_transactions(root / "Data", root / "artifacts" / "cache")
    train_context, train_labels = slice_window(recent, TRAIN_WINDOW)
    final_context, final_labels = slice_window(recent, FINAL_WINDOW)
    customers, articles = load_metadata(root / "Data")
    encoder = fit_feature_encoder(customers, articles)
    train_features = build_snapshot_features(
        train_context, customers, articles, encoder, TRAIN_WINDOW.label_start
    )
    final_features = build_snapshot_features(
        final_context, customers, articles, encoder, FINAL_WINDOW.label_start
    )
    if not np.array_equal(train_articles, train_features.article_ids.astype(str)):
        raise ValueError("Saved training article rows do not match four-week features")
    if not np.array_equal(final_articles, final_features.article_ids.astype(str)):
        raise ValueError("Saved final article rows do not match four-week features")
    train_truth_all = truth_by_customer(train_labels)
    final_truth_all = truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in train_users}
    final_truth = {user: final_truth_all[user] for user in final_users}
    user_rows, positions, sampling = _select_labeled_examples(
        train_users,
        train_union.items,
        train_features.article_ids,
        train_truth,
        config.negatives_per_positive,
        config.seed + 10,
    )
    cats, dense = _pair_arrays(
        train_features, train_users, train_union, user_rows, positions
    )
    train_x = _tree_features(cats, dense)
    labels = _labels(
        train_features, train_users, train_union, train_truth, user_rows, positions
    )
    groups = np.bincount(user_rows)
    groups = groups[groups > 0]
    from lightgbm import LGBMRanker

    lightgbm = LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=config.lgbm_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        lambdarank_truncation_level=12,
        random_state=config.seed + 12,
        n_jobs=8,
        verbosity=-1,
    )
    lightgbm.fit(
        train_x,
        labels,
        group=groups.tolist(),
        categorical_feature=list(range(cats.shape[1])),
    )
    from catboost import CatBoostClassifier

    categorical_columns = cats.shape[1]
    catboost = CatBoostClassifier(
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.learning_rate,
        loss_function="Logloss",
        random_seed=config.seed + 13,
        l2_leaf_reg=3.0,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )
    catboost.fit(
        _catboost_frame(train_x, categorical_columns),
        labels,
        cat_features=list(range(categorical_columns)),
    )
    lightgbm_scores, catboost_scores = _score_models(
        lightgbm,
        catboost,
        categorical_columns,
        final_features,
        final_users,
        final_union,
        config.score_batch_size,
    )
    fused_scores = _rank_fusion(lightgbm_scores, catboost_scores)
    ranked = {
        "lightgbm": _ranked(lightgbm_scores, final_union.items),
        "catboost": _ranked(catboost_scores, final_union.items),
        "equal_rank_fusion": _ranked(fused_scores, final_union.items),
    }
    active = set(final_context["customer_id"].astype(str).unique())
    metrics = {
        name: _metrics(final_users, rows, final_features, final_truth, active)
        for name, rows in ranked.items()
    }
    output_dir = root / "artifacts" / "own_four_week_ranker_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in ranked.items():
        pd.DataFrame(
            {
                "customer_id": final_users,
                "prediction": [
                    " ".join(final_features.article_ids[row[:12]]) for row in rows
                ],
            }
        ).to_csv(output_dir / f"{name}_top12.csv", index=False)
    result = {
        "scope": "project's own saved four-week 1,012-candidate pool; no retrieval changes",
        "candidate_source": "artifacts/hybrid_expanded_repeat_experiment",
        "config": asdict(config),
        "windows": {
            "train": asdict(TRAIN_WINDOW),
            "final": asdict(FINAL_WINDOW),
        },
        "training": {
            **sampling,
            "examples": len(labels),
            "positives": int(labels.sum()),
        },
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result_path

