"""Parallel multi-source retrieval, controlled repeats, and DLRM ranking."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .candidates import (
    SOURCE_NAMES,
    CandidateSource,
    CandidateUnion,
    build_co_purchase_map,
    co_purchase_candidates,
    fit_als,
    fit_two_tower_source,
    popularity_candidates,
    repeat_candidates,
    segment_popularity_candidates,
    two_tower_candidates,
    union_candidates,
)
from .data import (
    ARTICLE_CAT_COLUMNS,
    USER_CAT_COLUMNS,
    SnapshotFeatures,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_transactions_since,
    sample_customers,
    slice_window,
    truth_by_customer,
    window_with_history,
)
from .metrics import evaluate_rankings
from .models import DLRMRanker, fit_binary_model, fit_mf
from .pipeline import _select_labeled_examples


@dataclass(frozen=True)
class HybridConfig:
    profile: str
    candidate_policy: str = "split_new_repeat"
    history_days: int = 28
    seed: int = 42
    max_train_users: int = 6_000
    max_eval_users: int = 2_000
    max_mf_pairs: int | None = None
    mf_k: int = 500
    als_k: int = 500
    co_purchase_k: int = 100
    popularity_k: int = 100
    segment_k: int = 100
    two_tower_k: int = 200
    union_k: int = 1_000
    final_union_k: int | None = None
    repeat_k: int = 12
    mf_dim: int = 32
    mf_epochs: int = 3
    als_factors: int = 64
    als_iterations: int = 8
    als_regularization: float = 0.01
    max_two_tower_pairs: int = 100_000
    two_tower_epochs: int = 2
    dlrm_epochs: int = 2
    lgbm_estimators: int = 250
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    negatives_per_positive: int = 20
    batch_size: int = 4_096
    retrieval_batch_size: int = 128
    learning_rate: float = 0.003


HYBRID_PROFILES = {
    "hybrid_smoke": HybridConfig(
        profile="hybrid_smoke",
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        mf_k=50,
        als_k=50,
        co_purchase_k=30,
        popularity_k=30,
        segment_k=30,
        two_tower_k=50,
        union_k=100,
        repeat_k=5,
        mf_dim=16,
        mf_epochs=1,
        als_factors=16,
        als_iterations=2,
        max_two_tower_pairs=10_000,
        two_tower_epochs=1,
        dlrm_epochs=1,
        lgbm_estimators=50,
        negatives_per_positive=5,
        batch_size=2_048,
        retrieval_batch_size=64,
    ),
    "hybrid_experiment": HybridConfig(profile="hybrid_experiment"),
    "hybrid_unsplit_smoke": HybridConfig(
        profile="hybrid_unsplit_smoke",
        candidate_policy="direct_union",
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        mf_k=50,
        als_k=50,
        co_purchase_k=30,
        popularity_k=30,
        segment_k=30,
        two_tower_k=50,
        union_k=100,
        repeat_k=0,
        mf_dim=16,
        mf_epochs=1,
        als_factors=16,
        als_iterations=2,
        max_two_tower_pairs=10_000,
        two_tower_epochs=1,
        dlrm_epochs=1,
        lgbm_estimators=50,
        negatives_per_positive=5,
        batch_size=2_048,
        retrieval_batch_size=64,
    ),
    "hybrid_unsplit_experiment": HybridConfig(
        profile="hybrid_unsplit_experiment",
        candidate_policy="direct_union",
        repeat_k=0,
    ),
    "hybrid_allow_seen_repeat_smoke": HybridConfig(
        profile="hybrid_allow_seen_repeat_smoke",
        candidate_policy="allow_seen_plus_repeat",
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        mf_k=50,
        als_k=50,
        co_purchase_k=30,
        popularity_k=30,
        segment_k=30,
        two_tower_k=50,
        union_k=100,
        repeat_k=5,
        mf_dim=16,
        mf_epochs=1,
        als_factors=16,
        als_iterations=2,
        max_two_tower_pairs=10_000,
        two_tower_epochs=1,
        dlrm_epochs=1,
        lgbm_estimators=50,
        negatives_per_positive=5,
        batch_size=2_048,
        retrieval_batch_size=64,
    ),
    "hybrid_allow_seen_repeat_experiment": HybridConfig(
        profile="hybrid_allow_seen_repeat_experiment",
        candidate_policy="allow_seen_plus_repeat",
    ),
    "hybrid_expanded_repeat_smoke": HybridConfig(
        profile="hybrid_expanded_repeat_smoke",
        candidate_policy="split_new_repeat",
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        mf_k=50,
        als_k=50,
        co_purchase_k=30,
        popularity_k=30,
        segment_k=30,
        two_tower_k=50,
        union_k=100,
        final_union_k=105,
        repeat_k=5,
        mf_dim=16,
        mf_epochs=1,
        als_factors=16,
        als_iterations=2,
        max_two_tower_pairs=10_000,
        two_tower_epochs=1,
        dlrm_epochs=1,
        lgbm_estimators=50,
        negatives_per_positive=5,
        batch_size=2_048,
        retrieval_batch_size=64,
    ),
    "hybrid_expanded_repeat_experiment": HybridConfig(
        profile="hybrid_expanded_repeat_experiment",
        candidate_policy="split_new_repeat",
        final_union_k=1_012,
    ),
    "hybrid_allow_seen_repeat_expanded_experiment": HybridConfig(
        profile="hybrid_allow_seen_repeat_expanded_experiment",
        candidate_policy="allow_seen_plus_repeat",
        final_union_k=1_012,
    ),
    "hybrid_six_week_smoke": HybridConfig(
        profile="hybrid_six_week_smoke",
        history_days=42,
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        mf_k=50,
        als_k=50,
        co_purchase_k=30,
        popularity_k=30,
        segment_k=30,
        two_tower_k=50,
        union_k=100,
        final_union_k=105,
        repeat_k=5,
        mf_dim=16,
        mf_epochs=1,
        als_factors=16,
        als_iterations=2,
        max_two_tower_pairs=10_000,
        two_tower_epochs=1,
        dlrm_epochs=1,
        lgbm_estimators=50,
        negatives_per_positive=5,
        batch_size=2_048,
        retrieval_batch_size=64,
    ),
    "hybrid_six_week_experiment": HybridConfig(
        profile="hybrid_six_week_experiment",
        history_days=42,
        candidate_policy="split_new_repeat",
        final_union_k=1_012,
    ),
    "hybrid_eight_week_smoke": HybridConfig(
        profile="hybrid_eight_week_smoke",
        history_days=56,
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        mf_k=50,
        als_k=50,
        co_purchase_k=30,
        popularity_k=30,
        segment_k=30,
        two_tower_k=50,
        union_k=100,
        final_union_k=105,
        repeat_k=5,
        mf_dim=16,
        mf_epochs=1,
        als_factors=16,
        als_iterations=2,
        max_two_tower_pairs=10_000,
        two_tower_epochs=1,
        dlrm_epochs=1,
        lgbm_estimators=50,
        negatives_per_positive=5,
        batch_size=2_048,
        retrieval_batch_size=64,
    ),
    "hybrid_eight_week_experiment": HybridConfig(
        profile="hybrid_eight_week_experiment",
        history_days=56,
        candidate_policy="split_new_repeat",
        final_union_k=1_012,
    ),
}


DISCOVERY_NAMES = set(SOURCE_NAMES) - {"repeat"}


@dataclass
class SnapshotCandidates:
    sources: list[CandidateSource]
    new_union: CandidateUnion
    full_union: CandidateUnion
    training: dict
    runtime: dict[str, float]


def _fit_bpr(
    context: pd.DataFrame,
    features: SnapshotFeatures,
    config: HybridConfig,
    seed: int,
):
    pairs = context[["customer_id", "article_id"]].drop_duplicates()
    customer_ids = np.sort(context["customer_id"].astype(str).unique())
    return fit_mf(
        customer_ids=customer_ids,
        item_ids=features.article_ids,
        popularity=features.item_popularity,
        pair_customers=pairs["customer_id"].astype(str).to_numpy(),
        pair_items=pairs["article_id"].astype(str).to_numpy(),
        dim=config.mf_dim,
        epochs=config.mf_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=seed,
        max_pairs=config.max_mf_pairs,
    )


def _build_snapshot_candidates(
    context: pd.DataFrame,
    cutoff: str,
    features: SnapshotFeatures,
    customers: pd.DataFrame,
    encoder,
    target_customers: list[str],
    config: HybridConfig,
    seed: int,
) -> SnapshotCandidates:
    runtime: dict[str, float] = {}
    training: dict = {}

    mark = time.perf_counter()
    bpr, bpr_losses = _fit_bpr(context, features, config, seed)
    bpr_rows, _ = bpr.candidates(target_customers, config.mf_k)
    runtime["bpr"] = time.perf_counter() - mark
    training["bpr_losses"] = bpr_losses

    mark = time.perf_counter()
    als = fit_als(
        context,
        features,
        config.als_factors,
        config.als_iterations,
        config.als_regularization,
        seed + 1,
    )
    als_rows = als.candidates(
        target_customers, config.als_k, config.retrieval_batch_size
    )
    runtime["implicit_als"] = time.perf_counter() - mark

    mark = time.perf_counter()
    co_map = build_co_purchase_map(context, features)
    co_rows = co_purchase_candidates(
        context, features, target_customers, co_map, config.co_purchase_k
    )
    runtime["co_purchase"] = time.perf_counter() - mark
    training["co_purchase_items_with_neighbors"] = len(co_map)

    mark = time.perf_counter()
    pop_rows = popularity_candidates(features, len(target_customers), config.popularity_k)
    segment_rows = segment_popularity_candidates(
        context, customers, features, target_customers, config.segment_k
    )
    runtime["popularity_sources"] = time.perf_counter() - mark

    mark = time.perf_counter()
    user_cards = [encoder.cardinality(name) for name in USER_CAT_COLUMNS]
    item_cards = [encoder.cardinality(name) for name in ARTICLE_CAT_COLUMNS]
    tower, tower_losses, tower_stats = fit_two_tower_source(
        context,
        features,
        user_cards,
        item_cards,
        config.max_two_tower_pairs,
        config.two_tower_epochs,
        config.batch_size,
        config.learning_rate,
        seed + 2,
    )
    tower_rows = two_tower_candidates(
        tower,
        features,
        target_customers,
        config.two_tower_k,
        config.retrieval_batch_size,
    )
    runtime["two_tower"] = time.perf_counter() - mark
    training["two_tower_losses"] = tower_losses
    training["two_tower_examples"] = tower_stats

    discovery = [
        CandidateSource("bpr_mf", bpr_rows),
        CandidateSource("implicit_als", als_rows),
        CandidateSource("co_purchase", co_rows),
        CandidateSource("popularity", pop_rows),
        CandidateSource("segment_popularity", segment_rows),
        CandidateSource("two_tower", tower_rows),
    ]
    mark = time.perf_counter()
    if config.candidate_policy in {"split_new_repeat", "allow_seen_plus_repeat"}:
        repeat_mark = time.perf_counter()
        repeat_rows = repeat_candidates(
            context, features, target_customers, cutoff, config.repeat_k
        )
        runtime["repeat"] = time.perf_counter() - repeat_mark
        repeat = CandidateSource("repeat", repeat_rows)
        sources = [*discovery, repeat]
        mark = time.perf_counter()
        remove_history_from = (
            DISCOVERY_NAMES
            if config.candidate_policy == "split_new_repeat"
            else set()
        )
        new_union = union_candidates(
            target_customers,
            discovery,
            features,
            config.union_k,
            remove_history_from=remove_history_from,
        )
        full_union = union_candidates(
            target_customers,
            sources,
            features,
            config.final_union_k or config.union_k,
            remove_history_from=remove_history_from,
        )
    elif config.candidate_policy == "direct_union":
        sources = discovery
        full_union = union_candidates(
            target_customers,
            sources,
            features,
            config.union_k,
            remove_history_from=set(),
        )
        new_union = full_union
    else:
        raise ValueError(f"Unknown candidate policy: {config.candidate_policy}")
    runtime["candidate_unions"] = time.perf_counter() - mark
    return SnapshotCandidates(
        sources=sources,
        new_union=new_union,
        full_union=full_union,
        training=training,
        runtime=runtime,
    )


def _pair_arrays(
    features: SnapshotFeatures,
    customers: list[str],
    candidates: CandidateUnion,
    user_rows: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    item_rows = candidates.items[user_rows, positions]
    pair_customers = [customers[row] for row in user_rows]
    metadata_rows = np.fromiter(
        (features.customer_to_row.get(customer, -1) for customer in pair_customers),
        dtype=np.int64,
    )
    user_cats = np.zeros((len(user_rows), features.user_cats.shape[1]), dtype=np.int64)
    user_dense = np.zeros((len(user_rows), features.user_dense.shape[1]), dtype=np.float32)
    present = metadata_rows >= 0
    user_cats[present] = features.user_cats[metadata_rows[present]]
    user_dense[present] = features.user_dense[metadata_rows[present]]
    item_cats = features.item_cats[item_rows]
    item_dense = features.item_dense[item_rows]
    source_rr = candidates.source_rr[user_rows, positions]
    source_count = (source_rr > 0).sum(axis=1, keepdims=True).astype(np.float32)
    repeat_flag = np.fromiter(
        (
            features.article_ids[item] in features.purchased.get(customer, set())
            for customer, item in zip(pair_customers, item_rows, strict=True)
        ),
        dtype=np.float32,
    )[:, None]
    union_rank = (positions / max(candidates.items.shape[1] - 1, 1)).astype(np.float32)[:, None]
    dense = np.column_stack(
        [user_dense, item_dense, source_rr, source_count, repeat_flag, union_rank]
    ).astype(np.float32)
    cats = np.column_stack([user_cats, item_cats]).astype(np.int64)
    return cats, dense


def _score_dlrm(
    model: DLRMRanker,
    features: SnapshotFeatures,
    customers: list[str],
    candidates: CandidateUnion,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    width = candidates.items.shape[1]
    output = np.empty((len(customers), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(customers), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(customers))
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        rows = flat_users[start:stop]
        positions = flat_positions[start:stop]
        cats, dense = _pair_arrays(features, customers, candidates, rows, positions)
        with torch.no_grad():
            output.reshape(-1)[start:stop] = model(
                torch.as_tensor(cats, dtype=torch.long, device=device),
                torch.as_tensor(dense, dtype=torch.float32, device=device),
            ).cpu().numpy()
    return output


def _tree_features(cats: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """LightGBM input with categorical fields first and dense fields second."""
    return np.column_stack([cats, dense]).astype(np.float32)


def _tree_feature_names(source_names: tuple[str, ...]) -> list[str]:
    return [
        *(f"user_{name}" for name in USER_CAT_COLUMNS),
        *(f"item_{name}" for name in ARTICLE_CAT_COLUMNS),
        "user_log_activity",
        "user_days_since_purchase",
        "user_age",
        "user_preferred_channel",
        "item_log_popularity",
        "item_average_price",
        "item_days_since_purchase",
        *(f"rr_{name}" for name in source_names),
        "source_support_count",
        "previously_purchased_flag",
        "union_rank",
    ]


def _score_lightgbm(
    model: Any,
    features: SnapshotFeatures,
    customers: list[str],
    candidates: CandidateUnion,
    batch_size: int,
) -> np.ndarray:
    width = candidates.items.shape[1]
    output = np.empty((len(customers), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(customers), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(customers))
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        rows = flat_users[start:stop]
        positions = flat_positions[start:stop]
        cats, dense = _pair_arrays(features, customers, candidates, rows, positions)
        output.reshape(-1)[start:stop] = model.predict(
            _tree_features(cats, dense)
        ).astype(np.float32)
    return output


def _rankings(
    customers: list[str], rows: np.ndarray, article_ids: np.ndarray
) -> dict[str, list[str]]:
    return {
        customer: article_ids[rows[user_row]].tolist()
        for user_row, customer in enumerate(customers)
    }


def _metrics(
    customers: list[str],
    rows: np.ndarray,
    article_ids: np.ndarray,
    truth: dict[str, set[str]],
    known: set[str],
) -> dict:
    return {
        "candidate_k": rows.shape[1],
        "cohorts": evaluate_rankings(
            _rankings(customers, rows, article_ids), truth, known, rows.shape[1]
        ),
    }


def _filtered_source_rows(
    source: CandidateSource,
    customers: list[str],
    features: SnapshotFeatures,
) -> np.ndarray:
    return union_candidates(
        customers,
        [source],
        features,
        source.items.shape[1],
        remove_history_from={source.name} if source.name != "repeat" else set(),
    ).items


def _save_candidate_union(
    path: Path,
    customers: list[str],
    features: SnapshotFeatures,
    candidates: CandidateUnion,
) -> None:
    """Persist a candidate pool and provenance without touching other runs."""
    np.savez_compressed(
        path,
        customer_ids=np.asarray(customers),
        article_ids=features.article_ids,
        candidate_item_rows=candidates.items,
        source_rr=candidates.source_rr,
        source_names=np.asarray(candidates.source_names),
    )


def run_hybrid_experiment(
    root: Path, config: HybridConfig, save_candidates: bool = False
) -> Path:
    started = time.perf_counter()
    artifacts = root / "artifacts"
    run_dir = artifacts / config.profile
    run_dir.mkdir(parents=True, exist_ok=True)

    mark = time.perf_counter()
    train_window = window_with_history("2020-09-09", "2020-09-16", config.history_days)
    final_window = window_with_history("2020-09-16", "2020-09-23", config.history_days)
    transactions = load_transactions_since(
        root / "Data", artifacts / "cache", train_window.context_start
    )
    customers, articles = load_metadata(root / "Data")
    encoder = fit_feature_encoder(customers, articles)
    train_context, train_labels = slice_window(transactions, train_window)
    final_context, final_labels = slice_window(transactions, final_window)
    train_features = build_snapshot_features(
        train_context, customers, articles, encoder, train_window.label_start
    )
    final_features = build_snapshot_features(
        final_context, customers, articles, encoder, final_window.label_start
    )
    load_seconds = time.perf_counter() - mark

    train_known = set(train_context["customer_id"].astype(str).unique())
    train_users = sample_customers(train_labels, train_known, config.max_train_users, config.seed)
    train_truth_all = truth_by_customer(train_labels)
    train_truth = {customer: train_truth_all[customer] for customer in train_users}

    mark = time.perf_counter()
    train_candidates = _build_snapshot_candidates(
        train_context,
        train_window.label_start,
        train_features,
        customers,
        encoder,
        train_users,
        config,
        config.seed,
    )
    train_retrieval_seconds = time.perf_counter() - mark

    mark = time.perf_counter()
    user_rows, positions, example_stats = _select_labeled_examples(
        train_users,
        train_candidates.full_union.items,
        train_features.article_ids,
        train_truth,
        config.negatives_per_positive,
        config.seed + 10,
    )
    dlrm_cats, dlrm_dense = _pair_arrays(
        train_features,
        train_users,
        train_candidates.full_union,
        user_rows,
        positions,
    )
    dlrm_labels = np.fromiter(
        (
            train_features.article_ids[
                train_candidates.full_union.items[user_row, position]
            ]
            in train_truth[train_users[user_row]]
            for user_row, position in zip(user_rows, positions, strict=True)
        ),
        dtype=np.float32,
    )
    cardinalities = [encoder.cardinality(name) for name in USER_CAT_COLUMNS] + [
        encoder.cardinality(name) for name in ARTICLE_CAT_COLUMNS
    ]
    dlrm = DLRMRanker(cardinalities, dlrm_dense.shape[1])
    dlrm, dlrm_losses = fit_binary_model(
        dlrm,
        (
            torch.as_tensor(dlrm_cats, dtype=torch.long),
            torch.as_tensor(dlrm_dense, dtype=torch.float32),
        ),
        torch.as_tensor(dlrm_labels, dtype=torch.float32),
        config.dlrm_epochs,
        config.batch_size,
        config.learning_rate,
        config.seed + 11,
    )
    dlrm_training_seconds = time.perf_counter() - mark

    mark = time.perf_counter()
    lgbm_features = _tree_features(dlrm_cats, dlrm_dense)
    lgbm_groups = np.bincount(user_rows, minlength=len(train_users))
    lgbm_groups = lgbm_groups[lgbm_groups > 0]
    from lightgbm import LGBMRanker

    lightgbm = LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=config.lgbm_estimators,
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
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
        lgbm_features,
        dlrm_labels.astype(np.int32),
        group=lgbm_groups.tolist(),
        categorical_feature=list(range(dlrm_cats.shape[1])),
    )
    lightgbm_training_seconds = time.perf_counter() - mark

    final_known = set(final_context["customer_id"].astype(str).unique())
    eval_users = sample_customers(final_labels, final_known, config.max_eval_users, config.seed)
    final_truth_all = truth_by_customer(final_labels)
    final_truth = {customer: final_truth_all[customer] for customer in eval_users}

    mark = time.perf_counter()
    final_candidates = _build_snapshot_candidates(
        final_context,
        final_window.label_start,
        final_features,
        customers,
        encoder,
        eval_users,
        config,
        config.seed + 20,
    )
    final_retrieval_seconds = time.perf_counter() - mark
    if save_candidates:
        _save_candidate_union(
            run_dir / "train_candidate_pool.npz",
            train_users,
            train_features,
            train_candidates.full_union,
        )
        _save_candidate_union(
            run_dir / "final_candidate_pool.npz",
            eval_users,
            final_features,
            final_candidates.full_union,
        )
    mark = time.perf_counter()
    dlrm_scores = _score_dlrm(
        dlrm,
        final_features,
        eval_users,
        final_candidates.full_union,
        config.batch_size,
    )
    dlrm_order = np.argsort(-dlrm_scores, axis=1, kind="stable")
    final_dlrm_items = np.take_along_axis(
        final_candidates.full_union.items, dlrm_order, axis=1
    )
    dlrm_scoring_seconds = time.perf_counter() - mark
    mark = time.perf_counter()
    lightgbm_scores = _score_lightgbm(
        lightgbm,
        final_features,
        eval_users,
        final_candidates.full_union,
        config.batch_size,
    )
    lightgbm_order = np.argsort(-lightgbm_scores, axis=1, kind="stable")
    final_lightgbm_items = np.take_along_axis(
        final_candidates.full_union.items, lightgbm_order, axis=1
    )
    lightgbm_scoring_seconds = time.perf_counter() - mark

    metrics: dict[str, dict] = {}
    for source in final_candidates.sources:
        if source.name == "repeat":
            repeat_rankings = {
                customer: [
                    final_features.article_ids[item]
                    for item in source.items[row]
                    if item >= 0
                ]
                for row, customer in enumerate(eval_users)
            }
            metrics[source.name] = {
                "candidate_k": source.items.shape[1],
                "cohorts": evaluate_rankings(
                    repeat_rankings,
                    final_truth,
                    final_known,
                    source.items.shape[1],
                ),
            }
        elif config.candidate_policy == "split_new_repeat":
            filtered = _filtered_source_rows(source, eval_users, final_features)
            metrics[source.name] = _metrics(
                eval_users, filtered, final_features.article_ids, final_truth, final_known
            )
        else:
            metrics[source.name] = _metrics(
                eval_users,
                source.items,
                final_features.article_ids,
                final_truth,
                final_known,
            )
    if config.candidate_policy == "split_new_repeat":
        metrics["new_item_union"] = _metrics(
            eval_users,
            final_candidates.new_union.items,
            final_features.article_ids,
            final_truth,
            final_known,
        )
        metrics["new_plus_repeat_union"] = _metrics(
            eval_users,
            final_candidates.full_union.items,
            final_features.article_ids,
            final_truth,
            final_known,
        )
    elif config.candidate_policy == "allow_seen_plus_repeat":
        metrics["allow_seen_discovery_union"] = _metrics(
            eval_users,
            final_candidates.new_union.items,
            final_features.article_ids,
            final_truth,
            final_known,
        )
        metrics["allow_seen_plus_repeat_union"] = _metrics(
            eval_users,
            final_candidates.full_union.items,
            final_features.article_ids,
            final_truth,
            final_known,
        )
    else:
        metrics["direct_candidate_union"] = _metrics(
            eval_users,
            final_candidates.full_union.items,
            final_features.article_ids,
            final_truth,
            final_known,
        )
    metrics["dlrm"] = _metrics(
        eval_users,
        final_dlrm_items,
        final_features.article_ids,
        final_truth,
        final_known,
    )
    metrics["lightgbm_lambdarank"] = _metrics(
        eval_users,
        final_lightgbm_items,
        final_features.article_ids,
        final_truth,
        final_known,
    )

    relevant_pairs = sum(len(items) for items in final_truth.values())
    catalog = set(final_features.article_ids)
    summary = {
        "config": asdict(config),
        "architecture": {
            "candidate_policy": config.candidate_policy,
            "new_item_union_capacity": config.union_k,
            "final_union_capacity": config.final_union_k or config.union_k,
            "discovery_sources": list(SOURCE_NAMES[:-1]),
            "discovery_policy": {
                "split_new_repeat": "remove exact articles purchased in snapshot history",
                "direct_union": "allow each generator to return both previously purchased and new articles",
                "allow_seen_plus_repeat": "allow each generator to return both previously purchased and new articles, while retaining the dedicated repeat source",
            }[config.candidate_policy],
            "repeat_policy": (
                "frequency times 14-day exponential recency decay; bounded repeat_k"
                if config.candidate_policy
                in {"split_new_repeat", "allow_seen_plus_repeat"}
                else "no separate repeat generator"
            ),
            "union_priority": "sum of per-source reciprocal ranks, deterministic item-row tie break",
            "rankers": "DLRM and LightGBM LambdaRank over identical user, item, source provenance, repeat, support-count, and union-rank features",
        },
        "windows": {
            "ranker_training": asdict(train_window),
            "final_evaluation": asdict(final_window),
        },
        "dataset": {
            "train_context_rows": len(train_context),
            "train_label_rows": len(train_labels),
            "final_context_rows": len(final_context),
            "final_validation_rows": len(final_labels),
            "train_catalog_items": len(train_features.article_ids),
            "final_catalog_items": len(final_features.article_ids),
            "train_customers_sampled": len(train_users),
            "evaluation_customers_sampled": len(eval_users),
            "evaluation_history_customers": sum(user in final_known for user in eval_users),
            "evaluation_cold_start_customers": sum(user not in final_known for user in eval_users),
            "validation_relevant_pairs": relevant_pairs,
            "validation_pairs_in_catalog": sum(
                len(items & catalog) for items in final_truth.values()
            ),
        },
        "training": {
            "candidate_objectives": {
                "bpr_mf": "pairwise BPR on context purchases and uniform unobserved negatives",
                "implicit_als": "confidence-weighted sparse interaction reconstruction",
                "two_tower": "BCE on context purchases and uniform full-catalog negatives",
                "dlrm": "weighted BCE on next-week positives and negatives sampled from the matching candidate union",
                "lightgbm": "LambdaRank grouped by customer on the same candidate-aware examples; truncation level 12",
            },
            "train_snapshot": train_candidates.training,
            "final_snapshot": final_candidates.training,
            "dlrm_losses": dlrm_losses,
            "dlrm_examples": {
                **example_stats,
                "rows": len(dlrm_labels),
                "positives": int(dlrm_labels.sum()),
            },
            "lightgbm": {
                "groups": len(lgbm_groups),
                "estimators": config.lgbm_estimators,
                "feature_importance_gain": dict(
                    sorted(
                        zip(
                            _tree_feature_names(train_candidates.full_union.source_names),
                            lightgbm.booster_.feature_importance(importance_type="gain").tolist(),
                            strict=True,
                        ),
                        key=lambda pair: -pair[1],
                    )
                ),
            },
        },
        "metrics": metrics,
        "runtime_seconds": {
            "load_and_features": load_seconds,
            "train_snapshot_retrieval": train_retrieval_seconds,
            "dlrm_training": dlrm_training_seconds,
            "lightgbm_training": lightgbm_training_seconds,
            "final_snapshot_retrieval": final_retrieval_seconds,
            "final_dlrm_scoring": dlrm_scoring_seconds,
            "final_lightgbm_scoring": lightgbm_scoring_seconds,
            "train_source_detail": train_candidates.runtime,
            "final_source_detail": final_candidates.runtime,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "saved_candidate_pools": (
            ["train_candidate_pool.npz", "final_candidate_pool.npz"]
            if save_candidates
            else []
        ),
    }
    result_path = run_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    prediction_dir = artifacts / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for ranker_name, ranked_items in (
        ("dlrm", final_dlrm_items),
        ("lightgbm", final_lightgbm_items),
    ):
        pd.DataFrame(
            {
                "customer_id": eval_users,
                "prediction": [
                    " ".join(final_features.article_ids[rows[:12]])
                    for rows in ranked_items
                ],
            }
        ).to_csv(
            prediction_dir / f"{config.profile}_{ranker_name}_top12.csv", index=False
        )
    # Generic profile output points to the strongest verified ranker in this run.
    pd.DataFrame(
        {
            "customer_id": eval_users,
            "prediction": [
                " ".join(final_features.article_ids[rows[:12]])
                for rows in final_lightgbm_items
            ],
        }
    ).to_csv(prediction_dir / f"{config.profile}_top12.csv", index=False)
    return result_path
