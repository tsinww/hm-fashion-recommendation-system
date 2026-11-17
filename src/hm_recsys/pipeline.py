"""End-to-end experiment orchestration for the staged recommender."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    ARTICLE_CAT_COLUMNS,
    FINAL_WINDOW,
    TRAIN_WINDOW,
    USER_CAT_COLUMNS,
    SnapshotFeatures,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_recent_transactions,
    sample_customers,
    slice_window,
    truth_by_customer,
)
from .metrics import evaluate_rankings
from .models import DLRMRanker, TwoTower, fit_binary_model, fit_mf


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    seed: int = 42
    max_train_users: int = 6_000
    max_eval_users: int = 2_000
    max_mf_pairs: int | None = None
    candidate_k: int = 1_000
    rerank_k: int = 100
    negatives_per_positive: int = 20
    mf_dim: int = 32
    mf_epochs: int = 3
    downstream_epochs: int = 2
    batch_size: int = 4_096
    learning_rate: float = 0.003


PROFILES = {
    "smoke": ExperimentConfig(
        profile="smoke",
        max_train_users=300,
        max_eval_users=200,
        max_mf_pairs=50_000,
        candidate_k=100,
        rerank_k=30,
        negatives_per_positive=5,
        mf_dim=16,
        mf_epochs=1,
        downstream_epochs=1,
        batch_size=2_048,
    ),
    "experiment": ExperimentConfig(profile="experiment"),
}


def _fit_snapshot_mf(
    context: pd.DataFrame, features: SnapshotFeatures, config: ExperimentConfig, seed: int
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


def _select_labeled_examples(
    customer_ids: list[str],
    candidate_items: np.ndarray,
    item_ids: np.ndarray,
    truth: dict[str, set[str]],
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Sample positives and hard negatives only from the preceding stage's candidates."""
    rng = np.random.default_rng(seed)
    customer_rows: list[np.ndarray] = []
    candidate_positions: list[np.ndarray] = []
    users_with_retrieved_positive = 0
    total_relevant = 0
    retrieved_relevant = 0
    for user_row, customer_id in enumerate(customer_ids):
        relevant = truth.get(customer_id, set())
        total_relevant += len(relevant)
        labels = np.fromiter(
            (item_ids[item] in relevant for item in candidate_items[user_row]), dtype=bool
        )
        positives = np.flatnonzero(labels)
        retrieved_relevant += len(positives)
        if not len(positives):
            continue
        users_with_retrieved_positive += 1
        negatives = np.flatnonzero(~labels)
        n_negative = min(len(negatives), max(1, negatives_per_positive * len(positives)))
        chosen_negative = rng.choice(negatives, size=n_negative, replace=False)
        chosen = np.concatenate([positives, chosen_negative])
        customer_rows.append(np.full(len(chosen), user_row, dtype=np.int64))
        candidate_positions.append(chosen.astype(np.int64))
    if not customer_rows:
        raise RuntimeError("No positive labels were retrieved; increase candidate_k or inspect the split.")
    stats = {
        "users": len(customer_ids),
        "users_with_retrieved_positive": users_with_retrieved_positive,
        "relevant_pairs": total_relevant,
        "retrieved_relevant_pairs": retrieved_relevant,
        "retrieved_pair_fraction": retrieved_relevant / max(total_relevant, 1),
    }
    return np.concatenate(customer_rows), np.concatenate(candidate_positions), stats


def _pair_feature_arrays(
    features: SnapshotFeatures,
    customer_ids: list[str],
    candidate_items: np.ndarray,
    candidate_scores: np.ndarray,
    user_rows: np.ndarray,
    candidate_positions: np.ndarray,
    secondary_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_items = candidate_items[user_rows, candidate_positions]
    pair_customers = [customer_ids[row] for row in user_rows]
    metadata_rows = np.fromiter(
        (features.customer_to_row.get(customer, -1) for customer in pair_customers), dtype=np.int64
    )
    user_cats = np.zeros((len(metadata_rows), features.user_cats.shape[1]), dtype=np.int64)
    user_dense = np.zeros((len(metadata_rows), features.user_dense.shape[1]), dtype=np.float32)
    present = metadata_rows >= 0
    user_cats[present] = features.user_cats[metadata_rows[present]]
    user_dense[present] = features.user_dense[metadata_rows[present]]
    item_cats = features.item_cats[pair_items]
    item_dense = features.item_dense[pair_items]
    primary = candidate_scores[user_rows, candidate_positions].astype(np.float32)
    secondary = (
        np.zeros_like(primary)
        if secondary_scores is None
        else secondary_scores[user_rows, candidate_positions].astype(np.float32)
    )
    repeats = np.fromiter(
        (
            features.article_ids[item] in features.purchased.get(customer, set())
            for customer, item in zip(pair_customers, pair_items, strict=True)
        ),
        dtype=np.float32,
    )
    ranks = candidate_positions.astype(np.float32) / max(candidate_items.shape[1] - 1, 1)
    dense = np.column_stack([user_dense, item_dense, primary, secondary, repeats, ranks]).astype(
        np.float32
    )
    cats = np.column_stack([user_cats, item_cats]).astype(np.int64)
    return user_cats, item_cats, cats, dense


def _score_two_tower(
    model: TwoTower,
    features: SnapshotFeatures,
    customer_ids: list[str],
    candidate_items: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    output = np.empty(candidate_items.shape, dtype=np.float32)
    width = candidate_items.shape[1]
    flat_users = np.repeat(np.arange(len(customer_ids), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(customer_ids))
    dummy_scores = np.zeros(candidate_items.shape, dtype=np.float32)
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        rows = flat_users[start:stop]
        positions = flat_positions[start:stop]
        user_cats, item_cats, _, _ = _pair_feature_arrays(
            features, customer_ids, candidate_items, dummy_scores, rows, positions
        )
        metadata_rows = np.fromiter(
            (features.customer_to_row.get(customer_ids[row], -1) for row in rows), dtype=np.int64
        )
        user_dense = np.zeros((len(rows), features.user_dense.shape[1]), dtype=np.float32)
        present = metadata_rows >= 0
        user_dense[present] = features.user_dense[metadata_rows[present]]
        item_dense = features.item_dense[candidate_items[rows, positions]]
        with torch.no_grad():
            scores = model(
                torch.as_tensor(user_cats, dtype=torch.long, device=device),
                torch.as_tensor(user_dense, dtype=torch.float32, device=device),
                torch.as_tensor(item_cats, dtype=torch.long, device=device),
                torch.as_tensor(item_dense, dtype=torch.float32, device=device),
            )
        output.reshape(-1)[start:stop] = scores.detach().cpu().numpy()
    return output


def _score_dlrm(
    model: DLRMRanker,
    features: SnapshotFeatures,
    customer_ids: list[str],
    candidate_items: np.ndarray,
    mf_scores: np.ndarray,
    tt_scores: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    output = np.empty(candidate_items.shape, dtype=np.float32)
    width = candidate_items.shape[1]
    flat_users = np.repeat(np.arange(len(customer_ids), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(customer_ids))
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        rows = flat_users[start:stop]
        positions = flat_positions[start:stop]
        _, _, cats, dense = _pair_feature_arrays(
            features,
            customer_ids,
            candidate_items,
            mf_scores,
            rows,
            positions,
            tt_scores,
        )
        with torch.no_grad():
            scores = model(
                torch.as_tensor(cats, dtype=torch.long, device=device),
                torch.as_tensor(dense, dtype=torch.float32, device=device),
            )
        output.reshape(-1)[start:stop] = scores.detach().cpu().numpy()
    return output


def _reorder(
    candidate_items: np.ndarray,
    primary_scores: np.ndarray,
    keep: int | None = None,
    *aligned_scores: np.ndarray,
) -> tuple[np.ndarray, ...]:
    keep = candidate_items.shape[1] if keep is None else min(keep, candidate_items.shape[1])
    order = np.argsort(-primary_scores, axis=1, kind="stable")[:, :keep]
    rows = np.arange(len(candidate_items))[:, None]
    values: list[np.ndarray] = [candidate_items[rows, order], primary_scores[rows, order]]
    values += [score[rows, order] for score in aligned_scores]
    return tuple(values)


def _rankings(
    customer_ids: list[str], candidate_items: np.ndarray, article_ids: np.ndarray
) -> dict[str, list[str]]:
    return {
        customer: article_ids[candidate_items[row]].tolist()
        for row, customer in enumerate(customer_ids)
    }


def _metric_block(
    rankings: dict[str, list[str]],
    truth: dict[str, set[str]],
    known: set[str],
    candidate_k: int,
) -> dict:
    result = evaluate_rankings(rankings, truth, known, candidate_k)
    return {"candidate_k": candidate_k, "cohorts": result}


def run_experiment(root: Path, config: ExperimentConfig) -> Path:
    started = time.perf_counter()
    stage_times: dict[str, float] = {}
    artifacts = root / "artifacts"
    run_dir = artifacts / config.profile
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "Data"

    mark = time.perf_counter()
    transactions = load_recent_transactions(data_dir, artifacts / "cache")
    customers, articles = load_metadata(data_dir)
    encoder = fit_feature_encoder(customers, articles)
    train_context, train_labels = slice_window(transactions, TRAIN_WINDOW)
    final_context, final_labels = slice_window(transactions, FINAL_WINDOW)
    train_features = build_snapshot_features(
        train_context, customers, articles, encoder, TRAIN_WINDOW.label_start
    )
    final_features = build_snapshot_features(
        final_context, customers, articles, encoder, FINAL_WINDOW.label_start
    )
    stage_times["load_and_features"] = time.perf_counter() - mark

    train_known = set(train_context["customer_id"].astype(str).unique())
    train_users = sample_customers(
        train_labels, train_known, config.max_train_users, config.seed
    )
    train_truth_all = truth_by_customer(train_labels)
    train_truth = {customer: train_truth_all[customer] for customer in train_users}

    mark = time.perf_counter()
    train_mf, train_mf_losses = _fit_snapshot_mf(
        train_context, train_features, config, config.seed
    )
    train_candidates, train_mf_scores = train_mf.candidates(
        train_users, config.candidate_k
    )
    stage_times["train_snapshot_mf_and_candidates"] = time.perf_counter() - mark

    mark = time.perf_counter()
    user_rows, positions, tt_example_stats = _select_labeled_examples(
        train_users,
        train_candidates,
        train_features.article_ids,
        train_truth,
        config.negatives_per_positive,
        config.seed,
    )
    tt_user_cats, tt_item_cats, _, _ = _pair_feature_arrays(
        train_features,
        train_users,
        train_candidates,
        train_mf_scores,
        user_rows,
        positions,
    )
    metadata_rows = np.fromiter(
        (train_features.customer_to_row.get(train_users[row], -1) for row in user_rows),
        dtype=np.int64,
    )
    tt_user_dense = np.zeros((len(metadata_rows), train_features.user_dense.shape[1]), dtype=np.float32)
    present_metadata = metadata_rows >= 0
    tt_user_dense[present_metadata] = train_features.user_dense[metadata_rows[present_metadata]]
    tt_item_dense = train_features.item_dense[train_candidates[user_rows, positions]]
    tt_labels = np.fromiter(
        (
            train_features.article_ids[train_candidates[row, position]] in train_truth[train_users[row]]
            for row, position in zip(user_rows, positions, strict=True)
        ),
        dtype=np.float32,
    )
    user_cards = [encoder.cardinality(name) for name in USER_CAT_COLUMNS]
    item_cards = [encoder.cardinality(name) for name in ARTICLE_CAT_COLUMNS]
    two_tower = TwoTower(user_cards, item_cards, tt_user_dense.shape[1], tt_item_dense.shape[1])
    two_tower, tt_losses = fit_binary_model(
        two_tower,
        (
            torch.as_tensor(tt_user_cats, dtype=torch.long),
            torch.as_tensor(tt_user_dense, dtype=torch.float32),
            torch.as_tensor(tt_item_cats, dtype=torch.long),
            torch.as_tensor(tt_item_dense, dtype=torch.float32),
        ),
        torch.as_tensor(tt_labels, dtype=torch.float32),
        config.downstream_epochs,
        config.batch_size,
        config.learning_rate,
        config.seed + 1,
    )
    train_tt_scores = _score_two_tower(
        two_tower, train_features, train_users, train_candidates, config.batch_size
    )
    narrowed_items, narrowed_tt_scores, narrowed_mf_scores = _reorder(
        train_candidates, train_tt_scores, config.rerank_k, train_mf_scores
    )
    stage_times["two_tower_training"] = time.perf_counter() - mark

    mark = time.perf_counter()
    dlrm_user_rows, dlrm_positions, dlrm_example_stats = _select_labeled_examples(
        train_users,
        narrowed_items,
        train_features.article_ids,
        train_truth,
        config.negatives_per_positive,
        config.seed + 2,
    )
    _, _, dlrm_cats, dlrm_dense = _pair_feature_arrays(
        train_features,
        train_users,
        narrowed_items,
        narrowed_mf_scores,
        dlrm_user_rows,
        dlrm_positions,
        narrowed_tt_scores,
    )
    dlrm_labels = np.fromiter(
        (
            train_features.article_ids[narrowed_items[row, position]] in train_truth[train_users[row]]
            for row, position in zip(dlrm_user_rows, dlrm_positions, strict=True)
        ),
        dtype=np.float32,
    )
    dlrm = DLRMRanker(user_cards + item_cards, dlrm_dense.shape[1])
    dlrm, dlrm_losses = fit_binary_model(
        dlrm,
        (
            torch.as_tensor(dlrm_cats, dtype=torch.long),
            torch.as_tensor(dlrm_dense, dtype=torch.float32),
        ),
        torch.as_tensor(dlrm_labels, dtype=torch.float32),
        config.downstream_epochs,
        config.batch_size,
        config.learning_rate,
        config.seed + 3,
    )
    stage_times["dlrm_training"] = time.perf_counter() - mark

    mark = time.perf_counter()
    final_mf, final_mf_losses = _fit_snapshot_mf(
        final_context, final_features, config, config.seed + 4
    )
    final_known = set(final_context["customer_id"].astype(str).unique())
    eval_users = sample_customers(final_labels, final_known, config.max_eval_users, config.seed)
    final_truth_all = truth_by_customer(final_labels)
    final_truth = {customer: final_truth_all[customer] for customer in eval_users}
    final_candidates, final_mf_scores = final_mf.candidates(eval_users, config.candidate_k)
    final_tt_scores = _score_two_tower(
        two_tower, final_features, eval_users, final_candidates, config.batch_size
    )
    final_narrowed, final_narrowed_tt, final_narrowed_mf = _reorder(
        final_candidates, final_tt_scores, config.rerank_k, final_mf_scores
    )
    final_dlrm_scores = _score_dlrm(
        dlrm,
        final_features,
        eval_users,
        final_narrowed,
        final_narrowed_mf,
        final_narrowed_tt,
        config.batch_size,
    )
    final_dlrm_items, _ = _reorder(final_narrowed, final_dlrm_scores)
    stage_times["final_snapshot_training_and_scoring"] = time.perf_counter() - mark

    popular = np.tile(
        np.argsort(-final_features.item_popularity, kind="stable")[: config.candidate_k],
        (len(eval_users), 1),
    )
    methods = {
        "recent_popularity": _metric_block(
            _rankings(eval_users, popular, final_features.article_ids),
            final_truth,
            final_known,
            popular.shape[1],
        ),
        "mf": _metric_block(
            _rankings(eval_users, final_candidates, final_features.article_ids),
            final_truth,
            final_known,
            final_candidates.shape[1],
        ),
        "mf_two_tower": _metric_block(
            _rankings(eval_users, final_narrowed, final_features.article_ids),
            final_truth,
            final_known,
            final_narrowed.shape[1],
        ),
        "mf_two_tower_dlrm": _metric_block(
            _rankings(eval_users, final_dlrm_items, final_features.article_ids),
            final_truth,
            final_known,
            final_dlrm_items.shape[1],
        ),
    }

    all_relevant_pairs = sum(len(value) for value in final_truth.values())
    in_catalog_pairs = sum(
        len(value & set(final_features.article_ids)) for value in final_truth.values()
    )
    summary = {
        "config": asdict(config),
        "windows": {
            "downstream_training": asdict(TRAIN_WINDOW),
            "final_evaluation": asdict(FINAL_WINDOW),
        },
        "dataset": {
            "train_context_rows": len(train_context),
            "train_label_rows": len(train_labels),
            "final_context_rows": len(final_context),
            "final_validation_rows": len(final_labels),
            "train_catalog_items": len(train_features.article_ids),
            "final_catalog_items": len(final_features.article_ids),
            "final_context_customers": final_context["customer_id"].nunique(),
            "final_validation_customers": final_labels["customer_id"].nunique(),
            "final_validation_history_customers": len(
                set(final_labels["customer_id"].astype(str).unique()) & final_known
            ),
            "final_validation_cold_start_customers": len(
                set(final_labels["customer_id"].astype(str).unique()) - final_known
            ),
            "train_customers_sampled": len(train_users),
            "evaluation_customers_sampled": len(eval_users),
            "evaluation_history_customers": sum(customer in final_known for customer in eval_users),
            "evaluation_cold_start_customers": sum(customer not in final_known for customer in eval_users),
            "validation_relevant_pairs": all_relevant_pairs,
            "validation_pairs_in_eligible_catalog": in_catalog_pairs,
            "validation_pair_catalog_coverage": in_catalog_pairs / max(all_relevant_pairs, 1),
        },
        "training": {
            "mf_objective": "BPR pairwise loss with a uniform unobserved-item negative",
            "two_tower_objective": "weighted BCE on positives and hard negatives sampled from MF candidates",
            "dlrm_objective": "weighted BCE on positives and hard negatives sampled after two-tower narrowing",
            "train_mf_losses": train_mf_losses,
            "final_mf_losses": final_mf_losses,
            "two_tower_losses": tt_losses,
            "dlrm_losses": dlrm_losses,
            "two_tower_examples": {**tt_example_stats, "rows": len(tt_labels), "positives": int(tt_labels.sum())},
            "dlrm_examples": {**dlrm_example_stats, "rows": len(dlrm_labels), "positives": int(dlrm_labels.sum())},
        },
        "metrics": methods,
        "runtime_seconds": {**stage_times, "total": time.perf_counter() - started},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(final_mf.device),
        },
    }
    result_path = run_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    prediction_dir = artifacts / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "customer_id": eval_users,
            "prediction": [
                " ".join(final_features.article_ids[row[:12]]) for row in final_dlrm_items
            ],
        }
    )
    prediction_frame.to_csv(prediction_dir / f"{config.profile}_top12.csv", index=False)
    return result_path
