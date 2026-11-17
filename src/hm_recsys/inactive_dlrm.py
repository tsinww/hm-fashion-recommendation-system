"""DLRM ranker ablation over a previously saved inactive-user candidate pool."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .candidates import CandidateUnion
from .data import (
    ARTICLE_CAT_COLUMNS,
    FINAL_WINDOW,
    TRAIN_WINDOW,
    USER_CAT_COLUMNS,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_recent_transactions,
    slice_window,
    truth_by_customer,
)
from .inactive_pipeline import (
    InactiveUserState,
    _inactive_pair_features,
    _metrics_for_subgroups,
    _rankings,
    _user_state,
)
from .models import DLRMRanker, available_device, seed_everything


@dataclass(frozen=True)
class InactiveDLRMConfig:
    seed: int = 42
    negatives_per_positive: int = 20
    epochs: int = 3
    batch_size: int = 2_048
    score_batch_size: int = 8_192
    learning_rate: float = 1e-3
    embedding_dim: int = 8


def _load_union(path: Path) -> tuple[list[str], np.ndarray, CandidateUnion]:
    saved = np.load(path, allow_pickle=True)
    users = saved["customer_ids"].astype(str).tolist()
    article_ids = saved["article_ids"].astype(str)
    union = CandidateUnion(
        items=saved["candidate_item_rows"],
        source_rr=saved["source_rr"],
        source_names=tuple(saved["source_names"].astype(str).tolist()),
    )
    return users, article_ids, union


def _pairwise_positions(
    users: list[str],
    candidates: CandidateUnion,
    article_ids: np.ndarray,
    truth: dict[str, set[str]],
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Pair every retrieved positive with hard negatives from the same union."""
    rng = np.random.default_rng(seed)
    pair_users: list[np.ndarray] = []
    positive_positions: list[np.ndarray] = []
    negative_positions: list[np.ndarray] = []
    relevant_pairs = 0
    retrieved_pairs = 0
    users_with_positive = 0
    for user_row, user in enumerate(users):
        relevant = truth[user]
        relevant_pairs += len(relevant)
        is_positive = np.fromiter(
            (article_ids[item] in relevant for item in candidates.items[user_row]),
            dtype=bool,
        )
        positives = np.flatnonzero(is_positive)
        negatives = np.flatnonzero(~is_positive)
        retrieved_pairs += len(positives)
        if not len(positives) or not len(negatives):
            continue
        users_with_positive += 1
        for positive in positives:
            count = min(negatives_per_positive, len(negatives))
            sampled = rng.choice(negatives, size=count, replace=False)
            pair_users.append(np.full(count, user_row, dtype=np.int64))
            positive_positions.append(np.full(count, positive, dtype=np.int64))
            negative_positions.append(sampled.astype(np.int64))
    if not pair_users:
        raise RuntimeError("The saved training pool contains no retrieved positives")
    return (
        np.concatenate(pair_users),
        np.concatenate(positive_positions),
        np.concatenate(negative_positions),
        {
            "users": len(users),
            "users_with_retrieved_positive": users_with_positive,
            "relevant_pairs": relevant_pairs,
            "retrieved_relevant_pairs": retrieved_pairs,
            "retrieved_pair_fraction": retrieved_pairs / max(relevant_pairs, 1),
            "training_pairs": int(sum(len(rows) for rows in pair_users)),
        },
    )


def _fit_pairwise_dlrm(
    model: DLRMRanker,
    positive_cats: np.ndarray,
    positive_dense: np.ndarray,
    negative_cats: np.ndarray,
    negative_dense: np.ndarray,
    config: InactiveDLRMConfig,
) -> tuple[DLRMRanker, list[float]]:
    seed_everything(config.seed)
    device = available_device()
    model = model.to(device)
    dataset = TensorDataset(
        torch.as_tensor(positive_cats, dtype=torch.long),
        torch.as_tensor(positive_dense, dtype=torch.float32),
        torch.as_tensor(negative_cats, dtype=torch.long),
        torch.as_tensor(negative_dense, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    losses: list[float] = []
    for _ in range(config.epochs):
        model.train()
        total = 0.0
        for positive_cat, positive_den, negative_cat, negative_den in loader:
            positive_score = model(positive_cat.to(device), positive_den.to(device))
            negative_score = model(negative_cat.to(device), negative_den.to(device))
            loss = torch.nn.functional.softplus(negative_score - positive_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(positive_cat)
        losses.append(total / len(dataset))
    model.eval()
    return model, losses


def _score_dlrm(
    model: DLRMRanker,
    features,
    users: list[str],
    candidates: CandidateUnion,
    state: InactiveUserState,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    width = candidates.items.shape[1]
    scores = np.empty((len(users), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(users), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(users))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat_users), batch_size):
            stop = min(start + batch_size, len(flat_users))
            cats, dense = _inactive_pair_features(
                features,
                users,
                candidates,
                state,
                flat_users[start:stop],
                flat_positions[start:stop],
            )
            scores.reshape(-1)[start:stop] = model(
                torch.as_tensor(cats, dtype=torch.long, device=device),
                torch.as_tensor(dense, dtype=torch.float32, device=device),
            ).cpu().numpy()
    return scores


def _history_cache(cache_dir: Path) -> pd.DataFrame:
    paths = sorted(cache_dir.glob("inactive_target_history_*.pkl"))
    if not paths:
        raise FileNotFoundError("Run the inactive candidate experiment before the DLRM ablation")
    return pd.concat([pd.read_pickle(path) for path in paths], ignore_index=True).drop_duplicates()


def run_inactive_dlrm(root: Path, config: InactiveDLRMConfig) -> Path:
    started = time.perf_counter()
    source_dir = root / "artifacts" / "inactive_knn_experiment"
    output_dir = root / "artifacts" / "inactive_dlrm_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_users, train_articles, train_union = _load_union(source_dir / "train_candidate_pool.npz")
    final_users, final_articles, final_union = _load_union(source_dir / "final_candidate_pool.npz")
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
        raise ValueError("Saved training article rows no longer match snapshot features")
    if not np.array_equal(final_articles, final_features.article_ids.astype(str)):
        raise ValueError("Saved final article rows no longer match snapshot features")

    target_history = _history_cache(root / "artifacts" / "cache")
    train_state, _ = _user_state(
        train_users,
        target_history,
        TRAIN_WINDOW.context_start,
        TRAIN_WINDOW.label_start,
        180,
        train_features,
    )
    final_state, _ = _user_state(
        final_users,
        target_history,
        FINAL_WINDOW.context_start,
        FINAL_WINDOW.label_start,
        180,
        final_features,
    )
    train_truth_all = truth_by_customer(train_labels)
    final_truth_all = truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in train_users}
    final_truth = {user: final_truth_all[user] for user in final_users}

    pair_users, positive_positions, negative_positions, training_stats = _pairwise_positions(
        train_users,
        train_union,
        train_features.article_ids,
        train_truth,
        config.negatives_per_positive,
        config.seed + 1,
    )
    positive_cats, positive_dense = _inactive_pair_features(
        train_features, train_users, train_union, train_state, pair_users, positive_positions
    )
    negative_cats, negative_dense = _inactive_pair_features(
        train_features, train_users, train_union, train_state, pair_users, negative_positions
    )
    cardinalities = [encoder.cardinality(name) for name in USER_CAT_COLUMNS] + [
        encoder.cardinality(name) for name in ARTICLE_CAT_COLUMNS
    ]
    # Seed before constructing embedding/MLP layers so their initial weights are reproducible.
    seed_everything(config.seed)
    model = DLRMRanker(cardinalities, positive_dense.shape[1], config.embedding_dim)
    training_started = time.perf_counter()
    model, losses = _fit_pairwise_dlrm(
        model,
        positive_cats,
        positive_dense,
        negative_cats,
        negative_dense,
        config,
    )
    training_seconds = time.perf_counter() - training_started

    scoring_started = time.perf_counter()
    scores = _score_dlrm(
        model,
        final_features,
        final_users,
        final_union,
        final_state,
        config.score_batch_size,
    )
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_items = np.take_along_axis(final_union.items, order, axis=1)
    rankings = _rankings(final_users, ranked_items, final_features.article_ids)
    metrics = _metrics_for_subgroups(
        rankings, final_truth, final_state, final_union.items.shape[1]
    )
    scoring_seconds = time.perf_counter() - scoring_started

    pd.DataFrame(
        {
            "customer_id": final_users,
            "prediction": [
                " ".join(final_features.article_ids[rows[:12]]) for rows in ranked_items
            ],
        }
    ).to_csv(output_dir / "inactive_pairwise_dlrm_top12.csv", index=False)
    torch.save(model.state_dict(), output_dir / "inactive_pairwise_dlrm.pt")
    source_results = json.loads((source_dir / "results.json").read_text())
    raw_metrics = source_results["metrics"]["inactive_union_plus_repeats"]["cohorts"]
    lightgbm_metrics = source_results["metrics"]["inactive_lightgbm"]["cohorts"]
    summary = {
        "config": asdict(config),
        "objective": "pairwise softplus(s_negative - s_positive) within each customer's saved candidate pool",
        "candidate_pool": "artifacts/inactive_knn_experiment; retrieval was not rerun",
        "training": {**training_stats, "losses": losses},
        "metrics": metrics,
        "comparison_same_candidate_pool": {
            cohort: {
                "raw_reciprocal_rank": raw_metrics[cohort],
                "inactive_lightgbm": lightgbm_metrics[cohort],
                "pairwise_dlrm": metrics[cohort],
            }
            for cohort in metrics
        },
        "runtime_seconds": {
            "training": training_seconds,
            "final_scoring": scoring_seconds,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(next(model.parameters()).device),
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return result_path
