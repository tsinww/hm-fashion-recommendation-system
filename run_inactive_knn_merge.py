#!/usr/bin/env python3
"""Merge bounded KNN candidates into the stronger preserved inactive pool."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import (  # noqa: E402
    FINAL_WINDOW,
    TRAIN_WINDOW,
    load_recent_transactions,
    slice_window,
    truth_by_customer,
)
from hm_recsys.metrics import average_precision_at_k, recall_at_k  # noqa: E402


QUOTAS = (25, 50, 100, 200)
KNN_SCALES = (0.1, 0.25, 0.5, 1.0)


def load_pool(path: Path):
    saved = np.load(path, allow_pickle=True)
    return {
        "users": saved["customer_ids"].astype(str).tolist(),
        "articles": saved["article_ids"].astype(str),
        "items": saved["candidate_item_rows"],
        "source_rr": saved["source_rr"],
        "source_names": saved["source_names"].astype(str).tolist(),
    }


def align_old_pool(old, users: list[str]):
    rows = {user: row for row, user in enumerate(old["users"])}
    selected = np.asarray([rows[user] for user in users], dtype=np.int64)
    return old["items"][selected]


def knn_order(pool) -> np.ndarray:
    weights = np.ones(len(pool["source_names"]), dtype=np.float32)
    weights[pool["source_names"].index("behavioral_knn")] = 0.25
    weights[pool["source_names"].index("metadata_knn")] = 0.25
    scores = np.einsum("ukc,c->uk", pool["source_rr"], weights, optimize=True)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(pool["items"], order, axis=1)


def merge_one(old_items: np.ndarray, knn_items: np.ndarray, quota: int, scale: float):
    candidates = old_items.astype(np.int64).tolist()
    positions = {int(item): position for position, item in enumerate(candidates)}
    scores = [1.0 / (position + 1) for position in range(len(candidates))]
    for rank, raw_item in enumerate(knn_items[:quota]):
        item = int(raw_item)
        contribution = scale / (rank + 1)
        position = positions.get(item)
        if position is None:
            positions[item] = len(candidates)
            candidates.append(item)
            scores.append(contribution)
        else:
            scores[position] += contribution
    order = np.argsort(-np.asarray(scores), kind="stable")
    return np.asarray(candidates, dtype=np.int64)[order]


def merge_all(old_items, knn_items, quota: int, scale: float):
    return [
        merge_one(old_items[row], knn_items[row], quota, scale)
        for row in range(len(old_items))
    ]


def metrics(users, rankings, articles, truth, top12_only: bool = False):
    maps, recalls, candidate_recalls = [], [], []
    for row, user in enumerate(users):
        ranked_articles = articles[rankings[row]].tolist()
        maps.append(average_precision_at_k(ranked_articles, truth[user], 12))
        recalls.append(recall_at_k(ranked_articles, truth[user], 12))
        if not top12_only:
            candidate_recalls.append(
                recall_at_k(ranked_articles, truth[user], len(ranked_articles))
            )
    result = {
        "map_at_12": float(np.mean(maps)),
        "recall_at_12": float(np.mean(recalls)),
    }
    if candidate_recalls:
        result["candidate_recall"] = float(np.mean(candidate_recalls))
        result["mean_candidate_count"] = float(np.mean([len(row) for row in rankings]))
    return result


def subgroup_metrics(users, rankings, articles, truth, state_path: Path):
    saved = np.load(state_path, allow_pickle=True)
    state_users = saved["customer_ids"].astype(str).tolist()
    state_row = {user: row for row, user in enumerate(state_users)}
    has180 = saved["has_history_180d"]
    has_any = saved["has_any_history"]
    groups = {
        "all_inactive": users,
        "history_within_180d": [user for user in users if has180[state_row[user]] == 1],
        "stale_history_over_180d": [
            user for user in users
            if has180[state_row[user]] == 0 and has_any[state_row[user]] == 1
        ],
        "no_prior_transaction": [user for user in users if has_any[state_row[user]] == 0],
    }
    user_row = {user: row for row, user in enumerate(users)}
    output = {}
    for name, group_users in groups.items():
        group_rankings = [rankings[user_row[user]] for user in group_users]
        output[name] = {
            "customers": len(group_users),
            **metrics(group_users, group_rankings, articles, truth),
        }
    return output


def main() -> None:
    old_dir = ROOT / "artifacts" / "hybrid_expanded_repeat_experiment"
    knn_dir = ROOT / "artifacts" / "inactive_knn_experiment"
    old_train = load_pool(old_dir / "train_candidate_pool.npz")
    old_final = load_pool(old_dir / "final_candidate_pool.npz")
    knn_train = load_pool(knn_dir / "train_candidate_pool.npz")
    knn_final = load_pool(knn_dir / "final_candidate_pool.npz")
    if not np.array_equal(old_train["articles"], knn_train["articles"]):
        raise ValueError("Training article row spaces differ")
    if not np.array_equal(old_final["articles"], knn_final["articles"]):
        raise ValueError("Final article row spaces differ")
    train_old_items = align_old_pool(old_train, knn_train["users"])
    final_old_items = align_old_pool(old_final, knn_final["users"])
    train_knn_items = knn_order(knn_train)
    final_knn_items = knn_order(knn_final)
    recent = load_recent_transactions(ROOT / "Data", ROOT / "artifacts" / "cache")
    _, train_labels = slice_window(recent, TRAIN_WINDOW)
    _, final_labels = slice_window(recent, FINAL_WINDOW)
    train_truth_all, final_truth_all = truth_by_customer(train_labels), truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in knn_train["users"]}
    final_truth = {user: final_truth_all[user] for user in knn_final["users"]}
    trials = []
    for quota, scale in product(QUOTAS, KNN_SCALES):
        rankings = merge_all(train_old_items, train_knn_items, quota, scale)
        result = metrics(
            knn_train["users"], rankings, knn_train["articles"], train_truth, top12_only=True
        )
        trials.append({"knn_quota": quota, "knn_scale": scale, **result})
    selected = max(
        trials,
        key=lambda row: (
            row["map_at_12"], row["recall_at_12"], -row["knn_quota"], -row["knn_scale"]
        ),
    )
    final_rankings = merge_all(
        final_old_items,
        final_knn_items,
        selected["knn_quota"],
        selected["knn_scale"],
    )
    baseline_rankings = [row.copy() for row in final_old_items]
    state_path = knn_dir / "final_user_state.npz"
    output = {
        "selection_protocol": "quota/scale selected on Sep 9-15 inactive labels and frozen for Sep 16-22",
        "knn_internal_weights": {"behavioral_knn": 0.25, "metadata_knn": 0.25, "other_knn_pool_sources": 1.0},
        "selected": selected,
        "final_metrics": {
            "preserved_old_raw_pool": subgroup_metrics(
                knn_final["users"], baseline_rankings, knn_final["articles"], final_truth, state_path
            ),
            "old_pool_plus_bounded_knn": subgroup_metrics(
                knn_final["users"], final_rankings, knn_final["articles"], final_truth, state_path
            ),
        },
        "grid_trials": sorted(trials, key=lambda row: -row["map_at_12"]),
    }
    output_dir = ROOT / "artifacts" / "inactive_old_plus_knn"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_width = max(len(row) for row in final_rankings)
    padded = np.full((len(final_rankings), max_width), -1, dtype=np.int64)
    lengths = np.empty(len(final_rankings), dtype=np.int64)
    for row, ranking in enumerate(final_rankings):
        padded[row, : len(ranking)] = ranking
        lengths[row] = len(ranking)
    np.savez_compressed(
        output_dir / "final_candidate_pool.npz",
        customer_ids=np.asarray(knn_final["users"]),
        article_ids=knn_final["articles"],
        candidate_item_rows=padded,
        candidate_lengths=lengths,
    )
    pd.DataFrame(
        {
            "customer_id": knn_final["users"],
            "prediction": [
                " ".join(knn_final["articles"][ranking[:12]])
                for ranking in final_rankings
            ],
        }
    ).to_csv(output_dir / "top12.csv", index=False)
    path = output_dir / "results.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Results written to {path}")


if __name__ == "__main__":
    main()

