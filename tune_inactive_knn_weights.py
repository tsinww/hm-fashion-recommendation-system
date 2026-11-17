#!/usr/bin/env python3
"""Tune KNN reciprocal-rank weights on the earlier snapshot, then freeze/evaluate."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

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


WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def load_pool(path: Path):
    saved = np.load(path, allow_pickle=True)
    return {
        "users": saved["customer_ids"].astype(str).tolist(),
        "articles": saved["article_ids"].astype(str),
        "items": saved["candidate_item_rows"],
        "source_rr": saved["source_rr"],
        "source_names": saved["source_names"].astype(str).tolist(),
    }


def order_with_weights(pool, weights: np.ndarray) -> np.ndarray:
    scores = np.einsum("ukc,c->uk", pool["source_rr"], weights, optimize=True)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(pool["items"], order, axis=1)


def metrics(pool, rows: np.ndarray, truth: dict[str, set[str]]) -> dict[str, float]:
    maps, recalls = [], []
    for row, user in enumerate(pool["users"]):
        ranking = pool["articles"][rows[row]].tolist()
        maps.append(average_precision_at_k(ranking, truth[user], 12))
        recalls.append(recall_at_k(ranking, truth[user], 12))
    return {
        "map_at_12": float(np.mean(maps)),
        "recall_at_12": float(np.mean(recalls)),
    }


def main() -> None:
    source = ROOT / "artifacts" / "inactive_knn_experiment"
    train_pool = load_pool(source / "train_candidate_pool.npz")
    final_pool = load_pool(source / "final_candidate_pool.npz")
    if train_pool["source_names"] != final_pool["source_names"]:
        raise ValueError("Train/final source order differs")
    recent = load_recent_transactions(ROOT / "Data", ROOT / "artifacts" / "cache")
    _, train_labels = slice_window(recent, TRAIN_WINDOW)
    _, final_labels = slice_window(recent, FINAL_WINDOW)
    train_truth_all = truth_by_customer(train_labels)
    final_truth_all = truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in train_pool["users"]}
    final_truth = {user: final_truth_all[user] for user in final_pool["users"]}
    behavioral = train_pool["source_names"].index("behavioral_knn")
    metadata = train_pool["source_names"].index("metadata_knn")
    trials = []
    for behavior_weight, metadata_weight in product(WEIGHT_GRID, repeat=2):
        weights = np.ones(len(train_pool["source_names"]), dtype=np.float32)
        weights[behavioral] = behavior_weight
        weights[metadata] = metadata_weight
        result = metrics(train_pool, order_with_weights(train_pool, weights), train_truth)
        trials.append(
            {
                "behavioral_knn_weight": behavior_weight,
                "metadata_knn_weight": metadata_weight,
                **result,
            }
        )
    # MAP selects the setting; Recall breaks exact ties, then simpler/smaller weights.
    selected = max(
        trials,
        key=lambda row: (
            row["map_at_12"],
            row["recall_at_12"],
            -(row["behavioral_knn_weight"] + row["metadata_knn_weight"]),
        ),
    )
    selected_weights = np.ones(len(train_pool["source_names"]), dtype=np.float32)
    selected_weights[behavioral] = selected["behavioral_knn_weight"]
    selected_weights[metadata] = selected["metadata_knn_weight"]
    default_weights = np.ones(len(train_pool["source_names"]), dtype=np.float32)
    output = {
        "selection_protocol": "weights selected only on Sep 9-15 earlier-snapshot labels; frozen for Sep 16-22",
        "source_names": train_pool["source_names"],
        "selected_weights": {
            name: float(weight)
            for name, weight in zip(train_pool["source_names"], selected_weights, strict=True)
        },
        "earlier_snapshot_selected_metrics": {
            "map_at_12": selected["map_at_12"],
            "recall_at_12": selected["recall_at_12"],
        },
        "final_metrics": {
            "default_equal_weights": metrics(
                final_pool,
                order_with_weights(final_pool, default_weights),
                final_truth,
            ),
            "earlier_selected_knn_weights": metrics(
                final_pool,
                order_with_weights(final_pool, selected_weights),
                final_truth,
            ),
        },
        "grid_trials": sorted(trials, key=lambda row: -row["map_at_12"]),
    }
    output_dir = ROOT / "artifacts" / "inactive_knn_weight_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Results written to {path}")


if __name__ == "__main__":
    main()

