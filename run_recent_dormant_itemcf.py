#!/usr/bin/env python3
"""Add time-decayed item-to-item candidates for recent-dormant customers."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import (  # noqa: E402
    FINAL_WINDOW,
    TRAIN_WINDOW,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_recent_transactions,
    load_transactions_since,
    slice_window,
    truth_by_customer,
)
from hm_recsys.metrics import average_precision_at_k, recall_at_k  # noqa: E402
from run_inactive_knn_merge import (  # noqa: E402
    align_old_pool,
    knn_order,
    load_pool,
    merge_all,
    merge_one,
)


ITEMCF_QUOTAS = (25, 50, 100, 200)
ITEMCF_SCALES = (0.1, 0.25, 0.5, 1.0)


def target_history() -> pd.DataFrame:
    paths = sorted((ROOT / "artifacts" / "cache").glob("inactive_target_history_*.pkl"))
    if not paths:
        raise FileNotFoundError("Inactive target-history cache is missing")
    return pd.concat([pd.read_pickle(path) for path in paths], ignore_index=True).drop_duplicates()


def generalized_basket_itemcf(
    transactions: pd.DataFrame,
    cutoff: str,
    start: str,
    article_to_row: dict[str, int],
    neighbors_per_item: int = 50,
    max_basket_items: int = 20,
) -> dict[str, list[tuple[int, int]]]:
    """Map any historical seed article to current-catalog same-basket neighbors."""
    frame = transactions.loc[
        (transactions["t_dat"] >= pd.Timestamp(start))
        & (transactions["t_dat"] < pd.Timestamp(cutoff)),
        ["customer_id", "t_dat", "article_id"],
    ].drop_duplicates()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["article_id"] = frame["article_id"].astype(str)
    frame["basket"] = frame.groupby(
        ["customer_id", "t_dat"], observed=True, sort=False
    ).ngroup().astype(np.int32)
    sizes = frame.groupby("basket", observed=True)["article_id"].transform("size")
    frame = frame.loc[(sizes >= 2) & (sizes <= max_basket_items), ["basket", "article_id"]]
    right = frame.loc[frame["article_id"].isin(article_to_row), ["basket", "article_id"]].copy()
    right["neighbor"] = right["article_id"].map(article_to_row).astype(np.int32)
    paired = frame.merge(right[["basket", "article_id", "neighbor"]], on="basket", suffixes=("_seed", "_neighbor"))
    paired = paired.loc[paired["article_id_seed"] != paired["article_id_neighbor"]]
    counts = (
        paired.groupby(["article_id_seed", "neighbor"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["article_id_seed", "count", "neighbor"],
            ascending=[True, False, True],
        )
        .groupby("article_id_seed", observed=True)
        .head(neighbors_per_item)
    )
    output: dict[str, list[tuple[int, int]]] = {}
    for article, group in counts.groupby("article_id_seed", observed=True, sort=False):
        output[str(article)] = list(
            zip(
                group["neighbor"].astype(int).tolist(),
                group["count"].astype(int).tolist(),
                strict=True,
            )
        )
    return output


def dormant_state_and_profile(
    users: list[str], history: pd.DataFrame, context_start: str
) -> tuple[dict[str, str], pd.DataFrame]:
    boundary = pd.Timestamp(context_start)
    profile_start = boundary - pd.Timedelta(days=180)
    frame = history.loc[
        history["customer_id"].astype(str).isin(users)
        & (history["t_dat"] < boundary)
    ].copy()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["article_id"] = frame["article_id"].astype(str)
    recent = frame.loc[frame["t_dat"] >= profile_start].copy()
    any_users = set(frame["customer_id"])
    recent_users = set(recent["customer_id"])
    state = {
        user: (
            "history_within_180d"
            if user in recent_users
            else "stale_history_over_180d"
            if user in any_users
            else "no_prior_transaction"
        )
        for user in users
    }
    return state, recent


def itemcf_candidates(
    users: list[str],
    profile: pd.DataFrame,
    itemcf: dict[str, list[tuple[int, int]]],
    prediction_cutoff: str,
    max_candidates: int = 200,
) -> list[np.ndarray]:
    cutoff = pd.Timestamp(prediction_cutoff)
    stats = (
        profile.groupby(["customer_id", "article_id"], observed=True)
        .agg(count=("article_id", "size"), last=("t_dat", "max"))
        .reset_index()
    )
    by_user = {user: group for user, group in stats.groupby("customer_id", observed=True)}
    output: list[np.ndarray] = []
    for user in users:
        scores: Counter[int] = Counter()
        group = by_user.get(user)
        if group is not None:
            for row in group.itertuples(index=False):
                days = max((cutoff - row.last).days, 0)
                history_weight = float(row.count) * np.exp(-days / 60.0)
                for neighbor, co_count in itemcf.get(str(row.article_id), []):
                    scores[int(neighbor)] += history_weight * np.log1p(co_count)
        output.append(
            np.asarray([item for item, _ in scores.most_common(max_candidates)], dtype=np.int64)
        )
    return output


def add_rank_source(base: np.ndarray, source: np.ndarray, quota: int, scale: float) -> np.ndarray:
    return merge_one(base, source, quota, scale)


def evaluate(users, rankings, articles, truth):
    maps, recalls, candidate_recalls = [], [], []
    for row, user in enumerate(users):
        recs = articles[rankings[row]].tolist()
        maps.append(average_precision_at_k(recs, truth[user], 12))
        recalls.append(recall_at_k(recs, truth[user], 12))
        candidate_recalls.append(recall_at_k(recs, truth[user], len(recs)))
    return {
        "customers": len(users),
        "candidate_recall": float(np.mean(candidate_recalls)),
        "map_at_12": float(np.mean(maps)),
        "recall_at_12": float(np.mean(recalls)),
        "mean_candidate_count": float(np.mean([len(row) for row in rankings])),
    }


def cohort_metrics(users, rankings, articles, truth, states):
    row_by_user = {user: row for row, user in enumerate(users)}
    groups = {
        "all_inactive": users,
        "history_within_180d": [u for u in users if states[u] == "history_within_180d"],
        "stale_history_over_180d": [u for u in users if states[u] == "stale_history_over_180d"],
        "no_prior_transaction": [u for u in users if states[u] == "no_prior_transaction"],
    }
    return {
        name: evaluate(
            group,
            [rankings[row_by_user[user]] for user in group],
            articles,
            truth,
        )
        for name, group in groups.items()
    }


def main() -> None:
    started = time.perf_counter()
    old_dir = ROOT / "artifacts" / "hybrid_expanded_repeat_experiment"
    knn_dir = ROOT / "artifacts" / "inactive_knn_experiment"
    old_train, old_final = load_pool(old_dir / "train_candidate_pool.npz"), load_pool(old_dir / "final_candidate_pool.npz")
    knn_train, knn_final = load_pool(knn_dir / "train_candidate_pool.npz"), load_pool(knn_dir / "final_candidate_pool.npz")
    train_old = align_old_pool(old_train, knn_train["users"])
    final_old = align_old_pool(old_final, knn_final["users"])
    train_base = merge_all(train_old, knn_order(knn_train), 25, 0.5)
    final_base = merge_all(final_old, knn_order(knn_final), 25, 0.5)

    recent = load_recent_transactions(ROOT / "Data", ROOT / "artifacts" / "cache")
    train_context, train_labels = slice_window(recent, TRAIN_WINDOW)
    final_context, final_labels = slice_window(recent, FINAL_WINDOW)
    customers, articles = load_metadata(ROOT / "Data")
    encoder = fit_feature_encoder(customers, articles)
    train_features = build_snapshot_features(train_context, customers, articles, encoder, TRAIN_WINDOW.label_start)
    final_features = build_snapshot_features(final_context, customers, articles, encoder, FINAL_WINDOW.label_start)
    if not np.array_equal(train_features.article_ids.astype(str), knn_train["articles"]):
        raise ValueError("Training article rows differ")
    if not np.array_equal(final_features.article_ids.astype(str), knn_final["articles"]):
        raise ValueError("Final article rows differ")

    history = target_history()
    train_states, train_profile = dormant_state_and_profile(knn_train["users"], history, TRAIN_WINDOW.context_start)
    final_states, final_profile = dormant_state_and_profile(knn_final["users"], history, FINAL_WINDOW.context_start)
    long_transactions = load_transactions_since(
        ROOT / "Data", ROOT / "artifacts" / "cache", "2020-02-14"
    )
    train_itemcf = generalized_basket_itemcf(
        long_transactions,
        TRAIN_WINDOW.label_start,
        "2020-02-14",
        train_features.article_to_row,
    )
    final_itemcf = generalized_basket_itemcf(
        long_transactions,
        FINAL_WINDOW.label_start,
        "2020-02-21",
        final_features.article_to_row,
    )
    train_source = itemcf_candidates(
        knn_train["users"], train_profile, train_itemcf, TRAIN_WINDOW.label_start
    )
    final_source = itemcf_candidates(
        knn_final["users"], final_profile, final_itemcf, FINAL_WINDOW.label_start
    )
    train_truth_all, final_truth_all = truth_by_customer(train_labels), truth_by_customer(final_labels)
    train_truth = {u: train_truth_all[u] for u in knn_train["users"]}
    final_truth = {u: final_truth_all[u] for u in knn_final["users"]}
    train_recent_users = [u for u in knn_train["users"] if train_states[u] == "history_within_180d"]
    train_row = {u: row for row, u in enumerate(knn_train["users"])}
    trials = []
    for quota, scale in product(ITEMCF_QUOTAS, ITEMCF_SCALES):
        rankings = [
            add_rank_source(train_base[row], train_source[row], quota, scale)
            for row in range(len(train_base))
        ]
        group_rankings = [rankings[train_row[u]] for u in train_recent_users]
        result = evaluate(
            train_recent_users, group_rankings, knn_train["articles"], train_truth
        )
        trials.append({"itemcf_quota": quota, "itemcf_scale": scale, **result})
    selected = max(
        trials,
        key=lambda row: (
            row["map_at_12"], row["recall_at_12"], -row["itemcf_quota"], -row["itemcf_scale"]
        ),
    )
    final_rankings = []
    for row, user in enumerate(knn_final["users"]):
        if final_states[user] == "history_within_180d":
            final_rankings.append(
                add_rank_source(
                    final_base[row],
                    final_source[row],
                    selected["itemcf_quota"],
                    selected["itemcf_scale"],
                )
            )
        else:
            final_rankings.append(final_base[row])
    output = {
        "selection_protocol": "itemCF quota/scale selected on Sep 9-15 recent-dormant cohort; frozen for Sep 16-22",
        "itemcf": {
            "relationship_history": "180 days before prediction cutoff",
            "relationship": "same-customer same-day co-purchase; historical seed may be outside current catalog; neighbor must be current",
            "profile_decay_days": 60,
        },
        "selected": selected,
        "coverage": {
            "train_recent_dormant": len(train_recent_users),
            "final_recent_dormant": sum(state == "history_within_180d" for state in final_states.values()),
            "final_users_with_itemcf_candidates": sum(len(row) > 0 for row in final_source),
        },
        "final_metrics": {
            "old_plus_knn_base": cohort_metrics(
                knn_final["users"], final_base, knn_final["articles"], final_truth, final_states
            ),
            "base_plus_recent_dormant_itemcf": cohort_metrics(
                knn_final["users"], final_rankings, knn_final["articles"], final_truth, final_states
            ),
        },
        "grid_trials": sorted(trials, key=lambda row: -row["map_at_12"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    output_dir = ROOT / "artifacts" / "recent_dormant_itemcf"
    output_dir.mkdir(parents=True, exist_ok=True)
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

