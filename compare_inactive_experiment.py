#!/usr/bin/env python3
"""Compare the inactive KNN experiment with preserved four-week baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import (  # noqa: E402
    FINAL_WINDOW,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_recent_transactions,
    slice_window,
    truth_by_customer,
)
from hm_recsys.inactive_pipeline import _metrics_for_subgroups, _user_state  # noqa: E402


def rankings(users: list[str], rows: np.ndarray, article_ids: np.ndarray):
    return {
        user: [article_ids[item] for item in rows[row] if item >= 0]
        for row, user in enumerate(users)
    }


def main() -> None:
    artifacts = ROOT / "artifacts"
    new_dir = artifacts / "inactive_knn_experiment"
    new_pool = np.load(new_dir / "final_candidate_pool.npz", allow_pickle=True)
    users = new_pool["customer_ids"].astype(str).tolist()

    recent = load_recent_transactions(ROOT / "Data", artifacts / "cache")
    context, labels = slice_window(recent, FINAL_WINDOW)
    customers, articles = load_metadata(ROOT / "Data")
    encoder = fit_feature_encoder(customers, articles)
    features = build_snapshot_features(
        context, customers, articles, encoder, FINAL_WINDOW.label_start
    )
    history_caches = sorted((artifacts / "cache").glob("inactive_target_history_*.pkl"))
    if not history_caches:
        raise FileNotFoundError("Inactive target-history cache is missing")
    target_history = pd.concat(
        [pd.read_pickle(path) for path in history_caches], ignore_index=True
    ).drop_duplicates()
    state, _ = _user_state(
        users,
        target_history,
        FINAL_WINDOW.context_start,
        FINAL_WINDOW.label_start,
        180,
        features,
    )
    np.savez_compressed(
        new_dir / "final_user_state.npz",
        customer_ids=np.asarray(users),
        has_history_180d=state.has_history_180d,
        has_any_history=state.has_any_history,
        days_since_last_purchase=state.days_since_last_purchase,
        historical_purchase_count=state.historical_purchase_count,
    )
    truth_all = truth_by_customer(labels)
    truth = {user: truth_all[user] for user in users}

    baseline_dir = artifacts / "hybrid_expanded_repeat_experiment"
    baseline_pool = np.load(
        baseline_dir / "final_candidate_pool.npz", allow_pickle=True
    )
    baseline_users = baseline_pool["customer_ids"].astype(str).tolist()
    baseline_row = {user: row for row, user in enumerate(baseline_users)}
    selected_rows = np.asarray([baseline_row[user] for user in users], dtype=np.int64)
    baseline_articles = baseline_pool["article_ids"].astype(str)
    baseline_candidates = rankings(
        users,
        baseline_pool["candidate_item_rows"][selected_rows],
        baseline_articles,
    )
    baseline_predictions_frame = pd.read_csv(
        artifacts
        / "predictions"
        / "hybrid_expanded_repeat_experiment_lightgbm_top12.csv",
        dtype=str,
    )
    baseline_predictions_all = {
        row.customer_id: row.prediction.split()
        for row in baseline_predictions_frame.itertuples()
    }
    baseline_predictions = {
        user: baseline_predictions_all[user] for user in users
    }

    new_results = json.loads((new_dir / "results.json").read_text())
    comparison = {
        "cohort_definitions": {
            "all_inactive": "no purchase in the final four-week context",
            "history_within_180d": "older transaction in the 180 days before context",
            "stale_history_over_180d": "prior transaction exists but none in that 180-day window",
            "no_prior_transaction": "no earlier transaction in the provided data",
        },
        "baseline_candidate_union": _metrics_for_subgroups(
            baseline_candidates, truth, state, baseline_pool["candidate_item_rows"].shape[1]
        ),
        "baseline_lightgbm": _metrics_for_subgroups(
            baseline_predictions, truth, state, 12
        ),
        "new_inactive_union": new_results["metrics"]["inactive_union_plus_repeats"]["cohorts"],
        "new_inactive_lightgbm": new_results["metrics"]["inactive_lightgbm"]["cohorts"],
        "new_global_popularity": new_results["metrics"]["global_popularity"]["cohorts"],
        "new_segment_popularity": new_results["metrics"]["segment_popularity"]["cohorts"],
        "new_behavioral_knn": new_results["metrics"]["behavioral_knn"]["cohorts"],
        "new_metadata_knn": new_results["metrics"]["metadata_knn"]["cohorts"],
    }
    output_dir = artifacts / "inactive_knn_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.json"
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
