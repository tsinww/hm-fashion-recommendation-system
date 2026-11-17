#!/usr/bin/env python3
"""Compare 28-, 42-, and 56-day runs on identical fixed customer cohorts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import Window, slice_window, truth_by_customer  # noqa: E402
from hm_recsys.metrics import evaluate_rankings  # noqa: E402


PROFILES = {
    "four_weeks": "hybrid_expanded_repeat_experiment",
    "six_weeks": "hybrid_six_week_experiment",
    "eight_weeks": "hybrid_eight_week_experiment",
}
WINDOWS = {
    "four_weeks": Window("2020-08-19", "2020-09-16", "2020-09-23"),
    "six_weeks": Window("2020-08-05", "2020-09-16", "2020-09-23"),
    "eight_weeks": Window("2020-07-22", "2020-09-16", "2020-09-23"),
}


def main() -> None:
    artifacts = ROOT / "artifacts"
    transactions = pd.read_pickle(
        artifacts / "cache" / "transactions_2020-07-15_to_2020-09-22.pkl"
    )
    contexts = {}
    labels = None
    for horizon, window in WINDOWS.items():
        context, horizon_labels = slice_window(transactions, window)
        contexts[horizon] = set(context["customer_id"].astype(str).unique())
        if labels is None:
            labels = horizon_labels
        elif set(labels.index) != set(horizon_labels.index):
            raise ValueError("History horizons do not share identical validation labels")
    assert labels is not None
    truth_all = truth_by_customer(labels)

    result: dict[str, object] = {
        "profiles": PROFILES,
        "windows": {name: window.__dict__ for name, window in WINDOWS.items()},
        "cohort_definitions": {
            "active_4w": "purchase in days 1-28 before the cutoff",
            "recovered_5_to_6w": "no purchase in days 1-28; purchase in days 29-42",
            "recovered_7_to_8w": "no purchase in days 1-42; purchase in days 43-56",
            "cold_after_8w": "no purchase in days 1-56",
        },
        "metrics": {},
    }
    reference_users: list[str] | None = None
    for horizon, profile in PROFILES.items():
        pool = np.load(
            artifacts / profile / "final_candidate_pool.npz", allow_pickle=True
        )
        users = pool["customer_ids"].astype(str).tolist()
        if reference_users is None:
            reference_users = users
        elif users != reference_users:
            raise ValueError("Profiles do not contain the same evaluation customers")

        article_ids = pool["article_ids"].astype(str)
        candidate_rankings = {
            user: article_ids[rows].tolist()
            for user, rows in zip(users, pool["candidate_item_rows"], strict=True)
        }
        predictions = pd.read_csv(
            artifacts / "predictions" / f"{profile}_lightgbm_top12.csv", dtype=str
        )
        top12 = {
            row.customer_id: row.prediction.split()
            for row in predictions.itertuples()
        }
        known_4w = contexts["four_weeks"]
        known_6w = contexts["six_weeks"]
        known_8w = contexts["eight_weeks"]
        cohorts = {
            "active_4w": [user for user in users if user in known_4w],
            "recovered_5_to_6w": [
                user for user in users if user not in known_4w and user in known_6w
            ],
            "recovered_7_to_8w": [
                user for user in users if user not in known_6w and user in known_8w
            ],
            "cold_after_8w": [user for user in users if user not in known_8w],
        }
        result["metrics"][horizon] = {}
        for cohort, cohort_users in cohorts.items():
            cohort_truth = {user: truth_all[user] for user in cohort_users}
            candidate = evaluate_rankings(
                {user: candidate_rankings[user] for user in cohort_users},
                cohort_truth,
                set(cohort_users),
                len(next(iter(candidate_rankings.values()))),
            )["overall"]
            ranking = evaluate_rankings(
                {user: top12[user] for user in cohort_users},
                cohort_truth,
                set(cohort_users),
                12,
            )["overall"]
            result["metrics"][horizon][cohort] = {
                "customers": len(cohort_users),
                "candidate_recall": candidate["candidate_recall"],
                "map_at_12": ranking["map_at_12"],
                "recall_at_12": ranking["recall_at_12"],
            }

    output_dir = artifacts / "history_horizon_comparison_4w_6w_8w"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
