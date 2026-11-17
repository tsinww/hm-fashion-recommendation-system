#!/usr/bin/env python3
"""Compare four- and eight-week runs on identical fixed customer cohorts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import FINAL_WINDOW, Window, slice_window, truth_by_customer  # noqa: E402
from hm_recsys.metrics import evaluate_rankings  # noqa: E402


PROFILES = {
    "four_weeks": "hybrid_expanded_repeat_experiment",
    "eight_weeks": "hybrid_eight_week_experiment",
}


def main() -> None:
    artifacts = ROOT / "artifacts"
    transactions = pd.read_pickle(
        artifacts / "cache" / "transactions_2020-07-15_to_2020-09-22.pkl"
    )
    context_4w, labels = slice_window(transactions, FINAL_WINDOW)
    context_8w, _ = slice_window(
        transactions, Window("2020-07-22", "2020-09-16", "2020-09-23")
    )
    known_4w = set(context_4w["customer_id"].astype(str).unique())
    known_8w = set(context_8w["customer_id"].astype(str).unique())
    truth_all = truth_by_customer(labels)

    result: dict[str, object] = {
        "profiles": PROFILES,
        "cohort_definitions": {
            "active_4w": "purchase in the final snapshot's most recent 28 days",
            "recovered_5_to_8w": "no purchase in 28 days but a purchase in days 29-56",
            "cold_after_8w": "no purchase in the final snapshot's 56-day context",
        },
        "metrics": {},
    }
    reference_users: list[str] | None = None
    for horizon, profile in PROFILES.items():
        run_dir = artifacts / profile
        pool = np.load(run_dir / "final_candidate_pool.npz", allow_pickle=True)
        users = pool["customer_ids"].astype(str).tolist()
        if reference_users is None:
            reference_users = users
        elif users != reference_users:
            raise ValueError("Profiles do not contain the same evaluation customers")

        truth = {user: truth_all[user] for user in users}
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
        cohorts = {
            "active_4w": [user for user in users if user in known_4w],
            "recovered_5_to_8w": [
                user for user in users if user not in known_4w and user in known_8w
            ],
            "cold_after_8w": [user for user in users if user not in known_8w],
        }
        result["metrics"][horizon] = {}
        for cohort, cohort_users in cohorts.items():
            cohort_truth = {user: truth[user] for user in cohort_users}
            candidate_metrics = evaluate_rankings(
                {user: candidate_rankings[user] for user in cohort_users},
                cohort_truth,
                set(cohort_users),
                len(next(iter(candidate_rankings.values()))),
            )["overall"]
            top12_metrics = evaluate_rankings(
                {user: top12[user] for user in cohort_users},
                cohort_truth,
                set(cohort_users),
                12,
            )["overall"]
            result["metrics"][horizon][cohort] = {
                "customers": len(cohort_users),
                "candidate_recall": candidate_metrics["candidate_recall"],
                "map_at_12": top12_metrics["map_at_12"],
                "recall_at_12": top12_metrics["recall_at_12"],
            }

    output_dir = artifacts / "history_horizon_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
