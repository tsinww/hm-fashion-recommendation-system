#!/usr/bin/env python3
"""Build and evaluate the project's best observed cohort-routed system.

The active route uses equal-rank LightGBM--CatBoost fusion over the saved
four-week 1,012-candidate pool.  The inactive route preserves that pool and
adds bounded KNN candidates plus time-decayed ItemCF for recent-dormant users.

The resulting score is a post-hoc diagnostic because the decision to route
active and inactive users differently was made after inspecting final-week
cohort results.  Use ``--rebuild`` to regenerate every upstream artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import (  # noqa: E402
    FINAL_WINDOW,
    load_recent_transactions,
    slice_window,
    truth_by_customer,
)
from hm_recsys.metrics import average_precision_at_k, recall_at_k  # noqa: E402


ACTIVE_PREDICTIONS = (
    ROOT
    / "artifacts"
    / "own_four_week_ranker_ensemble"
    / "equal_rank_fusion_top12.csv"
)
INACTIVE_PREDICTIONS = ROOT / "artifacts" / "recent_dormant_itemcf" / "top12.csv"
OUTPUT_DIR = ROOT / "artifacts" / "best_routed_system"


def read_predictions(path: Path) -> dict[str, list[str]]:
    """Read a two-column customer/prediction file without losing leading zeros."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run this script with --rebuild to create it."
        )
    frame = pd.read_csv(
        path,
        dtype={"customer_id": "string", "prediction": "string"},
        keep_default_na=False,
    )
    required = {"customer_id", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return {
        str(row.customer_id): str(row.prediction).split()
        for row in frame.itertuples(index=False)
    }


def route_predictions(
    evaluation_users: list[str],
    active_users: set[str],
    active_predictions: dict[str, list[str]],
    inactive_predictions: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Select the active or inactive ranking for every evaluation customer."""
    output: dict[str, list[str]] = {}
    for user in evaluation_users:
        source = active_predictions if user in active_users else inactive_predictions
        if user not in source:
            route = "active" if user in active_users else "inactive"
            raise KeyError(f"Missing {route} prediction for customer {user}")
        output[user] = source[user][:12]
    return output


def evaluate(
    predictions: dict[str, list[str]],
    truth: dict[str, set[str]],
    active_users: set[str],
) -> dict[str, dict[str, float | int]]:
    """Return macro MAP@12 and Recall@12 overall and by routing cohort."""
    rows: list[tuple[str, float, float]] = []
    for user, relevant in truth.items():
        cohort = "active" if user in active_users else "inactive"
        rows.append(
            (
                cohort,
                average_precision_at_k(predictions[user], relevant, 12),
                recall_at_k(predictions[user], relevant, 12),
            )
        )
    output: dict[str, dict[str, float | int]] = {}
    for cohort in ("overall", "active", "inactive"):
        selected = rows if cohort == "overall" else [r for r in rows if r[0] == cohort]
        values = np.asarray([r[1:] for r in selected], dtype=np.float64)
        output[cohort] = {
            "customers": len(selected),
            "map_at_12": float(np.nanmean(values[:, 0])),
            "recall_at_12": float(np.nanmean(values[:, 1])),
        }
    return output


def rebuild() -> None:
    """Regenerate candidate retrieval, rankers, and inactive-route artifacts."""
    commands = [
        [
            sys.executable,
            "run_hybrid_experiment.py",
            "--profile",
            "hybrid_expanded_repeat_experiment",
            "--save-candidates",
        ],
        [sys.executable, "run_own_ranker_ensemble.py"],
        [
            sys.executable,
            "run_inactive_experiment.py",
            "--profile",
            "inactive_knn_experiment",
        ],
        [sys.executable, "run_inactive_knn_merge.py"],
        [sys.executable, "run_recent_dormant_itemcf.py"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate every upstream artifact before routing and evaluation.",
    )
    args = parser.parse_args()
    if args.rebuild:
        rebuild()

    active_predictions = read_predictions(ACTIVE_PREDICTIONS)
    inactive_predictions = read_predictions(INACTIVE_PREDICTIONS)
    evaluation_users = list(active_predictions)

    transactions = load_recent_transactions(ROOT / "Data", ROOT / "artifacts" / "cache")
    context, labels = slice_window(transactions, FINAL_WINDOW)
    active_users = set(context["customer_id"].astype(str))
    all_truth = truth_by_customer(labels)
    missing_truth = [user for user in evaluation_users if user not in all_truth]
    if missing_truth:
        raise ValueError(f"{len(missing_truth)} evaluation customers have no labels")
    truth = {user: all_truth[user] for user in evaluation_users}
    predictions = route_predictions(
        evaluation_users,
        active_users,
        active_predictions,
        inactive_predictions,
    )
    metrics = evaluate(predictions, truth, active_users)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "customer_id": evaluation_users,
            "prediction": [" ".join(predictions[user]) for user in evaluation_users],
            "route": [
                "active_lgbm_catboost_fusion"
                if user in active_users
                else "inactive_knn_itemcf_rrf"
                for user in evaluation_users
            ],
        }
    ).to_csv(OUTPUT_DIR / "top12.csv", index=False)
    result = {
        "status": "post_hoc_diagnostic_not_untouched_model_selection",
        "warning": (
            "The component predictions are leakage-safe, but the active/inactive "
            "routing policy was chosen after inspecting final-week cohort results. "
            "Validate the frozen route on another temporal fold before treating it "
            "as an unbiased official result."
        ),
        "architecture": {
            "active": "equal reciprocal-rank fusion of LightGBM and CatBoost",
            "inactive": (
                "preserved 1,012-candidate pool plus bounded KNN; add time-decayed "
                "ItemCF for recent-dormant customers; weighted reciprocal-rank ordering"
            ),
        },
        "window": {
            "context_start": FINAL_WINDOW.context_start,
            "label_start": FINAL_WINDOW.label_start,
            "label_end": FINAL_WINDOW.label_end,
        },
        "metrics": metrics,
    }
    path = OUTPUT_DIR / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {path}")


if __name__ == "__main__":
    main()
