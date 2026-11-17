#!/usr/bin/env python3
"""Test seen-item filtering with the project's four-week BPR MF."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import load_recent_transactions, truth_by_customer  # noqa: E402
from hm_recsys.metrics import evaluate_rankings  # noqa: E402
from hm_recsys.models import fit_mf  # noqa: E402


EVALUATION_SEED = 42
MF_TRAINING_SEED = 46
HISTORY_START = "2020-08-19"
VALIDATION_START = "2020-09-16"
VALIDATION_END = "2020-09-23"
EVALUATION_CUSTOMERS = 2_000
CANDIDATE_K = 1_000
EMBEDDING_DIM = 32
NEGATIVES_PER_POSITIVE = 1
EPOCHS = 3
BATCH_SIZE = 4_096
LEARNING_RATE = 0.003


def rankings(
    customers: list[str], candidate_rows: np.ndarray, article_ids: np.ndarray
) -> dict[str, list[str]]:
    return {
        customer: article_ids[candidate_rows[row]].tolist()
        for row, customer in enumerate(customers)
    }


def seen_top12_fraction(
    predictions: dict[str, list[str]],
    purchased: dict[str, set[str]],
    customers: set[str] | None = None,
) -> float:
    selected = predictions if customers is None else {
        customer: items for customer, items in predictions.items() if customer in customers
    }
    seen = sum(
        item in purchased.get(customer, set())
        for customer, items in selected.items()
        for item in items[:12]
    )
    return seen / (12 * len(selected))


def main() -> None:
    started = time.perf_counter()
    transactions = load_recent_transactions(ROOT / "Data", ROOT / "artifacts" / "cache")
    history = transactions.loc[
        (transactions["t_dat"] >= HISTORY_START)
        & (transactions["t_dat"] < VALIDATION_START)
    ].copy()
    validation = transactions.loc[
        (transactions["t_dat"] >= VALIDATION_START)
        & (transactions["t_dat"] < VALIDATION_END)
    ].copy()

    pairs = history[["customer_id", "article_id"]].drop_duplicates()
    customer_ids = np.sort(pairs["customer_id"].astype(str).unique())
    article_ids = np.sort(pairs["article_id"].astype(str).unique())
    popularity_counts = (
        history["article_id"].astype(str).value_counts().reindex(article_ids, fill_value=0)
    ).to_numpy(dtype=np.float32)
    purchased = (
        history.groupby("customer_id", observed=True)["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )

    truth_all = truth_by_customer(validation)
    all_validation_customers = np.sort(validation["customer_id"].astype(str).unique())
    rng = np.random.default_rng(EVALUATION_SEED)
    evaluation_customers = np.sort(
        rng.choice(all_validation_customers, size=EVALUATION_CUSTOMERS, replace=False)
    ).tolist()
    truth = {customer: truth_all[customer] for customer in evaluation_customers}
    known = set(customer_ids)

    fit_started = time.perf_counter()
    model, losses = fit_mf(
        customer_ids=customer_ids,
        item_ids=article_ids,
        popularity=popularity_counts,
        pair_customers=pairs["customer_id"].astype(str).to_numpy(),
        pair_items=pairs["article_id"].astype(str).to_numpy(),
        dim=EMBEDDING_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=MF_TRAINING_SEED,
        max_pairs=None,
    )
    fit_seconds = time.perf_counter() - fit_started

    score_started = time.perf_counter()
    unfiltered_rows, _ = model.candidates(evaluation_customers, CANDIDATE_K)
    filtered_rows, _ = model.candidates(
        evaluation_customers, CANDIDATE_K, excluded_items=purchased
    )
    score_seconds = time.perf_counter() - score_started
    unfiltered_predictions = rankings(evaluation_customers, unfiltered_rows, article_ids)
    filtered_predictions = rankings(evaluation_customers, filtered_rows, article_ids)

    unfiltered_metrics = evaluate_rankings(
        unfiltered_predictions, truth, known, CANDIDATE_K
    )
    filtered_metrics = evaluate_rankings(filtered_predictions, truth, known, CANDIDATE_K)
    new_item_truth = {
        customer: items - purchased.get(customer, set())
        for customer, items in truth.items()
        if items - purchased.get(customer, set())
    }
    repeat_truth = {
        customer: items & purchased.get(customer, set())
        for customer, items in truth.items()
        if items & purchased.get(customer, set())
    }

    relevant_pairs = sum(len(items) for items in truth.values())
    exact_repeats = sum(
        len(items & purchased.get(customer, set())) for customer, items in truth.items()
    )
    known_count = sum(customer in known for customer in evaluation_customers)
    output = {
        "config": {
            "evaluation_seed": EVALUATION_SEED,
            "mf_training_seed": MF_TRAINING_SEED,
            "history_start": HISTORY_START,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
            "evaluation_customers": EVALUATION_CUSTOMERS,
            "candidate_k": CANDIDATE_K,
            "embedding_dim": EMBEDDING_DIM,
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "BPR pairwise loss",
            "setup": "four_week_project_mf",
        },
        "dataset": {
            "history_rows": len(history),
            "history_distinct_pairs": len(pairs),
            "history_customers": len(customer_ids),
            "eligible_articles": len(article_ids),
            "validation_rows": len(validation),
            "validation_customers": len(all_validation_customers),
            "sample_history_customers": known_count,
            "sample_cold_start_customers": EVALUATION_CUSTOMERS - known_count,
            "sample_fallback_fraction": (EVALUATION_CUSTOMERS - known_count)
            / EVALUATION_CUSTOMERS,
            "sample_validation_distinct_pairs": relevant_pairs,
            "sample_exact_repeat_pairs": exact_repeats,
            "sample_exact_repeat_pair_fraction": exact_repeats / relevant_pairs,
            "catalog_fraction_retrieved": CANDIDATE_K / len(article_ids),
        },
        "training": {"epoch_losses": losses},
        "metrics": {
            "allow_seen_articles": {
                "candidate_k": CANDIDATE_K,
                "cohorts": unfiltered_metrics,
                "top12_seen_article_fraction": seen_top12_fraction(
                    unfiltered_predictions, purchased
                ),
                "history_customer_top12_seen_article_fraction": seen_top12_fraction(
                    unfiltered_predictions, purchased, known
                ),
                "new_item_metrics": evaluate_rankings(
                    unfiltered_predictions, new_item_truth, known, CANDIDATE_K
                ),
                "repeat_item_metrics": evaluate_rankings(
                    unfiltered_predictions, repeat_truth, known, CANDIDATE_K
                ),
            },
            "exclude_seen_articles": {
                "candidate_k": CANDIDATE_K,
                "cohorts": filtered_metrics,
                "top12_seen_article_fraction": seen_top12_fraction(
                    filtered_predictions, purchased
                ),
                "history_customer_top12_seen_article_fraction": seen_top12_fraction(
                    filtered_predictions, purchased, known
                ),
                "new_item_metrics": evaluate_rankings(
                    filtered_predictions, new_item_truth, known, CANDIDATE_K
                ),
                "repeat_item_metrics": evaluate_rankings(
                    filtered_predictions, repeat_truth, known, CANDIDATE_K
                ),
            },
        },
        "runtime_seconds": {
            "mf_training": fit_seconds,
            "candidate_scoring_both_policies": score_seconds,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(model.device),
        },
    }
    run_dir = ROOT / "artifacts" / "mf_seen_ablation"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    prediction_dir = ROOT / "artifacts" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name, predictions in (
        ("allow_seen", unfiltered_predictions),
        ("exclude_seen", filtered_predictions),
    ):
        pd.DataFrame(
            {
                "customer_id": evaluation_customers,
                "prediction": [" ".join(predictions[customer][:12]) for customer in evaluation_customers],
            }
        ).to_csv(prediction_dir / f"mf_{name}_top12.csv", index=False)
    print(f"Results written to {result_path}")


if __name__ == "__main__":
    main()
