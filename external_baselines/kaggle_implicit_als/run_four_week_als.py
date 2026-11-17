#!/usr/bin/env python3
"""Adapt the public Kaggle implicit-ALS notebook to this project's evaluation.

Source notebook: Julián Peller, "H&M - Implicit ALS model [0.014]".
The model and interaction construction follow the notebook; data paths, temporal
split, bounded candidate generation, cold-start fallback, and metrics are local
adaptations so its output is directly comparable with this project.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.data import load_recent_transactions, truth_by_customer  # noqa: E402
from hm_recsys.metrics import evaluate_rankings  # noqa: E402


HISTORY_START = "2020-08-19"
VALIDATION_START = "2020-09-16"
VALIDATION_END = "2020-09-23"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--customers", type=int, default=2_000)
    parser.add_argument("--candidate-k", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tag", default="full")
    return parser.parse_args()


def recommend(
    model: implicit.als.AlternatingLeastSquares,
    matrix: csr_matrix,
    evaluation_customers: list[str],
    user_to_row: dict[str, int],
    article_ids: np.ndarray,
    popularity_order: np.ndarray,
    k: int,
    batch_size: int,
    exclude_history: bool,
) -> dict[str, list[str]]:
    """Generate candidates in bounded batches and use popularity for unknown users."""
    k = min(k, len(article_ids))
    fallback = article_ids[popularity_order[:k]].tolist()
    predictions: dict[str, list[str]] = {}
    for start in range(0, len(evaluation_customers), batch_size):
        customer_batch = evaluation_customers[start : start + batch_size]
        known_customers = [c for c in customer_batch if c in user_to_row]
        for customer in customer_batch:
            if customer not in user_to_row:
                predictions[customer] = fallback.copy()
        if not known_customers:
            continue
        rows = np.asarray([user_to_row[c] for c in known_customers], dtype=np.int32)
        ids, _ = model.recommend(
            rows,
            matrix[rows],
            N=k,
            filter_already_liked_items=exclude_history,
        )
        for customer, item_rows in zip(known_customers, ids, strict=True):
            predictions[customer] = article_ids[item_rows].tolist()
    return predictions


def main() -> None:
    args = parse_args()
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

    user_rows, customer_index = pd.factorize(history["customer_id"].astype(str), sort=True)
    item_rows, article_index = pd.factorize(history["article_id"].astype(str), sort=True)
    customer_ids = customer_index.to_numpy(dtype=str)
    article_ids = article_index.to_numpy(dtype=str)
    matrix = csr_matrix(
        (
            np.ones(len(history), dtype=np.float32),
            (user_rows.astype(np.int32), item_rows.astype(np.int32)),
        ),
        shape=(len(customer_ids), len(article_ids)),
        dtype=np.float32,
    )
    user_to_row = {customer: row for row, customer in enumerate(customer_ids)}
    popularity_order = np.argsort(-np.asarray(matrix.sum(axis=0)).ravel(), kind="stable")

    validation_customers = np.sort(validation["customer_id"].astype(str).unique())
    rng = np.random.default_rng(SEED)
    evaluation_customers = np.sort(
        rng.choice(validation_customers, size=min(args.customers, len(validation_customers)), replace=False)
    ).tolist()
    all_truth = truth_by_customer(validation)
    truth = {customer: all_truth[customer] for customer in evaluation_customers}
    known = set(customer_ids)

    fit_started = time.perf_counter()
    model = implicit.als.AlternatingLeastSquares(
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.regularization,
        random_state=SEED,
    )
    model.fit(matrix, show_progress=True)
    fit_seconds = time.perf_counter() - fit_started

    score_started = time.perf_counter()
    allow_seen = recommend(
        model,
        matrix,
        evaluation_customers,
        user_to_row,
        article_ids,
        popularity_order,
        args.candidate_k,
        args.batch_size,
        exclude_history=False,
    )
    exclude_seen = recommend(
        model,
        matrix,
        evaluation_customers,
        user_to_row,
        article_ids,
        popularity_order,
        args.candidate_k,
        args.batch_size,
        exclude_history=True,
    )
    score_seconds = time.perf_counter() - score_started

    output = {
        "provenance": {
            "source": "Julián Peller, H&M - Implicit ALS model [0.014]",
            "url": "https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014",
            "adaptations": [
                "four-week history and untouched final validation week",
                "same deterministic 2,000-customer evaluation sample as project experiments",
                "candidate Recall@K, MAP@12, and Recall@12 with project metric code",
                "recent-popularity fallback for customers absent from the four-week history",
                "bounded recommendation batches",
            ],
        },
        "config": {
            "history_start": HISTORY_START,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
            "seed": SEED,
            "factors": args.factors,
            "iterations": args.iterations,
            "regularization": args.regularization,
            "evaluation_customers": len(evaluation_customers),
            "candidate_k": min(args.candidate_k, len(article_ids)),
            "batch_size": args.batch_size,
        },
        "dataset": {
            "history_rows": len(history),
            "history_customers": len(customer_ids),
            "eligible_articles": len(article_ids),
            "validation_rows": len(validation),
            "validation_customers": len(validation_customers),
            "sample_history_customers": sum(c in known for c in evaluation_customers),
            "sample_cold_start_customers": sum(c not in known for c in evaluation_customers),
        },
        "metrics": {
            "allow_seen_articles": evaluate_rankings(
                allow_seen, truth, known, min(args.candidate_k, len(article_ids))
            ),
            "exclude_seen_articles": evaluate_rankings(
                exclude_seen, truth, known, min(args.candidate_k, len(article_ids))
            ),
        },
        "runtime_seconds": {
            "training": fit_seconds,
            "candidate_generation_both_policies": score_seconds,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "implicit": implicit.__version__,
            "platform": platform.platform(),
        },
    }
    run_dir = ROOT / "artifacts" / "external_als"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / f"results_{args.tag}.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["metrics"], indent=2))
    print(f"Results written to {result_path}")


if __name__ == "__main__":
    main()
