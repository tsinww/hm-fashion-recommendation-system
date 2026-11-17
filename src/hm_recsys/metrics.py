"""Ranking metrics used consistently by every recommender stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of a customer's distinct relevant items present in the first k."""
    if not relevant:
        return np.nan
    return len(set(recommended[:k]) & relevant) / len(relevant)


def average_precision_at_k(
    recommended: Sequence[str], relevant: set[str], k: int = 12
) -> float:
    """Kaggle-style AP@k with distinct recommendations and binary relevance."""
    if not relevant:
        return np.nan
    score = 0.0
    hits = 0
    seen: set[str] = set()
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(len(relevant), k)


def evaluate_rankings(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, set[str]],
    known_customers: set[str],
    candidate_k: int,
) -> dict[str, dict[str, float | int]]:
    """Return the same metrics overall and for history/cold-start cohorts."""
    rows: list[tuple[str, float, float, float]] = []
    for customer_id, relevant in truth.items():
        recs = predictions.get(customer_id, ())
        cohort = "history" if customer_id in known_customers else "cold_start"
        rows.append(
            (
                cohort,
                recall_at_k(recs, relevant, candidate_k),
                average_precision_at_k(recs, relevant, 12),
                recall_at_k(recs, relevant, 12),
            )
        )

    output: dict[str, dict[str, float | int]] = {}
    for name in ("overall", "history", "cold_start"):
        selected = rows if name == "overall" else [row for row in rows if row[0] == name]
        values = np.asarray([row[1:] for row in selected], dtype=np.float64)
        output[name] = {
            "customers": len(selected),
            "candidate_recall": float(np.nanmean(values[:, 0])) if len(values) else np.nan,
            "map_at_12": float(np.nanmean(values[:, 1])) if len(values) else np.nan,
            "recall_at_12": float(np.nanmean(values[:, 2])) if len(values) else np.nan,
        }
    return output

