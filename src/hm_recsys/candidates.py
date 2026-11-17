"""Parallel candidate generators and deterministic candidate unions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import implicit
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from .data import SnapshotFeatures
from .models import TwoTower, fit_binary_model


SOURCE_NAMES = (
    "bpr_mf",
    "implicit_als",
    "co_purchase",
    "popularity",
    "segment_popularity",
    "two_tower",
    "repeat",
)


@dataclass
class CandidateSource:
    """A fixed-width ranked list represented by snapshot item-row indices."""

    name: str
    items: np.ndarray


@dataclass
class CandidateUnion:
    """Candidate rows plus reciprocal-rank provenance for every source."""

    items: np.ndarray
    source_rr: np.ndarray
    source_names: tuple[str, ...]


@dataclass
class FittedALS:
    model: implicit.als.AlternatingLeastSquares
    matrix: csr_matrix
    user_to_row: dict[str, int]
    item_ids: np.ndarray
    popularity_order: np.ndarray

    def candidates(
        self,
        customer_ids: list[str],
        k: int,
        batch_size: int = 128,
    ) -> np.ndarray:
        k = min(k, len(self.item_ids))
        output = np.empty((len(customer_ids), k), dtype=np.int64)
        fallback = self.popularity_order[:k]
        for start in range(0, len(customer_ids), batch_size):
            stop = min(start + batch_size, len(customer_ids))
            batch = customer_ids[start:stop]
            known_positions = [i for i, customer in enumerate(batch) if customer in self.user_to_row]
            for position, customer in enumerate(batch):
                if customer not in self.user_to_row:
                    output[start + position] = fallback
            if not known_positions:
                continue
            rows = np.asarray(
                [self.user_to_row[batch[position]] for position in known_positions],
                dtype=np.int32,
            )
            ids, _ = self.model.recommend(
                rows,
                self.matrix[rows],
                N=k,
                filter_already_liked_items=False,
            )
            for local, position in enumerate(known_positions):
                output[start + position] = ids[local]
        return output


def fit_als(
    context: pd.DataFrame,
    features: SnapshotFeatures,
    factors: int,
    iterations: int,
    regularization: float,
    seed: int,
) -> FittedALS:
    customer_ids = np.sort(context["customer_id"].astype(str).unique())
    user_to_row = {customer: row for row, customer in enumerate(customer_ids)}
    users = context["customer_id"].astype(str).map(user_to_row).to_numpy(dtype=np.int32)
    items = context["article_id"].astype(str).map(features.article_to_row).to_numpy(dtype=np.int32)
    matrix = csr_matrix(
        (np.ones(len(context), dtype=np.float32), (users, items)),
        shape=(len(customer_ids), len(features.article_ids)),
        dtype=np.float32,
    )
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=seed,
    )
    model.fit(matrix, show_progress=False)
    return FittedALS(
        model=model,
        matrix=matrix,
        user_to_row=user_to_row,
        item_ids=features.article_ids,
        popularity_order=np.argsort(-features.item_popularity, kind="stable"),
    )


def popularity_candidates(
    features: SnapshotFeatures, n_customers: int, k: int
) -> np.ndarray:
    top = np.argsort(-features.item_popularity, kind="stable")[: min(k, len(features.article_ids))]
    return np.tile(top, (n_customers, 1))


def _age_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[0, 19, 24, 29, 39, 49, 59, 69, 200],
        include_lowest=True,
    ).astype("string").fillna("__MISSING__")


def segment_popularity_candidates(
    context: pd.DataFrame,
    customers: pd.DataFrame,
    features: SnapshotFeatures,
    customer_ids: list[str],
    k: int,
) -> np.ndarray:
    """Top context articles within the customer's age bucket, with global fill."""
    k = min(k, len(features.article_ids))
    customer_age = customers[["customer_id", "age"]].copy()
    customer_age["customer_id"] = customer_age["customer_id"].astype(str)
    customer_age["age_bucket"] = _age_bucket(customer_age["age"])
    bucket_by_customer = customer_age.set_index("customer_id")["age_bucket"].to_dict()
    joined = context[["customer_id", "article_id"]].copy()
    joined["customer_id"] = joined["customer_id"].astype(str)
    joined["age_bucket"] = joined["customer_id"].map(bucket_by_customer).fillna("__MISSING__")
    counts = (
        joined.groupby(["age_bucket", "article_id"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["age_bucket", "count", "article_id"], ascending=[True, False, True])
    )
    global_top = np.argsort(-features.item_popularity, kind="stable")
    per_bucket: dict[str, list[int]] = defaultdict(list)
    for bucket, article in counts[["age_bucket", "article_id"]].itertuples(index=False):
        row = features.article_to_row.get(str(article))
        if row is not None and len(per_bucket[str(bucket)]) < k:
            per_bucket[str(bucket)].append(row)
    output = np.empty((len(customer_ids), k), dtype=np.int64)
    for row, customer in enumerate(customer_ids):
        ranked = per_bucket.get(str(bucket_by_customer.get(customer, "__MISSING__")), [])
        selected = list(ranked)
        selected_set = set(selected)
        for item in global_top:
            if int(item) not in selected_set:
                selected.append(int(item))
                selected_set.add(int(item))
                if len(selected) == k:
                    break
        output[row] = selected[:k]
    return output


def build_co_purchase_map(
    context: pd.DataFrame,
    features: SnapshotFeatures,
    neighbors_per_item: int = 30,
    max_basket_items: int = 20,
) -> dict[int, list[tuple[int, int]]]:
    """Count same-customer, same-day article co-occurrence without a dense item matrix."""
    baskets = context[["customer_id", "t_dat", "article_id"]].drop_duplicates().copy()
    baskets["item"] = baskets["article_id"].astype(str).map(features.article_to_row)
    baskets = baskets.dropna(subset=["item"])
    baskets["item"] = baskets["item"].astype(np.int32)
    baskets["basket"] = baskets.groupby(
        ["customer_id", "t_dat"], observed=True, sort=False
    ).ngroup().astype(np.int32)
    sizes = baskets.groupby("basket", observed=True)["item"].transform("size")
    baskets = baskets.loc[(sizes >= 2) & (sizes <= max_basket_items), ["basket", "item"]]
    paired = baskets.merge(baskets, on="basket", how="inner", suffixes=("_left", "_right"))
    paired = paired.loc[paired["item_left"] != paired["item_right"]]
    counts = (
        paired.groupby(["item_left", "item_right"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["item_left", "count", "item_right"],
            ascending=[True, False, True],
        )
        .groupby("item_left", observed=True)
        .head(neighbors_per_item)
    )
    result: dict[int, list[tuple[int, int]]] = {}
    for item, group in counts.groupby("item_left", observed=True, sort=False):
        result[int(item)] = list(
            zip(
                group["item_right"].astype(int).tolist(),
                group["count"].astype(int).tolist(),
                strict=True,
            )
        )
    return result


def co_purchase_candidates(
    context: pd.DataFrame,
    features: SnapshotFeatures,
    customer_ids: list[str],
    co_purchase_map: dict[int, list[tuple[int, int]]],
    k: int,
) -> np.ndarray:
    k = min(k, len(features.article_ids))
    recent = context.sort_values("t_dat", ascending=False).groupby(
        "customer_id", observed=True
    )["article_id"].agg(lambda values: list(dict.fromkeys(values.astype(str)))[:20])
    recent_by_customer = recent.to_dict()
    global_top = np.argsort(-features.item_popularity, kind="stable")
    output = np.empty((len(customer_ids), k), dtype=np.int64)
    for user_row, customer in enumerate(customer_ids):
        scores: Counter[int] = Counter()
        for history_rank, article in enumerate(recent_by_customer.get(customer, [])):
            item = features.article_to_row.get(article)
            if item is None:
                continue
            recency_weight = 1.0 / (history_rank + 1)
            for neighbor, count in co_purchase_map.get(item, []):
                scores[neighbor] += count * recency_weight
        ranked = [item for item, _ in scores.most_common(k)]
        selected = list(ranked)
        selected_set = set(selected)
        for item in global_top:
            if int(item) not in selected_set:
                selected.append(int(item))
                selected_set.add(int(item))
                if len(selected) == k:
                    break
        output[user_row] = selected[:k]
    return output


def repeat_candidates(
    context: pd.DataFrame,
    features: SnapshotFeatures,
    customer_ids: list[str],
    cutoff: str,
    k: int,
) -> np.ndarray:
    """Small repeat list ranked by frequency with exponential recency decay."""
    stats = context.groupby(["customer_id", "article_id"], observed=True).agg(
        count=("article_id", "size"), last_date=("t_dat", "max")
    ).reset_index()
    stats["score"] = stats["count"] * np.exp(
        -((pd.Timestamp(cutoff) - stats["last_date"]).dt.days.clip(lower=0) / 14.0)
    )
    stats = stats.sort_values(
        ["customer_id", "score", "article_id"], ascending=[True, False, True]
    )
    ranked = stats.groupby("customer_id", observed=True)["article_id"].agg(list).to_dict()
    output = np.full((len(customer_ids), k), -1, dtype=np.int64)
    for row, customer in enumerate(customer_ids):
        items = [
            features.article_to_row[str(article)]
            for article in ranked.get(customer, [])[:k]
            if str(article) in features.article_to_row
        ]
        output[row, : len(items)] = items
    return output


def _customer_feature_rows(
    features: SnapshotFeatures, customer_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.fromiter(
        (features.customer_to_row.get(customer, -1) for customer in customer_ids),
        dtype=np.int64,
    )
    cats = np.zeros((len(rows), features.user_cats.shape[1]), dtype=np.int64)
    dense = np.zeros((len(rows), features.user_dense.shape[1]), dtype=np.float32)
    present = rows >= 0
    cats[present] = features.user_cats[rows[present]]
    dense[present] = features.user_dense[rows[present]]
    return cats, dense


def fit_two_tower_source(
    context: pd.DataFrame,
    features: SnapshotFeatures,
    user_cardinalities: list[int],
    item_cardinalities: list[int],
    max_pairs: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[TwoTower, list[float], dict[str, int]]:
    """Train a full-catalog retrieval tower with uniform catalog negatives."""
    pairs = context[["customer_id", "article_id"]].drop_duplicates()
    if len(pairs) > max_pairs:
        pairs = pairs.sample(max_pairs, random_state=seed)
    customers = pairs["customer_id"].astype(str).tolist()
    positive_items = pairs["article_id"].astype(str).map(features.article_to_row).to_numpy(
        dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    negative_items = rng.integers(0, len(features.article_ids), size=len(pairs), dtype=np.int64)
    purchased = features.purchased
    invalid = np.fromiter(
        (
            features.article_ids[item] in purchased.get(customer, set())
            for customer, item in zip(customers, negative_items, strict=True)
        ),
        dtype=bool,
    )
    while invalid.any():
        negative_items[invalid] = rng.integers(
            0, len(features.article_ids), size=int(invalid.sum()), dtype=np.int64
        )
        invalid = np.fromiter(
            (
                features.article_ids[item] in purchased.get(customer, set())
                for customer, item in zip(customers, negative_items, strict=True)
            ),
            dtype=bool,
        )
    all_customers = customers + customers
    all_items = np.concatenate([positive_items, negative_items])
    user_cats, user_dense = _customer_feature_rows(features, all_customers)
    item_cats = features.item_cats[all_items]
    item_dense = features.item_dense[all_items]
    labels = np.concatenate(
        [np.ones(len(pairs), dtype=np.float32), np.zeros(len(pairs), dtype=np.float32)]
    )
    model = TwoTower(
        user_cardinalities,
        item_cardinalities,
        user_dense.shape[1],
        item_dense.shape[1],
    )
    model, losses = fit_binary_model(
        model,
        (
            torch.as_tensor(user_cats, dtype=torch.long),
            torch.as_tensor(user_dense, dtype=torch.float32),
            torch.as_tensor(item_cats, dtype=torch.long),
            torch.as_tensor(item_dense, dtype=torch.float32),
        ),
        torch.as_tensor(labels, dtype=torch.float32),
        epochs,
        batch_size,
        learning_rate,
        seed,
    )
    return model, losses, {"positive_pairs": len(pairs), "training_rows": len(labels)}


def two_tower_candidates(
    model: TwoTower,
    features: SnapshotFeatures,
    customer_ids: list[str],
    k: int,
    batch_size: int,
) -> np.ndarray:
    """Retrieve directly from the eligible catalog; never score all users at once."""
    device = next(model.parameters()).device
    k = min(k, len(features.article_ids))
    with torch.no_grad():
        item_vectors = model.encode_items(
            torch.as_tensor(features.item_cats, dtype=torch.long, device=device),
            torch.as_tensor(features.item_dense, dtype=torch.float32, device=device),
        )
    output = np.empty((len(customer_ids), k), dtype=np.int64)
    for start in range(0, len(customer_ids), batch_size):
        stop = min(start + batch_size, len(customer_ids))
        cats, dense = _customer_feature_rows(features, customer_ids[start:stop])
        with torch.no_grad():
            user_vectors = model.encode_users(
                torch.as_tensor(cats, dtype=torch.long, device=device),
                torch.as_tensor(dense, dtype=torch.float32, device=device),
            )
            scores = user_vectors @ item_vectors.T
            output[start:stop] = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
    return output


def union_candidates(
    customer_ids: list[str],
    sources: list[CandidateSource],
    features: SnapshotFeatures,
    max_candidates: int,
    remove_history_from: set[str],
) -> CandidateUnion:
    """Union sources, remove seen items from discovery sources, and retain provenance."""
    source_names = tuple(source.name for source in sources)
    width = min(max_candidates, len(features.article_ids))
    items = np.empty((len(customer_ids), width), dtype=np.int64)
    rr = np.zeros((len(customer_ids), width, len(sources)), dtype=np.float32)
    global_order = np.argsort(-features.item_popularity, kind="stable")
    for user_row, customer in enumerate(customer_ids):
        seen = features.purchased.get(customer, set())
        merged: dict[int, np.ndarray] = {}
        for source_index, source in enumerate(sources):
            for rank, item in enumerate(source.items[user_row]):
                item = int(item)
                if item < 0:
                    continue
                if (
                    source.name in remove_history_from
                    and features.article_ids[item] in seen
                ):
                    continue
                if item not in merged:
                    merged[item] = np.zeros(len(sources), dtype=np.float32)
                merged[item][source_index] = max(
                    merged[item][source_index], 1.0 / (rank + 1)
                )
        ranked = sorted(
            merged,
            key=lambda item: (-float(merged[item].sum()), item),
        )[:width]
        ranked_set = set(ranked)
        if len(ranked) < width:
            for item in global_order:
                item = int(item)
                if item in ranked_set or (
                    remove_history_from and features.article_ids[item] in seen
                ):
                    continue
                ranked.append(item)
                ranked_set.add(item)
                if len(ranked) == width:
                    break
        ranked = ranked[:width]
        items[user_row] = ranked
        for position, item in enumerate(ranked):
            if item in merged:
                rr[user_row, position] = merged[item]
    return CandidateUnion(items=items, source_rr=rr, source_names=source_names)
