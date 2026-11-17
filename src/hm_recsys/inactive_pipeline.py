"""Four-week-inactive retrieval with metadata and behavioral user KNN."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .candidates import (
    CandidateSource,
    CandidateUnion,
    popularity_candidates,
    segment_popularity_candidates,
    union_candidates,
)
from .data import (
    ARTICLE_CAT_COLUMNS,
    FINAL_WINDOW,
    TRAIN_WINDOW,
    SnapshotFeatures,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_recent_transactions,
    sample_customers,
    slice_window,
    truth_by_customer,
)
from .hybrid_pipeline import _pair_arrays, _tree_features
from .metrics import average_precision_at_k, recall_at_k
from .pipeline import _select_labeled_examples


INACTIVE_SOURCE_NAMES = (
    "behavioral_knn",
    "metadata_knn",
    "preference_popularity",
    "segment_popularity",
    "global_popularity",
    "historical_repeat",
)
DISCOVERY_SOURCE_NAMES = set(INACTIVE_SOURCE_NAMES) - {"historical_repeat"}


@dataclass(frozen=True)
class InactiveConfig:
    profile: str
    seed: int = 42
    max_train_users: int = 6_000
    max_eval_users: int = 2_000
    active_neighbor_pool: int = 50_000
    behavioral_neighbors: int = 50
    metadata_neighbors: int = 100
    behavioral_k: int = 300
    metadata_k: int = 400
    preference_k: int = 200
    segment_k: int = 200
    popularity_k: int = 300
    union_k: int = 1_000
    repeat_k: int = 12
    history_days: int = 180
    profile_half_life_days: float = 90.0
    max_neighbor_articles: int = 20
    negatives_per_positive: int = 20
    lgbm_estimators: int = 250
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    batch_size: int = 8_192


INACTIVE_PROFILES = {
    "inactive_knn_smoke": InactiveConfig(
        profile="inactive_knn_smoke",
        max_train_users=300,
        max_eval_users=200,
        active_neighbor_pool=5_000,
        behavioral_neighbors=20,
        metadata_neighbors=30,
        behavioral_k=60,
        metadata_k=80,
        preference_k=60,
        segment_k=50,
        popularity_k=80,
        union_k=200,
        repeat_k=5,
        negatives_per_positive=5,
        lgbm_estimators=50,
        batch_size=2_048,
    ),
    "inactive_knn_experiment": InactiveConfig(profile="inactive_knn_experiment"),
}


@dataclass
class InactiveUserState:
    users: list[str]
    has_history_180d: np.ndarray
    has_any_history: np.ndarray
    days_since_last_purchase: np.ndarray
    historical_purchase_count: np.ndarray
    historical_seen_current: dict[str, set[str]]


@dataclass
class InactiveCandidateSet:
    sources: list[CandidateSource]
    discovery_union: CandidateUnion
    full_union: CandidateUnion
    user_state: InactiveUserState
    runtime: dict[str, float]


def _sample_from_practical(
    labels: pd.DataFrame,
    practical_maximum: int,
    requested_maximum: int,
    seed: int,
) -> list[str]:
    practical = sample_customers(labels, set(), practical_maximum, seed)
    if requested_maximum >= len(practical):
        return practical
    rng = np.random.default_rng(seed + 101)
    return sorted(rng.choice(practical, requested_maximum, replace=False).tolist())


def _inactive_users(context: pd.DataFrame, users: list[str]) -> list[str]:
    active = set(context["customer_id"].astype(str).unique())
    return [user for user in users if user not in active]


def _target_history_cache(
    root: Path,
    target_users: set[str],
    cache_dir: Path,
) -> pd.DataFrame:
    """Cache all pre-final-cutoff transactions for the bounded target-user set."""
    digest = hashlib.sha1("\n".join(sorted(target_users)).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"inactive_target_history_{digest}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    dtypes = {
        "customer_id": "string",
        "article_id": "string",
        "price": "float32",
        "sales_channel_id": "int8",
    }
    for chunk in pd.read_csv(
        root / "Data" / "transactions_train.csv",
        dtype=dtypes,
        chunksize=1_000_000,
    ):
        rows = chunk.loc[
            (chunk["t_dat"] < FINAL_WINDOW.label_start)
            & chunk["customer_id"].astype(str).isin(target_users)
        ].copy()
        if not rows.empty:
            parts.append(rows)
    history = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not history.empty:
        history["t_dat"] = pd.to_datetime(history["t_dat"])
    history.to_pickle(cache_path)
    return history


def _article_attribute_matrix(
    articles: pd.DataFrame,
) -> tuple[csr_matrix, dict[str, int]]:
    """Five-field article one-hot matrix shared by user and item profiles."""
    article_ids = articles["article_id"].astype(str).to_numpy()
    article_to_row = {article: row for row, article in enumerate(article_ids)}
    rows: list[int] = []
    cols: list[int] = []
    offset = 0
    for column in ARTICLE_CAT_COLUMNS:
        values = articles[column].fillna("__MISSING__").astype(str)
        categories = {value: i for i, value in enumerate(sorted(values.unique()))}
        rows.extend(range(len(articles)))
        cols.extend(offset + values.map(categories).to_numpy(dtype=np.int32))
        offset += len(categories)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(articles), offset),
    )
    return normalize(matrix, norm="l2", axis=1), article_to_row


def _user_attribute_profiles(
    events: pd.DataFrame,
    user_ids: list[str],
    cutoff: pd.Timestamp,
    article_attributes: csr_matrix,
    article_to_row: dict[str, int],
    half_life_days: float,
) -> csr_matrix:
    user_to_row = {user: row for row, user in enumerate(user_ids)}
    if events.empty:
        return csr_matrix((len(user_ids), article_attributes.shape[1]), dtype=np.float32)
    frame = events[["customer_id", "article_id", "t_dat"]].copy()
    frame["user_row"] = frame["customer_id"].astype(str).map(user_to_row)
    frame["item_row"] = frame["article_id"].astype(str).map(article_to_row)
    frame = frame.dropna(subset=["user_row", "item_row"])
    if frame.empty:
        return csr_matrix((len(user_ids), article_attributes.shape[1]), dtype=np.float32)
    age = (cutoff - pd.to_datetime(frame["t_dat"])).dt.days.clip(lower=0)
    weights = np.exp(-np.log(2.0) * age.to_numpy(dtype=np.float32) / half_life_days)
    interaction = csr_matrix(
        (
            weights,
            (
                frame["user_row"].to_numpy(dtype=np.int32),
                frame["item_row"].to_numpy(dtype=np.int32),
            ),
        ),
        shape=(len(user_ids), article_attributes.shape[0]),
    )
    return normalize(interaction @ article_attributes, norm="l2", axis=1)


def _select_active_neighbors(
    context: pd.DataFrame, maximum: int
) -> list[str]:
    stats = (
        context.groupby("customer_id", observed=True)
        .agg(last_date=("t_dat", "max"), activity=("article_id", "size"))
        .reset_index()
        .sort_values(
            ["last_date", "activity", "customer_id"],
            ascending=[False, False, True],
        )
    )
    return stats["customer_id"].astype(str).head(maximum).tolist()


def _recent_articles_by_user(
    context: pd.DataFrame,
    active_users: list[str],
    features: SnapshotFeatures,
    maximum: int,
) -> list[list[int]]:
    active_set = set(active_users)
    frame = context.loc[
        context["customer_id"].astype(str).isin(active_set),
        ["customer_id", "article_id", "t_dat"],
    ].sort_values("t_dat", ascending=False)
    grouped = frame.groupby("customer_id", observed=True)["article_id"].agg(
        lambda values: list(dict.fromkeys(values.astype(str)))[:maximum]
    )
    mapping = grouped.to_dict()
    return [
        [features.article_to_row[item] for item in mapping.get(user, []) if item in features.article_to_row]
        for user in active_users
    ]


def _fill_ranked(
    ranked: list[int],
    fallback: np.ndarray,
    k: int,
) -> list[int]:
    selected = list(dict.fromkeys(ranked))
    seen = set(selected)
    for item in fallback:
        value = int(item)
        if value not in seen:
            selected.append(value)
            seen.add(value)
            if len(selected) == k:
                break
    return selected[:k]


def _neighbor_purchase_candidates(
    query_profiles,
    reference_profiles,
    reference_users: list[str],
    recent_items: list[list[int]],
    features: SnapshotFeatures,
    neighbors: int,
    k: int,
    postal_by_user: dict[str, str] | None = None,
    query_users: list[str] | None = None,
    reference_postal: list[str] | None = None,
) -> np.ndarray:
    """KNN retrieval: similarity-weight recent purchases of active neighbors."""
    k = min(k, len(features.article_ids))
    output = np.full((query_profiles.shape[0], k), -1, dtype=np.int64)
    if reference_profiles.shape[0] == 0:
        return output
    model = NearestNeighbors(
        n_neighbors=min(neighbors, reference_profiles.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=8,
    ).fit(reference_profiles)
    distances, indices = model.kneighbors(query_profiles)
    for row in range(query_profiles.shape[0]):
        if hasattr(query_profiles, "getrow"):
            available = query_profiles.getrow(row).nnz > 0
        else:
            available = bool(np.linalg.norm(query_profiles[row]) > 0)
        if not available:
            continue
        scores: Counter[int] = Counter()
        for distance, neighbor_row in zip(distances[row], indices[row], strict=True):
            similarity = max(0.0, 1.0 - float(distance))
            if (
                postal_by_user is not None
                and query_users is not None
                and reference_postal is not None
                and postal_by_user.get(query_users[row], "") == reference_postal[int(neighbor_row)]
            ):
                similarity += 0.25
            for rank, item in enumerate(recent_items[int(neighbor_row)]):
                scores[item] += similarity / (rank + 1)
        ranked = [item for item, _ in scores.most_common(k)]
        output[row, : len(ranked)] = ranked
    return output


def _metadata_matrix(
    customers: pd.DataFrame, user_ids: list[str]
) -> np.ndarray:
    frame = customers.set_index(customers["customer_id"].astype(str)).reindex(user_ids)
    age = frame["age"].to_numpy(dtype=np.float32)
    age_missing = np.isnan(age).astype(np.float32)
    median = np.nanmedian(age)
    age = np.nan_to_num(age, nan=median) / 100.0
    columns = [age, age_missing]
    for column in ("FN", "Active"):
        columns.append(frame[column].notna().to_numpy(dtype=np.float32))
    for column in ("club_member_status", "fashion_news_frequency"):
        values = frame[column].fillna("__MISSING__").astype(str)
        for category in sorted(customers[column].fillna("__MISSING__").astype(str).unique()):
            columns.append((values == category).to_numpy(dtype=np.float32))
    return normalize(np.column_stack(columns).astype(np.float32), norm="l2", axis=1)


def _user_state(
    users: list[str],
    target_history: pd.DataFrame,
    context_start: str,
    prediction_cutoff: str,
    history_days: int,
    features: SnapshotFeatures,
) -> tuple[InactiveUserState, pd.DataFrame]:
    history_boundary = pd.Timestamp(context_start)
    cutoff = pd.Timestamp(prediction_cutoff)
    frame = target_history.loc[
        (target_history["customer_id"].astype(str).isin(users))
        & (target_history["t_dat"] < history_boundary)
    ].copy()
    profile_start = history_boundary - pd.Timedelta(days=history_days)
    profile = frame.loc[frame["t_dat"] >= profile_start].copy()
    grouped_all = frame.groupby("customer_id", observed=True)
    last = grouped_all["t_dat"].max().to_dict()
    counts = grouped_all.size().to_dict()
    profile_users = set(profile["customer_id"].astype(str).unique())
    current_catalog = set(features.article_ids.astype(str))
    seen = (
        frame.loc[frame["article_id"].astype(str).isin(current_catalog)]
        .groupby("customer_id", observed=True)["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    days = np.asarray(
        [(cutoff - last[user]).days if user in last else np.nan for user in users],
        dtype=np.float32,
    )
    return (
        InactiveUserState(
            users=users,
            has_history_180d=np.asarray([user in profile_users for user in users], dtype=np.float32),
            has_any_history=np.asarray([user in last for user in users], dtype=np.float32),
            days_since_last_purchase=days,
            historical_purchase_count=np.asarray([counts.get(user, 0) for user in users], dtype=np.float32),
            historical_seen_current=seen,
        ),
        profile,
    )


def _preference_candidates(
    profiles: csr_matrix,
    current_item_attributes: csr_matrix,
    features: SnapshotFeatures,
    k: int,
    batch_size: int = 64,
) -> np.ndarray:
    k = min(k, len(features.article_ids))
    output = np.full((profiles.shape[0], k), -1, dtype=np.int64)
    popularity = np.log1p(features.item_popularity)
    popularity = popularity / max(float(popularity.max()), 1.0)
    for start in range(0, profiles.shape[0], batch_size):
        stop = min(start + batch_size, profiles.shape[0])
        scores = (profiles[start:stop] @ current_item_attributes.T).toarray()
        scores *= 1.0 + popularity[None, :]
        for local in range(stop - start):
            if profiles[start + local].nnz == 0:
                continue
            top = np.argpartition(-scores[local], kth=k - 1)[:k]
            top = top[np.argsort(-scores[local, top], kind="stable")]
            output[start + local] = top
    return output


def _historical_repeat_candidates(
    users: list[str],
    history: pd.DataFrame,
    features: SnapshotFeatures,
    cutoff: str,
    k: int,
) -> np.ndarray:
    output = np.full((len(users), k), -1, dtype=np.int64)
    if history.empty or k == 0:
        return output
    frame = history.loc[history["article_id"].astype(str).isin(features.article_to_row)].copy()
    if frame.empty:
        return output
    stats = frame.groupby(["customer_id", "article_id"], observed=True).agg(
        count=("article_id", "size"), last_date=("t_dat", "max")
    ).reset_index()
    age = (pd.Timestamp(cutoff) - stats["last_date"]).dt.days.clip(lower=0)
    stats["score"] = stats["count"] * np.exp(-age / 90.0)
    ranked = (
        stats.sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True])
        .groupby("customer_id", observed=True)
        .head(k)
        .groupby("customer_id", observed=True)["article_id"]
        .agg(list)
        .to_dict()
    )
    for row, user in enumerate(users):
        items = [features.article_to_row[str(item)] for item in ranked.get(user, [])]
        output[row, : len(items)] = items
    return output


def _build_inactive_candidates(
    context: pd.DataFrame,
    context_start: str,
    cutoff: str,
    features: SnapshotFeatures,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    target_users: list[str],
    target_history: pd.DataFrame,
    article_attributes: csr_matrix,
    article_to_global_row: dict[str, int],
    config: InactiveConfig,
) -> InactiveCandidateSet:
    runtime: dict[str, float] = {}
    mark = time.perf_counter()
    state, profile_history = _user_state(
        target_users,
        target_history,
        context_start,
        cutoff,
        config.history_days,
        features,
    )
    # Expose older seen items to discovery filtering and repeat features.
    features.purchased.update(state.historical_seen_current)
    active_users = _select_active_neighbors(context, config.active_neighbor_pool)
    recent_items = _recent_articles_by_user(
        context, active_users, features, config.max_neighbor_articles
    )
    runtime["state_and_active_pool"] = time.perf_counter() - mark

    mark = time.perf_counter()
    active_behavior = _user_attribute_profiles(
        context.loc[context["customer_id"].astype(str).isin(set(active_users))],
        active_users,
        pd.Timestamp(cutoff),
        article_attributes,
        article_to_global_row,
        config.profile_half_life_days,
    )
    target_behavior = _user_attribute_profiles(
        profile_history,
        target_users,
        pd.Timestamp(context_start),
        article_attributes,
        article_to_global_row,
        config.profile_half_life_days,
    )
    behavior_rows = _neighbor_purchase_candidates(
        target_behavior,
        active_behavior,
        active_users,
        recent_items,
        features,
        config.behavioral_neighbors,
        config.behavioral_k,
    )
    runtime["behavioral_knn"] = time.perf_counter() - mark

    mark = time.perf_counter()
    active_metadata = _metadata_matrix(customers, active_users)
    target_metadata = _metadata_matrix(customers, target_users)
    postal = customers.set_index(customers["customer_id"].astype(str))["postal_code"].astype(str).to_dict()
    metadata_rows = _neighbor_purchase_candidates(
        target_metadata,
        active_metadata,
        active_users,
        recent_items,
        features,
        config.metadata_neighbors,
        config.metadata_k,
        postal_by_user=postal,
        query_users=target_users,
        reference_postal=[postal.get(user, "") for user in active_users],
    )
    runtime["metadata_knn"] = time.perf_counter() - mark

    mark = time.perf_counter()
    current_global_rows = np.asarray(
        [article_to_global_row[item] for item in features.article_ids], dtype=np.int64
    )
    preference_rows = _preference_candidates(
        target_behavior,
        article_attributes[current_global_rows],
        features,
        config.preference_k,
    )
    segment_rows = segment_popularity_candidates(
        context, customers, features, target_users, config.segment_k
    )
    popularity_rows = popularity_candidates(features, len(target_users), config.popularity_k)
    repeat_rows = _historical_repeat_candidates(
        target_users, profile_history, features, cutoff, config.repeat_k
    )
    runtime["preference_and_fallbacks"] = time.perf_counter() - mark

    discovery = [
        CandidateSource("behavioral_knn", behavior_rows),
        CandidateSource("metadata_knn", metadata_rows),
        CandidateSource("preference_popularity", preference_rows),
        CandidateSource("segment_popularity", segment_rows),
        CandidateSource("global_popularity", popularity_rows),
    ]
    repeat = CandidateSource("historical_repeat", repeat_rows)
    mark = time.perf_counter()
    discovery_union = union_candidates(
        target_users,
        discovery,
        features,
        config.union_k,
        remove_history_from=DISCOVERY_SOURCE_NAMES,
    )
    full_union = union_candidates(
        target_users,
        [*discovery, repeat],
        features,
        config.union_k + config.repeat_k,
        remove_history_from=DISCOVERY_SOURCE_NAMES,
    )
    runtime["candidate_unions"] = time.perf_counter() - mark
    return InactiveCandidateSet(
        sources=[*discovery, repeat],
        discovery_union=discovery_union,
        full_union=full_union,
        user_state=state,
        runtime=runtime,
    )


def _inactive_extra_features(
    state: InactiveUserState, user_rows: np.ndarray
) -> np.ndarray:
    days = state.days_since_last_purchase[user_rows]
    days_missing = np.isnan(days).astype(np.float32)
    days_filled = np.nan_to_num(days, nan=9999.0)
    return np.column_stack(
        [
            state.has_history_180d[user_rows],
            state.has_any_history[user_rows],
            np.clip(days_filled, 0, 730) / 730.0,
            np.log1p(days_filled) / np.log1p(9999.0),
            days_missing,
            np.log1p(state.historical_purchase_count[user_rows]),
        ]
    ).astype(np.float32)


def _inactive_pair_features(
    features: SnapshotFeatures,
    users: list[str],
    candidates: CandidateUnion,
    state: InactiveUserState,
    user_rows: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cats, dense = _pair_arrays(features, users, candidates, user_rows, positions)
    return cats, np.column_stack([dense, _inactive_extra_features(state, user_rows)])


def _score_lightgbm(
    model: Any,
    features: SnapshotFeatures,
    users: list[str],
    candidates: CandidateUnion,
    state: InactiveUserState,
    batch_size: int,
) -> np.ndarray:
    width = candidates.items.shape[1]
    output = np.empty((len(users), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(users), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(users))
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        cats, dense = _inactive_pair_features(
            features,
            users,
            candidates,
            state,
            flat_users[start:stop],
            flat_positions[start:stop],
        )
        output.reshape(-1)[start:stop] = model.predict(_tree_features(cats, dense))
    return output


def _rankings(
    users: list[str], rows: np.ndarray, article_ids: np.ndarray
) -> dict[str, list[str]]:
    return {
        user: [article_ids[item] for item in rows[row] if item >= 0]
        for row, user in enumerate(users)
    }


def _metrics_for_subgroups(
    rankings: dict[str, list[str]],
    truth: dict[str, set[str]],
    state: InactiveUserState,
    candidate_k: int,
) -> dict[str, dict[str, float | int]]:
    groups = {
        "all_inactive": state.users,
        "history_within_180d": [
            user for i, user in enumerate(state.users) if state.has_history_180d[i] == 1
        ],
        "stale_history_over_180d": [
            user
            for i, user in enumerate(state.users)
            if state.has_history_180d[i] == 0 and state.has_any_history[i] == 1
        ],
        "no_prior_transaction": [
            user for i, user in enumerate(state.users) if state.has_any_history[i] == 0
        ],
    }
    output = {}
    for name, users in groups.items():
        candidate_recall = [recall_at_k(rankings[user], truth[user], candidate_k) for user in users]
        maps = [average_precision_at_k(rankings[user], truth[user], 12) for user in users]
        recalls = [recall_at_k(rankings[user], truth[user], 12) for user in users]
        output[name] = {
            "customers": len(users),
            "candidate_recall": float(np.nanmean(candidate_recall)) if users else np.nan,
            "map_at_12": float(np.nanmean(maps)) if users else np.nan,
            "recall_at_12": float(np.nanmean(recalls)) if users else np.nan,
        }
    return output


def _source_metrics(
    source: CandidateSource,
    users: list[str],
    features: SnapshotFeatures,
    truth: dict[str, set[str]],
    state: InactiveUserState,
) -> dict:
    rankings = _rankings(users, source.items, features.article_ids)
    return {
        "candidate_k": source.items.shape[1],
        "cohorts": _metrics_for_subgroups(
            rankings, truth, state, source.items.shape[1]
        ),
    }


def _save_union(
    path: Path,
    users: list[str],
    features: SnapshotFeatures,
    union: CandidateUnion,
) -> None:
    np.savez_compressed(
        path,
        customer_ids=np.asarray(users),
        article_ids=features.article_ids,
        candidate_item_rows=union.items,
        source_rr=union.source_rr,
        source_names=np.asarray(union.source_names),
    )


def run_inactive_experiment(
    root: Path, config: InactiveConfig, save_candidates: bool = True
) -> Path:
    started = time.perf_counter()
    artifacts = root / "artifacts"
    run_dir = artifacts / config.profile
    run_dir.mkdir(parents=True, exist_ok=True)

    recent = load_recent_transactions(root / "Data", artifacts / "cache")
    customers, articles = load_metadata(root / "Data")
    encoder = fit_feature_encoder(customers, articles)
    train_context, train_labels = slice_window(recent, TRAIN_WINDOW)
    final_context, final_labels = slice_window(recent, FINAL_WINDOW)

    practical_train = sample_customers(train_labels, set(), 6_000, config.seed)
    practical_eval = sample_customers(final_labels, set(), 2_000, config.seed)
    practical_train_inactive = _inactive_users(train_context, practical_train)
    practical_eval_inactive = _inactive_users(final_context, practical_eval)
    target_history = _target_history_cache(
        root,
        set(practical_train_inactive) | set(practical_eval_inactive),
        artifacts / "cache",
    )

    train_all = _sample_from_practical(
        train_labels, 6_000, config.max_train_users, config.seed
    )
    eval_all = _sample_from_practical(
        final_labels, 2_000, config.max_eval_users, config.seed
    )
    train_users = _inactive_users(train_context, train_all)
    eval_users = _inactive_users(final_context, eval_all)
    train_truth_all = truth_by_customer(train_labels)
    final_truth_all = truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in train_users}
    final_truth = {user: final_truth_all[user] for user in eval_users}

    train_features = build_snapshot_features(
        train_context, customers, articles, encoder, TRAIN_WINDOW.label_start
    )
    final_features = build_snapshot_features(
        final_context, customers, articles, encoder, FINAL_WINDOW.label_start
    )
    article_attributes, article_to_global_row = _article_attribute_matrix(articles)

    mark = time.perf_counter()
    train_candidates = _build_inactive_candidates(
        train_context,
        TRAIN_WINDOW.context_start,
        TRAIN_WINDOW.label_start,
        train_features,
        customers,
        articles,
        train_users,
        target_history,
        article_attributes,
        article_to_global_row,
        config,
    )
    train_retrieval_seconds = time.perf_counter() - mark

    user_rows, positions, example_stats = _select_labeled_examples(
        train_users,
        train_candidates.full_union.items,
        train_features.article_ids,
        train_truth,
        config.negatives_per_positive,
        config.seed + 10,
    )
    cats, dense = _inactive_pair_features(
        train_features,
        train_users,
        train_candidates.full_union,
        train_candidates.user_state,
        user_rows,
        positions,
    )
    labels = np.fromiter(
        (
            train_features.article_ids[
                train_candidates.full_union.items[user_row, position]
            ]
            in train_truth[train_users[user_row]]
            for user_row, position in zip(user_rows, positions, strict=True)
        ),
        dtype=np.int32,
    )
    groups = np.bincount(user_rows, minlength=len(train_users))
    groups = groups[groups > 0]
    mark = time.perf_counter()
    # Keep the native LightGBM import out of module import time so data and
    # candidate unit tests do not require OpenMP to be loaded.
    from lightgbm import LGBMRanker

    model = LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=config.lgbm_estimators,
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        lambdarank_truncation_level=12,
        random_state=config.seed + 12,
        n_jobs=8,
        verbosity=-1,
    )
    model.fit(
        _tree_features(cats, dense),
        labels,
        group=groups.tolist(),
        categorical_feature=list(range(cats.shape[1])),
    )
    training_seconds = time.perf_counter() - mark

    mark = time.perf_counter()
    final_candidates = _build_inactive_candidates(
        final_context,
        FINAL_WINDOW.context_start,
        FINAL_WINDOW.label_start,
        final_features,
        customers,
        articles,
        eval_users,
        target_history,
        article_attributes,
        article_to_global_row,
        config,
    )
    final_retrieval_seconds = time.perf_counter() - mark
    scores = _score_lightgbm(
        model,
        final_features,
        eval_users,
        final_candidates.full_union,
        final_candidates.user_state,
        config.batch_size,
    )
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_items = np.take_along_axis(final_candidates.full_union.items, order, axis=1)

    metrics = {
        source.name: _source_metrics(
            source,
            eval_users,
            final_features,
            final_truth,
            final_candidates.user_state,
        )
        for source in final_candidates.sources
    }
    for name, union in (
        ("inactive_discovery_union", final_candidates.discovery_union),
        ("inactive_union_plus_repeats", final_candidates.full_union),
    ):
        metrics[name] = {
            "candidate_k": union.items.shape[1],
            "cohorts": _metrics_for_subgroups(
                _rankings(eval_users, union.items, final_features.article_ids),
                final_truth,
                final_candidates.user_state,
                union.items.shape[1],
            ),
        }
    metrics["inactive_lightgbm"] = {
        "candidate_k": ranked_items.shape[1],
        "cohorts": _metrics_for_subgroups(
            _rankings(eval_users, ranked_items, final_features.article_ids),
            final_truth,
            final_candidates.user_state,
            ranked_items.shape[1],
        ),
    }

    if save_candidates:
        _save_union(
            run_dir / "train_candidate_pool.npz",
            train_users,
            train_features,
            train_candidates.full_union,
        )
        _save_union(
            run_dir / "final_candidate_pool.npz",
            eval_users,
            final_features,
            final_candidates.full_union,
        )
        np.savez_compressed(
            run_dir / "final_user_state.npz",
            customer_ids=np.asarray(eval_users),
            has_history_180d=final_candidates.user_state.has_history_180d,
            has_any_history=final_candidates.user_state.has_any_history,
            days_since_last_purchase=final_candidates.user_state.days_since_last_purchase,
            historical_purchase_count=final_candidates.user_state.historical_purchase_count,
        )

    predictions = pd.DataFrame(
        {
            "customer_id": eval_users,
            "prediction": [
                " ".join(final_features.article_ids[rows[:12]]) for rows in ranked_items
            ],
        }
    )
    predictions.to_csv(run_dir / "inactive_lightgbm_top12.csv", index=False)

    summary = {
        "config": asdict(config),
        "architecture": {
            "cohort": "no purchases in the previous four weeks",
            "eligible_catalog": "articles sold in the previous four weeks",
            "profile_history": "up to 180 days before the four-week context",
            "sources": list(INACTIVE_SOURCE_NAMES),
            "ranker": "LightGBM LambdaRank trained only on earlier-snapshot inactive customers",
        },
        "windows": {
            "ranker_training": asdict(TRAIN_WINDOW),
            "final_evaluation": asdict(FINAL_WINDOW),
        },
        "dataset": {
            "train_label_customers_sampled_before_inactive_filter": len(train_all),
            "train_inactive_customers": len(train_users),
            "evaluation_customers_sampled_before_inactive_filter": len(eval_all),
            "evaluation_inactive_customers": len(eval_users),
            "final_catalog_items": len(final_features.article_ids),
        },
        "training": {
            "examples": {**example_stats, "rows": len(labels), "positives": int(labels.sum())},
            "groups": len(groups),
            "feature_importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ).tolist(),
        },
        "metrics": metrics,
        "runtime_seconds": {
            "train_retrieval": train_retrieval_seconds,
            "lightgbm_training": training_seconds,
            "final_retrieval": final_retrieval_seconds,
            "train_source_detail": train_candidates.runtime,
            "final_source_detail": final_candidates.runtime,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "saved_candidate_pools": (
            ["train_candidate_pool.npz", "final_candidate_pool.npz", "final_user_state.npz"]
            if save_candidates
            else []
        ),
    }
    result_path = run_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return result_path
