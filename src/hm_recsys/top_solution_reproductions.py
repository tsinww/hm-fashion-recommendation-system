"""Practical, leakage-safe reproductions inspired by the published top solutions.

The winning teams did not release complete runnable code. These experiments
implement the architectures and feature families described in their write-ups,
while retaining this project's fixed customer sample and final validation week.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import CandidateUnion
from .data import (
    FINAL_WINDOW,
    Window,
    build_snapshot_features,
    fit_feature_encoder,
    load_metadata,
    load_transactions_since,
    sample_customers,
    slice_window,
    truth_by_customer,
    window_with_history,
)
from .hybrid_pipeline import _pair_arrays
from .metrics import evaluate_rankings
from .pipeline import _select_labeled_examples


FIRST_TRAIN = window_with_history("2020-09-09", "2020-09-16", 42)
FIRST_FINAL = window_with_history("2020-09-16", "2020-09-23", 42)
SECOND_STAGE1 = window_with_history("2020-09-02", "2020-09-09", 56)
SECOND_STAGE2 = window_with_history("2020-09-09", "2020-09-16", 56)
SECOND_FINAL = window_with_history("2020-09-16", "2020-09-23", 56)


@dataclass(frozen=True)
class TopSolutionConfig:
    seed: int = 42
    train_users: int = 6_000
    eval_users: int = 2_000
    first_candidate_k: int = 100
    second_initial_k: int = 600
    second_stage1_k: int = 130
    second_final_k: int = 330
    negatives_per_positive: int = 20
    estimators: int = 350
    learning_rate: float = 0.05
    num_leaves: int = 31
    score_batch_size: int = 100_000


@dataclass
class TemporalStats:
    item_7: np.ndarray
    item_30: np.ndarray
    item_all: np.ndarray
    pair_count: dict[tuple[str, str], int]
    pair_last: dict[tuple[str, str], pd.Timestamp]
    user_section_count: dict[tuple[str, str], int]
    article_section: dict[str, str]
    customer_age: dict[str, float]
    item_mean_age: dict[str, float]
    cutoff: pd.Timestamp


def _load_saved_union(path: Path, k: int) -> tuple[list[str], np.ndarray, CandidateUnion]:
    saved = np.load(path, allow_pickle=True)
    return (
        saved["customer_ids"].astype(str).tolist(),
        saved["article_ids"].astype(str),
        CandidateUnion(
            items=saved["candidate_item_rows"][:, :k],
            source_rr=saved["source_rr"][:, :k],
            source_names=tuple(saved["source_names"].astype(str).tolist()),
        ),
    )


def _temporal_stats(context, features, articles, customers, cutoff: str) -> TemporalStats:
    cutoff_ts = pd.Timestamp(cutoff)
    article_ids = features.article_ids.astype(str)
    item_to_row = features.article_to_row
    dates = pd.to_datetime(context["t_dat"])

    def item_counts(days: int | None) -> np.ndarray:
        frame = context if days is None else context.loc[dates >= cutoff_ts - pd.Timedelta(days=days)]
        counts = frame.groupby("article_id", observed=True).size()
        return np.asarray([counts.get(item, 0) for item in article_ids], dtype=np.float32)

    article_section = articles.set_index(articles["article_id"].astype(str))["section_name"].fillna("__MISSING__").astype(str).to_dict()
    enriched = context[["customer_id", "article_id", "t_dat"]].copy()
    enriched["customer_id"] = enriched["customer_id"].astype(str)
    enriched["article_id"] = enriched["article_id"].astype(str)
    enriched["section_name"] = enriched["article_id"].map(article_section).fillna("__MISSING__")
    pair = enriched.groupby(["customer_id", "article_id"], observed=True).agg(count=("article_id", "size"), last=("t_dat", "max"))
    section = enriched.groupby(["customer_id", "section_name"], observed=True).size()
    customer_age = customers.set_index(customers["customer_id"].astype(str))["age"].to_dict()
    buyer = enriched[["customer_id", "article_id"]].merge(
        customers[["customer_id", "age"]].assign(customer_id=lambda frame: frame.customer_id.astype(str)),
        on="customer_id",
        how="left",
    )
    item_mean_age = buyer.groupby("article_id", observed=True)["age"].mean().to_dict()
    return TemporalStats(
        item_7=item_counts(7),
        item_30=item_counts(30),
        item_all=item_counts(None),
        pair_count={key: int(value) for key, value in pair["count"].to_dict().items()},
        pair_last=pair["last"].to_dict(),
        user_section_count={key: int(value) for key, value in section.to_dict().items()},
        article_section=article_section,
        customer_age={str(key): float(value) for key, value in customer_age.items() if pd.notna(value)},
        item_mean_age={str(key): float(value) for key, value in item_mean_age.items() if pd.notna(value)},
        cutoff=cutoff_ts,
    )


def _model_features(features, users, union, user_rows, positions, stats: TemporalStats) -> np.ndarray:
    cats, dense = _pair_arrays(features, users, union, user_rows, positions)
    item_rows = union.items[user_rows, positions]
    item_ids = features.article_ids[item_rows].astype(str)
    customer_ids = [users[row] for row in user_rows]
    extra = np.zeros((len(user_rows), 8), dtype=np.float32)
    extra[:, 0] = np.log1p(stats.item_7[item_rows])
    extra[:, 1] = np.log1p(stats.item_30[item_rows])
    extra[:, 2] = np.log1p(stats.item_all[item_rows])
    for row, (customer, article) in enumerate(zip(customer_ids, item_ids, strict=True)):
        key = (customer, article)
        extra[row, 3] = np.log1p(stats.pair_count.get(key, 0))
        last = stats.pair_last.get(key)
        extra[row, 4] = min((stats.cutoff - last).days, 365) / 365 if last is not None else 1.0
        section = stats.article_section.get(article, "__MISSING__")
        extra[row, 5] = np.log1p(stats.user_section_count.get((customer, section), 0))
        age = stats.customer_age.get(customer)
        mean_age = stats.item_mean_age.get(article)
        extra[row, 6] = abs(age - mean_age) / 100 if age is not None and mean_age is not None else 1.0
        extra[row, 7] = float(key in stats.pair_count)
    return np.column_stack([cats, dense, extra]).astype(np.float32)


def _labels(users, union, features, truth, user_rows, positions) -> np.ndarray:
    return np.fromiter(
        (
            features.article_ids[union.items[user_row, position]] in truth[users[user_row]]
            for user_row, position in zip(user_rows, positions, strict=True)
        ),
        dtype=np.int32,
    )


def _fit_ranker(x, labels, user_rows, config: TopSolutionConfig, seed: int, classifier: bool):
    if classifier:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=config.estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            min_child_samples=50,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=8,
            verbosity=-1,
        )
        model.fit(x, labels)
        return model
    from lightgbm import LGBMRanker

    groups = np.bincount(user_rows)
    groups = groups[groups > 0]
    model = LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=config.estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        lambdarank_truncation_level=12,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )
    model.fit(x, labels, group=groups.tolist())
    return model


def _score(model, classifier, features, users, union, stats, batch_size) -> np.ndarray:
    width = union.items.shape[1]
    output = np.empty((len(users), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(users), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(users))
    for start in range(0, len(flat_users), batch_size):
        stop = min(start + batch_size, len(flat_users))
        x = _model_features(features, users, union, flat_users[start:stop], flat_positions[start:stop], stats)
        prediction = model.predict_proba(x)[:, 1] if classifier else model.predict(x)
        output.reshape(-1)[start:stop] = prediction
    return output


def _ranked_rows(scores, items) -> np.ndarray:
    return np.take_along_axis(items, np.argsort(-scores, axis=1, kind="stable"), axis=1)


def _metric_block(users, rows, features, truth, active, k) -> dict:
    rankings = {user: features.article_ids[rows[i]].tolist() for i, user in enumerate(users)}
    return evaluate_rankings(rankings, truth, active, k)


def _initial_candidates(context, users, features, articles, cutoff: str, k: int) -> CandidateUnion:
    cutoff_ts = pd.Timestamp(cutoff)
    frame = context[["customer_id", "article_id", "t_dat"]].copy()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["article_id"] = frame["article_id"].astype(str)
    frame["days"] = (cutoff_ts - pd.to_datetime(frame["t_dat"])).dt.days
    frame["weight"] = np.exp(-frame["days"].to_numpy(dtype=np.float32) / 14.0)
    popularity = frame.groupby("article_id", observed=True)["weight"].sum().sort_values(ascending=False)
    global_rows = [features.article_to_row[item] for item in popularity.index if item in features.article_to_row]
    article_section = articles.set_index(articles["article_id"].astype(str))["section_name"].fillna("__MISSING__").astype(str).to_dict()
    frame["section"] = frame["article_id"].map(article_section).fillna("__MISSING__")
    section_pop = frame.groupby(["section", "article_id"], observed=True)["weight"].sum().sort_values(ascending=False)
    section_ranked: dict[str, list[int]] = {}
    for (section, article), _ in section_pop.items():
        if article in features.article_to_row:
            section_ranked.setdefault(str(section), []).append(features.article_to_row[article])
    user_sections = (
        frame.groupby(["customer_id", "section"], observed=True).size().sort_values(ascending=False).groupby(level=0).head(3)
    )
    preferred: dict[str, list[str]] = {}
    for customer, section in user_sections.index:
        preferred.setdefault(str(customer), []).append(str(section))
    items = np.empty((len(users), k), dtype=np.int64)
    rr = np.zeros((len(users), k, 2), dtype=np.float32)
    for row, user in enumerate(users):
        selected: list[int] = []
        seen: set[int] = set()
        section_candidates: list[int] = []
        for section in preferred.get(user, []):
            section_candidates.extend(section_ranked.get(section, [])[:150])
        # Half global trends, half preference-conditioned popularity, then fill.
        for source_index, candidates in ((0, global_rows[: k // 2]), (1, section_candidates)):
            for source_rank, item in enumerate(candidates):
                if item in seen:
                    continue
                seen.add(item)
                selected.append(item)
                if len(selected) <= k:
                    rr[row, len(selected) - 1, source_index] = 1.0 / (source_rank + 1)
                if len(selected) == k:
                    break
            if len(selected) == k:
                break
        if len(selected) < k:
            for source_rank, item in enumerate(global_rows):
                if item in seen:
                    continue
                seen.add(item)
                selected.append(item)
                rr[row, len(selected) - 1, 0] = 1.0 / (source_rank + 1)
                if len(selected) == k:
                    break
        items[row] = selected
    return CandidateUnion(items, rr, ("recent_popularity", "section_popularity"))


def _stage2_union(context, users, features, stage1_items, width: int) -> CandidateUnion:
    frame = context[["customer_id", "article_id", "t_dat"]].copy()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["article_id"] = frame["article_id"].astype(str)
    history = (
        frame.groupby(["customer_id", "article_id"], observed=True)
        .agg(count=("article_id", "size"), last=("t_dat", "max"))
        .reset_index()
        .sort_values(["customer_id", "count", "last"], ascending=[True, False, False])
        .groupby("customer_id", observed=True)["article_id"]
        .agg(list)
        .to_dict()
    )
    global_order = np.argsort(-features.item_popularity, kind="stable")
    items = np.empty((len(users), width), dtype=np.int64)
    rr = np.zeros((len(users), width, 3), dtype=np.float32)
    for row, user in enumerate(users):
        selected: list[int] = []
        seen: set[int] = set()
        for rank, item in enumerate(stage1_items[row]):
            if item not in seen:
                seen.add(int(item)); selected.append(int(item)); rr[row, len(selected)-1, 0] = 1.0/(rank+1)
        for rank, article in enumerate(history.get(user, [])):
            item = features.article_to_row.get(str(article))
            if item is None or item in seen:
                continue
            seen.add(item); selected.append(item); rr[row, len(selected)-1, 1] = 1.0/(rank+1)
            if len(selected) == width:
                break
        if len(selected) < width:
            for rank, item in enumerate(global_order):
                item = int(item)
                if item in seen:
                    continue
                seen.add(item); selected.append(item); rr[row, len(selected)-1, 2] = 1.0/(rank+1)
                if len(selected) == width:
                    break
        items[row] = selected
    return CandidateUnion(items, rr, ("stage1_rank", "purchase_history", "popularity_fill"))


def _train_on_union(features, users, union, truth, stats, config, seed, classifier):
    user_rows, positions, sampling = _select_labeled_examples(
        users, union.items, features.article_ids, truth, config.negatives_per_positive, seed
    )
    x = _model_features(features, users, union, user_rows, positions, stats)
    labels = _labels(users, union, features, truth, user_rows, positions)
    return _fit_ranker(x, labels, user_rows, config, seed + 1, classifier), sampling, len(labels), int(labels.sum())


def run_first_place_inspired(root: Path, config: TopSolutionConfig) -> Path:
    started = time.perf_counter()
    source = root / "artifacts" / "hybrid_six_week_experiment"
    train_users, train_articles, train_union = _load_saved_union(source / "train_candidate_pool.npz", config.first_candidate_k)
    final_users, final_articles, final_union = _load_saved_union(source / "final_candidate_pool.npz", config.first_candidate_k)
    transactions = load_transactions_since(root / "Data", root / "artifacts" / "cache", FIRST_TRAIN.context_start)
    train_context, train_labels = slice_window(transactions, FIRST_TRAIN)
    final_context, final_labels = slice_window(transactions, FIRST_FINAL)
    customers, articles = load_metadata(root / "Data")
    encoder = fit_feature_encoder(customers, articles)
    train_features = build_snapshot_features(train_context, customers, articles, encoder, FIRST_TRAIN.label_start)
    final_features = build_snapshot_features(final_context, customers, articles, encoder, FIRST_FINAL.label_start)
    if not np.array_equal(train_articles, train_features.article_ids.astype(str)) or not np.array_equal(final_articles, final_features.article_ids.astype(str)):
        raise ValueError("Saved six-week pools do not match reconstructed snapshots")
    train_truth_all, final_truth_all = truth_by_customer(train_labels), truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in train_users}
    final_truth = {user: final_truth_all[user] for user in final_users}
    train_stats = _temporal_stats(train_context, train_features, articles, customers, FIRST_TRAIN.label_start)
    final_stats = _temporal_stats(final_context, final_features, articles, customers, FIRST_FINAL.label_start)
    model, sampling, examples, positives = _train_on_union(
        train_features, train_users, train_union, train_truth, train_stats, config, config.seed + 10, True
    )
    scores = _score(model, True, final_features, final_users, final_union, final_stats, config.score_batch_size)
    ranked = _ranked_rows(scores, final_union.items)
    active = set(final_context.customer_id.astype(str).unique())
    metrics = _metric_block(final_users, ranked, final_features, final_truth, active, config.first_candidate_k)
    output = root / "artifacts" / "first_place_inspired"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"customer_id": final_users, "prediction": [" ".join(final_features.article_ids[row[:12]]) for row in ranked]}).to_csv(output / "top12.csv", index=False)
    result = {
        "scope": "single-model reproduction inspired by the published first-place design; not the private 12-model ensemble",
        "published_design": "six-week training, Top-100 multi-strategy retrieval, interaction features, negative downsampling, LightGBM classifier",
        "config": asdict(config), "windows": {"train": asdict(FIRST_TRAIN), "final": asdict(FIRST_FINAL)},
        "training": {**sampling, "examples": examples, "positives": positives},
        "metrics": metrics, "runtime_seconds": time.perf_counter() - started,
    }
    path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); return path


def _catboost_frame(values: np.ndarray, categorical_columns: int) -> pd.DataFrame:
    """Represent encoded categorical columns as strings for native CatBoost handling."""
    frame = pd.DataFrame(values)
    for column in range(categorical_columns):
        frame[column] = frame[column].round().astype(np.int64).astype(str)
    return frame


def _rank_fusion(*score_arrays: np.ndarray) -> np.ndarray:
    """Equal-weight reciprocal-rank fusion, fixed before final evaluation."""
    fused = np.zeros_like(score_arrays[0], dtype=np.float32)
    for scores in score_arrays:
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty_like(order)
        ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
        fused += 1.0 / (ranks.astype(np.float32) + 1.0)
    return fused


def run_first_place_ensemble(root: Path, config: TopSolutionConfig) -> Path:
    """LightGBM + CatBoost rank blend on the same published-width Top-100 pool."""
    started = time.perf_counter()
    source = root / "artifacts" / "hybrid_six_week_experiment"
    train_users, train_articles, train_union = _load_saved_union(source / "train_candidate_pool.npz", config.first_candidate_k)
    final_users, final_articles, final_union = _load_saved_union(source / "final_candidate_pool.npz", config.first_candidate_k)
    transactions = load_transactions_since(root / "Data", root / "artifacts" / "cache", FIRST_TRAIN.context_start)
    train_context, train_labels = slice_window(transactions, FIRST_TRAIN)
    final_context, final_labels = slice_window(transactions, FIRST_FINAL)
    customers, articles = load_metadata(root / "Data")
    encoder = fit_feature_encoder(customers, articles)
    train_features = build_snapshot_features(train_context, customers, articles, encoder, FIRST_TRAIN.label_start)
    final_features = build_snapshot_features(final_context, customers, articles, encoder, FIRST_FINAL.label_start)
    if not np.array_equal(train_articles, train_features.article_ids.astype(str)) or not np.array_equal(final_articles, final_features.article_ids.astype(str)):
        raise ValueError("Saved six-week pools do not match reconstructed snapshots")
    train_truth_all, final_truth_all = truth_by_customer(train_labels), truth_by_customer(final_labels)
    train_truth = {user: train_truth_all[user] for user in train_users}
    final_truth = {user: final_truth_all[user] for user in final_users}
    train_stats = _temporal_stats(train_context, train_features, articles, customers, FIRST_TRAIN.label_start)
    final_stats = _temporal_stats(final_context, final_features, articles, customers, FIRST_FINAL.label_start)
    user_rows, positions, sampling = _select_labeled_examples(
        train_users, train_union.items, train_features.article_ids, train_truth,
        config.negatives_per_positive, config.seed + 40,
    )
    train_x = _model_features(train_features, train_users, train_union, user_rows, positions, train_stats)
    labels = _labels(train_users, train_union, train_features, train_truth, user_rows, positions)
    lightgbm = _fit_ranker(train_x, labels, user_rows, config, config.seed + 41, True)
    from catboost import CatBoostClassifier

    categorical_columns = train_features.user_cats.shape[1] + train_features.item_cats.shape[1]
    catboost = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=config.seed + 42,
        l2_leaf_reg=3.0,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )
    catboost.fit(
        _catboost_frame(train_x, categorical_columns),
        labels,
        cat_features=list(range(categorical_columns)),
    )
    lightgbm_scores = _score(
        lightgbm, True, final_features, final_users, final_union, final_stats,
        config.score_batch_size,
    )
    width = final_union.items.shape[1]
    catboost_scores = np.empty((len(final_users), width), dtype=np.float32)
    flat_users = np.repeat(np.arange(len(final_users), dtype=np.int64), width)
    flat_positions = np.tile(np.arange(width, dtype=np.int64), len(final_users))
    for start in range(0, len(flat_users), config.score_batch_size):
        stop = min(start + config.score_batch_size, len(flat_users))
        values = _model_features(
            final_features, final_users, final_union,
            flat_users[start:stop], flat_positions[start:stop], final_stats,
        )
        catboost_scores.reshape(-1)[start:stop] = catboost.predict_proba(
            _catboost_frame(values, categorical_columns)
        )[:, 1]
    fused_scores = _rank_fusion(lightgbm_scores, catboost_scores)
    rankings = {
        "lightgbm": _ranked_rows(lightgbm_scores, final_union.items),
        "catboost": _ranked_rows(catboost_scores, final_union.items),
        "equal_rank_fusion": _ranked_rows(fused_scores, final_union.items),
    }
    active = set(final_context.customer_id.astype(str).unique())
    metrics = {
        name: _metric_block(final_users, rows, final_features, final_truth, active, width)
        for name, rows in rankings.items()
    }
    output = root / "artifacts" / "first_place_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in rankings.items():
        pd.DataFrame({
            "customer_id": final_users,
            "prediction": [" ".join(final_features.article_ids[row[:12]]) for row in rows],
        }).to_csv(output / f"{name}_top12.csv", index=False)
    result = {
        "scope": "pre-specified equal-rank LightGBM/CatBoost blend on identical Top-100 candidates",
        "config": {**asdict(config), "catboost_iterations": 500, "catboost_depth": 8},
        "windows": {"train": asdict(FIRST_TRAIN), "final": asdict(FIRST_FINAL)},
        "training": {**sampling, "examples": len(labels), "positives": int(labels.sum())},
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def run_second_place_inspired(root: Path, config: TopSolutionConfig) -> Path:
    started = time.perf_counter()
    transactions = load_transactions_since(root / "Data", root / "artifacts" / "cache", SECOND_STAGE1.context_start)
    customers, articles = load_metadata(root / "Data")
    encoder = fit_feature_encoder(customers, articles)
    snapshots = []
    for window in (SECOND_STAGE1, SECOND_STAGE2, SECOND_FINAL):
        context, labels = slice_window(transactions, window)
        features = build_snapshot_features(context, customers, articles, encoder, window.label_start)
        users = sample_customers(labels, set(), config.train_users if window != SECOND_FINAL else config.eval_users, config.seed)
        truth_all = truth_by_customer(labels)
        truth = {user: truth_all[user] for user in users}
        candidates = _initial_candidates(context, users, features, articles, window.label_start, config.second_initial_k)
        stats = _temporal_stats(context, features, articles, customers, window.label_start)
        snapshots.append((window, context, features, users, truth, candidates, stats))
    _, _, s1_features, s1_users, s1_truth, s1_candidates, s1_stats = snapshots[0]
    stage1_model, s1_sampling, s1_examples, s1_positives = _train_on_union(
        s1_features, s1_users, s1_candidates, s1_truth, s1_stats, config, config.seed + 20, False
    )
    stage2_pools = []
    stage1_metrics = []
    for _, context, features, users, truth, candidates, stats in snapshots[1:]:
        scores = _score(stage1_model, False, features, users, candidates, stats, config.score_batch_size)
        ranked_initial = _ranked_rows(scores, candidates.items)
        top130 = ranked_initial[:, : config.second_stage1_k]
        stage2_pools.append(_stage2_union(context, users, features, top130, config.second_final_k))
        active = set(context.customer_id.astype(str).unique())
        stage1_metrics.append({
            "initial_600": _metric_block(users, ranked_initial, features, truth, active, config.second_initial_k),
            "selected_130": _metric_block(users, top130, features, truth, active, config.second_stage1_k),
        })
    _, train_context, train_features, train_users, train_truth, _, train_stats = snapshots[1]
    train_stage2 = stage2_pools[0]
    stage2_model, s2_sampling, s2_examples, s2_positives = _train_on_union(
        train_features, train_users, train_stage2, train_truth, train_stats, config, config.seed + 30, False
    )
    _, final_context, final_features, final_users, final_truth, _, final_stats = snapshots[2]
    final_stage2 = stage2_pools[1]
    final_scores = _score(stage2_model, False, final_features, final_users, final_stage2, final_stats, config.score_batch_size)
    final_ranked = _ranked_rows(final_scores, final_stage2.items)
    active = set(final_context.customer_id.astype(str).unique())
    final_metrics = _metric_block(final_users, final_ranked, final_features, final_truth, active, config.second_final_k)
    candidate_metrics = _metric_block(final_users, final_stage2.items, final_features, final_truth, active, config.second_final_k)
    output = root / "artifacts" / "second_place_inspired"; output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "final_candidate_pool.npz", customer_ids=np.asarray(final_users), article_ids=final_features.article_ids, candidate_item_rows=final_stage2.items, source_rr=final_stage2.source_rr, source_names=np.asarray(final_stage2.source_names))
    pd.DataFrame({"customer_id": final_users, "prediction": [" ".join(final_features.article_ids[row[:12]]) for row in final_ranked]}).to_csv(output / "top12.csv", index=False)
    result = {
        "scope": "reproduction inspired by the detailed second-place write-up; excludes teammate graph-walk candidates and private feature framework",
        "published_design": "about 600 popularity/section candidates -> LightGBM Top-130 -> add all history -> richer LightGBM",
        "config": asdict(config),
        "windows": {"stage1_train": asdict(SECOND_STAGE1), "stage2_train": asdict(SECOND_STAGE2), "final": asdict(SECOND_FINAL)},
        "training": {"stage1": {**s1_sampling, "examples": s1_examples, "positives": s1_positives}, "stage2": {**s2_sampling, "examples": s2_examples, "positives": s2_positives}},
        "metrics": {"stage1_final": stage1_metrics[1], "stage2_candidate_pool": candidate_metrics, "stage2_lightgbm": final_metrics},
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); return path

