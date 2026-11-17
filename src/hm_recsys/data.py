"""Data loading, temporal snapshots, and leakage-safe feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    context_start: str
    label_start: str
    label_end: str


TRAIN_WINDOW = Window("2020-08-12", "2020-09-09", "2020-09-16")
FINAL_WINDOW = Window("2020-08-19", "2020-09-16", "2020-09-23")


def window_with_history(label_start: str, label_end: str, history_days: int) -> Window:
    """Return a fixed-length context ending immediately before ``label_start``."""
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    start = pd.Timestamp(label_start) - pd.Timedelta(days=history_days)
    return Window(start.strftime("%Y-%m-%d"), label_start, label_end)


def load_transactions_since(
    data_dir: Path, cache_dir: Path, start_date: str
) -> pd.DataFrame:
    """Read transactions from ``start_date`` onward and cache that exact slice."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"transactions_{start_date}_to_2020-09-22.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    # A cache beginning earlier contains every row needed by a later snapshot.
    # Reuse the closest such cache; slice_window will enforce the exact boundary.
    requested_start = pd.Timestamp(start_date)
    reusable: list[tuple[pd.Timestamp, Path]] = []
    for candidate in cache_dir.glob("transactions_????-??-??_to_2020-09-22.pkl"):
        candidate_start = pd.Timestamp(candidate.name[13:23])
        if candidate_start <= requested_start:
            reusable.append((candidate_start, candidate))
    if reusable:
        _, reusable_path = max(reusable, key=lambda pair: pair[0])
        return pd.read_pickle(reusable_path)

    parts: list[pd.DataFrame] = []
    dtypes = {
        "customer_id": "string",
        "article_id": "string",
        "price": "float32",
        "sales_channel_id": "int8",
    }
    for chunk in pd.read_csv(
        data_dir / "transactions_train.csv", dtype=dtypes, chunksize=1_000_000
    ):
        recent = chunk.loc[chunk["t_dat"] >= start_date].copy()
        if not recent.empty:
            parts.append(recent)
    transactions = pd.concat(parts, ignore_index=True)
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions.to_pickle(cache_path)
    return transactions


def load_recent_transactions(data_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """Read the 3.2 GB source in chunks once, then reuse a compact local cache."""
    return load_transactions_since(data_dir, cache_dir, TRAIN_WINDOW.context_start)


def slice_window(transactions: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_start = pd.Timestamp(window.context_start)
    label_start = pd.Timestamp(window.label_start)
    label_end = pd.Timestamp(window.label_end)
    context = transactions.loc[
        (transactions["t_dat"] >= context_start) & (transactions["t_dat"] < label_start)
    ].copy()
    labels = transactions.loc[
        (transactions["t_dat"] >= label_start) & (transactions["t_dat"] < label_end)
    ].copy()
    return context, labels


def mode_or_missing(values: pd.Series) -> str:
    mode = values.dropna().mode()
    return "__MISSING__" if mode.empty else str(mode.iloc[0])


class CategoryEncoder:
    """Small dependency-free categorical encoder with index 0 reserved for OOV."""

    def __init__(self) -> None:
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, frames: Iterable[tuple[str, pd.Series]]) -> "CategoryEncoder":
        for name, series in frames:
            values = sorted(series.fillna("__MISSING__").astype(str).unique())
            self.maps[name] = {value: i + 1 for i, value in enumerate(values)}
        return self

    def transform(self, name: str, series: pd.Series) -> np.ndarray:
        mapping = self.maps[name]
        return (
            series.fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

    def cardinality(self, name: str) -> int:
        return len(self.maps[name]) + 1


ARTICLE_CAT_COLUMNS = [
    "product_type_name",
    "index_group_name",
    "garment_group_name",
    "colour_group_name",
    "section_name",
]
USER_CAT_COLUMNS = [
    "age_bucket",
    "club_member_status",
    "fashion_news_frequency",
    "preferred_index_group",
    "preferred_garment_group",
]


@dataclass
class SnapshotFeatures:
    customer_ids: np.ndarray
    article_ids: np.ndarray
    user_cats: np.ndarray
    user_dense: np.ndarray
    item_cats: np.ndarray
    item_dense: np.ndarray
    customer_to_row: dict[str, int]
    article_to_row: dict[str, int]
    purchased: dict[str, set[str]]
    item_popularity: np.ndarray


def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(
        data_dir / "customers.csv",
        usecols=[
            "customer_id",
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        ],
        dtype={"customer_id": "string"},
    )
    articles = pd.read_csv(
        data_dir / "articles.csv",
        usecols=["article_id", *ARTICLE_CAT_COLUMNS],
        dtype={"article_id": "string"},
    )
    return customers, articles


def fit_feature_encoder(customers: pd.DataFrame, articles: pd.DataFrame) -> CategoryEncoder:
    c = customers.copy()
    c["age_bucket"] = pd.cut(
        c["age"], bins=[0, 19, 24, 29, 39, 49, 59, 69, 200], include_lowest=True
    ).astype("string")
    frames: list[tuple[str, pd.Series]] = [(col, articles[col]) for col in ARTICLE_CAT_COLUMNS]
    frames += [
        ("age_bucket", c["age_bucket"]),
        ("club_member_status", c["club_member_status"]),
        ("fashion_news_frequency", c["fashion_news_frequency"]),
        ("preferred_index_group", articles["index_group_name"]),
        ("preferred_garment_group", articles["garment_group_name"]),
    ]
    return CategoryEncoder().fit(frames)


def build_snapshot_features(
    context: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    encoder: CategoryEncoder,
    cutoff: str,
) -> SnapshotFeatures:
    """Build features using only transactions strictly before the snapshot cutoff."""
    cutoff_ts = pd.Timestamp(cutoff)
    article_ids = np.sort(context["article_id"].dropna().astype(str).unique())
    item_frame = pd.DataFrame({"article_id": article_ids}).merge(
        articles, on="article_id", how="left"
    )
    article_to_row = {item: i for i, item in enumerate(article_ids)}

    item_cats = np.column_stack(
        [encoder.transform(col, item_frame[col]) for col in ARTICLE_CAT_COLUMNS]
    )
    item_stats = context.groupby("article_id", observed=True).agg(
        purchases=("article_id", "size"),
        avg_price=("price", "mean"),
        last_date=("t_dat", "max"),
    )
    aligned = item_frame[["article_id"]].join(item_stats, on="article_id")
    item_dense = np.column_stack(
        [
            np.log1p(aligned["purchases"].fillna(0).to_numpy(dtype=np.float32)),
            aligned["avg_price"].fillna(0).to_numpy(dtype=np.float32),
            ((cutoff_ts - aligned["last_date"]).dt.days.fillna(999).clip(0, 60) / 60).to_numpy(
                dtype=np.float32
            ),
        ]
    )
    popularity = aligned["purchases"].fillna(0).to_numpy(dtype=np.float32)

    customer_ids = customers["customer_id"].astype(str).to_numpy()
    customer_to_row = {customer: i for i, customer in enumerate(customer_ids)}
    user_frame = customers.copy()
    user_frame["age_bucket"] = pd.cut(
        user_frame["age"],
        bins=[0, 19, 24, 29, 39, 49, 59, 69, 200],
        include_lowest=True,
    ).astype("string")

    context_meta = context[["customer_id", "article_id", "t_dat", "sales_channel_id"]].merge(
        articles[["article_id", "index_group_name", "garment_group_name"]],
        on="article_id",
        how="left",
    )
    profile = context_meta.groupby("customer_id", observed=True).agg(
        activity=("article_id", "size"),
        last_date=("t_dat", "max"),
        preferred_channel=("sales_channel_id", "mean"),
        preferred_index_group=("index_group_name", mode_or_missing),
        preferred_garment_group=("garment_group_name", mode_or_missing),
    )
    user_frame = user_frame.join(profile, on="customer_id")
    user_cats = np.column_stack(
        [encoder.transform(col, user_frame[col]) for col in USER_CAT_COLUMNS]
    )
    age = user_frame["age"].fillna(user_frame["age"].median()).clip(0, 100) / 100
    user_dense = np.column_stack(
        [
            np.log1p(user_frame["activity"].fillna(0).to_numpy(dtype=np.float32)),
            ((cutoff_ts - user_frame["last_date"]).dt.days.fillna(999).clip(0, 60) / 60).to_numpy(
                dtype=np.float32
            ),
            age.to_numpy(dtype=np.float32),
            (user_frame["preferred_channel"].fillna(1) - 1).to_numpy(dtype=np.float32),
        ]
    )
    purchased = (
        context.groupby("customer_id", observed=True)["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    return SnapshotFeatures(
        customer_ids=customer_ids,
        article_ids=article_ids,
        user_cats=user_cats,
        user_dense=user_dense,
        item_cats=item_cats,
        item_dense=item_dense,
        customer_to_row=customer_to_row,
        article_to_row=article_to_row,
        purchased=purchased,
        item_popularity=popularity,
    )


def truth_by_customer(labels: pd.DataFrame) -> dict[str, set[str]]:
    return (
        labels.groupby("customer_id", observed=True)["article_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )


def sample_customers(
    labels: pd.DataFrame, known: set[str], maximum: int | None, seed: int
) -> list[str]:
    """Deterministic random sample; cohort proportions remain representative."""
    ids = np.sort(labels["customer_id"].astype(str).unique())
    if maximum is None or len(ids) <= maximum:
        return ids.tolist()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=maximum, replace=False)).tolist()
