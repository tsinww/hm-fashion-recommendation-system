"""PyTorch models for retrieval and candidate-aware re-ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MatrixFactorization(nn.Module):
    """Implicit-feedback MF trained with sampled pairwise BPR loss."""

    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, dim)
        self.item_embeddings = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.05)
        nn.init.normal_(self.item_embeddings.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embeddings(users) * self.item_embeddings(items)).sum(dim=1)


@dataclass
class FittedMF:
    model: MatrixFactorization
    user_to_index: dict[str, int]
    item_ids: np.ndarray
    item_to_index: dict[str, int]
    popularity_order: np.ndarray
    device: torch.device

    @torch.no_grad()
    def candidates(
        self,
        customer_ids: list[str],
        k: int,
        batch_size: int = 64,
        excluded_items: dict[str, set[str]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score only one bounded user batch at a time, never a full user-item matrix."""
        k = min(k, len(self.item_ids))
        out_items = np.empty((len(customer_ids), k), dtype=np.int64)
        out_scores = np.empty((len(customer_ids), k), dtype=np.float32)
        item_vectors = self.model.item_embeddings.weight.detach()
        fallback = self.popularity_order[:k]
        for start in range(0, len(customer_ids), batch_size):
            stop = min(start + batch_size, len(customer_ids))
            batch = customer_ids[start:stop]
            known_positions = [i for i, customer in enumerate(batch) if customer in self.user_to_index]
            for i, customer in enumerate(batch):
                if customer not in self.user_to_index:
                    out_items[start + i] = fallback
                    out_scores[start + i] = np.linspace(0.0, -1.0, k, dtype=np.float32)
            if known_positions:
                indices = torch.tensor(
                    [self.user_to_index[batch[i]] for i in known_positions],
                    dtype=torch.long,
                    device=self.device,
                )
                user_vectors = self.model.user_embeddings(indices)
                scores = user_vectors @ item_vectors.T
                if excluded_items:
                    for local, position in enumerate(known_positions):
                        excluded = excluded_items.get(batch[position], set())
                        excluded_indices = [
                            self.item_to_index[item]
                            for item in excluded
                            if item in self.item_to_index
                        ]
                        if excluded_indices:
                            scores[local, excluded_indices] = -torch.inf
                values, items = torch.topk(scores, k=k, dim=1)
                for local, position in enumerate(known_positions):
                    out_items[start + position] = items[local].cpu().numpy()
                    out_scores[start + position] = values[local].cpu().numpy()
        return out_items, out_scores


def fit_mf(
    customer_ids: np.ndarray,
    item_ids: np.ndarray,
    popularity: np.ndarray,
    pair_customers: np.ndarray,
    pair_items: np.ndarray,
    dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_pairs: int | None = None,
) -> tuple[FittedMF, list[float]]:
    seed_everything(seed)
    device = available_device()
    user_to_index = {value: i for i, value in enumerate(customer_ids)}
    item_to_index = {value: i for i, value in enumerate(item_ids)}
    u = np.fromiter((user_to_index[x] for x in pair_customers), dtype=np.int64)
    i = np.fromiter((item_to_index[x] for x in pair_items), dtype=np.int64)
    observed_keys = np.unique(u * len(item_ids) + i)
    if max_pairs is not None and len(u) > max_pairs:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(u), size=max_pairs, replace=False)
        u, i = u[keep], i[keep]

    model = MatrixFactorization(len(customer_ids), len(item_ids), dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(epochs):
        negatives = rng.integers(0, len(item_ids), size=len(i), dtype=np.int64)
        invalid = np.isin(u * len(item_ids) + negatives, observed_keys)
        while invalid.any():
            negatives[invalid] = rng.integers(0, len(item_ids), size=int(invalid.sum()))
            invalid = np.isin(u * len(item_ids) + negatives, observed_keys)
        order = rng.permutation(len(u))
        total = 0.0
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            users = torch.as_tensor(u[rows], dtype=torch.long, device=device)
            positives = torch.as_tensor(i[rows], dtype=torch.long, device=device)
            negs = torch.as_tensor(negatives[rows], dtype=torch.long, device=device)
            loss = -torch.nn.functional.logsigmoid(
                model(users, positives) - model(users, negs)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(rows)
        losses.append(total / len(u))
    model.eval()
    return (
        FittedMF(
            model=model,
            user_to_index=user_to_index,
            item_ids=item_ids,
            item_to_index=item_to_index,
            popularity_order=np.argsort(-popularity, kind="stable"),
            device=device,
        ),
        losses,
    )


def fit_mf_bce(
    customer_ids: np.ndarray,
    item_ids: np.ndarray,
    popularity: np.ndarray,
    pair_customers: np.ndarray,
    pair_items: np.ndarray,
    dim: int,
    negatives_per_positive: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[FittedMF, list[float]]:
    """Notebook-style pointwise MF with fixed uniform unobserved negatives."""
    seed_everything(seed)
    device = available_device()
    user_to_index = {value: i for i, value in enumerate(customer_ids)}
    item_to_index = {value: i for i, value in enumerate(item_ids)}
    positives_u = np.fromiter((user_to_index[x] for x in pair_customers), dtype=np.int64)
    positives_i = np.fromiter((item_to_index[x] for x in pair_items), dtype=np.int64)
    observed_keys = np.unique(positives_u * len(item_ids) + positives_i)

    rng = np.random.default_rng(seed)
    negatives_u = np.repeat(positives_u, negatives_per_positive)
    negatives_i = rng.integers(0, len(item_ids), size=len(negatives_u), dtype=np.int64)
    invalid = np.isin(negatives_u * len(item_ids) + negatives_i, observed_keys)
    while invalid.any():
        negatives_i[invalid] = rng.integers(0, len(item_ids), size=int(invalid.sum()))
        invalid = np.isin(negatives_u * len(item_ids) + negatives_i, observed_keys)

    train_u = np.concatenate([positives_u, negatives_u])
    train_i = np.concatenate([positives_i, negatives_i])
    labels = np.concatenate(
        [
            np.ones(len(positives_u), dtype=np.float32),
            np.zeros(len(negatives_u), dtype=np.float32),
        ]
    )
    model = MatrixFactorization(len(customer_ids), len(item_ids), dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        total = 0.0
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            users = torch.as_tensor(train_u[rows], dtype=torch.long, device=device)
            items = torch.as_tensor(train_i[rows], dtype=torch.long, device=device)
            targets = torch.as_tensor(labels[rows], dtype=torch.float32, device=device)
            loss = loss_fn(model(users, items), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(rows)
        losses.append(total / len(labels))
    model.eval()
    return (
        FittedMF(
            model=model,
            user_to_index=user_to_index,
            item_ids=item_ids,
            item_to_index=item_to_index,
            popularity_order=np.argsort(-popularity, kind="stable"),
            device=device,
        ),
        losses,
    )


class TwoTower(nn.Module):
    def __init__(
        self,
        user_cardinalities: list[int],
        item_cardinalities: list[int],
        user_dense_dim: int,
        item_dense_dim: int,
        embedding_dim: int = 8,
        tower_dim: int = 32,
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in user_cardinalities]
        )
        self.item_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in item_cardinalities]
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(len(user_cardinalities) * embedding_dim + user_dense_dim, 64),
            nn.ReLU(),
            nn.Linear(64, tower_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(len(item_cardinalities) * embedding_dim + item_dense_dim, 64),
            nn.ReLU(),
            nn.Linear(64, tower_dim),
        )

    def encode_users(self, cats: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = [layer(cats[:, i]) for i, layer in enumerate(self.user_embeddings)]
        return torch.nn.functional.normalize(self.user_mlp(torch.cat([*embedded, dense], dim=1)))

    def encode_items(self, cats: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = [layer(cats[:, i]) for i, layer in enumerate(self.item_embeddings)]
        return torch.nn.functional.normalize(self.item_mlp(torch.cat([*embedded, dense], dim=1)))

    def forward(
        self,
        user_cats: torch.Tensor,
        user_dense: torch.Tensor,
        item_cats: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        return 8.0 * (self.encode_users(user_cats, user_dense) * self.encode_items(item_cats, item_dense)).sum(1)


class DLRMRanker(nn.Module):
    """Compact DLRM: embedded sparse fields, dense bottom MLP, pairwise interactions."""

    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int = 8) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities]
        )
        self.bottom = nn.Sequential(
            nn.Linear(dense_dim, 32), nn.ReLU(), nn.Linear(32, embedding_dim), nn.ReLU()
        )
        fields = len(cardinalities) + 1
        interactions = fields * (fields - 1) // 2
        self.top = nn.Sequential(
            nn.Linear(embedding_dim + interactions, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, cats: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        dense_vector = self.bottom(dense)
        vectors = [dense_vector] + [layer(cats[:, i]) for i, layer in enumerate(self.embeddings)]
        stack = torch.stack(vectors, dim=1)
        dots = torch.bmm(stack, stack.transpose(1, 2))
        row, col = torch.triu_indices(len(vectors), len(vectors), offset=1, device=cats.device)
        interactions = dots[:, row, col]
        return self.top(torch.cat([dense_vector, interactions], dim=1)).squeeze(1)


def fit_binary_model(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[nn.Module, list[float]]:
    seed_everything(seed)
    device = available_device()
    model = model.to(device)
    dataset = TensorDataset(*tensors, labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    positives = max(float(labels.sum()), 1.0)
    pos_weight = torch.tensor([(len(labels) - positives) / positives], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            features = [value.to(device) for value in batch[:-1]]
            targets = batch[-1].to(device)
            logits = model(*features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(targets)
        losses.append(total / len(dataset))
    model.eval()
    return model, losses
