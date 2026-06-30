"""
SVD bias model — vectorized mini-batch SGD.

Formula: r_ui ~= mu + b_u + b_i + p_u . q_i

Optimization: each epoch randomly samples a user batch, using matrix multiplication
to update all latent vectors in the batch at once, avoiding per-sample Python loops.

Speedup: 10-20x depending on batch_size.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional

_SVD_FACTORS = 50
_SVD_LR = 0.01
_SVD_REG = 0.02
_RANDOM_SEED = 42


class SVDModel:
    """SVD model with user/item biases and latent factor dot product."""

    def __init__(self, n_users: int, n_items: int,
                 k: int = 50, lr: float = 0.01, reg: float = 0.02,
                 random_state: int = 42):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.global_mean: float = 3.0
        rng = np.random.RandomState(random_state)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.P = rng.normal(0, 0.1, (n_users, k)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, k)).astype(np.float32)
        self.user_idx: Dict[int, int] = {}
        self.item_idx: Dict[int, int] = {}
        self.fallback_counts: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame, epochs: int = 50, verbose: bool = True,
            batch_size: int = 4096) -> None:
        """Vectorized mini-batch SGD training."""
        data = df[["user_id", "item_id", "rating"]].copy()
        data["user_id"] = data["user_id"].apply(
            lambda x: int(x) if str(x).isdigit() else hash(str(x)) % 10**8)
        data["item_id"] = data["item_id"].apply(
            lambda x: int(x) if str(x).isdigit() else hash(str(x)) % 10**8)

        users = sorted(data["user_id"].unique())
        items = sorted(data["item_id"].unique())
        n_users = len(users)
        n_items = len(items)

        if len(self.bu) != n_users or len(self.bi) != n_items:
            rng = np.random.RandomState(_RANDOM_SEED)
            self.bu = np.zeros(n_users, dtype=np.float32)
            self.bi = np.zeros(n_items, dtype=np.float32)
            self.P = rng.normal(0, 0.1, (n_users, self.k)).astype(np.float32)
            self.Q = rng.normal(0, 0.1, (n_items, self.k)).astype(np.float32)

        self.user_idx = {int(u): i for i, u in enumerate(users)}
        self.item_idx = {int(m): j for j, m in enumerate(items)}
        self.global_mean = float(data["rating"].mean())

        u_indices = np.array([self.user_idx[int(r.user_id)] for r in data.itertuples()], dtype=np.int32)
        i_indices = np.array([self.item_idx[int(r.item_id)] for r in data.itertuples()], dtype=np.int32)
        ratings = np.array([r.rating for r in data.itertuples()], dtype=np.float32)

        n_ratings = len(ratings)

        for epoch in range(epochs):
            perm = np.random.permutation(n_ratings)
            total_loss = 0.0

            for start in range(0, n_ratings, batch_size):
                idx = perm[start:start + batch_size]
                u_batch = u_indices[idx]
                i_batch = i_indices[idx]
                r_batch = ratings[idx]

                pred = (self.global_mean
                        + self.bu[u_batch]
                        + self.bi[i_batch]
                        + np.sum(self.P[u_batch] * self.Q[i_batch], axis=1))

                err = r_batch - pred
                total_loss += np.sum(err ** 2)

                self.bu[u_batch] += self.lr * (err - self.reg * self.bu[u_batch])
                self.bi[i_batch] += self.lr * (err - self.reg * self.bi[i_batch])

                q_i = self.Q[i_batch]
                p_u = self.P[u_batch]

                self.P[u_batch] += self.lr * (err[:, None] * q_i - self.reg * p_u)
                self.Q[i_batch] += self.lr * (err[:, None] * p_u - self.reg * q_i)

            avg_loss = total_loss / n_ratings
            if verbose:
                print(f"  SVD Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    def predict(self, user_id: int, movie_id: int) -> float:
        u = self.user_idx.get(int(user_id))
        i = self.item_idx.get(int(movie_id))

        if u is not None and i is not None:
            self.fallback_counts["full_svd"] = self.fallback_counts.get("full_svd", 0) + 1
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
        elif u is not None:
            self.fallback_counts["user_bias_only"] = self.fallback_counts.get("user_bias_only", 0) + 1
            pred = self.global_mean + self.bu[u]
        elif i is not None:
            self.fallback_counts["item_bias_only"] = self.fallback_counts.get("item_bias_only", 0) + 1
            pred = self.global_mean + self.bi[i]
        else:
            self.fallback_counts["global_mean_only"] = self.fallback_counts.get("global_mean_only", 0) + 1
            pred = self.global_mean

        return float(np.clip(pred, 1.0, 5.0))

    def reset_fallback_counts(self) -> None:
        self.fallback_counts = {"full_svd": 0, "user_bias_only": 0,
                                "item_bias_only": 0, "global_mean_only": 0}

    def get_fallback_counts(self) -> Dict[str, int]:
        return dict(self.fallback_counts)

    def get_user_vector(self, user_id: int) -> np.ndarray:
        u = self.user_idx.get(int(user_id))
        return self.P[u].astype(np.float32) if u is not None else np.zeros(self.k, dtype=np.float32)

    def get_item_vector(self, item_id: int) -> np.ndarray:
        i = self.item_idx.get(int(item_id))
        return self.Q[i].astype(np.float32) if i is not None else np.zeros(self.k, dtype=np.float32)

    def get_pair_features(self, user_id: int, item_id: int) -> np.ndarray:
        return np.concatenate([self.get_user_vector(user_id), self.get_item_vector(item_id)])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SVDModel":
        with open(path, "rb") as f:
            return pickle.load(f)
