"""
User-based Collaborative Filtering (UserCF) baseline.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.neighbors import NearestNeighbors
from models.base_model import BaseModel


class UserCF(BaseModel):
    """User-based Collaborative Filtering with unified interface."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        cfg = config or {}
        self.k: int = cfg.get("cf_k", 30)
        self.min_common: int = cfg.get("cf_min_common", 5)
        self.user_mean: Optional[Dict[int, float]] = None
        self.rating_matrix: Optional[np.ndarray] = None
        self.user_ids: Optional[np.ndarray] = None
        self.movie_ids: Optional[np.ndarray] = None
        self.user_idx: Dict[int, int] = {}
        self.movie_idx: Dict[int, int] = {}
        self.knn: Optional[NearestNeighbors] = None
        self.centered: Optional[np.ndarray] = None

    def fit(self, train_data: pd.DataFrame, valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None, verbose: bool = True, **kwargs) -> "UserCF":
        if verbose:
            print("[CF] Building rating matrix...")
        pivot = train_data.pivot_table(index="user_id", columns="item_id", values="rating")
        self.user_ids = pivot.index.values
        self.movie_ids = pivot.columns.values
        self.rating_matrix = pivot.values
        self.user_mean = {int(uid): np.nanmean(self.rating_matrix[i])
                          for i, uid in enumerate(self.user_ids)}
        self.user_idx = {int(uid): i for i, uid in enumerate(self.user_ids)}
        self.movie_idx = {int(mid): j for j, mid in enumerate(self.movie_ids)}

        centered = self.rating_matrix.copy()
        for i in range(len(self.user_ids)):
            row = centered[i]
            mask = ~np.isnan(row)
            centered[i, mask] = row[mask] - self.user_mean.get(int(self.user_ids[i]), 3.0)
            centered[i, ~mask] = 0.0

        self.knn = NearestNeighbors(n_neighbors=min(self.k+1, len(self.user_ids)),
                                     metric="cosine", n_jobs=-1)
        self.knn.fit(centered)
        self.centered = centered
        if verbose:
            print(f"[CF] Ready: {len(self.user_ids)} users, {len(self.movie_ids)} movies")
        return self

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        preds = np.full(len(test_data), self.global_mean, dtype=np.float32)
        for i, (_, row) in enumerate(test_data.iterrows()):
            uid, mid = int(row["user_id"]), int(row["item_id"])
            if uid not in self.user_idx or mid not in self.movie_idx:
                continue
            ui, mj = self.user_idx[uid], self.movie_idx[mid]
            mu = self.user_mean.get(uid, self.global_mean)

            dist, idx = self.knn.kneighbors(self.centered[ui].reshape(1, -1))
            numer, denom = 0.0, 0.0
            for d, j in zip(dist[0], idx[0]):
                if j == ui:
                    continue
                sim = 1.0 - d
                if sim <= 0:
                    continue
                val = self.rating_matrix[j, mj]
                if np.isnan(val):
                    continue
                numer += sim * (val - self.user_mean.get(int(self.user_ids[j]), self.global_mean))
                denom += abs(sim)
            preds[i] = mu + numer / denom if denom > 0 else mu
        return np.clip(preds, 1.0, 5.0)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"k": self.k, "min_common": self.min_common,
                         "user_mean": self.user_mean, "rating_matrix": self.rating_matrix,
                         "user_ids": self.user_ids, "movie_ids": self.movie_ids,
                         "user_idx": self.user_idx, "movie_idx": self.movie_idx,
                         "centered": self.centered}, f)

    def load_checkpoint(self, path: str) -> "UserCF":
        with open(path, "rb") as f:
            d = pickle.load(f)
        for k, v in d.items():
            setattr(self, k, v)
        self.knn = NearestNeighbors(n_neighbors=min(self.k+1, len(self.user_ids)),
                                     metric="cosine", n_jobs=-1)
        self.knn.fit(self.centered)
        self._is_fitted = True
        return self
