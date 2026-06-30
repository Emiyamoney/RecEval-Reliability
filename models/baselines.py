"""
Baseline models: GlobalMean, UserBias, ItemBias, UserItemBias.

All computations use training data only — no data leakage.

Usage:
  model = GlobalMean()
  model.fit(train_df)
  preds = model.predict(test_df)
  results = model.evaluate(test_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from models.base_model import BaseModel


class GlobalMean(BaseModel):
    """Global mean baseline — predicts the same global mean for all users/items."""

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            **kwargs) -> "GlobalMean":
        self._compute_global_stats(train_data)
        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        n = len(test_data)
        return np.full(n, self.global_mean, dtype=np.float32)


class UserBias(BaseModel):
    """User bias model: r_ui = mu + b_u"""

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            reg: float = 5.0,
            **kwargs) -> "UserBias":
        self._compute_global_stats(train_data)

        user_means = train_data.groupby("user_id")["rating"].mean()
        user_counts = train_data.groupby("user_id")["rating"].count()
        for uid, mean_r in user_means.items():
            n_u = user_counts[uid]
            self.user_bias[int(uid)] = float(
                (mean_r - self.global_mean) * n_u / (n_u + reg)
            )

        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        preds = np.full(len(test_data), self.global_mean, dtype=np.float32)
        uids = test_data["user_id"].values.astype(int)
        bias = np.array([self.user_bias.get(uid, 0.0) for uid in uids], dtype=np.float32)
        preds += bias
        return self._clip_predictions(preds)


class ItemBias(BaseModel):
    """Item bias model: r_ui = mu + b_i"""

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            reg: float = 5.0,
            **kwargs) -> "ItemBias":
        self._compute_global_stats(train_data)

        item_means = train_data.groupby("item_id")["rating"].mean()
        item_counts = train_data.groupby("item_id")["rating"].count()
        for iid, mean_r in item_means.items():
            n_i = item_counts[iid]
            self.item_bias[int(iid)] = float(
                (mean_r - self.global_mean) * n_i / (n_i + reg)
            )

        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        preds = np.full(len(test_data), self.global_mean, dtype=np.float32)
        iids = test_data["item_id"].values.astype(int)
        bias = np.array([self.item_bias.get(iid, 0.0) for iid in iids], dtype=np.float32)
        preds += bias
        return self._clip_predictions(preds)


class UserItemBias(BaseModel):
    """User + Item bias model: r_ui = mu + b_u + b_i"""

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            reg: float = 5.0,
            n_iter: int = 10,
            lr: float = 0.01,
            **kwargs) -> "UserItemBias":
        """Alternating least squares style training for user/item biases."""
        self._compute_global_stats(train_data)

        users = train_data["user_id"].unique()
        items = train_data["item_id"].unique()
        for u in users:
            self.user_bias[int(u)] = 0.0
        for i in items:
            self.item_bias[int(i)] = 0.0

        user_ratings: Dict[int, list] = {}
        item_ratings: Dict[int, list] = {}
        for _, row in train_data.iterrows():
            uid, iid, r = int(row["user_id"]), int(row["item_id"]), float(row["rating"])
            user_ratings.setdefault(uid, []).append((iid, r))
            item_ratings.setdefault(iid, []).append((uid, r))

        for _ in range(n_iter):
            for uid, ratings in user_ratings.items():
                s = 0.0
                for iid, r in ratings:
                    s += r - self.global_mean - self.item_bias.get(iid, 0.0)
                self.user_bias[uid] = s / (len(ratings) + reg)

            for iid, ratings in item_ratings.items():
                s = 0.0
                for uid, r in ratings:
                    s += r - self.global_mean - self.user_bias.get(uid, 0.0)
                self.item_bias[iid] = s / (len(ratings) + reg)

        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        preds = np.full(len(test_data), self.global_mean, dtype=np.float32)
        uids = test_data["user_id"].values.astype(int)
        iids = test_data["item_id"].values.astype(int)
        u_bias = np.array([self.user_bias.get(uid, 0.0) for uid in uids], dtype=np.float32)
        i_bias = np.array([self.item_bias.get(iid, 0.0) for iid in iids], dtype=np.float32)
        preds += u_bias + i_bias
        return self._clip_predictions(preds)
