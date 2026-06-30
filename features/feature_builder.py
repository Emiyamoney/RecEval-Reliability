"""
Feature engineering builder v2 — dataset-agnostic thin wrapper with feature caching.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class FeatureBuilder:
    """Unified feature builder wrapping dataset-specific adapters."""

    def __init__(self, adapter=None):
        self.adapter = adapter
        self._fitted = False
        self.global_mean: float = 3.0
        self._cache: Dict = {}

    @staticmethod
    def _safe_id(val) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return hash(str(val)) % (10**9)

    def fit(self, train_df: pd.DataFrame, adapter=None, dataset_config=None) -> "FeatureBuilder":
        if adapter is not None:
            self.adapter = adapter
        if self.adapter is None:
            raise RuntimeError("FeatureBuilder.fit: adapter required")
        self.adapter.fit(train_df)
        self.global_mean = self.adapter.global_mean
        self._fitted = True
        fa = self.adapter.get_feature_availability()
        print(f"[FeatureBuilder] Fitted: dataset={fa['dataset']}, profile_dim={fa['profile_dim']}, behavior_dim={fa['behavior_dim']}")
        return self

    def build_features(self, df: pd.DataFrame, mode: str = "combined",
                       is_train: bool = False, svd_model=None,
                       use_cache: bool = True) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder not fitted")
        cache_key = (mode, is_train, id(df))
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        if mode == "profile":
            result = self.adapter.build_profile_features(df)
        elif mode == "behavior":
            result = self.adapter.build_behavior_features(df, is_train=is_train)
        elif mode == "combined":
            p = self.adapter.build_profile_features(df)
            b = self.adapter.build_behavior_features(df, is_train=is_train)
            result = np.concatenate([p, b], axis=1) if p.shape[1] > 0 and b.shape[1] > 0 else (p if p.shape[1] > 0 else b)
            if svd_model is not None:
                cf = self.adapter.build_cf_features(df, svd_model)
                result = np.concatenate([result, cf], axis=1) if result.shape[1] > 0 else cf
            if result.shape[1] == 0:
                result = np.zeros((len(df), 1), dtype=np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if use_cache:
            self._cache[cache_key] = result
        return result

    @property
    def user_behavior_stats(self) -> Dict:
        return self.adapter.user_stats if self.adapter else {}

    @property
    def item_behavior_stats(self) -> Dict:
        return self.adapter.item_stats if self.adapter else {}

    @property
    def profile_dim(self) -> int:
        return self.adapter.schema.profile_dim if self.adapter else 0

    @property
    def behavior_dim(self) -> int:
        return self.adapter.schema.behavior_dim if self.adapter else 0

    def get_feature_availability(self) -> Dict:
        return self.adapter.get_feature_availability() if self.adapter else {}
