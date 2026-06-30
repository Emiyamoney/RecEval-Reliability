"""
Unified model base class — interface all models must implement.

fit(train_data, valid_data=None, **kwargs) -> self
predict(test_data) -> np.ndarray
evaluate(test_data) -> dict
save_checkpoint(path)
load_checkpoint(path)
"""

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Unified prediction result container with fallback stats and warnings."""
    predictions: np.ndarray
    fallback_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Base class for all recommendation models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._is_fitted = False
        self.global_mean: float = 3.0
        self.user_bias: Dict[int, float] = {}
        self.item_bias: Dict[int, float] = {}

    @abstractmethod
    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            **kwargs) -> "BaseModel":
        """Train the model.

        Args:
            train_data: Training DataFrame with user_id, item_id, rating, and feature columns.
            valid_data: Validation DataFrame (optional).
            feature_config: Feature configuration dict.

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict(self, test_data: pd.DataFrame,
                **kwargs) -> np.ndarray:
        """Predict ratings.

        Args:
            test_data: Test DataFrame.

        Returns:
            Predicted ratings array of shape [N].
        """
        ...

    def evaluate(self, test_data: pd.DataFrame,
                 metrics: Optional[List[str]] = None,
                 **kwargs) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_data: Test DataFrame.
            metrics: List of metric names ['rmse', 'mae', 'mse'].

        Returns:
            Dict mapping metric name to value.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        preds = self.predict(test_data, **kwargs)
        trues = test_data["rating"].values.astype(np.float32)

        if metrics is None:
            metrics = ["rmse", "mae", "mse"]

        results: Dict[str, float] = {}
        for m in metrics:
            if m == "rmse":
                results["rmse"] = float(np.sqrt(mean_squared_error(trues, preds)))
            elif m == "mae":
                results["mae"] = float(mean_absolute_error(trues, preds))
            elif m == "mse":
                results["mse"] = float(mean_squared_error(trues, preds))

        results["n_samples"] = len(preds)
        return results

    def evaluate_grouped(self, test_data: pd.DataFrame,
                         group_col: str = "activity_group",
                         metrics: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, Dict[str, float]]:
        """Evaluate grouped by activity (cold/warm/medium)."""
        results: Dict[str, Dict[str, float]] = {}
        for grp_name, grp_df in test_data.groupby(group_col):
            results[str(grp_name)] = self.evaluate(grp_df, metrics, **kwargs)
        return results

    def save_checkpoint(self, path: str) -> None:
        """Save model to disk via pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_checkpoint(path: str) -> "BaseModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def is_fitted(self) -> bool:
        return self._is_fitted

    def _compute_global_stats(self, train_data: pd.DataFrame) -> None:
        """Compute global statistics from training data (no leakage)."""
        self.global_mean = float(train_data["rating"].mean())
        self.rating_min = float(train_data["rating"].min())
        self.rating_max = float(train_data["rating"].max())

    def _clip_predictions(self, preds: np.ndarray,
                          lo: Optional[float] = None,
                          hi: Optional[float] = None) -> np.ndarray:
        """Clip predictions to rating range (auto-detect from train data)."""
        lo = lo if lo is not None else getattr(self, "rating_min", 1.0)
        hi = hi if hi is not None else getattr(self, "rating_max", 5.0)
        return np.clip(preds, lo, hi)

    def _get_user_fallback(self, user_id: int) -> float:
        """Fallback prediction for unseen users."""
        return self.user_bias.get(user_id, 0.0) + self.global_mean

    def _get_item_fallback(self, item_id: int) -> float:
        """Fallback prediction for unseen items."""
        return self.item_bias.get(item_id, 0.0) + self.global_mean
