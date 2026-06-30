"""
SVD / NeuMF / DeepFM model wrappers.

These models are dispatched via unified_trainer.train_model/predict_model.
Wrappers serve as containers only — they do not implement their own fit/predict.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from models.base_model import BaseModel


class SVDWrapper(BaseModel):
    """SVD model container — actual training done by unified_trainer.train_model."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._svd = None

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            **kwargs) -> "SVDWrapper":
        from trainers.unified_trainer import train_model
        return train_model(self, "svd", train_data, valid_data, **kwargs)

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("SVD model not fitted. Use train_model() first.")
        from trainers.unified_trainer import predict_model
        return predict_model(self, "svd", test_data, **kwargs)


class NeuMFWrapper(BaseModel):
    """NeuMF model container — actual training done by unified_trainer.train_model."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._neumf = None
        self._neumf_info: Dict = {}

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            **kwargs) -> "NeuMFWrapper":
        from trainers.unified_trainer import train_model
        return train_model(self, "neumf", train_data, valid_data, **kwargs)

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        if self._neumf is None:
            raise RuntimeError("NeuMF model not fitted. Use train_model() first.")
        import torch
        from torch.utils.data import DataLoader
        from trainers.neumf_trainer import NeuMFDataset, collate_neumf

        info = self._neumf_info
        device = next(self._neumf.parameters()).device
        ds = NeuMFDataset(test_data, info["user2idx"], info["item2idx"])
        loader = DataLoader(ds, batch_size=2048, collate_fn=collate_neumf)

        self._neumf.eval()
        preds = []
        with torch.no_grad():
            for u, i, _ in loader:
                u, i = u.to(device), i.to(device)
                p = torch.clamp(self._neumf(u, i), 1.0, 5.0).cpu()
                preds.extend(p.tolist())

        if len(preds) != len(test_data):
            raise RuntimeError(
                f"NeuMF predict length mismatch: expected {len(test_data)}, got {len(preds)}"
            )
        return np.array(preds, dtype=np.float32)


class DeepFMWrapper(BaseModel):
    """DeepFM model container — actual training done by unified_trainer.train_model."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._deepfm = None
        self._deepfm_info: Dict = {}

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_config: Optional[Dict] = None,
            **kwargs) -> "DeepFMWrapper":
        from trainers.unified_trainer import train_model
        return train_model(self, "deepfm", train_data, valid_data, **kwargs)

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        if self._deepfm is None:
            raise RuntimeError("DeepFM model not fitted. Use train_model() first.")
        from trainers.unified_trainer import predict_model
        return predict_model(self, "deepfm", test_data, **kwargs)
