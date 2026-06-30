"""
Hybrid Model — Profile + Behavior + Collaborative Embeddings.

Indiscriminate fusion model used to validate the stable-but-not-optimal hypothesis.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
from models.base_model import BaseModel


class HybridMLP(nn.Module):
    """Hybrid MLP: profile + behavior + CF."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HybridModel(BaseModel):
    """Hybrid: Profile + Behavior + CF embeddings."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model: Optional[HybridMLP] = None
        self.feature_builder = None
        self.input_dim: int = 0

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_builder=None,
            svd_model=None,
            hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.3,
            lr: float = 0.005, epochs: int = 30, batch_size: int = 2048,
            device=None, verbose: bool = True,
            **kwargs) -> "HybridModel":
        self.feature_builder = feature_builder
        self._compute_global_stats(train_data)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train = feature_builder.build_features(
            train_data, mode="combined", is_train=True, svd_model=svd_model
        )
        y_train = train_data["rating"].values.astype(np.float32)
        self.input_dim = X_train.shape[1]

        self.model = HybridMLP(self.input_dim, hidden_dims, dropout).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if verbose:
            print(f"[Hybrid] input_dim={self.input_dim}, epochs={epochs}")

        self.model.train()
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)

            if valid_data is not None and len(valid_data) > 0:
                X_val = feature_builder.build_features(
                    valid_data, mode="combined", svd_model=svd_model
                )
                y_val = valid_data["rating"].values.astype(np.float32)
                val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
                val_loader = DataLoader(val_ds, batch_size=batch_size)
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y_true in val_loader:
                        x, y_true = x.to(device), y_true.to(device)
                        val_loss += criterion(self.model(x), y_true).item() * x.size(0)
                val_loss /= len(valid_data)
                self.model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"  Best val loss: {best_val_loss:.4f}")

        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame, svd_model=None, **kwargs) -> np.ndarray:
        if self.model is None or self.feature_builder is None:
            raise RuntimeError("Model not fitted")

        device = next(self.model.parameters()).device
        X_test = self.feature_builder.build_features(
            test_data, mode="combined", svd_model=svd_model
        )

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test).to(device)).cpu().numpy()

        return self._clip_predictions(preds)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "global_mean": self.global_mean,
            "hidden_dims": list(self.hidden_dims) if hasattr(self, 'hidden_dims') else [256, 128, 64],
            "dropout": self.dropout if hasattr(self, 'dropout') else 0.3,
        }, path)

    def load_checkpoint(self, path: str) -> "HybridModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self.global_mean = ckpt.get("global_mean", 3.0)
        hidden_dims = ckpt.get("hidden_dims", [256, 128, 64])
        dropout = ckpt.get("dropout", 0.3)
        self.model = HybridMLP(self.input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.model.load_state_dict(ckpt["state_dict"])
        self._is_fitted = True
        return self
