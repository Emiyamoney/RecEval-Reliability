"""
Feature-Subset MLP — generic feature-subset MLP base + Profile/Behavior consumers.

Eliminates duplication between ProfileOnlyModel and BehaviorOnlyModel.
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


class ProfileMLP(nn.Module):
    """Generic MLP: Linear -> BN -> ReLU -> Dropout -> ... -> Linear(1)"""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), dropout: float = 0.3):
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


class FeatureSubsetMLP(BaseModel):
    """Generic feature-subset MLP: uses different feature groups based on mode (profile/behavior/combined)."""

    def __init__(self, config: Optional[Dict] = None, mode: str = "profile"):
        super().__init__(config)
        self.mode = mode
        self.model: Optional[ProfileMLP] = None
        self.feature_builder = None
        self.input_dim: int = 0
        self.hidden_dims: Tuple[int, ...] = (128, 64)
        self.dropout: float = 0.3

    def _build_features(self, df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        if self.mode == "behavior":
            return self.feature_builder.build_features(df, mode="behavior", is_train=is_train)
        elif self.mode == "profile":
            return self.feature_builder.build_features(df, mode="profile")
        elif self.mode == "combined":
            return self.feature_builder.build_features(df, mode="combined", is_train=is_train)
        raise ValueError(f"Unknown mode: {self.mode}")

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            feature_builder=None,
            hidden_dims: Tuple[int, ...] = (128, 64),
            dropout: float = 0.3,
            lr: float = 0.005, epochs: int = 20, batch_size: int = 2048,
            device=None, verbose: bool = True,
            **kwargs) -> "FeatureSubsetMLP":
        self.feature_builder = feature_builder
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self._compute_global_stats(train_data)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train = self._build_features(train_data, is_train=True)
        y_train = train_data["rating"].values.astype(np.float32)
        self.input_dim = X_train.shape[1]

        if self.input_dim == 0:
            raise ValueError(
                f"{self.__class__.__name__}.fit: {self.mode} input_dim=0. "
                f"Dataset has no {self.mode} features."
            )

        self.model = ProfileMLP(self.input_dim, hidden_dims, dropout).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if verbose:
            print(f"[{self.__class__.__name__}] input_dim={self.input_dim}, epochs={epochs}")

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
                val_loss = self._eval_loss(valid_data, device, batch_size)
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

    def predict(self, test_data: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.model is None or self.feature_builder is None:
            raise RuntimeError("Model not fitted")

        device = next(self.model.parameters()).device
        X_test = self._build_features(test_data)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test).to(device)).cpu().numpy()

        return self._clip_predictions(preds)

    def _eval_loss(self, df: pd.DataFrame, device: torch.device, batch_size: int) -> float:
        X = self._build_features(df)
        y = df["rating"].values.astype(np.float32)
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(ds, batch_size=batch_size)
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y_true in loader:
                x, y_true = x.to(device), y_true.to(device)
                total += nn.MSELoss()(self.model(x), y_true).item() * x.size(0)
        self.model.train()
        return total / max(len(df), 1)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "global_mean": self.global_mean,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "mode": self.mode,
        }, path)

    def load_checkpoint(self, path: str) -> "FeatureSubsetMLP":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self.global_mean = ckpt.get("global_mean", 3.0)
        self.hidden_dims = tuple(ckpt.get("hidden_dims", [128, 64]))
        self.dropout = ckpt.get("dropout", 0.3)
        self.model = ProfileMLP(self.input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout)
        self.model.load_state_dict(ckpt["state_dict"])
        self._is_fitted = True
        return self


class ProfileOnlyModel(FeatureSubsetMLP):
    """Profile-Only: uses demographic + content features only."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config, mode="profile")


class BehaviorOnlyModel(FeatureSubsetMLP):
    """Behavior-Only: uses interaction features only."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config, mode="behavior")
