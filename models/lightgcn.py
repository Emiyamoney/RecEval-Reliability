"""
LightGCN — graph convolutional collaborative filtering (paper-level implementation).

Core components:
  1. Build user-item bipartite graph from train_data (scipy.sparse)
  2. Normalized adjacency: D^{-1/2} A D^{-1/2} (standard LightGCN form)
  3. K-layer propagation: E^{(k+1)} = (D^{-1/2} A D^{-1/2}) E^{(k)}, no transforms, no nonlinearity
  4. Final embedding = mean(E^{(0)}, E^{(1)}, ..., E^{(K)})
  5. Explicit rating prediction: MSE loss + dot product
  6. Cold-start: fallback only for unseen users/items, warm-start uses main path

Leakage control:
  - Graph built from train_data only
  - valid/test interactions never enter the graph
  - Prediction stats: warm_start / user_cold / item_cold / both_cold counts
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
from collections import defaultdict

from scipy.sparse import coo_matrix, csr_matrix
from models.base_model import BaseModel


class LightGCNModel(nn.Module):
    """LightGCN graph convolution network with normalized propagation."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_users + n_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

        self.register_buffer("norm_adj_indices", None)
        self.register_buffer("norm_adj_values", None)
        self.graph_ready: bool = False

    def set_graph(self, norm_adj: csr_matrix, device: torch.device) -> None:
        """Convert scipy CSR normalized adjacency matrix to PyTorch sparse tensor."""
        coo = norm_adj.tocoo()
        indices = torch.stack([
            torch.tensor(coo.row, dtype=torch.long),
            torch.tensor(coo.col, dtype=torch.long),
        ], dim=0)
        values = torch.tensor(coo.data, dtype=torch.float32)

        self.register_buffer("norm_adj_indices", indices.to(device))
        self.register_buffer("norm_adj_values", values.to(device))
        self.graph_ready = True

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute K-layer graph propagation, returning (final_user_emb, final_item_emb).

        E^{(k+1)} = (D^{-1/2} A D^{-1/2}) E^{(k)}
        Final E = mean(E^{(0)}, ..., E^{(K)})
        """
        if not self.graph_ready:
            raise RuntimeError("Graph not set. Call set_graph() first.")

        device = self.embedding.weight.device
        n_total = self.n_users + self.n_items
        size = torch.Size([n_total, n_total])

        ego_embeddings = self.embedding.weight
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            sparse_adj = torch.sparse_coo_tensor(
                self.norm_adj_indices, self.norm_adj_values, size
            ).coalesce()
            ego_embeddings = torch.sparse.mm(sparse_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)

        user_emb = final_embeddings[:self.n_users]
        item_emb = final_embeddings[self.n_users:]
        return user_emb, item_emb

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor,
                user_emb: Optional[torch.Tensor] = None,
                item_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict ratings via dot product of final embeddings.

        Args:
            user_idx: [B] user indices in graph (0..n_users-1).
            item_idx: [B] item indices in graph (0..n_items-1).
            user_emb: Pre-propagated user embeddings (optional, for speed).
            item_emb: Pre-propagated item embeddings (optional, for speed).
        """
        if user_emb is None or item_emb is None:
            user_emb, item_emb = self.propagate()

        u = user_emb[user_idx]
        i = item_emb[item_idx]
        return (u * i).sum(dim=-1)


class LightGCN(BaseModel):
    """LightGCN graph convolution recommendation model.

    Training:
      - Build bipartite graph from train_data only
      - Normalized adjacency matrix propagation
      - MSE loss for end-to-end embedding optimization

    Prediction:
      - Warm-start: graph convolution embedding dot product
      - User cold-start: item_bias fallback
      - Item cold-start: global_mean fallback
      - Both cold-start: global_mean
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model: Optional[LightGCNModel] = None
        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.n_users: int = 0
        self.n_items: int = 0

        self._cached_user_emb: Optional[torch.Tensor] = None
        self._cached_item_emb: Optional[torch.Tensor] = None

        self._cold_stats: Dict[str, int] = {}

    @staticmethod
    def _safe_int(val) -> int:
        """Safely convert value to integer."""
        try:
            return int(val)
        except (ValueError, TypeError):
            return hash(str(val)) % (10 ** 9)

    def fit(self, train_data: pd.DataFrame,
            valid_data: Optional[pd.DataFrame] = None,
            embedding_dim: int = 64, n_layers: int = 3,
            lr: float = 0.001, reg: float = 0.0001,
            epochs: int = 50, batch_size: int = 4096,
            device=None, verbose: bool = True,
            **kwargs) -> "LightGCN":
        """Train LightGCN model.

        Steps:
          1. Build user/item index mappings
          2. Build train-only bipartite graph
          3. Compute normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
          4. End-to-end train embeddings (MSE loss)
          5. Cache propagated embeddings
        """
        self._compute_global_stats(train_data)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.time()

        all_users = sorted(train_data["user_id"].unique())
        all_items = sorted(train_data["item_id"].unique())
        self.user2idx = {self._safe_int(u): i for i, u in enumerate(all_users)}
        self.item2idx = {self._safe_int(i): j for j, i in enumerate(all_items)}
        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)

        n_edges = len(train_data)

        if verbose:
            print(f"[LightGCN] Graph: {self.n_users} users, {self.n_items} items, "
                  f"{n_edges} edges | emb={embedding_dim}, layers={n_layers}")

        user_ids = train_data["user_id"].values
        item_ids = train_data["item_id"].values
        u_idx = np.array([self.user2idx[self._safe_int(u)] for u in user_ids])
        i_idx = np.array([self.item2idx[self._safe_int(i)] for i in item_ids])

        rows = np.concatenate([u_idx, i_idx + self.n_users])
        cols = np.concatenate([i_idx + self.n_users, u_idx])
        data_vals = np.ones(len(rows), dtype=np.float32)
        A = coo_matrix((data_vals, (rows, cols)),
                       shape=(self.n_users + self.n_items, self.n_users + self.n_items),
                       dtype=np.float32).tocsr()

        degrees = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)

        D_inv_sqrt = csr_matrix(
            (deg_inv_sqrt, (np.arange(len(deg_inv_sqrt)), np.arange(len(deg_inv_sqrt)))),
            shape=A.shape, dtype=np.float32
        )
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        nnz = A_norm.nnz
        density = nnz / (A_norm.shape[0] * A_norm.shape[1])
        if verbose:
            print(f"[LightGCN] NormAdj: {nnz} nonzeros, density={density:.6f}")

        self.model = LightGCNModel(
            self.n_users, self.n_items, embedding_dim, n_layers
        ).to(device)
        self.model.set_graph(A_norm, device)

        user_indices = torch.tensor(
            [self.user2idx[self._safe_int(uid)] for uid in train_data["user_id"]],
            dtype=torch.long
        )
        item_indices = torch.tensor(
            [self.item2idx[self._safe_int(iid)] for iid in train_data["item_id"]],
            dtype=torch.long
        )
        ratings = torch.tensor(train_data["rating"].values, dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(user_indices, item_indices, ratings)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.model.train()
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for u, i, r in train_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                user_emb, item_emb = self.model.propagate()
                pred = self.model(u, i, user_emb=user_emb, item_emb=item_emb)
                loss = criterion(pred, r)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)

            val_str = ""
            if valid_data is not None and len(valid_data) > 0:
                with torch.no_grad():
                    val_user_emb, val_item_emb = user_emb.detach(), item_emb.detach()
                val_loss = self._eval_loss(valid_data, device, batch_size, criterion,
                                           cached_user_emb=val_user_emb, cached_item_emb=val_item_emb)
                val_str = f", val_loss={val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}

            with torch.no_grad():
                emb_norm = self.model.embedding.weight.norm().item()

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}"
                      f"{val_str}, |E|={emb_norm:.2f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"  Best val loss: {best_val_loss:.4f}")

        self.model.eval()
        with torch.no_grad():
            self._cached_user_emb, self._cached_item_emb = self.model.propagate()

        item_means = train_data.groupby("item_id")["rating"].mean()
        for iid, mean_r in item_means.items():
            self.item_bias[self._safe_int(iid)] = float(mean_r - self.global_mean)

        self._is_fitted = True
        elapsed = time.time() - t0
        if verbose:
            print(f"  LightGCN done: time={elapsed:.0f}s, "
                  f"graph_nodes={self.n_users+self.n_items}, edges={n_edges*2}")

        return self

    def _eval_loss(self, df: pd.DataFrame, device: torch.device,
                   batch_size: int, criterion,
                   cached_user_emb: Optional[torch.Tensor] = None,
                   cached_item_emb: Optional[torch.Tensor] = None) -> float:
        """Compute MSE loss on validation set."""
        user_indices = torch.tensor(
            [self.user2idx.get(self._safe_int(uid), -1) for uid in df["user_id"]],
            dtype=torch.long
        )
        item_indices = torch.tensor(
            [self.item2idx.get(self._safe_int(iid), -1) for iid in df["item_id"]],
            dtype=torch.long
        )
        ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        valid_mask = (user_indices >= 0) & (item_indices >= 0)
        if not valid_mask.all():
            user_indices = user_indices[valid_mask]
            item_indices = item_indices[valid_mask]
            ratings = ratings[valid_mask]

        if len(user_indices) == 0:
            return float("inf")

        ds = TensorDataset(user_indices, item_indices, ratings)
        loader = DataLoader(ds, batch_size=batch_size)

        self.model.eval()
        if cached_user_emb is not None and cached_item_emb is not None:
            user_emb, item_emb = cached_user_emb, cached_item_emb
        else:
            with torch.no_grad():
                user_emb, item_emb = self.model.propagate()
        total = 0.0
        n = 0
        with torch.no_grad():
            for u, i, r in loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = self.model(u, i, user_emb=user_emb, item_emb=item_emb)
                total += criterion(pred, r).item() * u.size(0)
                n += u.size(0)
        return total / max(n, 1)

    def predict(self, test_data: pd.DataFrame, batch_size: int = 4096,
                **kwargs) -> np.ndarray:
        """Predict ratings (vectorized).

        Warm-start (both user+item in graph): graph convolution dot product
        User cold-start (user not in graph):  item_bias fallback
        Item cold-start (item not in graph):  global_mean fallback
        Both cold-start:                     global_mean
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = self._cached_user_emb.device
        self.model.eval()

        n_total = len(test_data)
        preds = np.full(n_total, self.global_mean, dtype=np.float32)
        stats: Dict[str, int] = defaultdict(int)

        with torch.no_grad():
            user_emb, item_emb = self.model.propagate()

        user_ids = test_data["user_id"].values
        item_ids = test_data["item_id"].values
        u_idx = np.array([self.user2idx.get(self._safe_int(u), -1) for u in user_ids])
        i_idx = np.array([self.item2idx.get(self._safe_int(i), -1) for i in item_ids])

        warm_mask = (u_idx >= 0) & (i_idx >= 0)
        user_cold_mask = (u_idx < 0) & (i_idx >= 0)
        item_cold_mask = (u_idx >= 0) & (i_idx < 0)
        both_cold_mask = (u_idx < 0) & (i_idx < 0)

        stats["warm_start"] = int(warm_mask.sum())
        stats["user_cold"] = int(user_cold_mask.sum())
        stats["item_cold"] = int(item_cold_mask.sum())
        stats["both_cold"] = int(both_cold_mask.sum())

        for mask, label in [(user_cold_mask, "user_cold"), (item_cold_mask, "item_cold"), (both_cold_mask, "both_cold")]:
            if mask.sum() == 0:
                continue
            indices = np.where(mask)[0]
            for idx in indices:
                uid = self._safe_int(user_ids[idx])
                iid = self._safe_int(item_ids[idx])
                if label == "user_cold":
                    preds[idx] = self.global_mean + self.item_bias.get(iid, 0.0)
                elif label == "item_cold":
                    preds[idx] = self.global_mean + self.user_bias.get(uid, 0.0) + self.item_bias.get(iid, 0.0)
                else:
                    preds[idx] = self.global_mean + self.user_bias.get(uid, 0.0) + self.item_bias.get(iid, 0.0)

        if warm_mask.sum() > 0:
            warm_u = torch.tensor(u_idx[warm_mask], dtype=torch.long).to(device)
            warm_i = torch.tensor(i_idx[warm_mask], dtype=torch.long).to(device)

            all_warm_preds = []
            for start in range(0, len(warm_u), batch_size):
                end = min(start + batch_size, len(warm_u))
                with torch.no_grad():
                    p = self.model(warm_u[start:end], warm_i[start:end],
                                   user_emb=user_emb, item_emb=item_emb).cpu().numpy()
                all_warm_preds.append(p)

            preds[warm_mask] = np.concatenate(all_warm_preds)

        self._cold_stats = dict(stats)
        if kwargs.get("verbose", True):
            print(f"  [LightGCN predict] warm={stats['warm_start']}, "
                  f"user_cold={stats['user_cold']}, "
                  f"item_cold={stats['item_cold']}, "
                  f"both_cold={stats['both_cold']}")

        return self._clip_predictions(preds)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.model.embedding_dim,
            "n_layers": self.model.n_layers,
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "global_mean": self.global_mean,
            "item_bias": self.item_bias,
            "_cold_stats": self._cold_stats,
            "norm_adj_indices": self.model.norm_adj_indices,
            "norm_adj_values": self.model.norm_adj_values,
        }, path)

    def load_checkpoint(self, path: str) -> "LightGCN":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.n_users = ckpt["n_users"]
        self.n_items = ckpt["n_items"]
        self.user2idx = ckpt["user2idx"]
        self.item2idx = ckpt["item2idx"]
        self.global_mean = ckpt["global_mean"]
        self.item_bias = ckpt.get("item_bias", {})
        self._cold_stats = ckpt.get("_cold_stats", {})

        emb_dim = ckpt["embedding_dim"]
        n_layers = ckpt["n_layers"]
        self.model = LightGCNModel(self.n_users, self.n_items, emb_dim, n_layers)
        self.model.load_state_dict(ckpt["state_dict"])

        if ckpt.get("norm_adj_indices") is not None:
            self.model.norm_adj_indices = ckpt["norm_adj_indices"]
            self.model.norm_adj_values = ckpt["norm_adj_values"]
            self.model.graph_ready = True

        self._is_fitted = True
        if self.model.graph_ready:
            self.model.eval()
            with torch.no_grad():
                self._cached_user_emb, self._cached_item_emb = self.model.propagate()
        return self

    def get_cold_stats(self) -> Dict[str, int]:
        """Return cold-start statistics from the most recent predict call."""
        return dict(self._cold_stats)
