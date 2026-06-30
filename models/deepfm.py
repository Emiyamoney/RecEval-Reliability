"""
DeepFM — Factorization Machine + Deep Neural Network.

Sparse fields: user_id, item_id, gender, age_bucket, occupation (embeddings)
Dense fields: movie audience vector + user genre preference + group_avg + activity_index

movie_vec:  dynamically aggregated from train_df gender/age columns (5 dims)
user_pref:  dynamically aggregated from train_df genre_* columns (N dims, dataset-dependent)
dense_dim = movie_vec_dim(5) + user_pref_dim(N) + 2
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class FeaturesConfig:
    """Configuration for DeepFM feature dimensions."""

    def __init__(self, n_users: int, n_items: int, use_occupation: bool = True,
                 emb_dim: int = 16, movie_vec_dim: int = 5, user_pref_dim: int = 0):
        self.sparse_fields: List[Tuple[str, int, int]] = [
            ("user_id", n_users + 1, emb_dim),
            ("item_id", n_items + 1, emb_dim),
            ("gender", 3, emb_dim),
            ("age_bucket", 4, emb_dim),
        ]
        if use_occupation:
            self.sparse_fields.append(("occupation", 22, emb_dim))
        self.total_sparse_dim = len(self.sparse_fields) * emb_dim
        self.movie_vec_dim = movie_vec_dim
        self.user_pref_dim = user_pref_dim
        self.dense_dim = movie_vec_dim + user_pref_dim + 2


class DeepFM(nn.Module):
    """DeepFM: FM first-order + FM second-order + Deep network."""

    def __init__(self, feat_config: FeaturesConfig, deep_layers: tuple = (128, 64), dropout: float = 0.3):
        super().__init__()
        self.feat_config = feat_config

        self.embeddings = nn.ModuleList()
        for name, vocab_size, emb_dim in feat_config.sparse_fields:
            self.embeddings.append(nn.Embedding(vocab_size, emb_dim))

        self.linear_sparse = nn.ModuleList()
        for name, vocab_size, emb_dim in feat_config.sparse_fields:
            self.linear_sparse.append(nn.Embedding(vocab_size, 1))
        self.linear_dense = nn.Linear(feat_config.dense_dim, 1, bias=False)

        total_emb_dim = sum(emb_dim for _, _, emb_dim in feat_config.sparse_fields)
        deep_input_dim = total_emb_dim + feat_config.dense_dim
        deep_modules = []
        prev_dim = deep_input_dim
        for h in deep_layers:
            deep_modules.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.deep_net = nn.Sequential(*deep_modules)

        fusion_dim = 1 + prev_dim
        self.output = nn.Linear(fusion_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, sparse_indices: list, dense_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            sparse_indices: list of tensors, each shape [batch].
            dense_features: tensor [batch, dense_dim].
        """
        fm_first = self.linear_dense(dense_features)
        for i, emb in enumerate(self.linear_sparse):
            fm_first = fm_first + emb(sparse_indices[i])

        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(sparse_indices[i]))
        emb_stack = torch.stack(embeddings, dim=1)
        sum_square = emb_stack.sum(dim=1).pow(2)
        square_sum = emb_stack.pow(2).sum(dim=1)
        fm_second = 0.5 * (sum_square - square_sum).sum(dim=-1, keepdim=True)

        emb_concat = torch.cat(embeddings, dim=-1)
        deep_in = torch.cat([emb_concat, dense_features], dim=-1)
        deep_out = self.deep_net(deep_in)

        fm_out = fm_first + fm_second
        fused = torch.cat([fm_out, deep_out], dim=-1)
        return self.output(fused).squeeze(-1)
