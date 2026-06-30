"""
NeuMF (Neural Matrix Factorization) — collaborative filtering neural baseline.

GMF branch + MLP branch dual-tower fusion, using UserID/MovieID only.
"""

import torch
import torch.nn as nn


class NeuMF(nn.Module):
    """Neural Matrix Factorization with GMF + MLP branches."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32,
                 mlp_layers: tuple = (64, 32, 16), dropout: float = 0.2):
        super().__init__()
        self.user_emb_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(n_items, embedding_dim)

        self.user_emb_mlp = nn.Embedding(n_users, mlp_layers[0])
        self.item_emb_mlp = nn.Embedding(n_items, mlp_layers[0])

        mlp_dims = [mlp_layers[0] * 2] + list(mlp_layers[1:])
        mlp_modules = []
        for i in range(len(mlp_dims) - 1):
            mlp_modules.extend([
                nn.Linear(mlp_dims[i], mlp_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*mlp_modules)

        fusion_dim = embedding_dim + mlp_layers[-1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_gmf = self.user_emb_gmf(user_idx)
        i_gmf = self.item_emb_gmf(item_idx)
        gmf_out = u_gmf * i_gmf

        u_mlp = self.user_emb_mlp(user_idx)
        i_mlp = self.item_emb_mlp(item_idx)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.fusion(fused).squeeze(-1)
