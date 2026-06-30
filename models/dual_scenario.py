"""Dual-Scenario Adaptive Framework — optimized with joint training + vectorized gating."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
from models.base_model import BaseModel
from models.profile_mlp import ProfileMLP


class SoftGatingNet(nn.Module):
    """Simple scalar gating: sigmoid(w * log1p(n) + b)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w * torch.log1p(n) + self.b).squeeze(-1)


class SoftGatingMLP(nn.Module):
    """MLP-based gating network for feature-rich input."""

    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DualScenarioModel(BaseModel):
    """Dual-Scenario model with hard switch or soft gating between profile and behavior streams."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        cfg = config or {}
        self.variant: str = cfg.get("variant", "hard_switch")
        self.tau: int = cfg.get("tau", 15)
        self.gating_variant: str = cfg.get("gating_variant", "mlp")
        self.profile_model: Optional[ProfileMLP] = None
        self.behavior_model: Optional[ProfileMLP] = None
        self.gating_net: Optional[nn.Module] = None
        self.feature_builder = None
        self.profile_input_dim: int = 0
        self.behavior_input_dim: int = 0
        self.hidden_dims: Tuple[int, ...] = (128, 64)
        self.dropout: float = 0.3
        self.user_n_interactions: Dict[int, int] = {}

    def fit(self, train_data, valid_data=None, feature_builder=None, tau=None,
            hidden_dims=(128, 64), dropout=0.3, lr=0.005, epochs=20,
            batch_size=2048, device=None, verbose=True, **kwargs):
        self.feature_builder = feature_builder
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        if tau is not None:
            self.tau = tau
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._compute_global_stats(train_data)
        self.user_n_interactions = {
            int(uid): s.get("interaction_count", 0)
            for uid, s in feature_builder.user_behavior_stats.items()}
        Xp = feature_builder.build_features(train_data, mode="profile")
        Xb = feature_builder.build_features(train_data, mode="behavior", is_train=True)
        self.profile_input_dim = Xp.shape[1]
        self.behavior_input_dim = Xb.shape[1]
        self.profile_model = ProfileMLP(self.profile_input_dim, hidden_dims, dropout).to(device)
        self.behavior_model = ProfileMLP(self.behavior_input_dim, hidden_dims, dropout).to(device)
        if verbose:
            print(f"[DualScenario] variant={self.variant}, p_dim={self.profile_input_dim}, b_dim={self.behavior_input_dim}")
        self._train_joint(train_data, feature_builder, lr, epochs, batch_size, device, verbose)
        if self.variant == "soft_gating":
            self._train_gating(train_data, feature_builder, lr, epochs, batch_size, device, verbose)
        self._is_fitted = True
        return self

    def _train_joint(self, train_data, fb, lr, epochs, batch_size, device, verbose):
        Xp = fb.build_features(train_data, mode="profile")
        Xb = fb.build_features(train_data, mode="behavior", is_train=True)
        y = train_data["rating"].values.astype(np.float32)
        opt_p = optim.Adam(self.profile_model.parameters(), lr=lr)
        opt_b = optim.Adam(self.behavior_model.parameters(), lr=lr)
        crit = nn.MSELoss()
        lp = DataLoader(TensorDataset(torch.tensor(Xp), torch.tensor(y)),
                        batch_size=batch_size, shuffle=True)
        lb = DataLoader(TensorDataset(torch.tensor(Xb), torch.tensor(y)),
                        batch_size=batch_size, shuffle=True)
        self.profile_model.train()
        self.behavior_model.train()
        for ep in range(epochs):
            ep_p, ep_b = 0.0, 0.0
            for (xp, yp), (xb, yb) in zip(lp, lb):
                xp, yp = xp.to(device), yp.to(device)
                xb, yb = xb.to(device), yb.to(device)
                pred = self.profile_model(xp)
                loss = crit(pred, yp)
                opt_p.zero_grad()
                loss.backward()
                opt_p.step()
                ep_p += loss.item()
                pred = self.behavior_model(xb)
                loss = crit(pred, yb)
                opt_b.zero_grad()
                loss.backward()
                opt_b.step()
                ep_b += loss.item()
            if verbose and (ep + 1) % max(1, epochs // 5) == 0:
                n = max(len(lp), 1)
                print(f"  [joint] Epoch {ep+1}/{epochs}: p_loss={ep_p/n:.4f}, b_loss={ep_b/n:.4f}")
        self.profile_model.eval()
        self.behavior_model.eval()

    def _train_gating(self, train_data, fb, lr, epochs, batch_size, device, verbose):
        n_int = np.array([self.user_n_interactions.get(int(uid), 0)
                          for uid in train_data["user_id"]], dtype=np.float32)
        if self.gating_variant == "mlp":
            gf = self._build_gating_features(train_data, n_int)
            self.gating_net = SoftGatingMLP(gf.shape[1]).to(device)
        else:
            gf = n_int
            self.gating_net = SoftGatingNet().to(device)
        opt = optim.Adam([
            {'params': self.profile_model.parameters(), 'lr': lr},
            {'params': self.behavior_model.parameters(), 'lr': lr},
            {'params': self.gating_net.parameters(), 'lr': 0.05},
        ])
        crit = nn.MSELoss()
        Xp = fb.build_features(train_data, mode="profile")
        Xb = fb.build_features(train_data, mode="behavior", is_train=True)
        y = train_data["rating"].values.astype(np.float32)
        self.profile_model.train()
        self.behavior_model.train()
        ds = TensorDataset(torch.tensor(Xp), torch.tensor(Xb),
                           torch.tensor(y), torch.tensor(gf))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        n_epochs = min(max(epochs, 15), 30)
        for ep in range(n_epochs):
            el = 0.0
            for xp, xb, yb, g in dl:
                xp = xp.to(device)
                xb = xb.to(device)
                yb = yb.to(device)
                g = g.to(device)
                a = self.gating_net(g)
                pred = (1 - a) * self.profile_model(xp) + a * self.behavior_model(xb)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                el += loss.item()
            if verbose:
                print(f"  [Gating] Epoch {ep+1}/{n_epochs}: loss={el/max(len(dl),1):.4f}")

    def _build_gating_features(self, train_data, n_int):
        uids = train_data["user_id"].values.astype(int)
        iids = train_data["item_id"].values.astype(int)
        N = len(train_data)
        feats = np.zeros((N, 5), dtype=np.float32)
        feats[:, 0] = np.log1p(n_int)
        feats[:, 1] = np.array([
            self.feature_builder.user_behavior_stats.get(uid, {}).get(
                "mean_rating", self.global_mean)
            for uid in uids], dtype=np.float32)
        feats[:, 2] = np.array([
            self.feature_builder.user_behavior_stats.get(uid, {}).get(
                "rating_std", 0.0)
            for uid in uids], dtype=np.float32)
        feats[:, 3] = np.array([
            self.feature_builder.item_behavior_stats.get(iid, {}).get(
                "mean_rating", self.global_mean)
            for iid in iids], dtype=np.float32)
        feats[:, 4] = np.array([
            self.feature_builder.item_behavior_stats.get(iid, {}).get(
                "rating_std", 0.0)
            for iid in iids], dtype=np.float32)
        return feats

    def predict(self, test_data, **kwargs):
        if self.feature_builder is None:
            raise RuntimeError("Not fitted")
        device = next(self.profile_model.parameters()).device
        Xp = self.feature_builder.build_features(test_data, mode="profile")
        Xb = self.feature_builder.build_features(test_data, mode="behavior")
        n_int = np.array([self.user_n_interactions.get(int(uid), 0)
                          for uid in test_data["user_id"]], dtype=np.float32)
        self.profile_model.eval()
        self.behavior_model.eval()
        with torch.no_grad():
            pp = self.profile_model(torch.tensor(Xp).to(device)).cpu().numpy()
            bp = self.behavior_model(torch.tensor(Xb).to(device)).cpu().numpy()
            if self.variant == "hard_switch":
                preds = np.where(n_int <= self.tau, pp, bp)
            elif self.variant == "soft_gating":
                if self.gating_net is None:
                    preds = (1 - 0.5) * pp + 0.5 * bp
                else:
                    self.gating_net.eval()
                    if self.gating_variant == "mlp":
                        gf = self._build_gating_features(test_data, n_int)
                        a = self.gating_net(
                            torch.tensor(gf).to(device)
                        ).cpu().numpy()
                    else:
                        a = self.gating_net(torch.tensor(n_int).to(device)).cpu().numpy()
                    preds = (1 - a) * pp + a * bp
            elif self.variant == "fixed_weight":
                w = getattr(self, 'fixed_profile_weight', 0.5)
                preds = (1 - w) * pp + w * bp
        return self._clip_predictions(preds)

    def get_gating_curve(self, n_range=(0, 200)):
        nv = np.arange(n_range[0], n_range[1] + 1, dtype=np.float32)
        if self.gating_net is not None:
            device = next(self.gating_net.parameters()).device
            self.gating_net.eval()
            with torch.no_grad():
                if self.gating_variant == "mlp":
                    d = torch.zeros(len(nv), 5, device=device)
                    d[:, 0] = torch.log1p(torch.tensor(nv, device=device))
                    av = self.gating_net(d).cpu().numpy()
                else:
                    av = self.gating_net(torch.tensor(nv).to(device)).cpu().numpy()
        else:
            av = (nv > self.tau).astype(np.float32)
        return nv, av

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = {
            "profile_state": self.profile_model.state_dict(),
            "behavior_state": self.behavior_model.state_dict(),
            "profile_input_dim": self.profile_input_dim,
            "behavior_input_dim": self.behavior_input_dim,
            "variant": self.variant, "tau": self.tau,
            "hidden_dims": list(self.hidden_dims), "dropout": self.dropout,
            "user_n_interactions": self.user_n_interactions,
            "global_mean": self.global_mean,
        }
        if self.gating_net is not None:
            d["gating_state"] = self.gating_net.state_dict()
            d["gating_variant"] = self.gating_variant
        torch.save(d, path)

    def load_checkpoint(self, path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        self.variant = c.get("variant", "hard_switch")
        self.tau = c.get("tau", 15)
        self.hidden_dims = tuple(c.get("hidden_dims", [128, 64]))
        self.dropout = c.get("dropout", 0.3)
        self.user_n_interactions = c.get("user_n_interactions", {})
        self.global_mean = c.get("global_mean", 3.0)
        self.profile_input_dim = c["profile_input_dim"]
        self.behavior_input_dim = c["behavior_input_dim"]
        self.profile_model = ProfileMLP(self.profile_input_dim,
                                         hidden_dims=self.hidden_dims,
                                         dropout=self.dropout)
        self.profile_model.load_state_dict(c["profile_state"])
        self.behavior_model = ProfileMLP(self.behavior_input_dim,
                                          hidden_dims=self.hidden_dims,
                                          dropout=self.dropout)
        self.behavior_model.load_state_dict(c["behavior_state"])
        if "gating_state" in c:
            self.gating_variant = c.get("gating_variant", "simple")
            if self.gating_variant == "mlp":
                self.gating_net = SoftGatingMLP()
            else:
                self.gating_net = SoftGatingNet()
            self.gating_net.load_state_dict(c["gating_state"])
        self._is_fitted = True
        return self
