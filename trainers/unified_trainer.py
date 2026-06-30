"""
统一训练器 — 封装所有 13 个模型的训练+预测流程

使用 models.registry 作为唯一模型创建入口。
所有模型统一通过 train_model / predict_model 调度。

修复说明:
  - 移除 .rename() 补丁 (列名在 loader 层统一)
  - 移除 DeepFM 静默 fallback (fail fast)
  - 使用统一模型注册表
"""

import sys, os, time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional

from models.base_model import BaseModel
from models.registry import create_model, MODEL_REGISTRY


# ================================================================
# train_model — 统一训练调度
# ================================================================

def train_model(model: BaseModel, model_name: str,
                train_data: pd.DataFrame,
                valid_data: Optional[pd.DataFrame],
                feature_builder=None,
                config: Optional[Dict] = None,
                device=None, verbose: bool = True,
                runtime_cfg: Optional[Dict] = None):
    """
    统一训练入口 — 根据 model_name 分发到正确的训练路径

    Args:
        model:       create_model() 创建的模型实例
        model_name:  模型名 (registry key)
        train_data:  训练集 (snake_case 列名)
        valid_data:  验证集 (可选)
        feature_builder: FeatureBuilder 实例
        config:      训练配置 {epochs, lr, batch_size, ...}
        device:      torch device
        verbose:     是否打印日志

    Returns:
        训练后的 model (in-place 修改)
    """
    cfg = config or {}
    lr = cfg.get("lr", 0.005)
    epochs = cfg.get("epochs", 20)
    batch_size = cfg.get("batch_size", 2048)
    hidden_dims = cfg.get("hidden_dims", [128, 64])
    embedding_dim = cfg.get("embedding_dim", 64)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()

    # ================================================================
    # Baselines (global_mean, user_bias, item_bias, user_item_bias)
    # ================================================================
    if model_name in ("global_mean", "user_bias", "item_bias", "user_item_bias"):
        model.fit(train_data, valid_data)
        return model

    # ================================================================
    # SVD
    # ================================================================
    if model_name == "svd":
        from models.svd_model import SVDModel

        n_users = train_data["user_id"].nunique()
        n_items = train_data["item_id"].nunique()
        svd = SVDModel(n_users, n_items)
        svd.fit(train_data, epochs=epochs, verbose=verbose)

        # Store SVD in model for predict dispatch
        model._svd = svd
        model._svd_n_users = n_users
        model._svd_n_items = n_items
        model._is_fitted = True
        model.global_mean = float(train_data["rating"].mean())
        return model

    # ================================================================
    # NeuMF
    # ================================================================
    if model_name == "neumf":
        from trainers.neumf_trainer import train_neumf

        neumf_model, info = train_neumf(
            train_data, valid_data, device=device,
            epochs=epochs, lr=lr, batch_size=batch_size,
            embedding_dim=embedding_dim,
            mlp_layers=cfg.get("mlp_layers", (64, 32, 16)),
            dropout=cfg.get("dropout", 0.2),
            runtime_cfg=runtime_cfg,
        )
        model._neumf = neumf_model
        model._neumf_info = info
        model._is_fitted = True
        return model

    # ================================================================
    # DeepFM
    # ================================================================
    if model_name == "deepfm":
        from trainers.deepfm_trainer import train_deepfm

        gm = float(train_data["rating"].mean())

        # DeepFM trainer 使用实际数据列，不伪造缺失的 demographic 特征
        # 缺失列由 DeepFMDataset 内部的 UNK embedding (index 0) 处理
        train_d = train_data.copy()
        val_d = valid_data.copy() if valid_data is not None and len(valid_data) > 0 else None

        # 构建空映射 (对于没有 movie_vec / user_pref 的数据集)
        activity_map = {}
        group_stats = {}
        movie_mean_map = {}

        deepfm_model, info = train_deepfm(
            train_d, val_d, activity_map, group_stats, movie_mean_map,
            gm, device=device,
            epochs=epochs, lr=lr, batch_size=batch_size,
            deep_layers=cfg.get("deep_layers", (256, 128, 64)),
            dropout=cfg.get("dropout", 0.3),
            runtime_cfg=runtime_cfg,
        )
        model._deepfm = deepfm_model
        model._deepfm_info = info
        model._is_fitted = True
        model.global_mean = gm
        return model

    # ================================================================
    # LightGCN
    # ================================================================
    if model_name == "lightgcn":
        model.fit(train_data, valid_data,
                  embedding_dim=embedding_dim,
                  n_layers=cfg.get("n_layers", 3),
                  lr=lr, epochs=epochs, batch_size=batch_size,
                  reg=cfg.get("reg", 0.0001),
                  device=device, verbose=verbose)
        return model

    # ================================================================
    # UserCF (no feature_builder needed)
    # ================================================================
    if model_name == "cf_userknn":
        model.fit(train_data, valid_data, verbose=verbose)
        return model

    # ================================================================
    # Neural models (Profile/Behavior/Hybrid/Dual)
    # ================================================================
    fit_kwargs = {
        "feature_builder": feature_builder,
        "hidden_dims": hidden_dims,
        "dropout": cfg.get("dropout", 0.3),
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "verbose": verbose,
        "runtime_cfg": runtime_cfg,
    }
    if model_name in ("dual_hard_switch", "dual_soft_gating"):
        fit_kwargs["tau"] = cfg.get("tau")

    model.fit(train_data, valid_data, **fit_kwargs)
    return model


# ================================================================
# predict_model — 统一预测调度
# ================================================================

def predict_model(model: BaseModel, model_name: str,
                  test_data: pd.DataFrame,
                  feature_builder=None, device=None,
                  runtime_cfg: Optional[Dict] = None) -> np.ndarray:
    """
    统一预测入口 — 根据 model_name 分发到正确的预测路径

    Args:
        model:           训练后的模型实例
        model_name:      模型名
        test_data:       测试集 (snake_case 列名)
        feature_builder: FeatureBuilder 实例
        device:          torch device

    Returns:
        predicted ratings [N] (np.float32)

    Raises:
        RuntimeError: 模型未训练或预测失败 (不复 fallback)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_test = len(test_data)

    # ================================================================
    # SVD
    # ================================================================
    if model_name == "svd":
        if not hasattr(model, '_svd'):
            raise RuntimeError("SVD model not fitted. Call train_model first.")
        model._svd.reset_fallback_counts()
        user_ids = test_data["user_id"].values
        item_ids = test_data["item_id"].values
        preds = []
        for uid_raw, iid_raw in zip(user_ids, item_ids):
            try:
                uid = int(uid_raw)
            except (ValueError, TypeError):
                uid = hash(str(uid_raw)) % 10 ** 8
            try:
                iid = int(iid_raw)
            except (ValueError, TypeError):
                iid = hash(str(iid_raw)) % 10 ** 8
            p = model._svd.predict(uid, iid)
            preds.append(np.clip(p, 1.0, 5.0))
        fb = model._svd.get_fallback_counts()
        print(f"  [SVD predict] fallback: {fb}")
        return np.array(preds, dtype=np.float32)

    # ================================================================
    # NeuMF
    # ================================================================
    if model_name == "neumf":
        if not hasattr(model, '_neumf'):
            raise RuntimeError("NeuMF model not fitted. Call train_model first.")

        from torch.utils.data import DataLoader
        from trainers.neumf_trainer import NeuMFDataset, collate_neumf

        info = model._neumf_info
        ds = NeuMFDataset(test_data, info["user2idx"], info["item2idx"])
        loader = DataLoader(ds, batch_size=2048, collate_fn=collate_neumf)

        model._neumf.eval()
        preds = []
        with torch.no_grad():
            for u, i, _ in loader:
                u, i = u.to(device), i.to(device)
                p = torch.clamp(model._neumf(u, i), 1.0, 5.0).cpu()
                preds.extend(p.tolist())

        if len(preds) != n_test:
            raise RuntimeError(
                f"NeuMF predict output length mismatch: "
                f"expected {n_test}, got {len(preds)}"
            )
        return np.array(preds, dtype=np.float32)

    # ================================================================
    # DeepFM — fail fast, no silent fallback
    # ================================================================
    if model_name == "deepfm":
        if not hasattr(model, '_deepfm'):
            raise RuntimeError("DeepFM model not fitted. Call train_model first.")

        from torch.utils.data import DataLoader
        from trainers.deepfm_trainer import DeepFMDataset, collate_deepfm

        info = model._deepfm_info
        gm = model.global_mean

        # 从 info 获取训练时保存的特征映射 (train 阶段已动态计算)
        movie_map = info.get("movie_map", {})
        user_pref = info.get("user_pref", {})
        movie_vec_dim = info.get("movie_vec_dim", 5)
        user_pref_dim = info.get("user_pref_dim", 0)

        # 使用真实数据列，缺失列由 DeepFMDataset UNK embedding 处理
        test_d = test_data.copy()

        # ---- 冷启动检测：向量化 ----
        user2idx = info["user2idx"]
        item2idx = info["item2idx"]
        user_ids = test_data["user_id"].values.astype(int)
        item_ids = test_data["item_id"].values.astype(int)
        cold_users = set(uid for uid in user_ids if uid not in user2idx)
        cold_items = set(iid for iid in item_ids if iid not in item2idx)

        n_cold_user = sum(1 for uid in user_ids if uid in cold_users)
        n_cold_item = sum(1 for iid in item_ids if iid in cold_items)
        n_cold_both = sum(1 for uid, iid in zip(user_ids, item_ids)
                          if uid in cold_users and iid in cold_items)
        n_warm = n_test - n_cold_user - n_cold_item + n_cold_both
        print(f"  [DeepFM predict] warm={n_warm}, user_cold={n_cold_user}, "
              f"item_cold={n_cold_item}, both_cold={n_cold_both}")

        try:
            ds = DeepFMDataset(
                test_d,
                user2idx, item2idx,
                info["gender2idx"], info["age2idx"],
                info.get("occ2idx", {}),
                movie_map, user_pref,
                {}, {}, {}, gm,
                use_occupation=info.get("use_occupation", True),
                group_avg_mode="train_stats",
                movie_vec_dim=movie_vec_dim, user_pref_dim=user_pref_dim,
            )
            loader = DataLoader(ds, batch_size=2048, collate_fn=collate_deepfm)

            model._deepfm.eval()
            preds = []
            with torch.no_grad():
                for sparse, dense, _ in loader:
                    sparse = [s.to(device) for s in sparse]
                    dense = dense.to(device)
                    p = torch.clamp(model._deepfm(sparse, dense), 1.0, 5.0).cpu()
                    preds.extend(p.tolist())

            if len(preds) != n_test:
                raise RuntimeError(
                    f"DeepFM predict output length mismatch: "
                    f"expected {n_test}, got {len(preds)}"
                )

            preds = np.array(preds, dtype=np.float32)

            # ---- 统计冷启动比例 (不使用 global_mean 覆盖) ----
            fallback_stats = {
                "warm": n_warm,
                "user_cold": n_cold_user,
                "item_cold": n_cold_item,
                "both_cold": n_cold_both,
                "global_mean_override": 0,
            }
            print(f"  [DeepFM predict] fallback: {fallback_stats}")

            return preds

        except Exception as e:
            raise RuntimeError(
                f"DeepFM predict failed: {e}\n"
                f"  n_test={n_test}, movie_vec_dim={movie_vec_dim}, "
                f"user_pref_dim={user_pref_dim}\n"
                f"  This is a REAL error, NOT a cold-start fallback scenario. "
                f"Fix the root cause."
            ) from e

    # ================================================================
    # All other models (LightGCN, Profile/Behavior/Hybrid/Dual)
    # ================================================================
    return model.predict(test_data)
