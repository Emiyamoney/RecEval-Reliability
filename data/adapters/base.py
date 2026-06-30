"""
数据集适配器基类 — 每个数据集实现自己的特征工程

每个数据集有独立的:
  - 加载逻辑 (load)
  - Profile 特征列和构建方式
  - Behavior 特征列和构建方式
  - 特征 schema 元数据

FeatureBuilder 通过适配器获取特征，不再硬编码列名。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class FeatureSchema:
    """数据集特征 schema — 描述可用特征和维度"""
    dataset_name: str = ""
    
    # Profile 特征
    profile_columns: List[str] = field(default_factory=list)
    profile_dim: int = 0
    has_demographic: bool = False
    
    # Behavior 特征
    behavior_columns: List[str] = field(default_factory=list)
    behavior_dim: int = 0
    
    # Item 特征
    item_feature_columns: List[str] = field(default_factory=list)
    
    # 缺失列 (该数据集不支持的常见特征)
    missing_columns: List[str] = field(default_factory=list)
    
    # 冷启动支持
    supports_cold_start: bool = True
    supports_warm_start: bool = True
    
    # 评分范围
    rating_min: float = 1.0
    rating_max: float = 5.0


class BaseDatasetAdapter(ABC):
    """数据集适配器基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.schema: FeatureSchema = FeatureSchema()
        self._fitted = False
        self.global_mean: float = 3.0
        
        # 行为统计 (从训练集计算)
        self.user_stats: Dict[int, Dict[str, float]] = {}
        self.item_stats: Dict[int, Dict[str, float]] = {}
        
        # 编码器
        self.categorical_encoders: Dict[str, Dict[str, int]] = {}
        self.scalers: Dict[str, Dict[str, float]] = {}

    # ================================================================
    # 必须实现
    # ================================================================

    @abstractmethod
    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        """加载原始数据，返回统一 DataFrame (user_id, item_id, rating + 特征列)"""
        ...

    @abstractmethod
    def get_dataset_name(self) -> str:
        """数据集名称"""
        ...

    @abstractmethod
    def get_profile_columns(self) -> List[str]:
        """本数据集的 profile 特征列名"""
        ...

    @abstractmethod
    def get_behavior_feature_names(self) -> List[str]:
        """本数据集的 behavior 特征名称"""
        ...

    # ================================================================
    # 可选覆盖 — 数据集特有特征工程
    # ================================================================

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据集特有预处理 (在 split 之前调用)"""
        return df

    def enrich_features(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """数据集特有特征增强 (如加载外部 CSV、构建交互特征)"""
        return df

    # ================================================================
    # Fit — 从训练集学习统计量
    # ================================================================

    def fit(self, train_df: pd.DataFrame):
        """在训练集上拟合所有统计量"""
        self.global_mean = float(train_df["rating"].mean())
        self.schema.rating_min = float(train_df["rating"].min())
        self.schema.rating_max = float(train_df["rating"].max())
        
        # 1. 拟合编码器
        self._fit_encoders(train_df)
        
        # 2. 计算行为统计
        self._compute_behavior_stats(train_df)
        
        # 3. 计算特征维度
        self.schema.profile_dim = self._compute_profile_dim(train_df)
        self.schema.behavior_dim = self._compute_behavior_dim()
        self.schema.profile_columns = self.get_profile_columns()
        self.schema.behavior_columns = self.get_behavior_feature_names()
        self.schema.missing_columns = self._detect_missing_columns(train_df)
        self.schema.dataset_name = self.get_dataset_name()
        
        self._fitted = True
        return self

    def _fit_encoders(self, train_df: pd.DataFrame):
        """为 categorical 列建立 vocab"""
        for col in self.get_profile_columns():
            if col in train_df.columns and train_df[col].dtype == object:
                vals = train_df[col].dropna().unique()
                self.categorical_encoders[col] = {
                    str(v): i + 1 for i, v in enumerate(sorted(vals))
                }

    def _compute_behavior_stats(self, train_df: pd.DataFrame):
        """计算用户/物品行为统计 (仅从训练集)"""
        # User stats
        for uid, grp in train_df.groupby("user_id"):
            uid = int(uid)
            ratings = grp["rating"].values
            self.user_stats[uid] = {
                "interaction_count": float(len(grp)),
                "mean_rating": float(ratings.mean()),
                "rating_std": float(ratings.std()) if len(ratings) > 1 else 0.0,
            }
        
        # Item stats
        for iid, grp in train_df.groupby("item_id"):
            iid = int(iid)
            ratings = grp["rating"].values
            self.item_stats[iid] = {
                "interaction_count": float(len(grp)),
                "mean_rating": float(ratings.mean()),
                "rating_std": float(ratings.std()) if len(ratings) > 1 else 0.0,
            }
        
        # 全局归一化参数
        all_counts = [s["interaction_count"] for s in self.user_stats.values()]
        if all_counts:
            self.scalers["user_interaction_count"] = {
                "mean": float(np.mean(all_counts)),
                "std": float(np.std(all_counts)) or 1.0,
            }

    def _compute_profile_dim(self, train_df: pd.DataFrame) -> int:
        """计算 profile 特征维度 — 子类可覆盖"""
        return len(self.get_profile_columns())

    def _compute_behavior_dim(self) -> int:
        """计算 behavior 特征维度"""
        return len(self.get_behavior_feature_names())

    def _detect_missing_columns(self, train_df: pd.DataFrame) -> List[str]:
        """检测常见但缺失的列"""
        common = ["gender", "age", "age_bucket", "occupation"]
        return [c for c in common if c not in train_df.columns]

    # ================================================================
    # 特征构建 — 数据集无关接口
    # ================================================================

    def build_profile_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        构建 profile 特征矩阵 [N, profile_dim]
        子类必须实现，因为每个数据集的 profile 列完全不同
        """
        raise NotImplementedError("Subclass must implement build_profile_features()")

    def build_behavior_features(self, df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        """
        构建 behavior 特征矩阵 [N, behavior_dim]
        默认实现：使用通用行为统计
        """
        feature_names = self.get_behavior_feature_names()
        n = len(df)
        feats = np.zeros((n, len(feature_names)), dtype=np.float32)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            u = self.user_stats.get(uid, {})
            i = self.item_stats.get(iid, {})
            
            for j, fname in enumerate(feature_names):
                feats[idx, j] = self._get_behavior_value(fname, u, i)
        
        return feats

    def _get_behavior_value(self, fname: str, u_stats: Dict, i_stats: Dict) -> float:
        """获取单个行为特征值"""
        if fname == "user_interaction_count":
            v = u_stats.get("interaction_count", 0.0)
            m = self.scalers.get("user_interaction_count", {}).get("mean", 1.0)
            s = self.scalers.get("user_interaction_count", {}).get("std", 1.0)
            return (v - m) / s
        elif fname == "item_interaction_count":
            return np.log1p(i_stats.get("interaction_count", 0.0))
        elif fname == "user_mean_rating":
            return u_stats.get("mean_rating", self.global_mean) - self.global_mean
        elif fname == "item_mean_rating":
            return i_stats.get("mean_rating", self.global_mean) - self.global_mean
        elif fname == "user_rating_std":
            return u_stats.get("rating_std", 0.0)
        elif fname == "item_rating_std":
            return i_stats.get("rating_std", 0.0)
        elif fname == "activity_index":
            return np.log1p(u_stats.get("interaction_count", 0.0))
        return 0.0

    def build_cf_features(self, df: pd.DataFrame, svd_model) -> np.ndarray:
        """构建 CF embedding 特征"""
        feats = []
        for _, row in df.iterrows():
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            uv = svd_model.get_user_vector(uid) if hasattr(svd_model, 'get_user_vector') else np.zeros(50)
            iv = svd_model.get_movie_vector(iid) if hasattr(svd_model, 'get_movie_vector') else np.zeros(50)
            feats.append(np.concatenate([uv, iv]))
        return np.array(feats, dtype=np.float32)

    # ================================================================
    # 元数据
    # ================================================================

    def get_schema(self) -> FeatureSchema:
        return self.schema

    def get_feature_availability(self) -> Dict:
        """返回特征可用性 summary"""
        return {
            "dataset": self.get_dataset_name(),
            "profile_dim": self.schema.profile_dim,
            "profile_columns": self.schema.profile_columns,
            "behavior_dim": self.schema.behavior_dim,
            "behavior_columns": self.schema.behavior_columns,
            "missing_columns": self.schema.missing_columns,
            "has_demographic": self.schema.has_demographic,
            "rating_range": [self.schema.rating_min, self.schema.rating_max],
        }

    def filter_valid_models(self, models: List[str], registry: Dict) -> Tuple[List[str], List[str]]:
        """根据特征可用性过滤模型，返回 (valid_models, skipped_models)"""
        valid, skipped = [], []
        for m in models:
            meta = registry.get(m, {})
            required = meta.get("required_features", [])
            skip = False
            if "profile" in required and self.schema.profile_dim == 0:
                skip = True
            if "behavior" in required and self.schema.behavior_dim == 0:
                skip = True
            if m == "profile_mlp" and self.schema.profile_dim == 0:
                skip = True
            if m == "behavior_mlp" and self.schema.behavior_dim == 0:
                skip = True
            if skip:
                skipped.append(m)
            else:
                valid.append(m)
        return valid, skipped

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _safe_int(val) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return hash(str(val)) % (10 ** 9)

    def _clip(self, preds: np.ndarray) -> np.ndarray:
        return np.clip(preds, self.schema.rating_min, self.schema.rating_max)
