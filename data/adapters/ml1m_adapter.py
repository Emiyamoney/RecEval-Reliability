"""
ML-1M 数据集适配器 — 丰富的 demographic + genre + 外部特征

特征:
  Profile: gender(2) + age_bucket(7) + occupation(21) + genre_*(18) = 48 dim
           可选 movie_audience_vec(5) + user_genre_pref(18) = 71 dim
  Behavior: 7 dim (user/item interaction_count, mean_rating, rating_std, activity_index)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from data.adapters.base import BaseDatasetAdapter, FeatureSchema
from data.loaders.ml1m_loader import ML1MLoader


class ML1MAdapter(BaseDatasetAdapter):
    """ML-1M 适配器 — 最丰富的特征集"""

    DATASET_NAME = "ml1m"
    AGE_BUCKETS = ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    
    # 18 genres in ML-1M
    GENRE_NAMES = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.schema.has_demographic = True
        
        # 外部特征缓存
        self._movie_vec: Dict[int, np.ndarray] = {}
        self._user_pref: Dict[int, np.ndarray] = {}
        self._user_pref_dim: int = 0
        
        # 特征维度 (在 fit 后确定)
        self._n_occupation: int = 21
        self._has_genre: bool = True

    def get_dataset_name(self) -> str:
        return self.DATASET_NAME

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        loader = ML1MLoader(self.config)
        df = loader.load(raw_dir=raw_dir or "data/raw/ml1m", sample_config=sample_config)
        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """ML-1M 特有预处理"""
        # 检测 genre_* 列
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        self._has_genre = len(genre_cols) > 0
        
        # 检测 occupation 范围
        if "occupation" in df.columns:
            vals = df["occupation"].dropna()
            if len(vals) > 0:
                self._n_occupation = int(vals.max()) + 1
        
        return df

    def enrich_features(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """加载外部 CSV 特征 (movie_audience_vector, user_genre_preference)"""
        # 仅在 fit 之后调用
        return df

    def load_external_features(self) -> bool:
        """加载外部 CSV 文件，返回是否成功"""
        # Movie audience vector
        mv_paths = [
            "data/raw/ml1m/movie_audience_vector_net_attitude.csv",
            "training set/movie_audience_vector_net_attitude.csv",
        ]
        for p in mv_paths:
            if os.path.exists(p):
                mv = pd.read_csv(p)
                for _, r in mv.iterrows():
                    self._movie_vec[int(r["MovieID"])] = np.array([
                        r["Vector_Male_Dim"], r["Vector_Female_Dim"],
                        r["Vector_Age_1-34"], r["Vector_Age_35-55"], r["Vector_Age_56+"],
                    ], dtype=np.float32)
                break
        
        # User genre preference
        uv_paths = [
            "data/raw/ml1m/training_user_genre_preference.csv",
            "training set/training_user_genre_preference.csv",
        ]
        for p in uv_paths:
            if os.path.exists(p):
                uv = pd.read_csv(p)
                for _, r in uv.iterrows():
                    self._user_pref[int(r["UserID"])] = r.drop("UserID").values.astype(np.float32)
                if self._user_pref:
                    self._user_pref_dim = len(next(iter(self._user_pref.values())))
                break
        
        return len(self._movie_vec) > 0 or len(self._user_pref) > 0

    def get_profile_columns(self) -> List[str]:
        """ML-1M profile 列: gender, age_bucket, occupation, genre_*"""
        cols = ["gender", "age_bucket", "occupation"]
        for g in self.GENRE_NAMES:
            clean = g.lower().replace(" ", "_").replace("-", "_").replace("'", "")
            cols.append(f"genre_{clean}")
        return cols

    def get_behavior_feature_names(self) -> List[str]:
        return [
            "user_interaction_count", "item_interaction_count",
            "user_mean_rating", "item_mean_rating",
            "user_rating_std", "item_rating_std",
            "activity_index",
        ]

    def _compute_profile_dim(self, train_df: pd.DataFrame) -> int:
        """ML-1M profile 维度: 2(gender) + 7(age) + 21(occ) + 18(genre) = 48"""
        dim = 0
        if "gender" in train_df.columns:
            dim += 2
        if "age_bucket" in train_df.columns:
            dim += len(self.AGE_BUCKETS)
        if "occupation" in train_df.columns:
            dim += self._n_occupation
        dim += len([c for c in train_df.columns if c.startswith("genre_")])
        return dim

    def build_profile_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        ML-1M Profile 特征:
          gender: M→[1,0], F→[0,1], else→[0,0] — 2 dim
          age_bucket: one-hot over AGE_BUCKETS — 7 dim
          occupation: one-hot over 0..20 — 21 dim
          genre_*: multi-hot — 18 dim
        """
        feats_list = []
        for _, row in df.iterrows():
            rfeats = []
            
            # Gender
            gender = str(row.get("gender", "unknown"))
            rfeats.extend([1.0, 0.0] if gender == "M" else
                          [0.0, 1.0] if gender == "F" else [0.0, 0.0])
            
            # Age bucket
            age_bucket = str(row.get("age_bucket", "unknown"))
            for b in self.AGE_BUCKETS:
                rfeats.append(1.0 if age_bucket == b else 0.0)
            
            # Occupation
            occ = int(row.get("occupation", 0)) if not pd.isna(row.get("occupation", 0)) else 0
            for i in range(self._n_occupation):
                rfeats.append(1.0 if i == occ else 0.0)
            
            # Genre multi-hot
            for g in self.GENRE_NAMES:
                clean = "genre_" + g.lower().replace(" ", "_").replace("-", "_").replace("'", "")
                val = row.get(clean, 0)
                rfeats.append(float(val) if not pd.isna(val) else 0.0)
            
            feats_list.append(rfeats)
        
        return np.array(feats_list, dtype=np.float32)

    def get_movie_vector(self, item_id: int) -> np.ndarray:
        """获取电影受众向量 (5 dim)，外部特征"""
        return self._movie_vec.get(item_id, np.zeros(5, dtype=np.float32))

    def get_user_preference(self, user_id: int) -> np.ndarray:
        """获取用户类型偏好向量，外部特征"""
        return self._user_pref.get(user_id, np.zeros(self._user_pref_dim, dtype=np.float32))

    def has_external_features(self) -> bool:
        return len(self._movie_vec) > 0 and len(self._user_pref) > 0
