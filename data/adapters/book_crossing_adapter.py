"""
Book-Crossing 数据集适配器 — 年龄 + 书籍元数据特征

特征:
  Profile: age(bucket+raw) + author(top) + year(bucket) + publisher(top) ≈ 50+ dim
  Behavior: 7 dim
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from data.adapters.base import BaseDatasetAdapter
from data.loaders.book_crossing_loader import BookCrossingLoader


class BookCrossingAdapter(BaseDatasetAdapter):
    """Book-Crossing 适配器 — 年龄 + 书籍元数据"""

    DATASET_NAME = "book_crossing"
    
    AGE_BUCKETS = ["<18", "18-24", "25-34", "35-44", "45-54", "55+", "unknown"]
    DECADE_BUCKETS = ["<1950", "1950-1979", "1980-1999", "2000+", "unknown"]

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.schema.has_demographic = True  # 年龄是真实的人口统计
        self._top_authors: List[str] = []
        self._top_publishers: List[str] = []

    def get_dataset_name(self) -> str:
        return self.DATASET_NAME

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        loader = BookCrossingLoader(self.config)
        df = loader.load(raw_dir=raw_dir or "data/raw/book_crossing", sample_config=sample_config)
        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Book-Crossing 特有预处理"""
        # 年龄缺失标记
        if "age" in df.columns:
            df["age_known"] = df["age"].notna().astype(np.float32)
            df["age"] = df["age"].fillna(0.0)
        
        # 年代分桶
        if "year" in df.columns:
            df["decade"] = df["year"].apply(
                lambda y: self._map_decade(y) if pd.notna(y) else "unknown"
            )
        else:
            df["decade"] = "unknown"
        
        # 构建 top authors / publishers
        for col, attr in [("author", "_top_authors"), ("publisher", "_top_publishers")]:
            if col in df.columns:
                counts = df[col].value_counts()
                setattr(self, attr, list(counts.head(30).index))
        
        return df

    def _map_decade(self, year) -> str:
        if pd.isna(year) or year <= 0:
            return "unknown"
        y = int(year)
        if y < 1950:
            return "<1950"
        elif y <= 1979:
            return "1950-1979"
        elif y <= 1999:
            return "1980-1999"
        else:
            return "2000+"

    def get_profile_columns(self) -> List[str]:
        """Book-Crossing profile 列"""
        return ["age", "age_bucket", "age_known", "author", "decade", "publisher"]

    def get_behavior_feature_names(self) -> List[str]:
        return [
            "user_interaction_count", "item_interaction_count",
            "user_mean_rating", "item_mean_rating",
            "user_rating_std", "item_rating_std",
            "activity_index",
        ]

    def _compute_profile_dim(self, train_df: pd.DataFrame) -> int:
        """Book-Crossing profile 维度: age_raw(1) + age_known(1) + age_bucket(7) + author(30) + decade(5) + publisher(30)"""
        dim = 2  # age_raw + age_known
        dim += len(self.AGE_BUCKETS)
        dim += len(self._top_authors) if self._top_authors else 30
        dim += len(self.DECADE_BUCKETS)
        dim += len(self._top_publishers) if self._top_publishers else 30
        return dim

    def build_profile_features(self, df: pd.DataFrame) -> np.ndarray:
        """Book-Crossing Profile 特征"""
        feats_list = []
        top_authors = self._top_authors or []
        top_publishers = self._top_publishers or []
        
        for _, row in df.iterrows():
            rfeats = []
            
            # Age (z-score normalized)
            age = float(row.get("age", 0))
            rfeats.append(age / 50.0)  # rough normalization
            
            # Age known flag
            rfeats.append(float(row.get("age_known", 0)))
            
            # Age bucket one-hot
            age_bucket = str(row.get("age_bucket", "unknown"))
            for b in self.AGE_BUCKETS:
                rfeats.append(1.0 if age_bucket == b else 0.0)
            
            # Author one-hot
            author = str(row.get("author", "unknown"))
            for a in top_authors:
                rfeats.append(1.0 if author == a else 0.0)
            if not top_authors:
                rfeats.extend([0.0] * 30)
            
            # Decade one-hot
            decade = str(row.get("decade", "unknown"))
            for d in self.DECADE_BUCKETS:
                rfeats.append(1.0 if decade == d else 0.0)
            
            # Publisher one-hot
            publisher = str(row.get("publisher", "unknown"))
            for p in top_publishers:
                rfeats.append(1.0 if publisher == p else 0.0)
            if not top_publishers:
                rfeats.extend([0.0] * 30)
            
            feats_list.append(rfeats)
        
        return np.array(feats_list, dtype=np.float32)
