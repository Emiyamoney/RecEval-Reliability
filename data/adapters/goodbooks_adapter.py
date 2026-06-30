"""
Goodbooks 数据集适配器 — 书籍元数据特征

特征:
  Profile: title_length, author_first, year (bucketed), genre (one-hot top tag)
           不含伪造的 demographic 特征
  Behavior: 7 dim
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from data.adapters.base import BaseDatasetAdapter
from data.loaders.goodbooks_loader import GoodbooksLoader


class GoodbooksAdapter(BaseDatasetAdapter):
    """Goodbooks 适配器 — 书籍元数据驱动"""

    DATASET_NAME = "goodbooks"
    
    # 年代分桶
    DECADE_BUCKETS = ["<1900", "1900-1949", "1950-1999", "2000-2009", "2010+", "unknown"]

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.schema.has_demographic = False
        self._top_genres: List[str] = []
        self._max_genres: int = 20

    def get_dataset_name(self) -> str:
        return self.DATASET_NAME

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        loader = GoodbooksLoader(self.config)
        df = loader.load(raw_dir=raw_dir or "data/raw/goodbooks", sample_config=sample_config)
        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Goodbooks 特有预处理 — 构建真实 profile 特征"""
        # 标题长度
        if "title" in df.columns:
            df["title_length"] = df["title"].fillna("").apply(len).astype(np.float32)
        else:
            df["title_length"] = 0.0
        
        # 作者首词 (简化)
        if "author" in df.columns:
            df["author_first"] = df["author"].fillna("").apply(
                lambda x: x.split(",")[0].strip() if x else "unknown"
            )
        else:
            df["author_first"] = "unknown"
        
        # 出版年代分桶
        if "year" in df.columns:
            df["decade"] = df["year"].apply(self._map_decade)
        else:
            df["decade"] = "unknown"
        
        # 构建 top genres 列表
        if "genre" in df.columns:
            genre_counts = df["genre"].value_counts()
            self._top_genres = list(genre_counts.head(self._max_genres).index)
            if "unknown" not in self._top_genres:
                self._top_genres.append("unknown")
        
        # 不添加伪造的 demographic 占位列
        # Goodbooks 没有真实的 gender/age/occupation
        
        return df

    def _map_decade(self, year) -> str:
        if pd.isna(year) or year <= 0:
            return "unknown"
        y = int(year)
        if y < 1900:
            return "<1900"
        elif y <= 1949:
            return "1900-1949"
        elif y <= 1999:
            return "1950-1999"
        elif y <= 2009:
            return "2000-2009"
        else:
            return "2010+"

    def get_profile_columns(self) -> List[str]:
        """Goodbooks profile 列: 全是书籍内容特征，无 demographic"""
        return ["title_length", "author_first", "decade"]

    def get_behavior_feature_names(self) -> List[str]:
        return [
            "user_interaction_count", "item_interaction_count",
            "user_mean_rating", "item_mean_rating",
            "user_rating_std", "item_rating_std",
            "activity_index",
        ]

    def _compute_profile_dim(self, train_df: pd.DataFrame) -> int:
        """Goodbooks profile 维度: 1(title_length) + author_onehot + decade_onehot + genre_onehot"""
        dim = 1  # title_length
        
        # author_first: 取 top 30 authors
        if "author_first" in train_df.columns:
            top_authors = train_df["author_first"].value_counts().head(30).index.tolist()
            dim += len(top_authors)
            self._top_authors = top_authors
        
        # decade buckets
        dim += len(self.DECADE_BUCKETS)
        
        # genre one-hot
        dim += len(self._top_genres)
        
        return dim

    def build_profile_features(self, df: pd.DataFrame) -> np.ndarray:
        """Goodbooks Profile 特征 — 全部基于书籍元数据"""
        feats_list = []
        top_authors = getattr(self, '_top_authors', [])
        
        for _, row in df.iterrows():
            rfeats = []
            
            # title_length (z-score normalized)
            tl = float(row.get("title_length", 0))
            rfeats.append(tl / 100.0)  # rough normalization
            
            # author_first one-hot
            author = str(row.get("author_first", "unknown"))
            for a in top_authors:
                rfeats.append(1.0 if author == a else 0.0)
            
            # decade one-hot
            decade = str(row.get("decade", "unknown"))
            for d in self.DECADE_BUCKETS:
                rfeats.append(1.0 if decade == d else 0.0)
            
            # genre one-hot
            genre = str(row.get("genre", "unknown"))
            for g in self._top_genres:
                rfeats.append(1.0 if genre == g else 0.0)
            
            feats_list.append(rfeats)
        
        return np.array(feats_list, dtype=np.float32)
