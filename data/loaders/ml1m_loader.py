"""
ML-1M 数据加载器
支持:
  1. 标准 MovieLens 1M 格式 (ratings.dat, users.dat, movies.dat)
  2. 项目已有的 CSV 格式 (training_total.csv)
输出统一格式: DataFrame with columns [user_id, item_id, rating, timestamp,
  gender, age, occupation, genres, ...]
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class ML1MLoader:
    """MovieLens 1M 数据加载器"""

    # ML-1M genres list (18 genres)
    GENRES_LIST = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    # Age bucket mapping (ML-1M standard)
    AGE_MAP = {
        1: "<18", 18: "18-24", 25: "25-34", 35: "35-44",
        45: "45-49", 50: "50-55", 56: "56+",
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.raw_config = config.get("raw", {}) if config else {}

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        """加载 ML-1M 数据，返回统一格式 DataFrame"""
        # 自动查找数据文件
        ratings_path = self._find_file(raw_dir, [
            "training_total.csv",
            "ratings.dat", "ratings.csv",
        ])

        if ratings_path and os.path.exists(ratings_path):
            if ratings_path.endswith(".dat"):
                df = self._load_standard_ml1m(os.path.dirname(ratings_path))
            else:
                df = self._load_existing_csv(ratings_path)
        else:
            raise FileNotFoundError(
                "ML-1M data not found. Run: python scripts/download_datasets.py\n"
                "Or place training_total.csv in data/raw/ml1m/"
            )

        # 应用采样
        if sample_config and sample_config.get("enabled"):
            df = self._sample(df, sample_config)

        return df

    def _find_file(self, raw_dir: str, candidates: list) -> Optional[str]:
        """在多个候选路径中查找文件"""
        for cand in candidates:
            # 1. raw_dir 相对路径
            path = os.path.join(raw_dir, cand) if raw_dir else cand
            if os.path.exists(path):
                return path
            # 2. 项目根目录相对路径
            if os.path.exists(cand):
                return cand
            # 3. data/raw/ml1m/
            path = os.path.join("data", "raw", "ml1m", cand)
            if os.path.exists(path):
                return path
            # 4. training set/
            path = os.path.join("training set", cand)
            if os.path.exists(path):
                return path
        return None

    def _load_standard_ml1m(self, data_dir: str) -> pd.DataFrame:
        """加载标准 ML-1M 格式 (ratings.dat + users.dat + movies.dat)"""
        print("[ML1MLoader] Loading standard ML-1M format...")

        # ratings.dat: UserID::MovieID::Rating::Timestamp
        ratings = pd.read_csv(
            os.path.join(data_dir, "ratings.dat"),
            sep="::", engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        # users.dat: UserID::Gender::Age::Occupation::Zip-code
        users = pd.read_csv(
            os.path.join(data_dir, "users.dat"),
            sep="::", engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
        )

        # movies.dat: MovieID::Title::Genres
        movies = pd.read_csv(
            os.path.join(data_dir, "movies.dat"),
            sep="::", engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )

        # Merge
        df = ratings.merge(users, on="user_id", how="left")
        df = df.merge(movies, on="item_id", how="left")

        # Parse genres to multi-hot
        df = self._parse_genres(df)

        # Map age to bucket labels
        df["age_bucket"] = df["age"].map(self.AGE_MAP).fillna("unknown")

        print(f"[ML1MLoader] Loaded: {len(df)} ratings, "
              f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df

    def _load_existing_csv(self, path: str) -> pd.DataFrame:
        """加载项目已有的 CSV 格式"""
        print(f"[ML1MLoader] Loading existing CSV: {path}")
        df = pd.read_csv(path)

        # 标准化列名
        col_map = {
            "UserID": "user_id", "MovieID": "item_id", "Rating": "rating",
            "Timestamp": "timestamp", "Gender": "gender", "Age": "age",
            "Occupation": "occupation", "Zip-code": "zip_code",
            "Genres": "genres", "Title": "title",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Parse genres
        if "genres" in df.columns:
            df = self._parse_genres(df)

        # Map age
        if "age" in df.columns:
            df["age_bucket"] = df["age"].apply(
                lambda a: self._map_age_to_bucket(a)
            )

        print(f"[ML1MLoader] Loaded: {len(df)} ratings, "
              f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df

    def _parse_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 genres 字符串解析为 multi-hot 编码"""
        if "genres" not in df.columns:
            return df

        # Split by "|"
        genre_sets = df["genres"].fillna("").str.split("|")

        for g in self.GENRES_LIST:
            clean = g.lower().replace(" ", "_").replace("-", "_").replace("'", "")
            col_name = f"genre_{clean}"
            df[col_name] = genre_sets.apply(lambda x: 1 if g in x else 0).astype(np.int8)

        return df

    def _map_age_to_bucket(self, age) -> str:
        """根据年龄值映射到 bucket"""
        if pd.isna(age):
            return "unknown"
        age = int(age)
        if age < 18:
            return "<18"
        elif age <= 24:
            return "18-24"
        elif age <= 34:
            return "25-34"
        elif age <= 44:
            return "35-44"
        elif age <= 49:
            return "45-49"
        elif age <= 55:
            return "50-55"
        else:
            return "56+"

    def _sample(self, df: pd.DataFrame, sample_config: Dict) -> pd.DataFrame:
        """采样以用于 smoke test"""
        max_users = sample_config.get("max_users", 500)
        max_items = sample_config.get("max_items", 500)
        max_interactions = sample_config.get("max_interactions", 5000)

        # 随机选择用户
        all_users = df["user_id"].unique()
        if len(all_users) > max_users:
            rng = np.random.RandomState(42)
            selected_users = rng.choice(all_users, max_users, replace=False)
            df = df[df["user_id"].isin(selected_users)]

        # 随机选择物品
        all_items = df["item_id"].unique()
        if len(all_items) > max_items:
            rng = np.random.RandomState(42)
            selected_items = rng.choice(all_items, max_items, replace=False)
            df = df[df["item_id"].isin(selected_items)]

        # 限制交互数
        if len(df) > max_interactions:
            df = df.sample(n=max_interactions, random_state=42)

        print(f"[ML1MLoader] Sampled: {len(df)} ratings, "
              f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df.reset_index(drop=True)


def load_ml1m(config: Dict, sample_config: Optional[Dict] = None,
              raw_dir: str = "") -> pd.DataFrame:
    """便捷函数"""
    loader = ML1MLoader(config)
    return loader.load(raw_dir=raw_dir, sample_config=sample_config)
