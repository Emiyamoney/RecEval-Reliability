"""
Book-Crossing 数据加载器

数据来源: Kaggle somnambwl/bookcrossing-dataset
  - Users.csv:   User-ID; Age
  - Books.csv:   ISBN; Title; Author; Year; Publisher
  - Ratings.csv: User-ID; ISBN; Rating

输出统一格式 (snake_case):
  user_id, item_id, rating, age, age_bucket, title, author, year, publisher

特征:
  - Demographic: age (数值), age_bucket (类别)
  - Item content: author (categorical), year (numerical), publisher (categorical)
  - 原始 rating 为 0-10 隐式评分, 不强制归一化
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional


class BookCrossingLoader:
    """Book-Crossing 数据加载器"""

    # 年龄分桶 (参考 ML-1M 风格)
    AGE_BUCKETS = [
        (0, 17, "<18"),
        (18, 24, "18-24"),
        (25, 34, "25-34"),
        (35, 44, "35-44"),
        (45, 54, "45-54"),
        (55, 100, "55+"),
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        """加载 Book-Crossing 数据，返回统一 DataFrame"""
        users_path = os.path.join(raw_dir, "Users.csv")
        books_path = os.path.join(raw_dir, "Books.csv")
        ratings_path = os.path.join(raw_dir, "Ratings.csv")

        for p in [users_path, books_path, ratings_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Book-Crossing data not found at {p}. "
                    f"Run: python scripts/download_datasets.py --dataset book_crossing"
                )

        print(f"[BookCrossing] Loading...")

        # ---- 加载 Ratings ----
        ratings = pd.read_csv(ratings_path, sep=";", encoding="latin-1",
                              dtype={"User-ID": int, "ISBN": str, "Rating": float})
        ratings = ratings.rename(columns={
            "User-ID": "user_id",
            "ISBN": "item_id",
            "Rating": "rating",
        })
        # 原始 rating 范围 0-10, 过滤掉 0, 归一化到 [1, 5] 与其他数据集对齐
        ratings = ratings[ratings["rating"] > 0].copy()
        ratings["rating"] = ratings["rating"].astype(np.float32)
        # 线性映射: [1, 10] → [1, 5]
        r_min, r_max = 1.0, 10.0
        ratings["rating"] = 1.0 + (ratings["rating"] - r_min) / (r_max - r_min) * 4.0
        ratings["rating"] = ratings["rating"].clip(1.0, 5.0)

        # ---- 加载 Users (demographic) ----
        # 数据可能有脏行 (User-ID 非数字), 用 str 读取再清洗
        users = pd.read_csv(users_path, sep=";", encoding="latin-1",
                            dtype=str, on_bad_lines="skip")
        users = users.rename(columns={"User-ID": "user_id"})
        # 清洗非数字 User-ID
        users["user_id"] = pd.to_numeric(users["user_id"].str.strip(), errors="coerce")
        users = users.dropna(subset=["user_id"])
        users["user_id"] = users["user_id"].astype(int)

        # 清洗年龄: 去除 NaN / <=0 / >100 的异常值
        if "Age" in users.columns:
            users = users.rename(columns={"Age": "age"})
            users["age"] = pd.to_numeric(users["age"].astype(str).str.strip(), errors="coerce")
            # 标记无效年龄为 NaN
            invalid_age = (users["age"].isna()) | (users["age"] <= 0) | (users["age"] > 100)
            users.loc[invalid_age, "age"] = np.nan

            # 年龄分桶
            users["age_bucket"] = users["age"].apply(self._map_age_to_bucket)

        # ---- 合并 User demographic 到 Ratings ----
        if "age" in users.columns:
            ratings = ratings.merge(users[["user_id", "age", "age_bucket"]],
                                    on="user_id", how="left")
        else:
            ratings["age"] = np.nan
            ratings["age_bucket"] = "unknown"

        # ---- 加载 Books (item metadata) ----
        print(f"[BookCrossing] Loading books metadata...")
        books = pd.read_csv(books_path, sep=";", encoding="latin-1",
                            on_bad_lines="skip",
                            dtype={"ISBN": str})
        books = books.rename(columns={
            "ISBN": "item_id",
            "Title": "title",
            "Author": "author",
            "Year": "year",
            "Publisher": "publisher",
        })

        # 清洗 Year
        if "year" in books.columns:
            books["year"] = pd.to_numeric(books["year"], errors="coerce")
            invalid_year = (books["year"].isna()) | (books["year"] <= 0) | (books["year"] > 2026)
            books.loc[invalid_year, "year"] = np.nan

        # ---- 合并 Book metadata 到 Ratings ----
        meta_cols = ["item_id"]
        for c in ["title", "author", "year", "publisher"]:
            if c in books.columns:
                meta_cols.append(c)
        ratings = ratings.merge(books[meta_cols], on="item_id", how="left")

        # ---- 稳定整数 ID 映射 ----
        ratings["raw_user_id"] = ratings["user_id"].astype(int)
        ratings["raw_item_id"] = ratings["item_id"].astype(str)
        ratings["user_id"] = pd.factorize(ratings["raw_user_id"], sort=True)[0]
        ratings["item_id"] = pd.factorize(ratings["raw_item_id"], sort=True)[0]

        # ---- 生成 timestamp (Book-Crossing 无时间, 用行号) ----
        if "timestamp" not in ratings.columns:
            ratings["timestamp"] = np.arange(len(ratings))

        # ---- 采样 ----
        if sample_config and sample_config.get("enabled"):
            ratings = self._sample(ratings, sample_config)

        n_users = ratings["user_id"].nunique()
        n_items = ratings["item_id"].nunique()
        n_with_age = ratings["age"].notna().sum()

        print(f"[BookCrossing] Loaded: {len(ratings)} ratings, "
              f"{n_users} users, {n_items} items, "
              f"{n_with_age}/{len(ratings)} with age")

        return ratings

    def _map_age_to_bucket(self, age) -> str:
        """年龄 → bucket 标签"""
        if pd.isna(age) or age <= 0 or age > 100:
            return "unknown"
        age = int(age)
        for lo, hi, label in self.AGE_BUCKETS:
            if lo <= age <= hi:
                return label
        return "55+"

    def _sample(self, df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
        """采样以用于 smoke test"""
        rng = np.random.RandomState(42)

        max_users = cfg.get("max_users", 500)
        max_items = cfg.get("max_items", 500)
        max_interactions = cfg.get("max_interactions", 5000)

        all_users = df["user_id"].unique()
        if len(all_users) > max_users:
            selected = rng.choice(all_users, max_users, replace=False)
            df = df[df["user_id"].isin(selected)]

        all_items = df["item_id"].unique()
        if len(all_items) > max_items:
            selected = rng.choice(all_items, max_items, replace=False)
            df = df[df["item_id"].isin(selected)]

        if len(df) > max_interactions:
            df = df.sample(n=max_interactions, random_state=42)

        return df.reset_index(drop=True)


def load_book_crossing(config: Optional[Dict] = None,
                       sample_config: Optional[Dict] = None,
                       raw_dir: str = "") -> pd.DataFrame:
    """便捷函数"""
    loader = BookCrossingLoader(config)
    return loader.load(
        raw_dir=raw_dir or "data/raw/book_crossing",
        sample_config=sample_config,
    )
