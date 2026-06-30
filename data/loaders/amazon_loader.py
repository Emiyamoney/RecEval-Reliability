"""
Amazon Movies and TV 数据加载器
支持:
  1. 2018 JSON 格式 (reviewerID, asin, overall, unixReviewTime)
  2. 2014 CSV 格式 (product/productId, review/userId, review/score, review/time)

注意: Amazon 缺少显式 demographic 用户画像
Profile 特征主要来自 item metadata + cold-start-safe user 特征
"""

import os, json
import pandas as pd
import numpy as np
from typing import Dict, Optional


class AmazonLoader:
    """Amazon Reviews 数据加载器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        reviews_path = self._find_file(raw_dir, [
            "ratings_Movies_and_TV.csv",
            "Movies_and_TV_5.json",
            "reviews.json",
            "Movies_and_TV.json",
        ])

        if not reviews_path:
            raise FileNotFoundError(
                "Amazon Movies data not found. Place ratings_Movies_and_TV.csv in data/raw/amazon_movies/"
            )

        print(f"[AmazonLoader] Loading: {reviews_path}")

        if reviews_path.endswith(".json") or reviews_path.endswith(".jsonl"):
            df = self._load_json(reviews_path)
        elif "ratings_" in reviews_path:
            # 2014 format CSV with headers like product/productId
            df = self._load_2014_csv(reviews_path)
        else:
            df = pd.read_csv(reviews_path)

        df = self._standardize(df)

        if sample_config and sample_config.get("enabled"):
            df = self._sample(df, sample_config)

        print(f"[AmazonLoader] Loaded: {len(df)} ratings, "
              f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df

    def _find_file(self, raw_dir, candidates):
        for cand in candidates:
            path = os.path.join(raw_dir, cand) if raw_dir else cand
            if os.path.exists(path):
                return path
        return None

    def _load_2014_csv(self, path: str) -> pd.DataFrame:
        """Load 2014 ratings-only CSV format: productId,userId,rating,timestamp"""
        print("[AmazonLoader] Detected 2014 CSV format...")
        # Try reading as simple 4-column CSV first (ratings-only format)
        df = pd.read_csv(path, header=None, nrows=1)
        ncols = df.shape[1]

        if ncols == 4:
            # Simple ratings-only: productId,userId,rating,timestamp
            df = pd.read_csv(path, header=None, names=["item_id", "user_id", "rating", "timestamp"])
            print(f"  Detected simple 4-column ratings CSV")
        elif ncols >= 8:
            # Full review format with 8 columns
            df = pd.read_csv(path, header=None, names=[
                "product_id", "user_id", "profile_name", "helpfulness",
                "rating", "time", "summary", "text"
            ])
            for col in ["product_id", "user_id", "rating", "time"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(f"{col.split('_')[0]}/", "", regex=False)
            df = df[["user_id", "product_id", "rating", "time"]].copy()
            df.columns = ["user_id", "item_id", "rating", "timestamp"]
        else:
            raise ValueError(f"Unexpected CSV format with {ncols} columns")

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0).astype(np.float32)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        df["user_id"] = df["user_id"].astype(str).str.strip()
        df["item_id"] = df["item_id"].astype(str).str.strip()

        return df

    def _load_json(self, path: str) -> pd.DataFrame:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(records)

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "reviewerID": "user_id", "asin": "item_id",
            "overall": "rating", "unixReviewTime": "timestamp",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["user_id", "item_id", "rating"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df["rating"] = df["rating"].astype(np.float32)

        # 保存原始字符串 ID
        df["raw_user_id"] = df["user_id"].astype(str).str.strip()
        df["raw_item_id"] = df["item_id"].astype(str).str.strip()

        # 稳定整数映射 (pandas.factorize sort=True 保证可复现)
        df["user_id"] = pd.factorize(df["raw_user_id"], sort=True)[0]
        df["item_id"] = pd.factorize(df["raw_item_id"], sort=True)[0]

        if "timestamp" not in df.columns:
            df["timestamp"] = 0

        # Amazon 无 demographic 数据，不添加伪造的占位列
        # gender/age/occupation 列仅当真实存在时才保留

        return df

    def _sample(self, df, cfg):
        rng = np.random.RandomState(42)
        users = df["user_id"].unique()
        if len(users) > cfg.get("max_users", 500):
            users = rng.choice(users, cfg["max_users"], replace=False)
            df = df[df["user_id"].isin(users)]
        items = df["item_id"].unique()
        if len(items) > cfg.get("max_items", 500):
            items = rng.choice(items, cfg["max_items"], replace=False)
            df = df[df["item_id"].isin(items)]
        if len(df) > cfg.get("max_interactions", 5000):
            df = df.sample(n=cfg["max_interactions"], random_state=42)
        return df.reset_index(drop=True)
