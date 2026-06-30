"""
Goodbooks-10k 数据加载器 (替代 Book-Crossing)
来源: https://github.com/zygmuntz/goodbooks-10k
数据: ratings.csv (6M ratings), books.csv (10K books), book_tags.csv, tags.csv
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional


class GoodbooksLoader:
    """Goodbooks-10k 数据加载器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def load(self, raw_dir: str = "", sample_config: Optional[Dict] = None) -> pd.DataFrame:
        ratings_path = os.path.join(raw_dir, "ratings.csv") if raw_dir else "data/raw/goodbooks/ratings.csv"

        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"Goodbooks data not found at {ratings_path}. "
                "Run: python scripts/download_datasets.py"
            )

        print(f"[Goodbooks] Loading ratings: {ratings_path}")

        # Load ratings
        ratings = pd.read_csv(ratings_path)
        ratings = ratings.rename(columns={
            "user_id": "user_id",
            "book_id": "item_id",
            "rating": "rating",
        })

        # Load books metadata
        books_path = os.path.join(os.path.dirname(ratings_path), "books.csv")
        if os.path.exists(books_path):
            print(f"[Goodbooks] Loading books metadata...")
            books = pd.read_csv(books_path)
            # Map work_id → metadata
            book_meta = {}
            for _, r in books.iterrows():
                book_meta[int(r["book_id"])] = {
                    "title": str(r.get("original_title", "")),
                    "authors": str(r.get("authors", "unknown")),
                    "year": float(r["original_publication_year"]) if pd.notna(r.get("original_publication_year")) else 0.0,
                    "avg_rating": float(r.get("average_rating", 0)),
                }

            # Add metadata columns
            ratings["title"] = ratings["item_id"].map(lambda x: book_meta.get(x, {}).get("title", ""))
            ratings["author"] = ratings["item_id"].map(lambda x: book_meta.get(x, {}).get("authors", "unknown"))
            ratings["year"] = ratings["item_id"].map(lambda x: book_meta.get(x, {}).get("year", 0.0))

        # Load tags
        tags_path = os.path.join(os.path.dirname(ratings_path), "book_tags.csv")
        tags_map_path = os.path.join(os.path.dirname(ratings_path), "tags.csv")
        if os.path.exists(tags_path) and os.path.exists(tags_map_path):
            print(f"[Goodbooks] Loading tags...")
            book_tags = pd.read_csv(tags_path)
            tags_map = pd.read_csv(tags_map_path)

            # Get top tag per book
            tag_names = dict(zip(tags_map["tag_id"], tags_map["tag_name"]))
            top_tags = book_tags.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
            top_tags = top_tags.groupby("goodreads_book_id").first()["tag_id"].reset_index()
            top_tags["genre"] = top_tags["tag_id"].map(tag_names)
            # Map goodreads_book_id → genre (approximate, using book_id)
            gb_to_genre = dict(zip(top_tags["goodreads_book_id"], top_tags["genre"]))

            # We need work_id → goodreads_book_id mapping from books.csv
            if os.path.exists(books_path):
                books_df = pd.read_csv(books_path)
                work_to_gb = dict(zip(books_df["book_id"], books_df["goodreads_book_id"]))
                ratings["genre"] = ratings["item_id"].map(
                    lambda x: gb_to_genre.get(work_to_gb.get(x, -1), "unknown")
                )
            else:
                ratings["genre"] = "unknown"

        # Standardize
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["item_id"] = ratings["item_id"].astype(int)
        ratings["rating"] = ratings["rating"].astype(np.float32)

        if "timestamp" not in ratings.columns:
            ratings["timestamp"] = np.arange(len(ratings))

        # Add placeholder demographics
        if "gender" not in ratings.columns:
            ratings["gender"] = "unknown"
        if "age" not in ratings.columns:
            ratings["age"] = 30
        if "occupation" not in ratings.columns:
            ratings["occupation"] = 0

        if sample_config and sample_config.get("enabled"):
            ratings = self._sample(ratings, sample_config)

        print(f"[Goodbooks] Loaded: {len(ratings)} ratings, "
              f"{ratings['user_id'].nunique()} users, {ratings['item_id'].nunique()} items")
        return ratings

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


def load_goodbooks(config=None, sample_config=None, raw_dir=""):
    return GoodbooksLoader(config).load(raw_dir=raw_dir or "data/raw/goodbooks", sample_config=sample_config)
