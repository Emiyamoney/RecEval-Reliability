"""
Unified data splitting module.

Supports three split types:
  1. strict_cold  — user-level split, test users never appear in train
  2. warm_random  — row-level random split, same user can span sets
  3. warm_temporal — time-ordered split: train < val < test

All behavior features are guaranteed to be computed from train only.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def make_strict_cold_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Strict Cold-Start Split: user-level, test users never in train."""
    rng = np.random.RandomState(seed)
    all_users = np.array(sorted(df["user_id"].unique()))
    rng.shuffle(all_users)

    n = len(all_users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_users = set(all_users[:n_train])
    val_users = set(all_users[n_train:n_train + n_val])
    test_users = set(all_users[n_train + n_val:])

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()
    test_df = df[df["user_id"].isin(test_users)].copy()

    summary = _build_summary(train_df, val_df, test_df)

    assert summary["train_test_overlap_users"] == 0, \
        f"Cold-start violation: {summary['train_test_overlap_users']} overlapping users!"
    assert summary["train_val_overlap_users"] == 0

    if save_dir:
        _save_split(train_df, val_df, test_df, save_dir, f"strict_cold_seed{seed}")
        print(f"[split] Strict cold-start split saved to {save_dir}/")

    _print_summary("Strict Cold-Start", summary)
    return train_df, val_df, test_df, summary


def make_warm_random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Warm-Start Row-Level Split: random row split, same user can span sets."""
    rng = np.random.RandomState(seed)
    n = len(df)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[indices[:n_train]].copy()
    val_df = df.iloc[indices[n_train:n_train + n_val]].copy()
    test_df = df.iloc[indices[n_train + n_val:]].copy()

    summary = _build_summary(train_df, val_df, test_df)
    summary["is_cold_start"] = False

    if save_dir:
        _save_split(train_df, val_df, test_df, save_dir, f"warm_random_seed{seed}")
        print(f"[split] Warm random split saved to {save_dir}/")

    _print_summary("Warm Random", summary)
    return train_df, val_df, test_df, summary


def make_warm_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Temporal Warm-Start Split: time-ordered, train < val < test."""
    if "timestamp" not in df.columns:
        raise ValueError("Temporal split requires 'timestamp' column in DataFrame")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df["timestamp"].max() <= val_df["timestamp"].min(), \
            "Temporal order violation: train timestamps overlap with val!"

    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df["timestamp"].max() <= test_df["timestamp"].min(), \
            "Temporal order violation: val timestamps overlap with test!"

    summary = _build_summary(train_df, val_df, test_df)
    summary["is_cold_start"] = False
    summary["train_time_range"] = (
        str(train_df["timestamp"].min()), str(train_df["timestamp"].max())
    )
    summary["val_time_range"] = (
        str(val_df["timestamp"].min()), str(val_df["timestamp"].max())
    )
    summary["test_time_range"] = (
        str(test_df["timestamp"].min()), str(test_df["timestamp"].max())
    )

    if save_dir:
        _save_split(train_df, val_df, test_df, save_dir, "warm_temporal")
        print(f"[split] Temporal warm split saved to {save_dir}/")

    _print_summary("Temporal Warm", summary)
    return train_df, val_df, test_df, summary


def create_all_splits(
    df: pd.DataFrame,
    config: Dict,
    save_dir: str,
    seed: int = 42,
) -> Dict[str, Tuple]:
    """Create all required splits based on configuration."""
    split_types = config.get("splits", ["strict_cold", "warm_random"])
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = round(1 - train_ratio - val_ratio, 2)

    results = {}
    for st in split_types:
        split_save_dir = os.path.join(save_dir, st)
        os.makedirs(split_save_dir, exist_ok=True)

        if st == "strict_cold":
            results["strict_cold"] = make_strict_cold_split(
                df, train_ratio, val_ratio, test_ratio,
                seed=seed, save_dir=split_save_dir,
            )
        elif st == "warm_random":
            results["warm_random"] = make_warm_random_split(
                df, train_ratio, val_ratio, test_ratio,
                seed=seed, save_dir=split_save_dir,
            )
        elif st == "warm_temporal":
            results["warm_temporal"] = make_warm_temporal_split(
                df, train_ratio, val_ratio, test_ratio,
                save_dir=split_save_dir,
            )

    return results


def _build_summary(train_df, val_df, test_df) -> Dict:
    """Build split summary statistics."""
    train_users = set(train_df["user_id"].unique()) if len(train_df) > 0 else set()
    val_users = set(val_df["user_id"].unique()) if len(val_df) > 0 else set()
    test_users = set(test_df["user_id"].unique()) if len(test_df) > 0 else set()
    train_items = set(train_df["item_id"].unique()) if len(train_df) > 0 else set()
    val_items = set(val_df["item_id"].unique()) if len(val_df) > 0 else set()
    test_items = set(test_df["item_id"].unique()) if len(test_df) > 0 else set()

    return {
        "train_ratings": len(train_df),
        "val_ratings": len(val_df),
        "test_ratings": len(test_df),
        "train_users": len(train_users),
        "val_users": len(val_users),
        "test_users": len(test_users),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "test_items": len(test_items),
        "train_val_overlap_users": len(train_users & val_users),
        "train_test_overlap_users": len(train_users & test_users),
        "val_test_overlap_users": len(val_users & test_users),
        "val_items_in_train": len(val_items & train_items),
        "test_items_in_train": len(test_items & train_items),
        "is_cold_start": True,
    }


def _save_split(train_df, val_df, test_df, save_dir: str, prefix: str) -> None:
    """Save splits to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, f"{prefix}_train.csv"), index=False)
    if len(val_df) > 0:
        val_df.to_csv(os.path.join(save_dir, f"{prefix}_valid.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, f"{prefix}_test.csv"), index=False)


def _print_summary(name: str, s: Dict) -> None:
    """Print split summary."""
    print(f"[split] {name}: "
          f"Train={s['train_ratings']}r/{s['train_users']}u/{s['train_items']}i | "
          f"Val={s['val_ratings']}r/{s['val_users']}u | "
          f"Test={s['test_ratings']}r/{s['test_users']}u")
    overlap = s["train_test_overlap_users"]
    if overlap > 0:
        print(f"[split]   train ∩ test users = {overlap} (warm-start)")
    else:
        print(f"[split]   train ∩ test users = 0 (strict cold-start OK)")
