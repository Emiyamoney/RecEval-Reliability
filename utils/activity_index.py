"""
Activity index computation — leakage-fixed version.

Computed from train_df only, supports save/load.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict


WEIGHT_SCALE_DEFAULT = 20.0


def compute_activity_index_from_df(train_df: pd.DataFrame, scale: float = WEIGHT_SCALE_DEFAULT) -> Dict[int, float]:
    """Compute activity index from train_df only (no val/test leakage)."""
    counts = train_df.groupby("UserID").size()
    if len(counts) == 0:
        return {}
    index_vals = np.log2(1.0 + counts.values / scale)
    max_val = index_vals.max()
    if max_val > 0:
        index_vals = index_vals / max_val
    return {int(uid): float(v) for uid, v in zip(counts.index, index_vals)}


def compute_activity_index(data_path: str, scale: float = WEIGHT_SCALE_DEFAULT) -> Dict[int, float]:
    """Legacy interface — compute from file (fallback only)."""
    df = pd.read_csv(data_path)
    return compute_activity_index_from_df(df, scale)


def save_activity_map(activity_map: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(activity_map, f)


def load_activity_map(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)
