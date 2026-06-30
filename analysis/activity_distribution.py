"""
用户活跃度分布分析
- 统计训练集中每个用户的交互量
- 输出 histogram, log-histogram, CDF, quantile table
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_activity_distribution(
    train_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str = "ml1m",
) -> Dict:
    """分析用户交互量分布"""
    os.makedirs(output_dir, exist_ok=True)

    # Count interactions per user
    user_counts = train_df.groupby("user_id").size()
    n_u = user_counts.values.astype(np.float64)

    # Basic stats
    stats = {
        "n_users": len(n_u),
        "mean": float(np.mean(n_u)),
        "std": float(np.std(n_u)),
        "min": int(np.min(n_u)),
        "max": int(np.max(n_u)),
        "median": float(np.median(n_u)),
    }

    # Quantiles
    quantiles = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
    for q in quantiles:
        stats[f"q{int(q*100)}"] = float(np.quantile(n_u, q))

    # Save quantile table
    quantile_df = pd.DataFrame([
        {"quantile": f"Q{int(q*100)}", "n_interactions": stats[f"q{int(q*100)}"]}
        for q in quantiles
    ])
    quantile_df.to_csv(os.path.join(output_dir, "user_activity_quantiles.csv"), index=False)

    # ---- Plot 1: Histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].hist(n_u, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(stats["median"], color="red", linestyle="--", label=f"Median={stats['median']:.0f}")
    axes[0].set_xlabel("Number of Interactions (n_u)")
    axes[0].set_ylabel("User Count")
    axes[0].set_title(f"User Activity Distribution — {dataset_name}")
    axes[0].legend()

    # Log scale
    axes[1].hist(np.log1p(n_u), bins=50, color="#2ecc71", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("log(1 + n_u)")
    axes[1].set_ylabel("User Count")
    axes[1].set_title("Log-Scale Activity Distribution")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "user_activity_distribution.png"), dpi=150)
    plt.close()

    # ---- Plot 2: CDF ----
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_n = np.sort(n_u)
    cdf = np.arange(1, len(sorted_n) + 1) / len(sorted_n)
    ax.plot(sorted_n, cdf, linewidth=2, color="#e74c3c")
    ax.set_xlabel("Number of Interactions (n_u)")
    ax.set_ylabel("Cumulative Fraction of Users")
    ax.set_title(f"User Activity CDF — {dataset_name}")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Mark quantiles
    for q in [0.25, 0.5, 0.75]:
        q_val = np.quantile(n_u, q)
        ax.axhline(q, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(q_val, color="gray", linestyle=":", alpha=0.5)
        ax.text(q_val, q, f" Q{int(q*100)}={q_val:.0f}", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "user_activity_cdf.png"), dpi=150)
    plt.close()

    print(f"[ActivityDist] {dataset_name}: users={stats['n_users']}, "
          f"mean={stats['mean']:.1f}, median={stats['median']:.0f}, "
          f"Q10={stats['q10']:.0f}, Q25={stats['q25']:.0f}, "
          f"Q50={stats['q50']:.0f}, Q75={stats['q75']:.0f}")

    return stats


def compute_activity_counts(train_df: pd.DataFrame) -> Dict[int, int]:
    """返回 {user_id: n_interactions}"""
    return train_df.groupby("user_id").size().to_dict()
