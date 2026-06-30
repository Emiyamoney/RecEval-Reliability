"""
实验图表生成
- 排名翻转分析
- 消融实验 heatmap
- Gating curve
- 模型对比图
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def plot_rank_reversal(
    results_df: pd.DataFrame,
    dataset: str,
    output_path: str,
):
    """排名翻转分析: cold vs warm 下各模型 RMSE 排名变化"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Aggregate: mean RMSE per model per scenario
    if "cold_rmse" not in results_df.columns:
        print("[Plot] No cold_rmse column, skipping rank reversal")
        return

    cold_agg = results_df.groupby("model")["cold_rmse"].mean().sort_values()
    warm_agg = results_df.groupby("model")["warm_rmse"].mean().sort_values()

    models = list(set(cold_agg.index) | set(warm_agg.index))
    cold_ranks = {m: list(cold_agg.index).index(m) + 1 if m in cold_agg.index else len(models)
                  for m in models}
    warm_ranks = {m: list(warm_agg.index).index(m) + 1 if m in warm_agg.index else len(models)
                  for m in models}

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.35

    cold_vals = [cold_ranks.get(m, len(models)) for m in models]
    warm_vals = [warm_ranks.get(m, len(models)) for m in models]

    bars1 = ax.bar(x - width/2, cold_vals, width, label="Cold-Start Rank", color="#3498db")
    bars2 = ax.bar(x + width/2, warm_vals, width, label="Warm-Start Rank", color="#e74c3c")

    ax.set_xlabel("Model")
    ax.set_ylabel("Rank (1 = Best)")
    ax.set_title(f"Rank Reversal: Cold vs Warm — {dataset}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.invert_yaxis()  # Rank 1 at top

    # Annotate flip direction
    for i, m in enumerate(models):
        cr = cold_ranks.get(m, 0)
        wr = warm_ranks.get(m, 0)
        if cr > 0 and wr > 0:
            arrow = "↑" if wr < cr else "↓" if wr > cr else "→"
            color = "green" if wr < cr else "red" if wr > cr else "gray"
            ax.text(i, max(cr, wr) + 0.5, arrow, ha="center", fontsize=14, color=color)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save rank CSV
    rank_df = pd.DataFrame({
        "model": models,
        "cold_rank": cold_vals,
        "warm_rank": warm_vals,
        "cold_rmse": [cold_agg.get(m, float("nan")) for m in models],
        "warm_rmse": [warm_agg.get(m, float("nan")) for m in models],
        "rank_change": [warm_ranks.get(m, 0) - cold_ranks.get(m, 0) for m in models],
    })
    rank_df.to_csv(output_path.replace(".png", ".csv"), index=False)

    print(f"[Plot] Rank reversal saved to {output_path}")


def plot_ablation_heatmap(
    ablation_results: pd.DataFrame,
    output_path: str,
):
    """消融实验 heatmap"""
    if len(ablation_results) == 0:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Pivot: features × metrics
    pivot = ablation_results.pivot_table(
        index="ablation_group",
        columns="metric",
        values="value",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax,
                cbar_kws={"label": "RMSE (lower is better)"})
    ax.set_title("Ablation Study: Feature Group Impact on RMSE")
    ax.set_ylabel("Feature Group")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gating_curve(
    n_values: np.ndarray,
    alpha_values: np.ndarray,
    tau: int = 15,
    output_path: str = "",
):
    """绘制 Soft Gating 曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_values, alpha_values, linewidth=2, color="#9b59b6")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="α=0.5 (equal weight)")
    ax.axvline(tau, color="red", linestyle="--", alpha=0.5, label=f"τ={tau}")

    ax.fill_between(n_values, 0, alpha_values, alpha=0.2, color="#9b59b6")
    ax.text(n_values.max() * 0.3, 0.15, "Profile-Weighted", fontsize=10, color="#3498db")
    ax.text(n_values.max() * 0.6, 0.85, "Behavior-Weighted", fontsize=10, color="#e74c3c")

    ax.set_xlabel("User Interaction Count (n)")
    ax.set_ylabel("α (Behavior Weight)")
    ax.set_title("Learned Gating Curve: α = σ(w·log(1+n) + b)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.close()


def plot_model_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Model Comparison",
):
    """多模型 RMSE/MAE 对比柱状图"""
    if len(results_df) == 0:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    agg = results_df.groupby("model").agg({"rmse": "mean", "mae": "mean"}).reset_index()
    agg = agg.sort_values("rmse")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(agg)))

    ax1.bar(agg["model"], agg["rmse"], color=colors, edgecolor="white")
    ax1.set_title("RMSE")
    ax1.set_xticklabels(agg["model"], rotation=45, ha="right")
    for i, v in enumerate(agg["rmse"]):
        ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    ax2.bar(agg["model"], agg["mae"], color=colors, edgecolor="white")
    ax2.set_title("MAE")
    ax2.set_xticklabels(agg["model"], rotation=45, ha="right")
    for i, v in enumerate(agg["mae"]):
        ax2.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
