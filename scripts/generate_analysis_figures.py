"""
Generate analysis figures from supplementary results.

Figures:
  fig_s1_cold_warm_significance.png  - Cold-Warm gap with significance markers
  fig_s2_protocol_correlation.png    - Protocol ranking correlation heatmap
  fig_s3_psi_summary.png             - Protocol Sensitivity Index per model
  fig_s4_ranking_metrics.png         - Recall@K and NDCG@K comparison

Usage:
  python -m scripts.generate_analysis_figures
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results")
OUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
#  Figure 1: Cold-Warm Significance
# ============================================================

def fig_cold_warm_significance():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "cold_warm_significance.csv"))

    # 选取 ML-1M warm_random (核心结果)
    subset = df[(df["dataset"] == "ml1m") & (df["split"] == "warm_random")].copy()
    subset = subset.sort_values("gap", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = []
    for _, row in subset.iterrows():
        if row["significant_001"]:
            colors.append("#2196F3")
        elif row["significant_005"]:
            colors.append("#64B5F6")
        else:
            colors.append("#BDBDBD")

    bars = ax.barh(subset["model"], subset["gap"], color=colors, edgecolor="white", height=0.7)

    # 显著性标记
    for i, (_, row) in enumerate(subset.iterrows()):
        if row["significant_001"]:
            ax.text(row["gap"] + 0.01, i, "**", va="center", fontsize=9, color="#1565C0")
        elif row["significant_005"]:
            ax.text(row["gap"] + 0.01, i, "*", va="center", fontsize=9, color="#1976D2")

    ax.set_xlabel("Cold-Warm RMSE Gap")
    ax.set_title("Cold-Warm Performance Gap (ML-1M, warm-random)")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="p < 0.01"),
        Patch(facecolor="#64B5F6", label="p < 0.05"),
        Patch(facecolor="#BDBDBD", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_s1_cold_warm_significance.png"))
    plt.close()
    print("fig_s1_cold_warm_significance.png saved")


# ============================================================
#  Figure 2: Protocol Ranking Correlation Heatmap
# ============================================================

def fig_protocol_correlation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "protocol_ranking_correlation.csv"))

    datasets = ["ml1m", "goodbooks", "book_crossing"]
    pairs = [
        ("strict_cold", "warm_random"),
        ("strict_cold", "warm_temporal"),
        ("warm_random", "warm_temporal"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_data = df[df["dataset"] == dataset]

        matrix = np.zeros((3, 3))
        for _, row in ds_data.iterrows():
            i = pairs.index((row["protocol_a"], row["protocol_b"]))
            matrix[i, 0] = row["spearman_rho"]
            matrix[i, 1] = row["mean_abs_rank_change"]
            matrix[i, 2] = row["max_abs_rank_change"]

        # 绘制 Spearman rho 热力图
        im = ax.imshow(matrix[:, 0:1].T, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(3))
        ax.set_xticklabels(["SC vs WR", "SC vs WT", "WR vs WT"], fontsize=8, rotation=45)
        ax.set_yticks([0])
        ax.set_yticklabels(["Spearman $\\rho$"])
        ax.set_title(dataset.upper().replace("_", "-"))

        # 标注数值
        for i in range(3):
            val = matrix[i, 0]
            ax.text(i, 0, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman $\\rho$")
    plt.suptitle("Protocol Pairwise Ranking Correlation", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_s2_protocol_correlation.png"), bbox_inches="tight")
    plt.close()
    print("fig_s2_protocol_correlation.png saved")


# ============================================================
#  Figure 3: PSI Summary
# ============================================================

def fig_psi_summary():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "psi_summary.csv"))

    datasets = ["ml1m", "goodbooks", "book_crossing"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_data = df[df["dataset"] == dataset].sort_values("psi", ascending=True)

        colors = ["#E53935" if v > 4 else ("#FFA726" if v > 2 else "#66BB6A")
                  for v in ds_data["psi"]]

        ax.barh(ds_data["model"], ds_data["psi"], color=colors, edgecolor="white", height=0.7)
        ax.set_xlabel("PSI")
        ax.set_title(dataset.upper().replace("_", "-"))
        ax.axvline(x=4, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E53935", label="PSI > 4 (high)"),
        Patch(facecolor="#FFA726", label="2 < PSI < 4 (medium)"),
        Patch(facecolor="#66BB6A", label="PSI < 2 (low)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("Protocol Sensitivity Index (PSI) per Model", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_s3_psi_summary.png"), bbox_inches="tight")
    plt.close()
    print("fig_s3_psi_summary.png saved")


# ============================================================
#  Figure 4: Ranking Metrics
# ============================================================

def fig_ranking_metrics():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_metrics_summary.csv"))

    # 选取 warm_random 协议
    wr = df[df["split"] == "warm_random"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Recall@10 by dataset
    ax = axes[0]
    for i, dataset in enumerate(["ml1m", "goodbooks", "book_crossing"]):
        ds_data = wr[wr["dataset"] == dataset].sort_values("recall@10_mean", ascending=True)
        y_pos = np.arange(len(ds_data)) + i * 0.25
        ax.barh(y_pos, ds_data["recall@10_mean"], height=0.25,
                label=dataset.upper().replace("_", "-"), alpha=0.85)
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ds_data["model"], fontsize=8)

    ax.set_xlabel("Recall@10")
    ax.set_title("Recall@10 by Model (warm-random)")
    ax.legend(fontsize=7)

    # NDCG@10 by dataset
    ax = axes[1]
    for i, dataset in enumerate(["ml1m", "goodbooks", "book_crossing"]):
        ds_data = wr[wr["dataset"] == dataset].sort_values("ndcg@10_mean", ascending=True)
        y_pos = np.arange(len(ds_data)) + i * 0.25
        ax.barh(y_pos, ds_data["ndcg@10_mean"], height=0.25,
                label=dataset.upper().replace("_", "-"), alpha=0.85)
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ds_data["model"], fontsize=8)

    ax.set_xlabel("NDCG@10")
    ax.set_title("NDCG@10 by Model (warm-random)")
    ax.legend(fontsize=7)

    plt.suptitle("Ranking Metrics Comparison", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_s4_ranking_metrics.png"), bbox_inches="tight")
    plt.close()
    print("fig_s4_ranking_metrics.png saved")


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Generating Analysis Figures")
    print("=" * 50)
    fig_cold_warm_significance()
    fig_protocol_correlation()
    fig_psi_summary()
    fig_ranking_metrics()
    print("\nAll figures saved to:", OUT_DIR)
