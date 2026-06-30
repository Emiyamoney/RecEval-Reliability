"""
协议间排名相关性分析

功能:
  计算模型排名在不同协议之间的 Spearman / Kendall 相关系数。
  量化 Protocol Sensitivity Index (PSI)。

输入:
  results/full/runs/{dataset}/{model}/{seed}/{split}/metrics.json

输出:
  analysis/results/protocol_ranking_correlation.csv
  analysis/results/psi_summary.csv

用法:
  python -m analysis.protocol_ranking_correlation
"""

import sys, os, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------- 项目根目录 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

RUNS_DIR = PROJECT_ROOT / "results" / "full" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results"
SEEDS = [2024, 2025, 2026, 2027, 2028]
PROTOCOLS = ["strict_cold", "warm_random", "warm_temporal"]


# ============================================================
#  数据加载
# ============================================================

def load_mean_rmse() -> pd.DataFrame:
    """加载所有 metrics.json，计算每个 (dataset, model, split) 的均值"""
    rows = []
    for metrics_path in RUNS_DIR.rglob("metrics.json"):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "dataset": m["dataset"],
            "model": m["model"],
            "split": m["split_type"],
            "seed": m["seed"],
            "rmse": m["rmse"],
        })
    df = pd.DataFrame(rows)
    # 按 (dataset, model, split) 分组求均值
    mean_df = df.groupby(["dataset", "model", "split"])["rmse"].mean().reset_index()
    return mean_df


# ============================================================
#  排名计算
# ============================================================

def compute_rankings(mean_df: pd.DataFrame) -> pd.DataFrame:
    """为每个 (dataset, split) 计算模型排名"""
    rankings = []
    for (dataset, split), group in mean_df.groupby(["dataset", "split"]):
        group_sorted = group.sort_values("rmse")
        for rank, (_, row) in enumerate(group_sorted.iterrows(), 1):
            rankings.append({
                "dataset": dataset,
                "model": row["model"],
                "split": split,
                "rmse": row["rmse"],
                "rank": rank,
            })
    return pd.DataFrame(rankings)


# ============================================================
#  相关性分析
# ============================================================

def compute_pairwise_correlation(rankings: pd.DataFrame) -> pd.DataFrame:
    """计算协议两两之间的 Spearman / Kendall 相关系数"""
    results = []
    datasets = rankings["dataset"].unique()

    for dataset in datasets:
        ds_data = rankings[rankings["dataset"] == dataset]
        splits = ds_data["split"].unique()

        for i, s1 in enumerate(splits):
            for j, s2 in enumerate(splits):
                if i >= j:
                    continue

                r1 = ds_data[ds_data["split"] == s1][["model", "rank"]].set_index("model")["rank"]
                r2 = ds_data[ds_data["split"] == s2][["model", "rank"]].set_index("model")["rank"]

                # 对齐模型
                common = r1.index.intersection(r2.index)
                if len(common) < 3:
                    continue

                spearman_rho, spearman_p = stats.spearmanr(r1[common], r2[common])
                kendall_tau, kendall_p = stats.kendalltau(r1[common], r2[common])

                # 排名变化
                rank_diff = (r1[common] - r2[common]).abs()

                results.append({
                    "dataset": dataset,
                    "protocol_a": s1,
                    "protocol_b": s2,
                    "n_models": len(common),
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "kendall_tau": kendall_tau,
                    "kendall_p": kendall_p,
                    "mean_abs_rank_change": rank_diff.mean(),
                    "max_abs_rank_change": rank_diff.max(),
                })

    return pd.DataFrame(results)


def compute_psi(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Protocol Sensitivity Index (PSI)
    PSI(m) = avg |rank_p1(m) - rank_p2(m)| across all protocol pairs
    """
    psi_results = []
    datasets = rankings["dataset"].unique()

    for dataset in datasets:
        ds_data = rankings[rankings["dataset"] == dataset]
        models = ds_data["model"].unique()

        for model in models:
            model_data = ds_data[ds_data["model"] == model]
            ranks = model_data.set_index("split")["rank"]

            # 所有协议对
            splits = model_data["split"].values
            diffs = []
            for i in range(len(splits)):
                for j in range(i + 1, len(splits)):
                    s1, s2 = splits[i], splits[j]
                    if s1 in ranks.index and s2 in ranks.index:
                        diffs.append(abs(ranks[s1] - ranks[s2]))

            psi = np.mean(diffs) if diffs else 0.0
            max_rank = ranks.max()
            min_rank = ranks.min()

            psi_results.append({
                "dataset": dataset,
                "model": model,
                "psi": psi,
                "rank_range": max_rank - min_rank,
                "best_protocol": ranks.idxmin(),
                "worst_protocol": ranks.idxmax(),
            })

    return pd.DataFrame(psi_results)


# ============================================================
#  LaTeX 表格输出
# ============================================================

def to_latex_correlation(df: pd.DataFrame, output_path: Path):
    """生成协议相关性 LaTeX 表格"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Protocol Pairwise Ranking Correlation (Spearman $\\rho$ / Kendall $\\tau$).}\n"
        "\\label{tab:protocol_corr}\n"
        "\\begin{tabular}{llccccc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Protocol Pair} & \\textbf{Spearman $\\rho$} "
        "& \\textbf{p-value} & \\textbf{Kendall $\\tau$} & \\textbf{Mean $|\\Delta$rank$|$} "
        "& \\textbf{Max $|\\Delta$rank$|$} \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for _, r in df.iterrows():
        label = f"{r['protocol_a']} vs {r['protocol_b']}"
        rows.append(
            f"{r['dataset']} & {label} & "
            f"{r['spearman_rho']:.3f} & {r['spearman_p']:.4f} & "
            f"{r['kendall_tau']:.3f} & {r['mean_abs_rank_change']:.2f} & "
            f"{r['max_abs_rank_change']:.0f} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(rows))
        f.write("\n")
        f.write(footer)

    print(f"LaTeX table -> {output_path}")


def to_latex_psi(df: pd.DataFrame, output_path: Path):
    """生成 PSI LaTeX 表格"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Protocol Sensitivity Index (PSI) per Model.}\n"
        "\\label{tab:psi}\n"
        "\\begin{tabular}{llccccc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{PSI} & \\textbf{Rank Range} "
        "& \\textbf{Best Protocol} & \\textbf{Worst Protocol} \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['dataset']} & {r['model']} & "
            f"{r['psi']:.2f} & {r['rank_range']:.0f} & "
            f"{r['best_protocol']} & {r['worst_protocol']} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(rows))
        f.write("\n")
        f.write(footer)

    print(f"LaTeX table -> {output_path}")


# ============================================================
#  主函数
# ============================================================

def main():
    print("=" * 60)
    print("Protocol Ranking Correlation Analysis")
    print("=" * 60)

    # 加载数据
    mean_df = load_mean_rmse()
    print(f"Loaded {len(mean_df)} (dataset, model, split) combinations")

    # 计算排名
    rankings = compute_rankings(mean_df)
    print(f"Computed rankings for {rankings['model'].nunique()} models")

    # 协议相关性
    corr_df = compute_pairwise_correlation(rankings)
    print(f"\nPairwise correlation ({len(corr_df)} pairs):")
    print(corr_df[["dataset", "protocol_a", "protocol_b", "spearman_rho", "mean_abs_rank_change"]].to_string())

    # PSI
    psi_df = compute_psi(rankings)
    print(f"\nPSI summary:")
    print(psi_df.groupby("dataset")["psi"].agg(["mean", "max"]).round(2))

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(OUTPUT_DIR / "protocol_ranking_correlation.csv", index=False)
    psi_df.to_csv(OUTPUT_DIR / "psi_summary.csv", index=False)

    to_latex_correlation(corr_df, OUTPUT_DIR / "protocol_ranking_correlation.tex")
    to_latex_psi(psi_df, OUTPUT_DIR / "psi_summary.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()
