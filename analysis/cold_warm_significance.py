"""
Cold-Warm 显著性检验

功能:
  对每个 (模型, 数据集, 协议) 组合，检验 cold_rmse 与 warm_rmse 差异的统计显著性。
  使用 5 个 seed 的 cold_rmse / warm_rmse 配对样本 t-test。

输入:
  results/full/runs/{dataset}/{model}/{seed}/{split}/metrics.json

输出:
  analysis/results/cold_warm_significance.csv

用法:
  python -m analysis.cold_warm_significance
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


# ============================================================
#  数据加载
# ============================================================

def load_all_metrics() -> pd.DataFrame:
    """递归读取所有 metrics.json，返回 DataFrame"""
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
            "cold_rmse": m.get("cold_rmse"),
            "warm_rmse": m.get("warm_rmse"),
            "cold_n": m.get("cold_n"),
            "warm_n": m.get("warm_n"),
        })
    return pd.DataFrame(rows)


# ============================================================
#  显著性检验
# ============================================================

def test_cold_warm_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 (dataset, model, split) 组合：
    - 收集 5 个 seed 的 cold_rmse 和 warm_rmse
    - 配对 t-test 检验 cold > warm
    - 计算 Cohen's d 效应量
    """
    results = []
    grouped = df.groupby(["dataset", "model", "split"])

    for (dataset, model, split), group in grouped:
        if len(group) < 3:
            continue

        cold = group["cold_rmse"].dropna().values
        warm = group["warm_rmse"].dropna().values

        if len(cold) < 3 or len(warm) < 3:
            continue

        # 配对 t-test (cold - warm > 0)
        diff = cold - warm
        t_stat, p_value = stats.ttest_1samp(diff, 0.0)
        # 单尾检验: cold > warm
        p_one_tail = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        # Cohen's d
        mean_diff = np.mean(cold) - np.mean(warm)
        pooled_std = np.sqrt((np.var(cold, ddof=1) + np.var(warm, ddof=1)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # 效应量解读
        if abs(cohens_d) < 0.2:
            interp = "negligible"
        elif abs(cohens_d) < 0.5:
            interp = "small"
        elif abs(cohens_d) < 0.8:
            interp = "medium"
        else:
            interp = "large"

        results.append({
            "dataset": dataset,
            "model": model,
            "split": split,
            "n_seeds": len(cold),
            "cold_rmse_mean": np.mean(cold),
            "cold_rmse_std": np.std(cold, ddof=1),
            "warm_rmse_mean": np.mean(warm),
            "warm_rmse_std": np.std(warm, ddof=1),
            "gap": mean_diff,
            "gap_pct": mean_diff / np.mean(warm) * 100,
            "t_statistic": t_stat,
            "p_value_two_tail": p_value,
            "p_value_one_tail": p_one_tail,
            "significant_005": p_one_tail < 0.05,
            "significant_001": p_one_tail < 0.01,
            "cohens_d": cohens_d,
            "effect_size": interp,
        })

    return pd.DataFrame(results)


# ============================================================
#  LaTeX 表格输出
# ============================================================

def to_latex(df: pd.DataFrame, output_path: Path):
    """生成 LaTeX 表格"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Cold-Warm Gap Statistical Significance (5-seed paired t-test).}\n"
        "\\label{tab:cold_warm_sig}\n"
        "\\begin{tabular}{llccccc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{Cold RMSE} & \\textbf{Warm RMSE} "
        "& \\textbf{Gap} & \\textbf{p-value} & \\textbf{Cohen's $d$} \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for _, r in df.iterrows():
        sig = "$^{\\ast\\ast}$" if r["significant_001"] else ("$^{\\ast}$" if r["significant_005"] else "")
        rows.append(
            f"{r['dataset']} & {r['model']} & "
            f"{r['cold_rmse_mean']:.4f} & {r['warm_rmse_mean']:.4f} & "
            f"{r['gap']:.4f} & {r['p_value_two_tail']:.4f}{sig} & "
            f"{r['cohens_d']:.2f} ({r['effect_size']}) \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\multicolumn{7}{l}{\\footnotesize $^{*} p<0.05$, $^{**} p<0.01$ (one-tailed paired t-test)}\n"
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
    print("Cold-Warm Statistical Significance Test")
    print("=" * 60)

    # 加载数据
    df = load_all_metrics()
    print(f"Loaded {len(df)} metric records")

    # 运行检验
    results = test_cold_warm_gap(df)
    print(f"Computed significance for {len(results)} (dataset, model, split) combinations")

    # 保存 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "cold_warm_significance.csv"
    results.to_csv(csv_path, index=False)
    print(f"CSV -> {csv_path}")

    # 保存 LaTeX
    latex_path = OUTPUT_DIR / "cold_warm_significance.tex"
    to_latex(results, latex_path)

    # 打印摘要
    n_sig_005 = results["significant_005"].sum()
    n_sig_001 = results["significant_001"].sum()
    n_total = len(results)
    print(f"\nSummary: {n_sig_005}/{n_total} significant at p<0.05, {n_sig_001}/{n_total} at p<0.01")

    # 按数据集汇总
    print("\nMean gap by dataset:")
    print(results.groupby("dataset")["gap"].agg(["mean", "std", "min", "max"]).round(4))


if __name__ == "__main__":
    main()
