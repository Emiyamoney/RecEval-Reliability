"""
Ranking Metrics 汇总分析

功能:
  聚合 427 个 metrics.json 中的 Recall@K、NDCG@K、HitRate@K，
  生成论文用的 ranking metrics 表格。

输入:
  results/full/runs/{dataset}/{model}/{seed}/{split}/metrics.json

输出:
  analysis/results/ranking_metrics_summary.csv
  analysis/results/ranking_metrics_by_dataset.csv

用法:
  python -m analysis.ranking_metrics_summary
"""

import sys, os, json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 项目根目录 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

RUNS_DIR = PROJECT_ROOT / "results" / "full" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "results"
K_VALUES = [5, 10, 20]


# ============================================================
#  数据加载
# ============================================================

def load_all_metrics() -> pd.DataFrame:
    """递归读取所有 metrics.json，提取 ranking metrics"""
    rows = []
    for metrics_path in RUNS_DIR.rglob("metrics.json"):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "dataset": m["dataset"],
            "model": m["model"],
            "split": m["split_type"],
            "seed": m["seed"],
            "rmse": m["rmse"],
            "cold_rmse": m.get("cold_rmse"),
            "warm_rmse": m.get("warm_rmse"),
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = m.get(f"recall@{k}")
            row[f"ndcg@{k}"] = m.get(f"ndcg@{k}")
            row[f"hit@{k}"] = m.get(f"hit@{k}")
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
#  聚合分析
# ============================================================

def aggregate_by_group(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """按指定列分组，计算均值和标准差（仅数值列）"""
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"] and df[c].dtype in ["float64", "int64", "float32"]]
    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = ["mean", "std"]

    agg = df.groupby(group_cols).agg(agg_dict).reset_index()

    # 扁平化多级列名
    agg.columns = ["_".join(c).rstrip("_") if c[1] else c[0] for c in agg.columns]
    return agg


def create_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建论文主表：每个 (dataset, model, split) 的 ranking metrics
    包含 mean ± std
    """
    group_cols = ["dataset", "model", "split"]
    agg = aggregate_by_group(df, group_cols)

    # 选取关键列
    key_cols = group_cols.copy()
    key_cols.append("rmse_mean")
    key_cols.append("rmse_std")
    for k in K_VALUES:
        key_cols.extend([f"recall@{k}_mean", f"recall@{k}_std",
                         f"ndcg@{k}_mean", f"ndcg@{k}_std"])

    return agg[key_cols]


def create_by_dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    按数据集汇总：每个 (dataset, model) 在所有协议上的平均表现
    """
    group_cols = ["dataset", "model"]
    agg = aggregate_by_group(df, group_cols)

    key_cols = group_cols.copy()
    key_cols.append("rmse_mean")
    key_cols.append("rmse_std")
    for k in K_VALUES:
        key_cols.extend([f"recall@{k}_mean", f"recall@{k}_std",
                         f"ndcg@{k}_mean", f"ndcg@{k}_std"])

    return agg[key_cols]


# ============================================================
#  LaTeX 输出
# ============================================================

def format_mean_std(mean: float, std: float) -> str:
    """格式化 mean ± std"""
    if pd.isna(mean) or pd.isna(std):
        return "---"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def to_latex_main(df: pd.DataFrame, output_path: Path):
    """生成论文主表 LaTeX"""
    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Ranking Metrics Summary (5-seed mean $\\pm$ std).}\n"
        "\\label{tab:ranking_metrics}\n"
        "\\begin{tabular}{lllccccc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{Split} & "
        "\\textbf{RMSE} & "
        "\\textbf{Recall@10} & \\textbf{NDCG@10} & "
        "\\textbf{Recall@20} & \\textbf{NDCG@20} \\\\\n"
        "\\midrule\n"
    )

    rows = []
    prev_dataset = None
    for _, r in df.iterrows():
        # 数据集分隔线
        if prev_dataset is not None and r["dataset"] != prev_dataset:
            rows.append("\\midrule")
        prev_dataset = r["dataset"]

        rmse = format_mean_std(r["rmse_mean"], r["rmse_std"])
        r10 = format_mean_std(r["recall@10_mean"], r["recall@10_std"])
        n10 = format_mean_std(r["ndcg@10_mean"], r["ndcg@10_std"])
        r20 = format_mean_std(r["recall@20_mean"], r["recall@20_std"])
        n20 = format_mean_std(r["ndcg@20_mean"], r["ndcg@20_std"])

        rows.append(
            f"{r['dataset']} & {r['model']} & {r['split']} & "
            f"{rmse} & {r10} & {n10} & {r20} & {n20} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
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
    print("Ranking Metrics Summary")
    print("=" * 60)

    # 加载数据
    df = load_all_metrics()
    print(f"Loaded {len(df)} metric records")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Splits: {df['split'].unique()}")

    # 创建主表
    main_table = create_main_table(df)
    print(f"\nMain table: {len(main_table)} rows")

    # 创建数据集汇总表
    by_dataset = create_by_dataset_table(df)
    print(f"By-dataset table: {len(by_dataset)} rows")

    # 保存 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(OUTPUT_DIR / "ranking_metrics_summary.csv", index=False)
    by_dataset.to_csv(OUTPUT_DIR / "ranking_metrics_by_dataset.csv", index=False)

    # 保存 LaTeX
    to_latex_main(main_table, OUTPUT_DIR / "ranking_metrics_summary.tex")

    # 打印 Top-5 by Recall@20
    print("\nTop-5 models by Recall@20 (warm_random):")
    wr = df[df["split"] == "warm_random"]
    top5 = wr.groupby("model")["recall@20"].mean().sort_values(ascending=False).head()
    print(top5.round(6))

    # 打印 Top-5 by NDCG@20
    print("\nTop-5 models by NDCG@20 (warm_random):")
    top5 = wr.groupby("model")["ndcg@20"].mean().sort_values(ascending=False).head()
    print(top5.round(4))

    print("\nDone.")


if __name__ == "__main__":
    main()
