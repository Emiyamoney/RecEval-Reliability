"""
result_final 数据格式转换脚本
将 md 报告转换为论文友好的 CSV + LaTeX 格式

输入: result_final/smoke/ 下的 summary.md + 各 report.md
输出: result_final/smoke/paper_formats/ 下的 CSV + LaTeX 文件
"""
import os, re, json, csv, glob
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(PROJECT_ROOT, "result_final", "smoke")
OUTPUT_DIR = os.path.join(RESULT_DIR, "paper_formats")


def parse_summary_md(path):
    """解析 summary.md 中的 Completed 表格"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    in_completed = False
    for line in content.split("\n"):
        if line.startswith("## Completed"):
            in_completed = True
            continue
        if line.startswith("## Failed"):
            in_completed = False
            continue
        if in_completed and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6 or parts[0] == "Run":
                continue
            run_name = parts[1]
            segs = run_name.split("__")
            if len(segs) >= 4:
                dataset = segs[0]
                model = segs[1]
                seed = segs[2].replace("seed", "")
                split = segs[3]
            else:
                dataset, model, seed, split = run_name, "", "", ""
            rows.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "split": split,
                "rmse": parts[2] if parts[2] != "N/A" else "",
                "mae": parts[3] if parts[3] != "N/A" else "",
                "cold_rmse": parts[4] if parts[4] != "N/A" else "",
                "warm_rmse": parts[5] if parts[5] != "N/A" else "",
                "time": parts[6] if len(parts) > 6 else "",
            })
    return rows


def parse_report_md(path):
    """解析单个 report.md"""
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    in_metrics = False
    for line in content.split("\n"):
        if "## 评估指标" in line or "## Metrics" in line:
            in_metrics = True
            continue
        if in_metrics and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0] != "Metric" and parts[0] != "指标":
                key = parts[1].lower().replace(" ", "_")
                val = parts[2]
                metrics[key] = val
    return metrics


def parse_failed_runs(path):
    """解析 Failed 部分"""
    failed = []
    in_failed = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## Failed"):
                in_failed = True
                continue
            if in_failed and line.startswith("- **"):
                m = re.match(r"- \*\*(.+?)\*\*: (.+)", line.strip())
                if m:
                    run_name, error = m.group(1), m.group(2)
                    segs = run_name.split("__")
                    if len(segs) >= 4:
                        failed.append({
                            "dataset": segs[0], "model": segs[1],
                            "seed": segs[2].replace("seed", ""), "split": segs[3],
                            "error": error,
                        })
            if in_failed and line.startswith("## ") and "Failed" not in line:
                break
    return failed


def generate_paper_tables(df):
    """生成论文用 LaTeX 表格"""
    tables = []

    for dataset in df["dataset"].unique():
        ds_df = df[df["dataset"] == dataset].copy()
        ds_df["rmse"] = pd.to_numeric(ds_df["rmse"], errors="coerce")

        # 按 model 分组，计算 mean±std
        agg = ds_df.groupby("model")["rmse"].agg(["mean", "std"]).sort_values("mean")
        agg["std"] = agg["std"].fillna(0)

        # 生成 LaTeX
        latex_lines = []
        latex_lines.append(f"% {dataset} — RMSE by model (mean ± std across seeds)")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{RMSE comparison on " + dataset.replace("_", "-") + "}")
        latex_lines.append(r"\label{tab:" + dataset + "_rmse}")
        latex_lines.append(r"\begin{tabular}{lrr}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Model & RMSE & Std \\")
        latex_lines.append(r"\midrule")
        for model, row in agg.iterrows():
            latex_lines.append(f"{model} & {row['mean']:.4f} & {row['std']:.4f} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        tables.append(("\n".join(latex_lines), dataset))

    return tables


def generate_cold_warm_table(df):
    """生成 cold vs warm 分组对比表"""
    latex_lines = []
    latex_lines.append(r"% Cold-start vs Warm-start RMSE comparison")
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Cold vs Warm RMSE by model}")
    latex_lines.append(r"\label{tab:cold_warm}")
    latex_lines.append(r"\begin{tabular}{llrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & Model & Cold RMSE & Warm RMSE \\")
    latex_lines.append(r"\midrule")

    for dataset in df["dataset"].unique():
        ds_df = df[df["dataset"] == dataset]
        for model in ds_df["model"].unique():
            m_df = ds_df[ds_df["model"] == model]
            cold = pd.to_numeric(m_df["cold_rmse"], errors="coerce").dropna()
            warm = pd.to_numeric(m_df["warm_rmse"], errors="coerce").dropna()
            cold_str = f"{cold.mean():.4f}" if len(cold) > 0 else "N/A"
            warm_str = f"{warm.mean():.4f}" if len(warm) > 0 else "N/A"
            ds_label = dataset.replace("_", "-")
            latex_lines.append(f"{ds_label} & {model} & {cold_str} & {warm_str} \\\\")
        if dataset != df["dataset"].unique()[-1]:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    return "\n".join(latex_lines)


def generate_seed42_table(df):
    """生成 seed=42 单独对比表（论文主表）"""
    s42 = df[df["seed"] == "42"].copy()
    s42["rmse"] = pd.to_numeric(s42["rmse"], errors="coerce")

    latex_lines = []
    latex_lines.append(r"% Seed=42 results — primary comparison table")
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Results with seed=42 (single run)}")
    latex_lines.append(r"\label{tab:seed42}")
    latex_lines.append(r"\begin{tabular}{llrrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & Model & Split & RMSE & MAE \\")
    latex_lines.append(r"\midrule")

    for dataset in s42["dataset"].unique():
        ds_df = s42[s42["dataset"] == dataset].sort_values("model")
        for _, row in ds_df.iterrows():
            mae = row["mae"] if row["mae"] else "N/A"
            ds_label = dataset.replace("_", "-")
            latex_lines.append(
                f"{ds_label} & {row['model']} & {row['split']} "
                f"& {row['rmse']:.4f} & {mae} \\\\"
            )
        if dataset != s42["dataset"].unique()[-1]:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    return "\n".join(latex_lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Convert] Output dir: {OUTPUT_DIR}")

    # 1. 解析 summary.md
    summary_path = os.path.join(RESULT_DIR, "summary.md")
    rows = parse_summary_md(summary_path)
    print(f"[Convert] Parsed {len(rows)} completed runs from summary.md")

    # 2. 扫描所有 report.md 补充数据
    report_files = glob.glob(os.path.join(RESULT_DIR, "runs", "*", "*", "*", "*", "report.md"))
    report_count = 0
    for rf in report_files:
        parts = rf.replace(RESULT_DIR, "").strip(os.sep).split(os.sep)
        if len(parts) >= 6:
            # runs/dataset/model/seed/split/report.md
            dataset, model, seed, split = parts[1], parts[2], parts[3].replace("seed", ""), parts[4]
            metrics = parse_report_md(rf)
            if metrics:
                report_count += 1
                # 补充 summary 中缺少的指标
                for row in rows:
                    if (row["dataset"] == dataset and row["model"] == model
                            and row["seed"] == seed and row["split"] == split):
                        for k, v in metrics.items():
                            if k not in row or not row[k]:
                                row[k] = v
                        break

    print(f"[Convert] Enhanced with {report_count} report.md files")

    # 3. 写入 CSV
    csv_path = os.path.join(OUTPUT_DIR, "all_results.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[Convert] CSV written: {csv_path} ({len(rows)} rows)")

    # 4. 写入失败运行
    failed = parse_failed_runs(summary_path)
    if failed:
        fail_csv = os.path.join(OUTPUT_DIR, "failed_runs.csv")
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(failed[0].keys()))
            w.writeheader()
            w.writerows(failed)
        print(f"[Convert] Failed runs CSV: {fail_csv} ({len(failed)} rows)")

    # 5. 生成 LaTeX 表格
    df = pd.DataFrame(rows)
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")

    # 按数据集的 RMSE 表
    tables = generate_paper_tables(df)
    for tex_content, dataset in tables:
        tex_path = os.path.join(OUTPUT_DIR, f"table_{dataset}_rmse.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        print(f"[Convert] LaTeX table: {tex_path}")

    # Cold vs Warm 对比表
    cold_warm_tex = generate_cold_warm_table(df)
    cw_path = os.path.join(OUTPUT_DIR, "table_cold_warm.tex")
    with open(cw_path, "w", encoding="utf-8") as f:
        f.write(cold_warm_tex)
    print(f"[Convert] LaTeX cold/warm table: {cw_path}")

    # Seed42 主表
    seed42_tex = generate_seed42_table(df)
    s42_path = os.path.join(OUTPUT_DIR, "table_seed42.tex")
    with open(s42_path, "w", encoding="utf-8") as f:
        f.write(seed42_tex)
    print(f"[Convert] LaTeX seed42 table: {s42_path}")

    # 6. 生成论文用多 seed 汇总表 (mean ± std)
    agg_rows = []
    for dataset in df["dataset"].unique():
        for model in df[df["dataset"] == dataset]["model"].unique():
            sub = df[(df["dataset"] == dataset) & (df["model"] == model)]
            for split in sub["split"].unique():
                ssub = sub[sub["split"] == split]
                rmse_vals = pd.to_numeric(ssub["rmse"], errors="coerce").dropna()
                mae_vals = pd.to_numeric(ssub["mae"], errors="coerce").dropna()
                if len(rmse_vals) > 0:
                    agg_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "split": split,
                        "n_seeds": len(rmse_vals),
                        "rmse_mean": f"{rmse_vals.mean():.4f}",
                        "rmse_std": f"{rmse_vals.std():.4f}" if len(rmse_vals) > 1 else "N/A",
                        "mae_mean": f"{mae_vals.mean():.4f}" if len(mae_vals) > 0 else "N/A",
                        "mae_std": f"{mae_vals.std():.4f}" if len(mae_vals) > 1 else "N/A",
                    })

    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(OUTPUT_DIR, "aggregated_results.csv")
    agg_df.to_csv(agg_csv, index=False, encoding="utf-8")
    print(f"[Convert] Aggregated CSV: {agg_csv} ({len(agg_rows)} rows)")

    # 生成聚合 LaTeX 表
    agg_latex_lines = []
    agg_latex_lines.append(r"% Aggregated results: mean ± std across seeds")
    agg_latex_lines.append(r"\begin{table*}[t]")
    agg_latex_lines.append(r"\centering")
    agg_latex_lines.append(r"\caption{Aggregated Results (mean $\pm$ std across 5 seeds)}")
    agg_latex_lines.append(r"\label{tab:aggregated}")
    agg_latex_lines.append(r"\begin{tabular}{lllrr}")
    agg_latex_lines.append(r"\toprule")
    agg_latex_lines.append(r"Dataset & Model & Split & RMSE & MAE \\")
    agg_latex_lines.append(r"\midrule")

    prev_ds = None
    for _, row in agg_df.iterrows():
        ds_label = row["dataset"].replace("_", "-")
        sep = r" \midrule" if prev_ds and prev_ds != row["dataset"] else ""
        if sep:
            agg_latex_lines.append(sep)
        mae_str = f"{row['mae_mean']}"
        if row["mae_std"] != "N/A":
            mae_str = f"{row['mae_mean']}$_{{{row['mae_std']}}}$"
        rmse_str = f"{row['rmse_mean']}"
        if row["rmse_std"] != "N/A":
            rmse_str = f"{row['rmse_mean']}$_{{{row['rmse_std']}}}$"
        agg_latex_lines.append(
            f"{ds_label} & {row['model']} & {row['split']} & {rmse_str} & {mae_str} \\\\"
        )
        prev_ds = row["dataset"]

    agg_latex_lines.append(r"\bottomrule")
    agg_latex_lines.append(r"\end{tabular}")
    agg_latex_lines.append(r"\end{table*}")

    agg_tex = os.path.join(OUTPUT_DIR, "table_aggregated.tex")
    with open(agg_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(agg_latex_lines))
    print(f"[Convert] LaTeX aggregated table: {agg_tex}")

    # 7. 转换 leakage check report
    leakage_path = os.path.join(RESULT_DIR, "leakage_check_report.md")
    if os.path.exists(leakage_path):
        leakage = parse_report_md(leakage_path)
        # 简单转为 JSON
        with open(leakage_path, "r", encoding="utf-8") as f:
            lc = f.read()
        lc_json = {"checks": [], "verdict": ""}
        for line in lc.split("\n"):
            m = re.match(r"- (.+?): PASS", line)
            if m:
                lc_json["checks"].append({"name": m.group(1), "status": "PASS"})
            m = re.match(r"- \*\*\[(\w+)\]\*\* (.+)", line)
            if m:
                lc_json["checks"].append({"name": m.group(2), "status": m.group(1)})
            if "VERDICT:" in line:
                lc_json["verdict"] = line.split("VERDICT:")[1].strip().rstrip("*").strip()
        lc_out = os.path.join(OUTPUT_DIR, "leakage_check.json")
        with open(lc_out, "w", encoding="utf-8") as f:
            json.dump(lc_json, f, indent=2, ensure_ascii=False)
        print(f"[Convert] Leakage check JSON: {lc_out}")

    # 8. 生成汇总统计
    stats = {
        "total_runs": len(rows),
        "failed_runs": len(failed),
        "datasets": df["dataset"].unique().tolist(),
        "models": df["model"].unique().tolist(),
        "splits": df["split"].unique().tolist(),
        "seeds": df["seed"].unique().tolist(),
    }
    stats_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Convert] Dataset info: {stats_path}")

    print(f"\n[Convert] Done! All files in: {OUTPUT_DIR}")
    print(f"  - all_results.csv: {len(rows)} rows (raw per-run)")
    print(f"  - aggregated_results.csv: {len(agg_rows)} rows (mean±std)")
    print(f"  - failed_runs.csv: {len(failed)} rows")
    print(f"  - table_*.tex: {len(tables)+3} LaTeX tables")
    print(f"  - leakage_check.json: structured leak report")


if __name__ == "__main__":
    main()
