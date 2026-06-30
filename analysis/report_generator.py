"""
自动实验报告生成器
从实际运行结果自动生成 Markdown 论文报告
不预设结论，所有数据来自实际运行
"""

import os, json, datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def generate_report(
    results_dir: str,
    output_path: str = "",
) -> str:
    """从 results/ 目录自动生成实验报告"""
    os.makedirs(os.path.dirname(output_path) or results_dir, exist_ok=True)

    if not output_path:
        output_path = os.path.join(results_dir, "report.md")

    # Collect data
    main_results = _load_csv(os.path.join(results_dir, "tables", "main_results.csv"))
    tau_results = _load_csv(os.path.join(results_dir, "tau_scan", "tau_scan_results.csv"))
    significance = _load_csv(os.path.join(results_dir, "statistics", "significance_tests.csv"))
    ablation = _load_csv(os.path.join(results_dir, "tables", "ablation_results.csv"))
    final_tau = _load_json(os.path.join(results_dir, "analysis", "final_tau.json"))

    L = []
    w = L.append

    # Title
    w("# Cold-Start vs Warm-Start: Adaptive Recommendation Experiment Report\n\n")
    w(f"> Auto-generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    w("---\n\n")

    # 1. Dataset Statistics
    w("## 1. Dataset Statistics\n\n")
    if len(main_results) > 0:
        for ds in main_results["dataset"].unique():
            ds_df = main_results[main_results["dataset"] == ds]
            w(f"### {ds}\n\n")
            w(f"- Total experiments: {len(ds_df)}\n")
            w(f"- Models evaluated: {ds_df['model'].nunique()}\n")
            w(f"- Split types: {', '.join(ds_df['split_type'].unique())}\n\n")

    # 2. User Activity Distribution
    w("## 2. User Activity Distribution\n\n")
    w("See `results/*/analysis/user_activity_distribution.png` for visualizations.\n\n")

    # 3. τ Selection
    w("## 3. τ (Cold/Warm Threshold) Selection\n\n")
    if final_tau:
        w(f"**Selected τ = {final_tau.get('selected_tau', 'N/A')}**\n\n")
        reason = final_tau.get("selection_reason", {})
        for k, v in reason.items():
            w(f"- **{k}**: {v}\n")
        w("\n")
    else:
        w("τ selection not yet completed. Run `scripts/run_tau_scan.py` first.\n\n")

    if len(tau_results) > 0:
        w("### τ Scan Results\n\n")
        w(tau_results.to_markdown(index=False))
        w("\n\n")

    # 4. Main Results
    w("## 4. Main Results\n\n")
    if len(main_results) > 0:
        # Aggregated mean ± std
        agg = main_results.groupby(["dataset", "split_type", "model"]).agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            cold_rmse_mean=("cold_rmse", "mean"),
            warm_rmse_mean=("warm_rmse", "mean"),
        ).reset_index()

        agg["rmse_display"] = agg.apply(
            lambda r: f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}", axis=1
        )
        agg["mae_display"] = agg.apply(
            lambda r: f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}", axis=1
        )

        w("### Mean ± Std Results\n\n")
        display = agg[["dataset", "split_type", "model", "rmse_display", "mae_display",
                        "cold_rmse_mean", "warm_rmse_mean"]]
        w(display.to_markdown(index=False))
        w("\n\n")

        # Best per scenario
        w("### Best Model per Scenario\n\n")
        for ds in main_results["dataset"].unique():
            for st in main_results["split_type"].unique():
                subset = main_results[
                    (main_results["dataset"] == ds) &
                    (main_results["split_type"] == st)
                ]
                if len(subset) == 0:
                    continue
                best = subset.loc[subset["rmse"].idxmin()]
                w(f"- **{ds}/{st}**: {best['model']} (RMSE={best['rmse']:.4f})\n")
        w("\n")

    # 5. Rank Reversal
    w("## 5. Cold vs Warm Rank Analysis\n\n")
    w("See `results/figures/rank_reversal_*.png` for visualization.\n\n")

    # 6. Ablation
    w("## 6. Ablation Study\n\n")
    if len(ablation) > 0:
        w(ablation.to_markdown(index=False))
        w("\n\n")
    else:
        w("Ablation study not yet completed.\n\n")

    # 7. Significance Tests
    w("## 7. Significance Tests\n\n")
    if len(significance) > 0:
        sig_summary = significance.groupby(["model_a", "model_b", "subset"]).agg(
            mean_p=("t_test_p", "mean"),
            n_sig=("is_significant", "sum"),
        ).reset_index()
        w(sig_summary.to_markdown(index=False))
        w("\n\n")
    else:
        w("Significance tests not yet completed.\n\n")

    # 8. Hypothesis Verification
    w("## 8. Hypothesis Verification\n\n")
    w("### Core Hypothesis\n\n")
    w("> Profile/Content/Metadata features are more important in cold-start scenarios; "
      "Behavior/Interaction History features are more important in warm-start scenarios.\n\n")

    # Auto-verify from results
    verified = _check_hypothesis(main_results)
    w(f"### Verification: **{verified['status']}**\n\n")
    w(verified["detail"])
    w("\n\n")

    # 9. Failures
    w("## 9. Failure Analysis\n\n")
    if len(main_results) > 0:
        w("Models with cold_rmse > warm_rmse (potential cold-start failures):\n\n")
        cold_failures = main_results[main_results["cold_rmse"] > main_results["warm_rmse"] * 1.5]
        if len(cold_failures) > 0:
            for _, r in cold_failures.iterrows():
                w(f"- {r['model']}: cold_rmse={r['cold_rmse']:.4f} vs warm_rmse={r['warm_rmse']:.4f}\n")
        else:
            w("No severe cold-start failures detected.\n")
    w("\n---\n\n")

    # Output files
    w("## Output Files\n\n")
    w("| File | Description |\n|------|-------------|\n")
    w("| `results/tables/main_results.csv` | Main experiment results |\n")
    w("| `results/tables/main_results_mean_std.csv` | Mean ± std per model |\n")
    w("| `results/tables/ablation_results.csv` | Ablation study |\n")
    w("| `results/statistics/significance_tests.csv` | Statistical tests |\n")
    w("| `results/figures/` | All figures |\n")
    w("| `results/report.md` | This report |\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(L)

    print(f"[ReportGenerator] Report saved to {output_path}")
    return output_path


def _check_hypothesis(main_results: pd.DataFrame) -> Dict:
    """从实验结果自动验证核心假设"""
    if len(main_results) == 0:
        return {"status": "INCONCLUSIVE (no data)", "detail": "No experiment results available."}

    # Check: is profile_mlp better than behavior_mlp in cold-start?
    profile_mlp = main_results[main_results["model"] == "profile_mlp"]
    behavior_mlp = main_results[main_results["model"] == "behavior_mlp"]

    evidence = []

    if len(profile_mlp) > 0 and len(behavior_mlp) > 0:
        p_cold = profile_mlp["cold_rmse"].mean()
        b_cold = behavior_mlp["cold_rmse"].mean()
        p_warm = profile_mlp["warm_rmse"].mean()
        b_warm = behavior_mlp["warm_rmse"].mean()

        if p_cold < b_cold:
            evidence.append(f"✅ Profile-Only better than Behavior-Only in cold-start "
                           f"(RMSE: {p_cold:.4f} < {b_cold:.4f})")
        else:
            evidence.append(f"❌ Profile-Only NOT better than Behavior-Only in cold-start "
                           f"(RMSE: {p_cold:.4f} >= {b_cold:.4f})")

        if b_warm < p_warm:
            evidence.append(f"✅ Behavior-Only better than Profile-Only in warm-start "
                           f"(RMSE: {b_warm:.4f} < {p_warm:.4f})")
        else:
            evidence.append(f"❌ Behavior-Only NOT better than Profile-Only in warm-start "
                           f"(RMSE: {b_warm:.4f} >= {p_warm:.4f})")

    # Check Dual-Scenario
    dual = main_results[main_results["model"].str.contains("dual")]
    if len(dual) > 0:
        dual_rmse = dual["rmse"].mean()
        evidence.append(f"Dual-Scenario overall RMSE: {dual_rmse:.4f}")

    # Determine status
    if "❌" in " ".join(evidence):
        status = "PARTIALLY SUPPORTED or NOT SUPPORTED"
    elif len(evidence) >= 2:
        status = "SUPPORTED"
    else:
        status = "INCONCLUSIVE"

    return {
        "status": status,
        "detail": "\n".join(f"- {e}" for e in evidence),
    }


def _load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
