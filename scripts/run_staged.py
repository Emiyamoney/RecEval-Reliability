"""
分阶段实验脚本 — 每跑完一个 (dataset, model, seed, split) 就生成 markdown 报告
用法:
  # 跑全部（分阶段，每个组合独立报告）
  python scripts/run_staged.py --config configs/full_gpu.yaml

  # 只跑一个
  python scripts/run_staged.py --config configs/full_gpu.yaml --dataset ml1m --model dual_soft_gating --seed 2024

  # 只跑基线
  python scripts/run_staged.py --config configs/full_gpu.yaml --dataset ml1m --model global_mean,user_bias,item_bias,user_item_bias

  # 断点续跑
  python scripts/run_staged.py --config configs/full_gpu.yaml --resume
"""

import sys, os, json, time, argparse, traceback
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.config_loader import load_config, get_dataset_config, get_model_config
from utils.splitters import make_strict_cold_split, make_warm_random_split, make_warm_temporal_split
from features.feature_builder import FeatureBuilder
from evaluation.leakage_checker import LeakageChecker
from models.registry import create_model, MODEL_REGISTRY
from trainers.unified_trainer import train_model, predict_model
from evaluation.experiment_evaluator import ExperimentEvaluator
from data.adapters.registry import get_adapter
from analysis.activity_distribution import compute_activity_counts


def parse_args():
    p = argparse.ArgumentParser(description="Staged Experiment Runner")
    p.add_argument("--config", type=str, default="configs/full_gpu.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    return p.parse_args()


def generate_run_markdown(run_dir: str, run_id: str, result: Dict, fb,
                          dataset: str, model_name: str, split_type: str,
                          seed: int, elapsed: float, device: str):
    """单次 run 的 markdown 报告"""
    fa = fb.get_feature_availability() if fb else {}
    lines = []
    w = lines.append

    w(f"# Run Report: {model_name}\n\n")
    w(f"| 项目 | 值 |\n|------|----|\n")
    w(f"| 数据集 | {dataset} |\n")
    w(f"| 模型 | {model_name} |\n")
    w(f"| 划分 | {split_type} |\n")
    w(f"| 种子 | {seed} |\n")
    w(f"| 设备 | {device} |\n")
    w(f"| 耗时 | {elapsed:.0f}s |\n")
    w(f"| 时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n")

    w(f"## 评估指标\n\n")
    w(f"| 指标 | 值 |\n|------|----:|\n")
    for k in ["rmse", "mae", "mse", "cold_rmse", "warm_rmse", "n_samples"]:
        if k in result:
            v = result[k]
            w(f"| {k} | {v:.4f} |\n" if isinstance(v, float) else f"| {k} | {v} |\n")
    w("\n")

    w(f"## 分组评估\n\n")
    for group_name in ["cold", "medium", "warm"]:
        key = f"group_{group_name}"
        if key in result:
            g = result[key]
            w(f"### {group_name.capitalize()}\n\n")
            w(f"| 指标 | 值 |\n|------|----:|\n")
            for k, v in g.items():
                w(f"| {k} | {v:.4f} |\n" if isinstance(v, float) else f"| {k} | {v} |\n")
            w("\n")

    w(f"## 特征信息\n\n")
    w(f"| 特征 | 维度 |\n|------|------|\n")
    w(f"| profile_dim | {fa.get('profile_dim', 'N/A')} |\n")
    w(f"| behavior_dim | {fa.get('behavior_dim', 'N/A')} |\n")
    w(f"| dataset | {fa.get('dataset', 'N/A')} |\n\n")

    if result.get("cold_stats"):
        w(f"## 冷启动统计\n\n")
        w(f"| 类型 | 数量 |\n|------|------|\n")
        for k, v in result["cold_stats"].items():
            w(f"| {k} | {v} |\n")
        w("\n")

    md_path = os.path.join(run_dir, "report.md")
    os.makedirs(run_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return md_path


def generate_summary_markdown(output_dir: str, all_results: List[Dict]):
    """全局汇总 markdown"""
    lines = []
    w = lines.append

    w(f"# Experiment Summary\n\n")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 按数据集分组
    by_dataset = {}
    for r in all_results:
        by_dataset.setdefault(r["dataset"], []).append(r)

    for dataset, runs in sorted(by_dataset.items()):
        w(f"## Dataset: {dataset}\n\n")
        w(f"| Model | Split | Seed | RMSE | MAE | Time |\n")
        w(f"|-------|-------|-----:|-----:|----:|-----:|\n")
        for r in sorted(runs, key=lambda x: (x["split"], x["model"], x["seed"])):
            m = r.get("metrics", {})
            w(f"| {r['model']} | {r['split']} | {r['seed']} "
              f"| {m.get('rmse', 0):.4f} | {m.get('mae', 0):.4f} "
              f"| {r.get('elapsed', '-')}s |\n")
        w("\n")

        # RMSE 均值±标准差
        from collections import defaultdict
        model_rmse = defaultdict(list)
        for r in runs:
            rmse = r.get("metrics", {}).get("rmse")
            if rmse is not None:
                model_rmse[r["model"]].append(rmse)

        if model_rmse:
            w(f"### RMSE Summary (mean ± std)\n\n")
            w(f"| Model | RMSE | Std |\n|-------|-----:|----:|\n")
            for m_name, rmses in sorted(model_rmse.items()):
                w(f"| {m_name} | {np.mean(rmses):.4f} | {np.std(rmses):.4f} |\n")
            w("\n")

    # 异常检测
    rmse_vals = [r.get("metrics", {}).get("rmse") for r in all_results if r.get("metrics", {}).get("rmse")]
    if rmse_vals:
        w(f"## Anomaly Check\n\n")
        w(f"- Min RMSE: {min(rmse_vals):.4f}\n")
        w(f"- Max RMSE: {max(rmse_vals):.4f}\n")
        w(f"- Mean RMSE: {np.mean(rmse_vals):.4f}\n")
        w(f"- Total runs: {len(all_results)}\n\n")

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return md_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device_str = cfg.get("device", "cpu")
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = cfg.get("output_dir", "results/staged")

    from utils.performance import setup_runtime, reset_gpu_memory_stats
    setup_runtime(cfg.get("runtime", {}))
    reset_gpu_memory_stats()

    print("=" * 60)
    print(f"Staged Experiment | Device: {device}")
    print("=" * 60)

    # 解析 datasets
    datasets_to_run = cfg.get("datasets", [])
    if args.dataset:
        datasets_to_run = [d for d in datasets_to_run
                           if (isinstance(d, str) and d == args.dataset) or
                              (isinstance(d, dict) and d.get("name") == args.dataset)]

    # 解析 models
    models_to_run = cfg.get("models", [])
    if args.model:
        models_to_run = [m for m in models_to_run
                         if (isinstance(m, str) and m in args.model.split(",")) or
                            (isinstance(m, dict) and m.get("name") in args.model.split(","))]
    skip_heavy = cfg.get("skip_heavy_models_in_smoke", [])
    models_to_run = [m for m in models_to_run
                     if (m if isinstance(m, str) else m.get("name", "")) not in skip_heavy]

    # 解析 seeds
    seeds = cfg.get("seeds", [42])
    if args.seed:
        seeds = [args.seed]

    # 解析 splits
    splits_to_run = cfg.get("splits", ["strict_cold", "warm_random"])
    if args.split:
        splits_to_run = [args.split]

    train_cfg = cfg.get("training", {})
    sample_cfg = cfg.get("sample", {"enabled": False})
    tau_raw = cfg.get("tau_scan", {}).get("values", 15)
    if isinstance(tau_raw, list):
        tau_val = tau_raw[0] if tau_raw else 15
    elif isinstance(tau_raw, str):
        tau_val = 15
    else:
        tau_val = int(tau_raw)
    evaluator = ExperimentEvaluator(tau=tau_val)

    # 断点续跑 manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {}
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    all_results = []
    total = len(datasets_to_run) * len(models_to_run) * len(seeds) * len(splits_to_run)
    done = 0

    print(f"\nPlan: {len(datasets_to_run)} datasets × {len(models_to_run)} models × "
          f"{len(seeds)} seeds × {len(splits_to_run)} splits = {total} runs\n")

    for ds_name in datasets_to_run:
        ds_name = ds_name if isinstance(ds_name, str) else ds_name.get("name", "unknown")
        ds_config = get_dataset_config(cfg, ds_name) or {}

        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        # Load data
        try:
            adapter = get_adapter(ds_name, ds_config)
            df = adapter.load(sample_config=sample_cfg)
            print(f"  Loaded: {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        except Exception as e:
            print(f"  [SKIP] Load failed: {e}")
            continue

        for seed in seeds:
            for split_type in splits_to_run:
                split_dir = os.path.join(output_dir, ds_name, split_type, f"seed{seed}")

                try:
                    if split_type == "strict_cold":
                        train_df, valid_df, test_df, _ = make_strict_cold_split(df, seed=seed, save_dir=split_dir)
                    elif split_type == "warm_random":
                        train_df, valid_df, test_df, _ = make_warm_random_split(df, seed=seed, save_dir=split_dir)
                    elif split_type == "warm_temporal":
                        train_df, valid_df, test_df, _ = make_warm_temporal_split(df, save_dir=split_dir)
                    else:
                        continue
                except Exception as e:
                    print(f"    [SKIP] Split {split_type} failed: {e}")
                    continue

                if len(valid_df) == 0:
                    valid_df = None

                fb = FeatureBuilder(adapter)
                fb.fit(train_df)
                activity_counts = compute_activity_counts(train_df)

                fa = fb.get_feature_availability()
                print(f"\n  [{split_type}] train={len(train_df)}, test={len(test_df)} "
                      f"| profile={fa['profile_dim']}, behavior={fa['behavior_dim']}")

                for m_entry in models_to_run:
                    m_name = m_entry if isinstance(m_entry, str) else m_entry.get("name", "unknown")
                    done += 1
                    run_id = f"{ds_name}__{m_name}__seed{seed}__{split_type}"
                    run_dir = os.path.join(output_dir, "runs", ds_name, m_name, f"seed{seed}", split_type)

                    # 断点检查
                    if args.resume and manifest.get(run_id, {}).get("status") == "completed" and not args.force:
                        print(f"    [{m_name}] ✓ skip (completed)")
                        continue

                    m_config = get_model_config(cfg, m_name) or (m_entry if isinstance(m_entry, dict) else {})

                    print(f"\n    [{done}/{total}] {m_name} | seed={seed} | {split_type}")
                    t0 = time.time()

                    try:
                        model = create_model(m_name, m_config)
                        model = train_model(model, m_name, train_df, valid_df,
                                            feature_builder=fb,
                                            config={**train_cfg, **m_config},
                                            device=device,
                                            runtime_cfg=cfg.get("runtime", {}),
                                            verbose=True)
                        preds = predict_model(model, m_name, test_df,
                                              feature_builder=fb, device=device,
                                              runtime_cfg=cfg.get("runtime", {}))
                        result = evaluator.evaluate(m_name, test_df, preds,
                                                    dataset=ds_name, split_type=split_type, seed=seed,
                                                    model_config=m_config, activity_counts=activity_counts)
                        elapsed = time.time() - t0

                        # 保存结果
                        os.makedirs(run_dir, exist_ok=True)
                        np.save(os.path.join(run_dir, "predictions.npy"), preds)
                        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)

                        # 生成单次 markdown 报告
                        md_path = generate_run_markdown(
                            run_dir, run_id, result, fb,
                            ds_name, m_name, split_type, seed, elapsed, str(device)
                        )

                        # 记录
                        manifest[run_id] = {
                            "status": "completed",
                            "metrics": {"rmse": result.get("rmse"), "mae": result.get("mae")},
                            "elapsed": f"{elapsed:.0f}",
                            "report": md_path,
                            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        all_results.append({
                            "dataset": ds_name, "model": m_name,
                            "split": split_type, "seed": seed,
                            "metrics": result, "elapsed": f"{elapsed:.0f}",
                        })

                        print(f"    [{m_name}] RMSE={result.get('rmse',0):.4f} "
                              f"MAE={result.get('mae',0):.4f} ({elapsed:.0f}s)")
                        print(f"    → report: {md_path}")

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"    [{m_name}] FAILED ({elapsed:.0f}s): {e}")
                        traceback.print_exc()
                        manifest[run_id] = {
                            "status": "failed",
                            "error": str(e),
                            "elapsed": f"{elapsed:.0f}",
                        }

                    # 每次 run 后保存 manifest + 全局汇总
                    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2, ensure_ascii=False)
                    generate_summary_markdown(output_dir, all_results)

    # 最终汇总
    generate_summary_markdown(output_dir, all_results)
    print(f"\n{'='*60}")
    print(f"Done! {len(all_results)}/{total} runs completed")
    print(f"Summary: {output_dir}/summary.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
