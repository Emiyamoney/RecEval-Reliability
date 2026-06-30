"""
主实验运行脚本 v3 — 数据集适配器模式 + 增量报告

用法:
  CPU smoke:  python scripts/run_experiment.py --config configs/smoke_cpu.yaml --resume
  GPU full:    python scripts/run_experiment.py --config configs/full_gpu.yaml --resume
  单数据集:     python scripts/run_experiment.py --config ... --dataset ml1m
  单模型:       python scripts/run_experiment.py --config ... --model svd
  强制重跑:     python scripts/run_experiment.py --config ... --force
  遇错即停:     python scripts/run_experiment.py --config ... --fail-fast
"""

import sys, os, json, time, argparse, traceback
from datetime import datetime
from typing import Dict, Optional
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
    p = argparse.ArgumentParser(description="Experiment Runner v3")
    p.add_argument("--config", type=str, default="configs/smoke_cpu.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", type=str, default=None, help="Override split type (strict_cold/warm_random/warm_temporal)")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--fail-fast", action="store_true", default=False)
    return p.parse_args()


def load_manifest(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": {}, "updated_at": None}


def save_manifest(m: Dict, path: str):
    m["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)


def run_status(m: Dict, rid: str) -> Optional[str]:
    return m["runs"].get(rid, {}).get("status")


def mark_run(m: Dict, rid: str, status: str, **kw):
    e = m["runs"].get(rid, {})
    e["status"] = status
    e.update(kw)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if status == "running":
        e["start_time"] = ts
    elif status in ("completed", "failed", "skipped"):
        e["end_time"] = ts
    m["runs"][rid] = e


def generate_run_report(run_dir: str, run_id: str, result: Dict, fb, elapsed: float):
    fa = fb.get_feature_availability() if fb else {}
    lines = [
        f"# Run Report: {run_id}\n\n",
        f"- **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- **Elapsed**: {elapsed:.0f}s\n\n",
        f"## Metrics\n\n",
        f"| Metric | Value |\n|--------|------:|\n",
        f"| RMSE | {result.get('rmse','N/A')} |\n",
        f"| MAE | {result.get('mae','N/A')} |\n",
        f"| MSE | {result.get('mse','N/A')} |\n",
        f"| Cold RMSE | {result.get('cold_rmse','N/A')} |\n",
        f"| Warm RMSE | {result.get('warm_rmse','N/A')} |\n\n",
        f"## Feature Availability\n\n```json\n{json.dumps(fa, indent=2, ensure_ascii=False)}\n```\n\n",
    ]
    rp = os.path.join(run_dir, "report.md")
    os.makedirs(run_dir, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return rp


def update_summary(manifest: Dict, output_dir: str):
    runs = manifest.get("runs", {})
    completed = {rid: r for rid, r in runs.items() if r.get("status") == "completed"}
    failed = {rid: r for rid, r in runs.items() if r.get("status") == "failed"}
    skipped = {rid: r for rid, r in runs.items() if r.get("status") == "skipped"}

    lines = [
        "# Experiment Summary\n\n",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"| Status | Count |\n|--------|-----:|\n",
        f"| Completed | {len(completed)} |\n",
        f"| Failed | {len(failed)} |\n",
        f"| Skipped | {len(skipped)} |\n\n",
    ]
    if completed:
        lines.append("## Completed\n\n| Run | RMSE | MAE | Cold RMSE | Warm RMSE | Time |\n|-----|-----:|----:|----------:|----------:|----:|\n")
        for rid, r in sorted(completed.items()):
            m = r.get("metrics", {})
            cold = m.get('cold_rmse', float('nan'))
            warm = m.get('warm_rmse', float('nan'))
            cold_s = f"{cold:.4f}" if cold == cold else "N/A"
            warm_s = f"{warm:.4f}" if warm == warm else "N/A"
            lines.append(f"| {rid} | {m.get('rmse',0):.4f} | {m.get('mae',0):.4f} | {cold_s} | {warm_s} | {r.get('elapsed','-')}s |\n")
    if failed:
        lines.append("\n## Failed\n\n")
        for rid, r in sorted(failed.items()):
            lines.append(f"- **{rid}**: {r.get('error','?')}\n")
    if skipped:
        lines.append("\n## Skipped\n\n")
        for rid, r in sorted(skipped.items()):
            lines.append(f"- **{rid}**: {r.get('skip_reason','?')}\n")

    # Anomaly check
    rmse_map = {}
    for rid, r in completed.items():
        v = r.get("metrics", {}).get("rmse")
        if v is not None:
            rmse_map.setdefault(round(v, 4), []).append(rid)
    anomalies = [f"- ⚠️ RMSE={k:.4f} shared by: {', '.join(v)}" for k, v in rmse_map.items() if len(v) > 1]
    lines.append("\n## Anomaly Checks\n\n")
    lines.extend(anomalies or ["✅ No identical RMSE anomalies.\n"])

    rp = os.path.join(output_dir, "summary.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mode = cfg.get("mode", "smoke")
    device_str = cfg.get("device", "cpu")
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = cfg.get("output_dir", f"outputs/{mode}")

    print("=" * 60)
    print(f"Runner v3 — {mode.upper()} | Device: {device} | Resume: {args.resume}")
    print("=" * 60)

    datasets_to_run = cfg.get("datasets", [])
    if args.dataset:
        datasets_to_run = [d for d in datasets_to_run
                           if (isinstance(d, dict) and d.get("name") == args.dataset) or d == args.dataset]

    models_to_run = cfg.get("models", [])
    if args.model:
        # 如果命令行指定的模型不在config里，加上它
        if args.model not in models_to_run:
            models_to_run.append(args.model)
        models_to_run = [m for m in models_to_run
                         if (isinstance(m, dict) and m.get("name") == args.model) or m == args.model]
    else:
        skip_heavy = cfg.get("skip_heavy_models_in_smoke", [])
        models_to_run = [m for m in models_to_run
                         if (m if isinstance(m, str) else m.get("name")) not in skip_heavy]

    seeds = cfg.get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    sample_cfg = cfg.get("sample", {"enabled": False})
    train_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})
    splits_to_run = cfg.get("splits", ["strict_cold", "warm_random"])
    if args.split:
        splits_to_run = [args.split]

    from utils.performance import setup_runtime, reset_gpu_memory_stats
    setup_runtime(runtime_cfg)
    reset_gpu_memory_stats()

    status_dir = os.path.join(output_dir, "status")
    manifest_path = os.path.join(status_dir, "manifest.json")
    manifest = load_manifest(manifest_path)
    evaluator = ExperimentEvaluator(tau=15)

    print(f"\nDatasets: {datasets_to_run}")
    print(f"Models: {models_to_run}")
    print(f"Seeds: {seeds}\n")

    for ds_name in datasets_to_run:
        ds_name = ds_name if isinstance(ds_name, str) else ds_name.get("name", "unknown")
        ds_config = get_dataset_config(cfg, ds_name) or {}

        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        # ---- Load via adapter ----
        try:
            adapter = get_adapter(ds_name, ds_config)
            df = adapter.load(sample_config=sample_cfg)
            if hasattr(adapter, 'load_external_features'):
                adapter.load_external_features()
            print(f"  Loaded: {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        except Exception as e:
            print(f"  [SKIP] Load failed: {e}")
            continue

        for seed in seeds:
            print(f"\n  --- Seed: {seed} ---")
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
                    print(f"    [SKIP] Split failed: {e}")
                    continue

                if len(valid_df) == 0:
                    valid_df = None

                # ---- Fit FeatureBuilder with adapter ----
                fb = FeatureBuilder(adapter)
                fb.fit(train_df)
                fa = fb.get_feature_availability()
                print(f"    [{split_type}] train={len(train_df)}, test={len(test_df)} | "
                      f"profile_dim={fa['profile_dim']}, behavior_dim={fa['behavior_dim']}")

                activity_counts = compute_activity_counts(train_df)

                # Filter invalid models (extract name from dict if needed)
                model_names = [m if isinstance(m, str) else m.get("name", "unknown") for m in models_to_run]
                valid_models, skipped_models = adapter.filter_valid_models(model_names, MODEL_REGISTRY)
                for m in skipped_models:
                    print(f"    [{m}] SKIPPED: no features available")
                    rid = f"{ds_name}__{m}__seed{seed}__{split_type}"
                    mark_run(manifest, rid, "skipped", skip_reason="no features", feature_availability=fa)

                for m_name in valid_models:
                    m_config = get_model_config(cfg, m_name) or {}
                    if not m_config:
                        m_config = {"name": m_name}
                    run_id = f"{ds_name}__{m_name}__seed{seed}__{split_type}"
                    run_dir = os.path.join(output_dir, "runs", ds_name, m_name, f"seed{seed}", split_type)

                    st = run_status(manifest, run_id)
                    if st == "completed" and not args.force and args.resume:
                        print(f"    [{m_name}] Completed → skip")
                        continue
                    if st in ("failed", "skipped") and not args.force:
                        print(f"    [{m_name}] {st} → skip (use --force)")
                        continue

                    mark_run(manifest, run_id, "running")
                    save_manifest(manifest, manifest_path)
                    t0 = time.time()

                    try:
                        print(f"\n    [{m_name}] Training...")
                        model = create_model(m_name, m_config)
                        model = train_model(model, m_name, train_df, valid_df,
                                           feature_builder=fb,
                                           config={**train_cfg, **m_config},
                                           device=device, runtime_cfg=runtime_cfg,
                                           verbose=(mode == "smoke"))
                        preds = predict_model(model, m_name, test_df,
                                              feature_builder=fb, device=device,
                                              runtime_cfg=runtime_cfg)
                        result = evaluator.evaluate(m_name, test_df, preds,
                                                    dataset=ds_name, split_type=split_type, seed=seed,
                                                    model_config=m_config, activity_counts=activity_counts)
                        elapsed = time.time() - t0
                        print(f"    [{m_name}] RMSE={result['rmse']:.4f} ({elapsed:.0f}s)")

                        os.makedirs(run_dir, exist_ok=True)
                        np.save(os.path.join(run_dir, "predictions.npy"), preds)
                        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        with open(os.path.join(run_dir, "feature_summary.json"), "w") as f:
                            json.dump(fa, f, indent=2, ensure_ascii=False)
                        generate_run_report(run_dir, run_id, result, fb, elapsed)

                        mark_run(manifest, run_id, "completed",
                                 metrics={
                                     "rmse": result["rmse"],
                                     "mae": result["mae"],
                                     "cold_rmse": result.get("cold_rmse", float("nan")),
                                     "warm_rmse": result.get("warm_rmse", float("nan")),
                                 },
                                 elapsed=f"{elapsed:.0f}", feature_availability=fa)
                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"    [{m_name}] FAILED ({elapsed:.0f}s): {e}")
                        if args.fail_fast:
                            traceback.print_exc()
                            save_manifest(manifest, manifest_path)
                            update_summary(manifest, output_dir)
                            return
                        mark_run(manifest, run_id, "failed", error=str(e), elapsed=f"{elapsed:.0f}")
                        os.makedirs(run_dir, exist_ok=True)
                        with open(os.path.join(run_dir, "error.log"), "w", encoding="utf-8") as f:
                            traceback.print_exc(file=f)

                    save_manifest(manifest, manifest_path)
                    update_summary(manifest, output_dir)

        # Leakage check
        try:
            train_check, val_check, test_check, _ = make_strict_cold_split(df, seed=42)
            lc = LeakageChecker()
            bf_cfg = ds_config.get("behavior_features", {"compute_from": "train_only"})
            lc.check_all(train_check, val_check, test_check, "strict_cold",
                        behavior_features=bf_cfg)
            lc.generate_report(os.path.join(output_dir, "leakage_check_report.md"))
            print(f"  Leakage check: PASSED")
        except RuntimeError as e:
            print(f"  LEAKAGE: {e}")

    update_summary(manifest, output_dir)
    print(f"\n{'='*60}\nDone! Output: {output_dir}/\nSummary: {output_dir}/summary.md\n{'='*60}")


if __name__ == "__main__":
    main()
