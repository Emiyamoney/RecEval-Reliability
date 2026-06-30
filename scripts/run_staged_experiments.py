"""
分阶段实验运行器 v2
一个阶段 = 一个seed + 一个模型 + 一个数据集（跑 strict_cold + warm_random）
每个阶段的两个 split 结果写入同一行 Excel

用法:
  python scripts/run_staged_experiments.py --config configs/full_gpu.yaml
  python scripts/run_staged_experiments.py --config configs/full_gpu.yaml --dataset ml1m
  python scripts/run_staged_experiments.py --config configs/full_gpu.yaml --model deepfm
  python scripts/run_staged_experiments.py --config configs/full_gpu.yaml --dataset ml1m --seed 42
  python scripts/run_staged_experiments.py --config configs/smoke_cpu.yaml
  python scripts/run_staged_experiments.py --summary-only
"""

import sys, os, json, time, argparse, traceback, math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.config_loader import load_config, get_dataset_config, get_model_config
from utils.splitters import make_strict_cold_split, make_warm_random_split
from features.feature_builder import FeatureBuilder
from evaluation.leakage_checker import LeakageChecker
from models.registry import create_model, MODEL_REGISTRY
from trainers.unified_trainer import train_model, predict_model
from evaluation.experiment_evaluator import ExperimentEvaluator
from data.adapters.registry import get_adapter
from analysis.activity_distribution import compute_activity_counts


def _clean(v):
    if isinstance(v, (dict, list)):
        return str(v)[:100]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v


class ExcelWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "experiment_log.xlsx")
        self.summary_path = os.path.join(output_dir, "paper_summary.xlsx")
        os.makedirs(output_dir, exist_ok=True)
        self.rows = []

    def add_row(self, row: dict):
        self.rows.append({k: _clean(v) for k, v in row.items()})
        df = pd.DataFrame(self.rows)
        df.to_excel(self.path, index=False, sheet_name="Experiment Log")
        print(f"  [Excel] Saved {len(self.rows)} rows -> {self.path}")

    def generate_summary(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        df = df.applymap(_clean)

        with pd.ExcelWriter(self.summary_path, engine="openpyxl") as w:
            # Sheet 1: mean±std
            num = ["rmse_strict", "rmse_warm", "mae_strict", "mae_warm",
                   "cold_rmse_strict", "cold_rmse_warm", "warm_rmse_strict", "warm_rmse_warm"]
            for c in num:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            grp = ["dataset", "model"]
            agg_dict = {}
            for c in num:
                if c in df.columns:
                    agg_dict[c + "_mean"] = (c, "mean")
                    agg_dict[c + "_std"] = (c, "std")
            agg_dict["n_seeds"] = ("seed", "count")
            if agg_dict:
                summary = df.groupby(grp).agg(**agg_dict).reset_index()
                summary = summary.applymap(_clean)
                summary.to_excel(w, index=False, sheet_name="Main Results")

            # Per-dataset sheets
            for ds in df["dataset"].unique():
                sheet = ds.replace("_", " ").title()[:31]
                sub = df[df["dataset"] == ds].copy()
                sub.to_excel(w, index=False, sheet_name=sheet)

        print(f"\n[Excel] Paper summary -> {self.summary_path}")


def run_one_stage(dataset, model, seed, cfg, device, excel, splits=None):
    """
    一个阶段: 一个seed + 一个模型 + 一个数据集
    跑 strict_cold 和 warm_random 两个 split，结果合并为一行
    """
    if splits is None:
        splits = cfg.get("splits", ["strict_cold", "warm_random"])
    mode = cfg.get("mode", "smoke")
    train_cfg = cfg.get("training", {})
    output_dir = cfg.get("output_dir", "results/full")

    stage_id = f"{dataset}__{model}__seed{seed}"
    print(f"\n{'='*60}")
    print(f"  Stage: {stage_id}  (splits: {splits})")
    print(f"{'='*60}")

    row = {
        "stage_id": stage_id,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "status": "running",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Check if already done
    metrics_path = os.path.join(output_dir, dataset, f"stage_{model}_seed{seed}.json")
    if os.path.exists(metrics_path) and "--force" not in sys.argv:
        try:
            with open(metrics_path, encoding="utf-8") as f:
                saved = json.load(f)
            row.update(saved)
            row["status"] = "completed (cached)"
            print(f"  [{stage_id}] Already done, loading from cache.")
            excel.add_row(row)
            return row
        except Exception:
            pass

    t0 = time.time()

    try:
        # Load dataset
        ds_config = get_dataset_config(cfg, dataset)
        adapter = get_adapter(dataset, ds_config)
        raw_dir = ds_config.get("raw", {})
        if isinstance(raw_dir, dict):
            raw_dir = raw_dir.get("ratings", "data/raw/" + dataset)
        df = adapter.load(raw_dir=raw_dir if isinstance(raw_dir, str) else "")
        print(f"  [{dataset}] Loaded {len(df)} rows")

        # Sample
        sample_cfg = cfg.get("sample", {})
        if sample_cfg.get("enabled", False):
            mu = sample_cfg.get("max_users", 500)
            mi = sample_cfg.get("max_items", 500)
            mr = sample_cfg.get("max_interactions", 5000)
            users = df["user_id"].value_counts().head(mu).index
            items = df["item_id"].value_counts().head(mi).index
            df = df[df["user_id"].isin(users) & df["item_id"].isin(items)]
            if len(df) > mr:
                df = df.sample(n=mr, random_state=seed)
            print(f"  [{dataset}] Sampled to {len(df)} rows")

        m_config = get_model_config(cfg, model) or {}
        m_config.update(train_cfg)
        m_config["name"] = model

        for split in splits:
            print(f"\n  --- {split} ---")
            split_id = f"{stage_id}__{split}"
            t1 = time.time()

            # Split
            if split == "strict_cold":
                train_df, valid_df, test_df, _ = make_strict_cold_split(df, seed=seed)
            elif split == "warm_random":
                train_df, valid_df, test_df, _ = make_warm_random_split(df, seed=seed)
            else:
                raise ValueError(f"Unknown split: {split}")
            print(f"  train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

            # Feature builder
            fb = FeatureBuilder(adapter)
            fb.fit(train_df)

            # Leakage check (non-fatal)
            try:
                lc = LeakageChecker()
                bf_cfg = ds_config.get("behavior_features", {})
                lc.check_all(train_df, valid_df, test_df, split, behavior_features=bf_cfg)
            except Exception as e:
                print(f"  [Leakage] {e}")

            activity_counts = compute_activity_counts(train_df)

            # Train
            model_inst = create_model(model, m_config)
            model_inst = train_model(
                model_inst, model, train_df, valid_df,
                feature_builder=fb, config=m_config,
                device=device, verbose=(mode == "smoke"),
            )

            # Predict
            preds = predict_model(model_inst, model, test_df,
                                  feature_builder=fb, device=device)

            # Evaluate
            evaluator = ExperimentEvaluator(tau=m_config.get("tau", 15))
            result = evaluator.evaluate(
                model, test_df, preds,
                dataset=dataset, split_type=split, seed=seed,
                model_config=m_config, activity_counts=activity_counts,
            )

            elapsed = time.time() - t1
            print(f"  [{split_id}] RMSE={result['rmse']:.4f} MAE={result['mae']:.4f} ({elapsed:.0f}s)")

            # Write to row
            suffix = "strict" if split == "strict_cold" else "warm"
            row[f"rmse_{suffix}"] = result["rmse"]
            row[f"mae_{suffix}"] = result["mae"]
            row[f"mse_{suffix}"] = result["mse"]
            row[f"cold_rmse_{suffix}"] = result.get("cold_rmse", "")
            row[f"warm_rmse_{suffix}"] = result.get("warm_rmse", "")
            row[f"medium_rmse_{suffix}"] = result.get("medium_rmse", "")
            row[f"n_samples_{suffix}"] = result.get("n_samples", 0)
            row[f"elapsed_{suffix}"] = f"{elapsed:.0f}s"

            # Save per-split metrics
            run_dir = os.path.join(output_dir, dataset, split, f"seed{seed}", model)
            os.makedirs(run_dir, exist_ok=True)
            np.save(os.path.join(run_dir, "predictions.npy"), preds)
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({k: _clean(v) for k, v in result.items()}, f, indent=2, ensure_ascii=False)

        total = time.time() - t0
        row["status"] = "completed"
        row["total_time"] = f"{total:.0f}s"
        row["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Cache stage result
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: _clean(v) for k, v in row.items() if k not in ("status", "start_time", "end_time")},
                      f, indent=2, ensure_ascii=False)

        print(f"\n  [{stage_id}] DONE ({total:.0f}s)")

    except Exception as e:
        elapsed = time.time() - t0
        row["status"] = "failed"
        row["error"] = str(e)[:300]
        row["total_time"] = f"{elapsed:.0f}s"
        print(f"\n  [{stage_id}] FAILED ({elapsed:.0f}s): {e}")
        traceback.print_exc()

    excel.add_row(row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/full_gpu.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device_str = cfg.get("device", "cpu")
    device = __import__("torch").device(
        "cuda" if device_str == "cuda" and __import__("torch").cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    output_dir = cfg.get("output_dir", "results/full")
    excel = ExcelWriter(output_dir)

    if args.summary_only:
        # Load all cached stage results
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.startswith("stage_") and f.endswith(".json"):
                    try:
                        with open(os.path.join(root, f), encoding="utf-8") as fh:
                            row = json.load(fh)
                        row["status"] = "completed (cached)"
                        excel.add_row(row)
                    except Exception:
                        pass
        excel.generate_summary()
        return

    datasets = [args.dataset] if args.dataset else cfg.get("datasets", ["ml1m"])
    models = [args.model] if args.model else cfg.get("models", ["global_mean"])
    seeds = [args.seed] if args.seed else cfg.get("seeds", [42])
    splits = cfg.get("splits", ["strict_cold", "warm_random"])

    stages = [(ds, md, sd) for ds in datasets for md in models for sd in seeds]
    print(f"\nTotal stages: {len(stages)}")
    print(f"  Datasets: {datasets}")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Splits: {splits}")

    ok, fail = 0, 0
    for i, (ds, md, sd) in enumerate(stages):
        print(f"\n[{i+1}/{len(stages)}] {ds} / {md} / seed{sd}")
        r = run_one_stage(ds, md, sd, cfg, device, excel, splits)
        if r and r.get("status") == "completed":
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Done: {ok} completed, {fail} failed")
    print(f"{'='*60}")

    excel.generate_summary()


if __name__ == "__main__":
    main()
