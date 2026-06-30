"""
τ 扫描分析脚本 (整合版)
用法: python scripts/run_tau_scan.py --dataset ml1m --config configs/full_gpu.yaml
"""
import sys, os, json, argparse, numpy as np, pandas as pd, torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.config_loader import load_config, get_dataset_config
from data.loaders.ml1m_loader import load_ml1m
from utils.splitters import make_strict_cold_split
from features.feature_builder import FeatureBuilder
from analysis.activity_distribution import analyze_activity_distribution, compute_activity_counts
from analysis.tau_analysis import elbow_method, gmm_analysis, tau_scan, select_tau, save_tau_result
from models.baselines import GlobalMean
from models.profile_mlp import ProfileOnlyModel, BehaviorOnlyModel
from models.hybrid_model import HybridModel
from models.dual_scenario import DualScenarioModel
from evaluation.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ml1m")
    p.add_argument("--config", type=str, default="configs/full_gpu.yaml")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ds_config = get_dataset_config(cfg, args.dataset) or {"name": args.dataset}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = f"results/tau_scan/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[TauScan] Dataset: {args.dataset}, Device: {device}")

    if args.dataset != "ml1m":
        print(f"[TauScan] {args.dataset} not supported"); return

    df = load_ml1m(ds_config)
    train_df, valid_df, test_df, _ = make_strict_cold_split(df, seed=args.seed)
    fb = FeatureBuilder(ds_config); fb.fit(train_df)
    activity_counts = compute_activity_counts(train_df)
    n_u_values = np.array(list(activity_counts.values()), dtype=np.float64)

    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    _ = analyze_activity_distribution(train_df, analysis_dir, args.dataset)

    print("[Step 1] Elbow...")
    elbow_info = elbow_method(n_u_values, analysis_dir)
    print(f"  τ≈{elbow_info['candidate_tau']}")

    print("[Step 2] GMM...")
    gmm_info = gmm_analysis(n_u_values, analysis_dir)
    print(f"  τ≈{gmm_info['candidate_tau']}")

    tau_candidates = [1,2,3,5,10,15,20,30,50,100]
    print(f"[Step 3] τ scan: {tau_candidates}")

    def eval_for_tau(train, valid, tau):
        results = {"tau": tau}
        models = {
            "profile_mlp": ProfileOnlyModel({}),
            "behavior_mlp": BehaviorOnlyModel({}),
            "hybrid": HybridModel({}),
            "dual_hard": DualScenarioModel({"variant":"hard_switch","tau":tau}),
        }
        for name, model in models.items():
            try:
                model.fit(train, valid, feature_builder=fb, hidden_dims=[32,16],
                         epochs=3, batch_size=512, device=device, verbose=False)
                preds = model.predict(valid)
                m = compute_all_metrics(valid["rating"].values, preds)
                results[f"{name}_rmse"] = m["rmse"]
            except: results[f"{name}_rmse"] = float("nan")
        results["overall_rmse"] = results.get("dual_hard_rmse", float("nan"))
        return results

    scan_results = tau_scan(train_df, valid_df, activity_counts, tau_candidates, eval_for_tau, output_dir)
    final_tau = select_tau(scan_results, elbow_info, gmm_info, {})
    save_tau_result(final_tau, os.path.join(output_dir, "final_tau.json"))
    print(f"\nSelected τ = {final_tau['selected_tau']}")


if __name__ == "__main__":
    main()
