"""
Complete Smoke Test — 13 models x 3 datasets x 2 splits = 78 experiments
"""
import sys, os, time, numpy as np, pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from utils.splitters import make_strict_cold_split, make_warm_random_split
from features.feature_builder import FeatureBuilder
from evaluation.metrics import compute_all_metrics
from analysis.activity_distribution import compute_activity_counts
from data.adapters.registry import get_adapter
from models.registry import create_model
from trainers.unified_trainer import train_model, predict_model

print(f"PyTorch {torch.__version__} | CPU")
DEVICE = torch.device("cpu")
EPOCHS = 3
EMB_DIM = 16

MODELS = [
    'global_mean', 'user_bias', 'item_bias', 'user_item_bias',
    'svd', 'neumf', 'deepfm', 'lightgcn',
    'profile_mlp', 'behavior_mlp', 'hybrid',
    'dual_hard_switch', 'dual_soft_gating',
]
SPLITS = ['strict_cold', 'warm_random']


def load_dataset(name):
    cfg = {'enabled': True, 'max_users': 200, 'max_items': 200, 'max_interactions': 1200}
    if name == 'ml1m':
        from data.loaders.ml1m_loader import load_ml1m
        return load_ml1m({'name': 'ml1m'}, sample_config=cfg)
    elif name == 'goodbooks':
        from data.loaders.goodbooks_loader import load_goodbooks
        return load_goodbooks(sample_config=cfg, raw_dir='data/raw/goodbooks')
    elif name == 'book_crossing':
        from data.loaders.book_crossing_loader import load_book_crossing
        return load_book_crossing(sample_config={'enabled': True, 'max_users': 200, 'max_items': 200, 'max_interactions': 1200})
    raise ValueError(name)


def create_feature_builder(dataset_name):
    """Create a FeatureBuilder with the proper adapter."""
    adapter = get_adapter(dataset_name)
    return FeatureBuilder(adapter)


def main():
    t0 = time.time()
    os.makedirs('results/smoke/tables', exist_ok=True)
    all_rows = []

    print("=" * 70)
    print(f"SMOKE TEST — 13 models × 3 datasets × 2 splits = 78 experiments")
    print(f"PyTorch {torch.__version__} CPU | epochs={EPOCHS} emb_dim={EMB_DIM}")
    print("=" * 70)

    for ds_name in ['ml1m', 'goodbooks', 'book_crossing']:
        print(f"\n{'='*70}\nDATASET: {ds_name}\n{'='*70}")
        df = load_dataset(ds_name)
        print(f"  {len(df)}r / {df['user_id'].nunique()}u / {df['item_id'].nunique()}i")

        for split_type in SPLITS:
            print(f"\n  [{split_type}]")
            if split_type == 'strict_cold':
                train_df, val_df, test_df, _ = make_strict_cold_split(df, seed=42)
            else:
                train_df, val_df, test_df, _ = make_warm_random_split(df, seed=42)
            print(f"    train={len(train_df)}r/{train_df['user_id'].nunique()}u  test={len(test_df)}r/{test_df['user_id'].nunique()}u")

            fb = create_feature_builder(ds_name)
            fb.fit(train_df)
            act_counts = compute_activity_counts(train_df)
            tau = max(1, int(np.median(list(act_counts.values())))) if act_counts else 3

            cfg = {'epochs': EPOCHS, 'batch_size': 128, 'embedding_dim': EMB_DIM,
                   'hidden_dims': [32, 16], 'lr': 0.01}

            for mname in MODELS:
                try:
                    model = create_model(mname)
                    model = train_model(model, mname, train_df, val_df if len(val_df) > 0 else None,
                                       feature_builder=fb, config=cfg, device=DEVICE, verbose=False)
                    preds = predict_model(model, mname, test_df, feature_builder=fb, device=DEVICE)
                except Exception as e:
                    print(f"    {mname:<20s} FAIL: {str(e)[:70]}")
                    all_rows.append({'dataset': ds_name, 'split_type': split_type, 'model': mname,
                                     'rmse': float('nan'), 'mae': float('nan'),
                                     'cold_rmse': float('nan'), 'warm_rmse': float('nan'),
                                     'n_test': len(test_df), 'tau_used': tau})
                    continue

                trues = test_df['rating'].values.astype(np.float32)
                preds_c = np.clip(preds, 1.0, 5.0)
                m = compute_all_metrics(trues, preds_c)
                n_ints = np.array([act_counts.get(uid, 0) for _, uid in test_df['user_id'].items()])
                cold_mask, warm_mask = n_ints <= tau, n_ints > tau
                cold_rmse = float(np.sqrt(np.mean((trues[cold_mask]-preds_c[cold_mask])**2))) if cold_mask.sum() > 0 else float('nan')
                warm_rmse = float(np.sqrt(np.mean((trues[warm_mask]-preds_c[warm_mask])**2))) if warm_mask.sum() > 0 else float('nan')

                all_rows.append({'dataset': ds_name, 'split_type': split_type, 'model': mname,
                                 'rmse': m['rmse'], 'mae': m['mae'],
                                 'cold_rmse': cold_rmse, 'warm_rmse': warm_rmse,
                                 'n_test': len(test_df), 'tau_used': tau})
                print(f"    {mname:<20s} RMSE={m['rmse']:.4f} cold={cold_rmse:.4f} warm={warm_rmse:.4f}")

    # Save
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('results/smoke/tables/main_results.csv', index=False)
    results_df.groupby(['dataset','split_type','model']).agg(
        rmse_mean=('rmse','mean'), rmse_std=('rmse','std'),
        mae_mean=('mae','mean'), mae_std=('mae','std'),
        cold_rmse_mean=('cold_rmse','mean'), warm_rmse_mean=('warm_rmse','mean'),
    ).reset_index().to_csv('results/smoke/tables/main_results_mean_std.csv', index=False)
    results_df.groupby('model')['rmse'].mean().sort_values().reset_index().to_csv('results/smoke/summary.csv', index=False)

    # Report
    lines = []
    w = lines.append
    w("# Smoke Test Report — All 13 Models\n\n")
    w(f"> PyTorch {torch.__version__} CPU | {time.time()-t0:.0f}s | {len(all_rows)} experiments\n\n---\n\n")
    w("## Model Ranking (Mean RMSE)\n\n")
    pivot = results_df.pivot_table(index='model', values='rmse', aggfunc='mean').sort_values('rmse')
    w(pivot.to_markdown(floatfmt=".4f"))
    w("\n\n## By Dataset\n\n")
    for ds in ['ml1m','goodbooks','book_crossing']:
        w(f"### {ds}\n\n")
        sub = results_df[results_df['dataset']==ds].pivot_table(index='model', columns='split_type', values='rmse', aggfunc='mean')
        w(sub.to_markdown(floatfmt=".4f"))
        w("\n\n")
    w("## Cold vs Warm\n\n| Model | Cold | Warm | Delta |\n|-------|-----:|-----:|------:|\n")
    for m in MODELS:
        sub = results_df[results_df['model']==m]
        c = sub['cold_rmse'].dropna().mean(); w_ = sub['warm_rmse'].dropna().mean()
        w(f"| {m} | {c:.4f} | {w_:.4f} | {c-w_:+.4f} |\n")
    w("\n## Best per Dataset\n\n")
    for ds in ['ml1m','goodbooks','amazon_movies']:
        sub = results_df[results_df['dataset']==ds].dropna(subset=['rmse'])
        if len(sub) > 0:
            best = sub.loc[sub['rmse'].idxmin()]
            w(f"- **{ds}**: {best['model']} RMSE={best['rmse']:.4f}\n")
    w("\n---\n*CPU smoke test — pipeline verification only, not for paper conclusions.*\n")
    with open('results/smoke/smoke_report.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n{'='*70}\nDONE — {time.time()-t0:.0f}s\n{'='*70}")
    for ds in ['ml1m','goodbooks','amazon_movies']:
        sub = results_df[results_df['dataset']==ds].dropna(subset=['rmse'])
        if len(sub) > 0:
            best = sub.loc[sub['rmse'].idxmin()]
            print(f"  {ds}: {best['model']} RMSE={best['rmse']:.4f}")


if __name__ == '__main__':
    main()
