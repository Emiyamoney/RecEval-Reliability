# Adaptive Dual-Scenario Recommendation Experiment Framework

## Overview

This framework implements a **paper-level experiment pipeline** to verify the hypothesis:

> In cold-start scenarios, **Profile / Content / Metadata** features are more important;  
> in warm-start scenarios, **Behavior / Interaction History** features are more important.  
> Therefore, recommendation models should **adaptively select or weight** strategies based on user activity.

Built on top of the existing Activity-Aware Hybrid Residual Recommendation Framework.

## Quick Start

### CPU Smoke Test (verify pipeline integrity)

```bash
python scripts/run_experiment.py --config configs/smoke_cpu.yaml
```

Expected output:
```
results/smoke/summary.csv
results/smoke/smoke_report.md
results/smoke/report.md
results/smoke/leakage_check_report.md
results/smoke/figures/model_comparison.png
```

### GPU Full Experiment

```bash
python scripts/run_experiment.py --config configs/full_gpu.yaml
```

Single dataset:
```bash
python scripts/run_experiment.py --config configs/full_gpu.yaml --dataset ml1m
```

Single model:
```bash
python scripts/run_experiment.py --config configs/full_gpu.yaml --dataset ml1m --model lightgcn
```

Resume from checkpoint:
```bash
python scripts/run_experiment.py --config configs/full_gpu.yaml --resume
```

### τ (Tau) Analysis

```bash
python scripts/run_tau_scan.py --dataset ml1m --config configs/full_gpu.yaml
```

### Generate Report

```bash
python scripts/generate_report.py --results_dir results/smoke
```

## Implemented Models

| Model | Type | Cold-Start Support | Description |
|-------|------|:---:|-------------|
| `global_mean` | Baseline | ✅ | Global rating mean |
| `user_bias` | Baseline | ✅ | User-specific bias |
| `item_bias` | Baseline | ✅ | Item-specific bias |
| `user_item_bias` | Baseline | ✅ | User + Item bias |
| `svd` | Matrix Factorization | ✅ | SVD with fallback for unseen users |
| `profile_mlp` | Neural | ✅ | Demographic + content features only |
| `behavior_mlp` | Neural | ⚠️ zero-fill | Behavior features only |
| `hybrid` | Neural | ✅ | Profile + Behavior + CF embeddings |
| `dual_hard_switch` | Adaptive | ✅ | Hard switch: profile if cold, behavior if warm |
| `dual_soft_gating` | Adaptive | ✅ | Learned gating: α = σ(w·log(1+n) + b) |
| `neumf` | Neural CF | ✅ | GMF + MLP fusion |
| `deepfm` | Neural CF | ✅ | FM + Deep network |
| `lightgcn` | Graph CF | ⚠️ fallback | LightGCN with item-bias fallback for cold users |

## Implemented Datasets

| Dataset | Loader | Profile Features | Size | Status |
|---------|--------|-----------------|------|--------|
| **MovieLens 1M** | `data/loaders/ml1m_loader.py` | Gender, Age, Occupation, Genres | ~6MB (本地) | ✅ Ready |
| **Amazon Movies & TV** | `data/loaders/amazon_loader.py` | Item metadata only (no demographics) | ~3GB (需下载) | ✅ Ready |
| **Book-Crossing** | `data/loaders/book_crossing_loader.py` | Age, Location, Author, Year, Publisher | ~25MB (需下载) | ✅ Ready |

### Important Notes on Datasets

- **Amazon**: No explicit user demographic profile. Profile features come from item metadata and cold-start-safe user statistics.
- **Book-Crossing**: Rating range is 1-10, scaled to 1-5. Extremely sparse (good for sparse-scenario validation). Download from [Kaggle](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset). Smoke test uses synthetic data.
- **ML-1M**: Already available locally in `training set/training_total.csv`.

## τ (Tau) Derivation Methods

The scientific τ selection process includes:

1. **User Activity Distribution Analysis** — histograms, log-histograms, CDF curves, quantile tables
2. **K-Means on log(1+n_u)** + Elbow method for optimal cluster count
3. **Gaussian Mixture Model** + BIC/AIC selection
4. **τ Scan** — evaluate models at candidate τ values, record cold/warm/overall RMSE
5. **Statistical Boundary** — paired t-test, Wilcoxon, Cohen's d

Selection priority:
1. Validation set overall RMSE optimal
2. No collapse on cold/warm subsets
3. Profile-vs-Behavior difference statistically significant
4. Consistent with Elbow/GMM candidates
5. Simple and interpretable

## Split Types

| Split | Description | Output |
|-------|-------------|--------|
| `strict_cold` | User-level: test users never in train | `data/processed/{dataset}/strict_cold/` |
| `warm_random` | Row-level random: same user can cross sets | `data/processed/{dataset}/warm_random/` |
| `warm_temporal` | Time-ordered: train < val < test | `data/processed/{dataset}/warm_temporal/` |

## Data Leakage Prevention

7 mandatory checks implemented in `src/data/leakage_checker.py`:

1. ✅ Strict cold-start test users NOT in train set
2. ✅ Behavior features computed from train only
3. ✅ Temporal split time ordering verified
4. ✅ Test ratings not used in feature aggregation
5. ✅ τ selected from train/val only, not test
6. ✅ Scaler/encoder fit on train only
7. ✅ Unseen categories mapped to UNK token

**If leakage is detected, training is ABORTED with an error.**

## Feature Groups

### Profile Features (cold-start friendly)
- Demographics: gender, age bucket, occupation
- Content: genres (multi-hot), categories, city, state
- Static metadata: account age, fans, etc.

### Behavior Features (warm-start effective)
- User/item interaction counts
- User/item mean ratings
- Rating standard deviations
- Recency features
- Time-decayed activity
- Activity index: `log1p(user_interaction_count) + ...`

### Missing Handling
- Unknown category → `<UNK>` token
- Missing numerical → 0 with missing indicator
- Cold-start user behavior → all zeros

## Project Structure

```
├── configs/                    # YAML configuration
│   ├── smoke_cpu.yaml          # CPU lightweight test
│   ├── full_gpu.yaml           # GPU full experiment
│   ├── datasets/               # Per-dataset configs
│   │   ├── ml1m.yaml
│   │   ├── amazon_movies.yaml
│   │   └── yelp.yaml
│   └── models/                 # Per-model hyperparams
│       ├── svd.yaml
│       ├── neumf.yaml
│       ├── deepfm.yaml
│       ├── lightgcn.yaml
│       ├── profile_mlp.yaml
│       ├── behavior_mlp.yaml
│       ├── hybrid.yaml
│       └── dual_scenario.yaml
├── models/                     # All models (existing + new)
│   ├── svd_model.py, mlp_model.py     # Original models
│   ├── neumf.py, deepfm.py, cf_model.py
│   ├── base_model.py           # BaseModel interface
│   ├── baselines.py            # GlobalMean, Bias models
│   ├── profile_mlp.py          # Profile-Only + Behavior-Only
│   ├── hybrid_model.py         # Hybrid model
│   ├── dual_scenario.py        # Hard Switch + Soft Gating
│   └── lightgcn.py             # LightGCN
├── trainers/                   # All trainers
│   ├── svd_trainer.py, mlp_trainer.py, ...
│   ├── unified_trainer.py      # Unified trainer (factory)
│   └── early_stopping.py
├── evaluation/                 # All evaluators
│   ├── evaluator.py, group_eval.py, ...
│   ├── experiment_evaluator.py # Group evaluator (cold/warm/medium)
│   ├── leakage_checker.py      # 7 mandatory checks
│   └── metrics.py              # RMSE/MAE + Ranking
├── data/                       # Data layer
│   ├── dataset.py              # Original RatingDataset
│   └── loaders/                # ml1m, amazon, book_crossing
├── features/                   # Feature engineering
│   ├── group_stats.py          # Original group stats
│   └── feature_builder.py      # Unified feature construction
├── utils/                      # Utilities
│   ├── activity_index.py, split.py  # Original utils
│   ├── config_loader.py        # YAML config loader
│   └── splitters.py            # 3 split types
├── analysis/                   # Analysis tools
│   ├── activity_distribution.py
│   ├── tau_analysis.py         # Elbow/KMeans/GMM/Scan
│   ├── significance.py         # t-test/Wilcoxon/Cohen's d
│   ├── plotting.py             # All figures
│   └── report_generator.py     # Auto report
├── scripts/
│   ├── preprocess.py
│   ├── run_experiment.py       # Main entry point
│   ├── run_tau_scan.py
│   └── generate_report.py
└── results/                    # Output directory
    ├── smoke/                  # Smoke test results
    ├── full/                   # Full experiment results
    │   ├── tables/
    │   ├── figures/
    │   └── report.md
    └── leakage_check_report.md
```

## Output Files

After full GPU experiment:
```
results/full/
├── tables/
│   ├── main_results.csv
│   ├── main_results_mean_std.csv
│   └── ablation_results.csv
├── figures/
│   ├── model_comparison.png
│   ├── rank_reversal_ml1m.png
│   ├── gating_curve.png
│   └── ablation_heatmap.png
├── statistics/
│   └── significance_tests.csv
├── report.md
└── leakage_check_report.md
```

## Dependencies

```
torch, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, pyyaml
```

Install:
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn pyyaml
```

## Acceptance Criteria

- [x] CPU smoke test runs: `python scripts/run_experiment.py --config configs/smoke_cpu.yaml`
- [x] Outputs: `summary.csv`, `smoke_report.md`, `leakage_check_report.md`
- [x] GPU full experiment: `python scripts/run_experiment.py --config configs/full_gpu.yaml`
- [x] All results from actual runs, not hardcoded
- [x] Report auto-generated from results
- [x] τ not selected from test set
- [x] No data leakage in feature construction
