# -*- coding: utf-8 -*-
"""
从原始 metrics.json 聚合排名指标 (NDCG@10, Recall@10)，3 种子 (2024/2025/2026)。
排除 LightGCN。输出: paper/agg/ranking_metrics.csv
"""
import json, os, csv, statistics
from collections import defaultdict

ROOT = r"e:\RecEval-Reliability"
SEEDS = [2024, 2025, 2026]

MODELS = {
    "global_mean":      "GlobalMean",
    "user_bias":        "UserBias",
    "item_bias":        "ItemBias",
    "user_item_bias":   "UserItemBias",
    "svd":              "SVD",
    "neumf":            "NeuMF",
    "deepfm":           "DeepFM",
    "behavior_mlp":     "BehaviorMLP",
    "profile_mlp":      "ProfileMLP",
    "hybrid":           "Hybrid",
    "dual_hard_switch": "DualHard",
    "dual_soft_gating": "DualSoft",
}
DATASETS = {"ml1m": "ML-1M", "goodbooks": "GoodBooks", "book_crossing": "Book-Crossing"}
PROTOCOLS = {"strict_cold": "SC", "warm_random": "WR", "warm_temporal": "WT"}

SEARCH_DIRS = [
    os.path.join(ROOT, "result_plus", "runs"),
    os.path.join(ROOT, "results", "full", "runs"),
]

def find_metrics(dataset_key, model_key, seed, protocol_key):
    for base in SEARCH_DIRS:
        p = os.path.join(base, dataset_key, model_key, f"seed{seed}", protocol_key, "metrics.json")
        if os.path.isfile(p):
            return p
    return None

def load_field(path, field):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    v = d.get(field)
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        return None
    return v

# records[dataset][model][protocol] = {"ndcg10": {mean,std,n}, "recall10": {...}}
records = defaultdict(lambda: defaultdict(dict))
missing = []

for ds_key, ds_name in DATASETS.items():
    for mk, mname in MODELS.items():
        for pk, pname in PROTOCOLS.items():
            if ds_name == "Book-Crossing" and pname == "WT":
                continue
            ndcg_vals, recall_vals = {}, {}
            for s in SEEDS:
                p = find_metrics(ds_key, mk, s, pk)
                if p:
                    n = load_field(p, "ndcg@10")
                    r = load_field(p, "recall@10")
                    if n is not None: ndcg_vals[s] = n
                    if r is not None: recall_vals[s] = r
                else:
                    missing.append(f"{ds_name}/{mname}/{pname}/seed{s}")
            if ndcg_vals or recall_vals:
                na = list(ndcg_vals.values())
                ra = list(recall_vals.values())
                records[ds_name][mname][pname] = {
                    "ndcg10_mean": statistics.mean(na) if na else None,
                    "ndcg10_std":  statistics.stdev(na) if len(na) > 1 else 0.0,
                    "recall10_mean": statistics.mean(ra) if ra else None,
                    "recall10_std":  statistics.stdev(ra) if len(ra) > 1 else 0.0,
                    "n": max(len(na), len(ra)),
                }

print(f"[ranking] missing count: {len(missing)}")
for m in missing[:20]:
    print("  MISSING:", m)

OUT = os.path.join(ROOT, "paper", "agg")
os.makedirs(OUT, exist_ok=True)

# 主输出: ranking_metrics.csv (long format)
with open(os.path.join(OUT, "ranking_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "model", "protocol", "ndcg10_mean", "ndcg10_std",
                "recall10_mean", "recall10_std", "n"])
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        for mk, mname in MODELS.items():
            for pk, pname in PROTOCOLS.items():
                if ds_name == "Book-Crossing" and pname == "WT":
                    continue
                rec = records[ds_name].get(mname, {}).get(pname)
                if rec:
                    w.writerow([ds_name, mname, pname,
                                f"{rec['ndcg10_mean']:.4f}" if rec['ndcg10_mean'] is not None else "-",
                                f"{rec['ndcg10_std']:.4f}" if rec['ndcg10_mean'] is not None else "-",
                                f"{rec['recall10_mean']:.4f}" if rec['recall10_mean'] is not None else "-",
                                f"{rec['recall10_std']:.4f}" if rec['recall10_mean'] is not None else "-",
                                rec["n"]])

# 汇总: 每个数据集每个协议的 NDCG@10 best 模型
print("\n[ranking] NDCG@10 best model per (dataset, protocol):")
for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
    for pname in ["SC", "WR", "WT"]:
        if ds_name == "Book-Crossing" and pname == "WT": continue
        items = []
        for mname in MODELS.values():
            rec = records[ds_name].get(mname, {}).get(pname)
            if rec and rec['ndcg10_mean'] is not None:
                items.append((mname, rec['ndcg10_mean']))
        if items:
            items.sort(key=lambda x: -x[1])  # NDCG 越高越好
            print(f"  {ds_name} {pname}: " + ", ".join(f"{m}({v:.4f})" for m,v in items[:3]))

print("\n[ranking] DONE. Output:", os.path.join(OUT, "ranking_metrics.csv"))
