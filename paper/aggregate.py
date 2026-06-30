# -*- coding: utf-8 -*-
"""
从原始 metrics.json 重新聚合 3 种子 (2024/2025/2026) 实验结果。
排除 LightGCN。计算 mean±std、排名、Kendall tau、Protocol Regret、PSI、BNR、功能退化验证。
输出: paper/agg/ 下的多个 csv + 一个 summary 文本。
"""
import json, os, glob, csv, statistics
from collections import defaultdict

ROOT = r"e:\RecEval-Reliability"
SEEDS = [2024, 2025, 2026]

# 模型目录名 -> 论文名 -> 类别 -> 复杂度(去掉Graph后1-4)
MODELS = {
    "global_mean":      ("GlobalMean",     "Bias",      1),
    "user_bias":        ("UserBias",       "Bias",      1),
    "item_bias":        ("ItemBias",       "Bias",      1),
    "user_item_bias":   ("UserItemBias",   "Bias",      1),
    "svd":              ("SVD",            "MF",        2),
    "neumf":            ("NeuMF",          "Deep",      3),
    "deepfm":           ("DeepFM",         "Deep",      3),
    "behavior_mlp":     ("BehaviorMLP",    "Deep",      3),
    "profile_mlp":      ("ProfileMLP",     "Deep",      3),
    "hybrid":           ("Hybrid",         "Deep",      3),
    "dual_hard_switch": ("DualHard",       "Adaptive",  4),
    "dual_soft_gating": ("DualSoft",       "Adaptive",  4),
}
DATASETS = {"ml1m": "ML-1M", "goodbooks": "GoodBooks", "book_crossing": "Book-Crossing"}
PROTOCOLS = {"strict_cold": "SC", "warm_random": "WR", "warm_temporal": "WT"}

# 搜索两个目录，result_plus 优先（较新、book_crossing 全）
SEARCH_DIRS = [
    os.path.join(ROOT, "result_plus", "runs"),
    os.path.join(ROOT, "results", "full", "runs"),
]

def find_metrics(dataset_key, model_key, seed, protocol_key):
    """在两个搜索目录中找 metrics.json，result_plus 优先。"""
    for base in SEARCH_DIRS:
        p = os.path.join(base, dataset_key, model_key, f"seed{seed}", protocol_key, "metrics.json")
        if os.path.isfile(p):
            return p
    return None

def load_rmse(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("rmse")

# ---------- 1. 聚合 mean±std ----------
# records[dataset][model][protocol] = {mean, std, n, values, seed_rmses}
records = defaultdict(lambda: defaultdict(dict))
missing = []
for ds_key, ds_name in DATASETS.items():
    for mk, (mname, cat, cx) in MODELS.items():
        for pk, pname in PROTOCOLS.items():
            if ds_name == "Book-Crossing" and pname == "WT":
                continue  # BC 无时间戳，无 WT
            vals = {}
            for s in SEEDS:
                p = find_metrics(ds_key, mk, s, pk)
                if p:
                    r = load_rmse(p)
                    if r is not None:
                        vals[s] = r
                else:
                    missing.append(f"{ds_name}/{mname}/{pname}/seed{s}")
            if vals:
                arr = list(vals.values())
                records[ds_name][mname][pname] = {
                    "mean": statistics.mean(arr),
                    "std": statistics.stdev(arr) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                    "seeds": vals,
                }
            else:
                missing.append(f"{ds_name}/{mname}/{pname} (no data any seed)")

print(f"[aggregate] missing count: {len(missing)}")
for m in missing[:20]:
    print("  MISSING:", m)

OUT = os.path.join(ROOT, "paper", "agg")
os.makedirs(OUT, exist_ok=True)

# 导出 mean±std 主表
with open(os.path.join(OUT, "main_mean_std.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "model", "category", "complexity", "protocol", "rmse_mean", "rmse_std", "n"])
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        for mk, (mname, cat, cx) in MODELS.items():
            for pk, pname in PROTOCOLS.items():
                if ds_name == "Book-Crossing" and pname == "WT":
                    continue
                rec = records[ds_name].get(mname, {}).get(pname)
                if rec:
                    w.writerow([ds_name, mname, cat, cx, pname,
                                f"{rec['mean']:.4f}", f"{rec['std']:.4f}", rec["n"]])

# ---------- 2. 排名（按 mean RMSE，升序，1=最好） ----------
# ranks[dataset][protocol][model] = rank
ranks = defaultdict(lambda: defaultdict(dict))
for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
    for pk, pname in PROTOCOLS.items():
        if ds_name == "Book-Crossing" and pname == "WT":
            continue
        items = [(m, records[ds_name][m][pname]["mean"])
                 for m in records[ds_name] if pname in records[ds_name][m]]
        items.sort(key=lambda x: x[1])
        for i, (m, _) in enumerate(items, 1):
            ranks[ds_name][pname][m] = i

# 导出排名表
with open(os.path.join(OUT, "ranks.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "model", "SC_rank", "WR_rank", "WT_rank", "max_rank_shift"])
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        for mk, (mname, cat, cx) in MODELS.items():
            row = [ds_name, mname]
            rs = []
            for pname in ["SC", "WR", "WT"]:
                r = ranks[ds_name].get(pname, {}).get(mname)
                row.append(r if r else "-")
                if r: rs.append(r)
            shift = max(rs) - min(rs) if len(rs) > 1 else 0
            row.append(shift)
            w.writerow(row)

# ---------- 3. Kendall tau（协议间排名一致性） ----------
def kendall_tau(a, b):
    # a, b: 等长 model->rank dict（相同模型集合）
    models = sorted(a.keys())
    x = [a[m] for m in models]
    y = [b[m] for m in models]
    n = len(models)
    conc, disc = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0: conc += 1
            elif dx * dy < 0: disc += 1
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else 0.0

with open(os.path.join(OUT, "kendall_tau.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "pair", "kendall_tau", "n_models"])
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        protos = [p for p in ["SC", "WR", "WT"] if p in ranks[ds_name]]
        for i in range(len(protos)):
            for j in range(i+1, len(protos)):
                a, b = protos[i], protos[j]
                # 取两个协议共有的模型
                common = set(ranks[ds_name][a].keys()) & set(ranks[ds_name][b].keys())
                da = {m: ranks[ds_name][a][m] for m in common}
                db = {m: ranks[ds_name][b][m] for m in common}
                tau = kendall_tau(da, db)
                w.writerow([ds_name, f"{a}-vs-{b}", f"{tau:.3f}", len(common)])

# ---------- 4. Protocol Regret ----------
# Regret[m,d,p] = RMSE[m,d,p] - RMSE[best,d,p];  best = min mean RMSE
# MeanRegret, WorstRegret per (model, dataset)
with open(os.path.join(OUT, "protocol_regret.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "model", "SC_regret", "WR_regret", "WT_regret", "mean_regret", "worst_regret"])
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        # best per protocol
        best = {}
        for pname in ["SC", "WR", "WT"]:
            if pname not in ranks[ds_name]: continue
            best[pname] = min(records[ds_name][m][pname]["mean"]
                              for m in records[ds_name] if pname in records[ds_name][m])
        for mk, (mname, cat, cx) in MODELS.items():
            regs = {}
            for pname in ["SC", "WR", "WT"]:
                if pname in ranks[ds_name] and mname in records[ds_name] and pname in records[ds_name][mname]:
                    regs[pname] = records[ds_name][mname][pname]["mean"] - best[pname]
            if regs:
                mr = statistics.mean(regs.values())
                wr = max(regs.values())
                row = [ds_name, mname]
                for pname in ["SC", "WR", "WT"]:
                    row.append(f"{regs[pname]:.4f}" if pname in regs else "-")
                row += [f"{mr:.4f}", f"{wr:.4f}"]
                w.writerow(row)

# ---------- 5. PSI (rank range) ----------
with open(os.path.join(OUT, "psi.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "ML-1M_PSI", "GoodBooks_PSI", "BC_PSI"])
    for mk, (mname, cat, cx) in MODELS.items():
        row = [mname]
        for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
            rs = [ranks[ds_name][p][mname] for p in ["SC", "WR", "WT"]
                  if p in ranks[ds_name] and mname in ranks[ds_name][p]]
            row.append(max(rs) - min(rs) if rs else 0)
        w.writerow(row)

# ---------- 6. BNR (Bias-Normalized RMSE, 原 CR) ----------
# BNR = RMSE[model, WR] / RMSE[best bias, WR];  best bias = min over Bias models
bias_models = ["GlobalMean", "UserBias", "ItemBias", "UserItemBias"]
with open(os.path.join(OUT, "bnr.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "ML-1M_BNR", "GoodBooks_BNR", "BC_BNR"])
    for mk, (mname, cat, cx) in MODELS.items():
        row = [mname]
        for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
            wr = records[ds_name].get(mname, {}).get("WR")
            if wr is None:
                row.append("-"); continue
            bias_best = min(records[ds_name][b]["WR"]["mean"] for b in bias_models
                            if b in records[ds_name] and "WR" in records[ds_name][b])
            row.append(f"{wr['mean']/bias_best:.2f}")
        w.writerow(row)

# ---------- 7. 功能退化验证: strict cold 下 UserBias vs GlobalMean ----------
with open(os.path.join(OUT, "functional_collapse.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "GlobalMean_SC_mean", "UserBias_SC_mean", "diff", "UserItemBias_SC_mean", "ItemBias_SC_mean", "diff2"])
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        gm = records[ds_name].get("GlobalMean", {}).get("SC", {}).get("mean")
        ub = records[ds_name].get("UserBias", {}).get("SC", {}).get("mean")
        uib = records[ds_name].get("UserItemBias", {}).get("SC", {}).get("mean")
        ib = records[ds_name].get("ItemBias", {}).get("SC", {}).get("mean")
        w.writerow([ds_name,
                    f"{gm:.4f}" if gm else "-",
                    f"{ub:.4f}" if ub else "-",
                    f"{abs(ub-gm):.5f}" if (gm and ub) else "-",
                    f"{uib:.4f}" if uib else "-",
                    f"{ib:.4f}" if ib else "-",
                    f"{abs(uib-ib):.5f}" if (uib and ib) else "-"])

# ---------- 8. Activity gap (cold vs warm RMSE, ML-1M WR) ----------
with open(os.path.join(OUT, "activity_gap.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "complexity", "cold_rmse", "warm_rmse", "gap"])
    for mk, (mname, cat, cx) in MODELS.items():
        # 从 seed2024 的 cold_rmse/warm_rmse 取（metrics.json 里有）
        p = find_metrics("ml1m", mk, 2024, "warm_random")
        if not p: continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        c = d.get("cold_rmse"); wrm = d.get("warm_rmse")
        if c is not None and wrm is not None:
            w.writerow([mname, cx, f"{c:.4f}", f"{wrm:.4f}", f"{c-wrm:.4f}"])

# ---------- 9. 稀疏度表 (WR, 各数据集) ----------
with open(os.path.join(OUT, "sparsity.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "BC_WR_mean", "GB_WR_mean", "ML1M_WR_mean"])
    for mk, (mname, cat, cx) in MODELS.items():
        row = [mname]
        for ds_key, ds_name in [("book_crossing","Book-Crossing"),("goodbooks","GoodBooks"),("ml1m","ML-1M")]:
            rec = records[ds_name].get(mname, {}).get("WR")
            row.append(f"{rec['mean']:.3f}" if rec else "-")
        w.writerow(row)

# ---------- 10. 输出 summary 文本 ----------
with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("=== 3-SEED AGGREGATION SUMMARY (exclude LightGCN) ===\n\n")
    f.write("## Functional collapse (SC): UserBias vs GlobalMean\n")
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        gm = records[ds_name].get("GlobalMean", {}).get("SC", {})
        ub = records[ds_name].get("UserBias", {}).get("SC", {})
        f.write(f"  {ds_name}: GlobalMean={gm.get('mean'):.4f}  UserBias={ub.get('mean'):.4f}  diff={abs(gm['mean']-ub['mean']):.5f}\n")
    f.write("\n## Kendall tau between protocols\n")
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        protos = [p for p in ["SC","WR","WT"] if p in ranks[ds_name]]
        for i in range(len(protos)):
            for j in range(i+1, len(protos)):
                a,b = protos[i], protos[j]
                common = set(ranks[ds_name][a]) & set(ranks[ds_name][b])
                da = {m:ranks[ds_name][a][m] for m in common}
                db = {m:ranks[ds_name][b][m] for m in common}
                f.write(f"  {ds_name} {a}-vs-{b}: tau={kendall_tau(da,db):.3f} (n={len(common)})\n")
    f.write("\n## Top-3 per protocol per dataset (by mean RMSE)\n")
    for ds_name in ["ML-1M", "GoodBooks", "Book-Crossing"]:
        for pname in ["SC","WR","WT"]:
            if pname not in ranks[ds_name]: continue
            top3 = sorted(ranks[ds_name][pname].items(), key=lambda x:x[1])[:3]
            names = [f"{m}(r{r})" for m,r in top3]
            f.write(f"  {ds_name} {pname}: {', '.join(names)}\n")

print("\n[aggregate] DONE. Output in", OUT)
print("Files:", os.listdir(OUT))
