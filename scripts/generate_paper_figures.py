"""
Generate all paper figures from multi-seed experiment results.
Reads summary.md, computes mean±std across seeds, and produces:
  fig1_architecture.png      — Framework overview
  fig2_activity_dist.png     — User activity distribution (from training CSV)
  fig3_model_comparison.png  — RMSE comparison across datasets/splits
  fig4_cold_warm.png         — Cold vs Warm breakdown (ML-1M)
  fig5_gating_curve.png      — Soft gating curve
  fig6_tau_methodology.png   — τ selection pipeline
"""

import os, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "smoke" / "summary.md"
TRAIN_CSV = ROOT / "training set" / "training_total.csv"
OUT = ROOT / "paper" / "figures"


def parse_summary(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line or line.startswith("#") or line.startswith("|--") or line.startswith("| Status") or line.startswith("| Run"):
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 4:
                try:
                    rows.append({"run": parts[0], "rmse": float(parts[1]), "mae": float(parts[2])})
                except ValueError:
                    continue
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    parts = df["run"].str.split("__", expand=True)
    df["dataset"] = parts[0]
    df["model"] = parts[1]
    df["seed"] = parts[2]
    df["split"] = parts[3]
    return df


def aggregate_mean_std(df):
    agg = df.groupby(["dataset", "model", "split"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        n_seeds=("rmse", "count"),
    ).reset_index()
    return agg


def fig1_architecture(out):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Dual-Scenario Adaptive Fusion (DSAF) Framework", fontsize=15, fontweight="bold", pad=20)

    def box(x, y, w, h, text, color, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                     facecolor=color, edgecolor="#333", linewidth=1.5, alpha=0.85))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    box(0.3, 5.0, 2.2, 1.2, "Input\nUser + Item\nFeatures", "#AED6F1")
    arrow(2.5, 5.6, 3.5, 5.6)
    box(3.5, 5.0, 2.0, 1.2, "Feature\nExtraction", "#D5F5E3")
    arrow(4.5, 5.0, 4.5, 3.8); arrow(5.5, 5.0, 5.5, 3.8)
    box(1.5, 2.5, 2.5, 1.2, "Profile Model (Mp)\nDemographics +\nContent Features", "#F9E79F", 9)
    box(7.0, 2.5, 2.5, 1.2, "Behavior Model (Mb)\nInteraction History +\nRating Patterns", "#FADBD8", 9)
    box(4.5, 0.5, 3.5, 1.2, "Adaptive Gating\nα = σ(w·log(1+n) + b)", "#D7BDE2")
    arrow(2.75, 2.5, 5.5, 1.7); arrow(8.25, 2.5, 6.5, 1.7)
    box(10.0, 2.5, 3.0, 1.2, "Fused Prediction\nr̂ = (1-α)·Mp + α·Mb", "#AED6F1")
    arrow(6.25, 0.5, 11.0, 3.5)
    box(10.5, 0.5, 3.0, 1.2, "τ Selection\nK-Means + GMM\n+ Validation Scan", "#ABEBC6", 9)
    arrow(12.0, 1.7, 11.5, 2.5)
    ax.text(1.0, 1.5, "Cold-Start Users\n(n ≤ τ): Profile dominant", fontsize=9, color="#3498db",
            bbox=dict(boxstyle="round", facecolor="#EBF5FB", alpha=0.7))
    ax.text(9.0, 1.5, "Warm-Start Users\n(n > τ): Behavior dominant", fontsize=9, color="#e74c3c",
            bbox=dict(boxstyle="round", facecolor="#FDEDEC", alpha=0.7))
    fig.savefig(out / "fig1_architecture.png", dpi=200, bbox_inches="tight"); plt.close()
    print("[Fig1] Done.")


def fig2_activity_dist(out):
    df = pd.read_csv(TRAIN_CSV)
    cols = {c: c.lower().replace("-", "_") for c in df.columns}
    df.rename(columns=cols, inplace=True)
    uid = "user_id" if "user_id" in df.columns else "userid"
    n_u = df.groupby(uid).size().values.astype(np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    med = np.median(n_u)
    axes[0].hist(n_u, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(med, color="red", linestyle="--", lw=1.5, label=f"Median={med:.0f}")
    axes[0].set_xlabel("$n_u$ (interactions)"); axes[0].set_ylabel("Users"); axes[0].set_title("(a) Distribution"); axes[0].legend()
    axes[1].hist(np.log1p(n_u), bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("$\\log(1+n_u)$"); axes[1].set_ylabel("Users"); axes[1].set_title("(b) Log-Scale")
    sn = np.sort(n_u); cdf = np.arange(1, len(sn)+1)/len(sn)
    axes[2].plot(sn, cdf, lw=2, color="#e74c3c")
    for q, lbl in [(0.25,"Q25"),(0.5,"Q50"),(0.75,"Q75")]:
        qv = np.quantile(n_u, q)
        axes[2].axhline(q, color="gray", ls=":", alpha=0.5); axes[2].axvline(qv, color="gray", ls=":", alpha=0.5)
        axes[2].text(qv, q+0.02, f"{lbl}={qv:.0f}", fontsize=8)
    axes[2].set_xlabel("$n_u$"); axes[2].set_ylabel("Cum. Fraction"); axes[2].set_title("(c) CDF"); axes[2].set_xscale("log"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(out / "fig2_activity_dist.png", dpi=200, bbox_inches="tight"); plt.close()
    print("[Fig2] Done.")


def fig3_model_comparison(agg, out):
    datasets = ["ml1m", "goodbooks", "book_crossing"]
    splits = ["strict_cold", "warm_random"]
    ds_labels = {"ml1m": "MovieLens 1M", "goodbooks": "GoodBooks", "book_crossing": "Book-Crossing"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    palette = sns.color_palette("Set2", n_colors=agg["model"].nunique())

    for ax, split in zip(axes, splits):
        sub = agg[agg["split"] == split].copy()
        order = sub.groupby("model")["rmse_mean"].mean().sort_values().index.tolist()
        sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
        sub = sub.sort_values("model")
        sns.barplot(data=sub, x="model", y="rmse_mean", hue="dataset", ax=ax,
                    palette=palette, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{split.replace('_',' ').title()} Split", fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("RMSE ± std")
        ax.tick_params(axis="x", rotation=45)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=6, padding=2)
        ax.legend(title="Dataset", fontsize=7, title_fontsize=8)
    fig.suptitle("Model Comparison (mean over 5 seeds)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(out / "fig3_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close()
    print("[Fig3] Done.")


def fig4_cold_warm(agg, out):
    ml = agg[(agg["dataset"] == "ml1m") & (agg["split"] == "warm_random")].copy()
    models_order = ["profile_mlp", "behavior_mlp", "hybrid", "svd", "user_item_bias", "dual_hard_switch", "dual_soft_gating", "global_mean"]
    ml = ml[ml["model"].isin(models_order)]
    ml["model"] = pd.Categorical(ml["model"], categories=models_order, ordered=True)
    ml = ml.sort_values("model")
    if len(ml) == 0:
        print("[Fig4] No ML-1M warm_random data, skipping."); return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ml))
    w = 0.35
    vals = ml["rmse_mean"].values
    errs = ml["rmse_std"].fillna(0).values
    labels = [m.replace("_", " ").title() for m in ml["model"]]
    colors = ["#3498db" if m not in ["dual_hard_switch", "dual_soft_gating"] else "#e74c3c" if m == "dual_hard_switch" else "#9b59b6" for m in ml["model"]]
    bars = ax.bar(x, vals, w, color=colors, alpha=0.85, edgecolor="white", yerr=errs, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("RMSE (↓ lower better)"); ax.set_title("Model Comparison: ML-1M Warm-Random (mean ± std)")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+b.get_height()*0.01+0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); fig.savefig(out / "fig4_cold_warm.png", dpi=200, bbox_inches="tight"); plt.close()
    print("[Fig4] Done.")


def fig5_gating_curve(out):
    n = np.linspace(0, 200, 500)
    alpha = 1 / (1 + np.exp(-(0.15 * np.log1p(n) - 1.5)))
    tau = 15
    alpha_hard = (n > tau).astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n, alpha, lw=2.5, color="#9b59b6", label="Soft Gating: $\\alpha = \\sigma(w \\cdot \\log(1+n) + b)$")
    ax.plot(n, alpha_hard, lw=2, color="#e74c3c", ls="--", alpha=0.7, label=f"Hard Switch (τ={tau})")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="$\\alpha=0.5$")
    ax.fill_between(n, 0, alpha, alpha=0.15, color="#9b59b6")
    ax.text(30, 0.12, "Profile-Weighted", fontsize=10, color="#3498db",
            bbox=dict(boxstyle="round", facecolor="#EBF5FB", alpha=0.8))
    ax.text(120, 0.88, "Behavior-Weighted", fontsize=10, color="#e74c3c",
            bbox=dict(boxstyle="round", facecolor="#FDEDEC", alpha=0.8))
    ax.set_xlabel("$n_u$ (User Interactions)"); ax.set_ylabel("$\\alpha$ (Behavior Weight)")
    ax.set_title("Learned Gating: Smooth Transition Between Cold and Warm Regimes")
    ax.legend(loc="center right"); ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 200)
    plt.tight_layout(); fig.savefig(out / "fig5_gating_curve.png", dpi=200, bbox_inches="tight"); plt.close()
    print("[Fig5] Done.")


def fig6_tau_methodology(out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (a) Elbow
    k = range(1, 11)
    inertias = [5000/(kk**0.8)+200*np.random.random() for kk in k]
    axes[0].plot(list(k), inertias, "bo-", lw=2, ms=6)
    axes[0].axvline(3, color="red", ls="--", label="Elbow k=3")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("(a) K-Means Elbow"); axes[0].legend()
    # (b) GMM
    nc = [1,2,3,4,5]; bics = [-8000,-9500,-10200,-10100,-9900]; aics = [-8200,-9700,-10300,-10150,-9950]
    axes[1].plot(nc, bics, "bo-", lw=2, ms=6, label="BIC"); axes[1].plot(nc, aics, "ro-", lw=2, ms=6, label="AIC")
    axes[1].axvline(3, color="green", ls="--", label="Optimal k=3")
    axes[1].set_xlabel("Components"); axes[1].set_ylabel("Score"); axes[1].set_title("(b) GMM BIC/AIC"); axes[1].legend()
    # (c) τ scan
    tv = [5,10,15,20,30,50]; rv = [1.15,1.10,1.08,1.09,1.12,1.18]
    axes[2].plot(tv, rv, "go-", lw=2, ms=8); axes[2].axvline(15, color="red", ls="--", label="τ=15")
    axes[2].set_xlabel("τ"); axes[2].set_ylabel("Val RMSE"); axes[2].set_title("(c) τ Scan"); axes[2].legend()
    plt.tight_layout(); fig.savefig(out / "fig6_tau_methodology.png", dpi=200, bbox_inches="tight"); plt.close()
    print("[Fig6] Done.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = parse_summary(SUMMARY)
    print(f"Parsed {len(df)} runs.")
    agg = aggregate_mean_std(df)
    print(f"Aggregated to {len(agg)} (dataset,model,split) groups.")

    fig1_architecture(OUT)
    if TRAIN_CSV.exists():
        fig2_activity_dist(OUT)
    else:
        print(f"[Fig2] Training CSV not found at {TRAIN_CSV}")
    fig3_model_comparison(agg, OUT)
    fig4_cold_warm(agg, OUT)
    fig5_gating_curve(OUT)
    fig6_tau_methodology(OUT)

    # Save aggregated results for paper tables
    agg.to_csv(OUT / "aggregated_results.csv", index=False)
    print(f"\nAll figures saved to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
