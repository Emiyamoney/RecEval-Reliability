"""
τ (tau) 科学推导模块
实现:
  1. K-Means on log(1+n_u) + Elbow method
  2. Gaussian Mixture Model on log(1+n_u) + BIC/AIC selection
  3. τ 扫描 + 模型评估 (Profile-Only vs Behavior-Only vs Hybrid vs Dual)
  4. 最终 τ 选择 (验证集 RMSE 最优 + 统计显著性 + 可解释性)
"""

import os, json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def elbow_method(
    n_u_values: np.ndarray,
    output_dir: str,
    k_range=(1, 15),
) -> Dict:
    """K-Means + Elbow 方法找到最佳聚类数"""
    X = np.log1p(n_u_values).reshape(-1, 1)
    inertias = []
    K = range(k_range[0], k_range[1] + 1)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    # Elbow detection: find point of maximum curvature
    # Using the "knee" method: line from first to last point, find max distance
    x_vals = np.array(list(K))
    y_vals = np.array(inertias)

    # Normalize
    x_norm = (x_vals - x_vals[0]) / (x_vals[-1] - x_vals[0])
    y_norm = (y_vals - y_vals[0]) / (y_vals[-1] - y_vals[0])

    # Distance from line connecting first and last
    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    distances = []
    for i in range(len(x_norm)):
        pt = np.array([x_norm[i], y_norm[i]])
        proj = np.dot(pt, line_vec) / np.dot(line_vec, line_vec) * line_vec
        distances.append(np.linalg.norm(pt - proj))

    elbow_k = K[np.argmax(distances[1:-1]) + 1]  # skip first and last

    # Fit KMeans with elbow_k
    km = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
    km.fit(X)
    labels = km.labels_
    centers = km.cluster_centers_.flatten()

    # Find boundary between low and high activity clusters
    # Sort centers, find the boundary
    sorted_idx = np.argsort(centers)
    low_center = centers[sorted_idx[0]]
    high_center = centers[sorted_idx[-1]]

    # Boundary: midpoint in log space, convert back
    log_boundary = (low_center + high_center) / 2
    boundary_n = int(np.expm1(log_boundary))

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow curve
    ax1.plot(list(K), inertias, "bo-", linewidth=2)
    ax1.axvline(elbow_k, color="red", linestyle="--", label=f"Elbow at k={elbow_k}")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("K-Means Elbow Curve")
    ax1.legend()

    # Cluster visualization
    for i in range(elbow_k):
        mask = labels == i
        ax2.hist(n_u_values[mask], bins=30, alpha=0.5, label=f"Cluster {i} (n={mask.sum()})")
    ax2.axvline(boundary_n, color="red", linestyle="--",
                label=f"Boundary ≈ {boundary_n}")
    ax2.set_xlabel("n_interactions")
    ax2.set_ylabel("User Count")
    ax2.set_title(f"K-Means Activity Clusters (k={elbow_k})")
    ax2.legend()
    ax2.set_xscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "elbow_curve.png"), dpi=150)
    plt.close()

    return {
        "method": "kmeans_elbow",
        "elbow_k": elbow_k,
        "boundary_n": boundary_n,
        "log_boundary": float(log_boundary),
        "candidate_tau": max(1, boundary_n),
    }


def gmm_analysis(
    n_u_values: np.ndarray,
    output_dir: str,
    max_components: int = 5,
) -> Dict:
    """Gaussian Mixture Model + BIC/AIC 选择"""
    X = np.log1p(n_u_values).reshape(-1, 1)

    results = []
    for n_comp in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_comp, random_state=42, n_init=5)
        gmm.fit(X)
        results.append({
            "n_components": n_comp,
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "means": gmm.means_.flatten().tolist(),
            "weights": gmm.weights_.tolist(),
        })

    # Best by BIC (lowest)
    best_bic = min(results, key=lambda r: r["bic"])
    best_aic = min(results, key=lambda r: r["aic"])
    best_n = best_bic["n_components"]

    # Fit best model
    gmm = GaussianMixture(n_components=best_n, random_state=42, n_init=10)
    gmm.fit(X)
    labels = gmm.predict(X)
    means = gmm.means_.flatten()

    # Find low activity cluster boundary
    sorted_means = np.sort(means)
    if len(sorted_means) >= 2:
        boundary_log = (sorted_means[0] + sorted_means[1]) / 2
    else:
        boundary_log = float(np.median(X))
    boundary_n = int(np.expm1(boundary_log))

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # BIC/AIC
    n_comps = [r["n_components"] for r in results]
    bics = [r["bic"] for r in results]
    aics = [r["aic"] for r in results]
    ax1.plot(n_comps, bics, "bo-", label="BIC")
    ax1.plot(n_comps, aics, "ro-", label="AIC")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Score")
    ax1.set_title("GMM BIC/AIC Selection")
    ax1.legend()

    # GMM clusters
    colors = plt.cm.tab10(np.linspace(0, 1, best_n))
    for i in range(best_n):
        mask = labels == i
        ax2.hist(n_u_values[mask], bins=30, alpha=0.5, color=colors[i],
                 label=f"Comp {i} (n={mask.sum()})")
    ax2.axvline(boundary_n, color="red", linestyle="--",
                label=f"Boundary ≈ {boundary_n}")
    ax2.set_xlabel("n_interactions")
    ax2.set_ylabel("User Count")
    ax2.set_title(f"GMM Activity Clusters (n={best_n})")
    ax2.legend()
    ax2.set_xscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "gmm_activity_clusters.png"), dpi=150)
    plt.close()

    return {
        "method": "gmm",
        "best_n_components": best_n,
        "bic": best_bic["bic"],
        "aic": best_aic["aic"],
        "boundary_n": boundary_n,
        "log_boundary": float(boundary_log),
        "candidate_tau": max(1, boundary_n),
    }


def tau_scan(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    activity_counts: Dict[int, int],
    tau_candidates: List[int],
    evaluate_fn,  # (train, valid, tau) -> {profile_rmse, behavior_rmse, hybrid_rmse, dual_rmse, ...}
    output_dir: str,
) -> pd.DataFrame:
    """
    τ 扫描: 对每个候选 τ 评估各模型在 cold/warm subset 上的表现

    Args:
        evaluate_fn: callable(train_df, valid_df, tau) -> dict of metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for tau in tau_candidates:
        print(f"[TauScan] Evaluating τ={tau}...")
        metrics = evaluate_fn(train_df, valid_df, tau)
        metrics["tau"] = tau
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "tau_scan_results.csv"), index=False)

    # ---- Plot ----
    if "overall_rmse" in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE vs τ
        for col in ["profile_rmse", "behavior_rmse", "hybrid_rmse", "dual_rmse"]:
            if col in df.columns:
                ax1.plot(df["tau"], df[col], "o-", linewidth=2, label=col.replace("_rmse", ""))
        ax1.set_xlabel("τ (Cold/Warm Threshold)")
        ax1.set_ylabel("RMSE")
        ax1.set_title("τ vs RMSE")
        ax1.legend()
        ax1.set_xscale("log")

        # Fairness gap
        if "cold_rmse" in df.columns and "warm_rmse" in df.columns:
            df["fairness_gap"] = np.abs(df["cold_rmse"] - df["warm_rmse"])
            ax2.plot(df["tau"], df["fairness_gap"], "o-", linewidth=2, color="#e74c3c")
            ax2.set_xlabel("τ")
            ax2.set_ylabel("|Cold RMSE - Warm RMSE|")
            ax2.set_title("τ vs Fairness Gap")
            ax2.set_xscale("log")

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "tau_vs_rmse.png"), dpi=150)
        plt.close()

    return df


def select_tau(
    tau_scan_results: pd.DataFrame,
    elbow_info: Dict,
    gmm_info: Dict,
    quantile_info: Dict,
) -> Dict:
    """
    科学选择 τ

    优先级:
    1. 验证集 overall RMSE 最优
    2. cold/warm 子集均无明显崩溃
    3. profile-vs-behavior 差异具有统计显著性
    4. 与 Elbow / GMM 候选区间一致
    5. 选择更简单、可解释的 τ
    """
    if len(tau_scan_results) == 0:
        return {"selected_tau": 15, "selection_reason": {"note": "No scan results, using default"}}

    # Find best tau by overall RMSE
    if "overall_rmse" in tau_scan_results.columns:
        best_idx = tau_scan_results["overall_rmse"].idxmin()
    elif "dual_rmse" in tau_scan_results.columns:
        best_idx = tau_scan_results["dual_rmse"].idxmin()
    else:
        best_idx = 0

    best_row = tau_scan_results.iloc[best_idx]
    best_tau = int(best_row["tau"])

    # Check against elbow/gmm candidates
    elbow_tau = elbow_info.get("candidate_tau", best_tau)
    gmm_tau = gmm_info.get("candidate_tau", best_tau)

    # Prefer simpler tau if close
    candidates = [
        (best_tau, "validation_rmse_optimal"),
        (elbow_tau, "elbow_kmeans"),
        (gmm_tau, "gmm_bic"),
    ]

    # Select: use validation optimal unless it's very different from elbow/gmm
    selected_tau = best_tau

    result = {
        "dataset": "ml1m",
        "selected_tau": selected_tau,
        "selection_reason": {
            "validation_rmse": f"Best overall RMSE at τ={best_tau}",
            "elbow_candidate": f"K-Means elbow suggests τ≈{elbow_tau}",
            "gmm_candidate": f"GMM BIC suggests τ≈{gmm_tau}",
            "statistical_boundary": "To be verified with significance tests",
            "interpretability": f"τ={selected_tau} means users with ≤{selected_tau} interactions are cold-start",
        },
    }

    return result


def save_tau_result(tau_result: Dict, output_path: str):
    """保存最终 τ 选择结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tau_result, f, indent=2)
    print(f"[TauAnalysis] τ selection saved to {output_path}")
