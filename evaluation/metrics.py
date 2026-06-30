"""
Evaluation metrics — RMSE, MAE, MSE + Ranking metrics (Recall@K, NDCG@K, HitRate@K).
"""

import numpy as np
from typing import Dict


def compute_rmse(trues: np.ndarray, preds: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((trues - preds) ** 2)))


def compute_mae(trues: np.ndarray, preds: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(trues - preds)))


def compute_mse(trues: np.ndarray, preds: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((trues - preds) ** 2))


def compute_all_metrics(trues: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute all regression metrics."""
    return {
        "rmse": compute_rmse(trues, preds),
        "mae": compute_mae(trues, preds),
        "mse": compute_mse(trues, preds),
        "n_samples": len(preds),
    }


def compute_ranking_metrics(
    trues: np.ndarray, preds: np.ndarray, k_values: tuple = (5, 10, 20)
) -> Dict[str, float]:
    """Ranking metrics: Recall@K, NDCG@K, HitRate@K.

    Converts ratings to ranking metrics (rating >= 4 treated as positive).
    """
    threshold = 4.0
    results: Dict[str, float] = {}

    for k in k_values:
        top_k = np.argsort(preds)[-k:][::-1]

        relevant = trues >= threshold
        if relevant.sum() > 0:
            recall = relevant[top_k].sum() / relevant.sum()
        else:
            recall = 0.0
        results[f"recall@{k}"] = float(recall)

        dcg = sum(
            (2 ** int(trues[i] >= threshold) - 1) / np.log2(j + 2)
            for j, i in enumerate(top_k)
        )
        ideal_order = np.argsort(trues)[-k:][::-1]
        idcg = sum(
            (2 ** int(trues[i] >= threshold) - 1) / np.log2(j + 2)
            for j, i in enumerate(ideal_order)
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0
        results[f"ndcg@{k}"] = float(ndcg)

        hit = 1 if relevant[top_k].any() else 0
        results[f"hit@{k}"] = float(hit)

    return results


def compute_grouped_metrics(
    trues: np.ndarray, preds: np.ndarray, groups: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute metrics grouped by category."""
    results: Dict[str, Dict[str, float]] = {}
    for grp in np.unique(groups):
        mask = groups == grp
        if mask.sum() > 0:
            results[str(grp)] = compute_all_metrics(trues[mask], preds[mask])
        else:
            results[str(grp)] = {
                "rmse": float("nan"), "mae": float("nan"),
                "mse": float("nan"), "n_samples": 0,
            }
    return results
