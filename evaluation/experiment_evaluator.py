"""
Unified evaluator — supports grouped evaluation (cold/warm/medium), multi-metric, multi-seed.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from evaluation.metrics import (
    compute_all_metrics, compute_ranking_metrics, compute_grouped_metrics,
)


class ExperimentEvaluator:
    """Experiment evaluator for multi-model, multi-seed comparisons."""

    def __init__(self, tau: int = 15):
        self.tau = tau
        self.all_results: List[Dict] = []

    def evaluate(
        self,
        model_name: str,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        dataset: str = "ml1m",
        split_type: str = "strict_cold",
        seed: int = 42,
        model_config: Optional[Dict] = None,
        activity_counts: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a model's output."""
        trues = test_data["rating"].values.astype(np.float32)

        overall = compute_all_metrics(trues, predictions)
        grouped = self._evaluate_grouped(trues, predictions, test_data, activity_counts)
        ranking = compute_ranking_metrics(trues, predictions)

        result = {
            "dataset": dataset,
            "split_type": split_type,
            "model": model_name,
            "seed": seed,
            "rmse": overall["rmse"],
            "mae": overall["mae"],
            "mse": overall["mse"],
            "n_samples": overall["n_samples"],
            "cold_rmse": grouped.get("cold", {}).get("rmse", float("nan")),
            "warm_rmse": grouped.get("warm", {}).get("rmse", float("nan")),
            "medium_rmse": grouped.get("medium", {}).get("rmse", float("nan")),
            "cold_n": grouped.get("cold", {}).get("n_samples", 0),
            "warm_n": grouped.get("warm", {}).get("n_samples", 0),
            "medium_n": grouped.get("medium", {}).get("n_samples", 0),
            **ranking,
            "model_config": model_config or {},
        }

        self.all_results.append(result)
        return result

    def _evaluate_grouped(
        self,
        trues: np.ndarray,
        preds: np.ndarray,
        test_data: pd.DataFrame,
        activity_counts: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Dict]:
        """Evaluate grouped by cold/warm/medium."""
        if activity_counts is None:
            return {}

        groups = np.array([
            self._classify(activity_counts.get(int(uid), 0))
            for uid in test_data["user_id"]
        ])

        return compute_grouped_metrics(trues, preds, groups)

    def _classify(self, n_interactions: int) -> str:
        """Classify user based on tau threshold."""
        if n_interactions == 0:
            return "cold"
        elif n_interactions <= self.tau:
            return "cold"
        elif n_interactions <= self.tau * 3:
            return "medium"
        else:
            return "warm"

    def get_results_df(self) -> pd.DataFrame:
        """Return all results as a DataFrame."""
        return pd.DataFrame(self.all_results)

    def save_results(self, path: str) -> None:
        """Save results to CSV and JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.get_results_df()
        df.to_csv(path, index=False)
        json_path = path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(self.all_results, f, indent=2)
        print(f"[Evaluator] Results saved to {path}")

    def summarize(self) -> str:
        """Generate a summary of all results."""
        if not self.all_results:
            return "No results yet."

        df = self.get_results_df()
        lines = []
        lines.append("=" * 60)
        lines.append("Experiment Results Summary")
        lines.append("=" * 60)

        for _, row in df.iterrows():
            lines.append(
                f"  {row['model']:<20s} "
                f"RMSE={row['rmse']:.4f} "
                f"MAE={row['mae']:.4f} "
                f"cold_RMSE={row['cold_rmse']:.4f} "
                f"warm_RMSE={row['warm_rmse']:.4f}"
            )

        return "\n".join(lines)
