"""
显著性检验模块
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d effect size
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


def paired_ttest(
    model_a_preds: np.ndarray,
    model_b_preds: np.ndarray,
    trues: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """Paired t-test: 比较两个模型在相同测试集上的 MSE 差异"""
    # Per-sample squared errors
    err_a = (trues - model_a_preds) ** 2
    err_b = (trues - model_b_preds) ** 2
    diff = err_a - err_b

    t_stat, p_value = stats.ttest_1samp(diff, 0.0)

    return {
        "test": "paired_ttest",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def wilcoxon_test(
    model_a_preds: np.ndarray,
    model_b_preds: np.ndarray,
    trues: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """Wilcoxon signed-rank test"""
    err_a = (trues - model_a_preds) ** 2
    err_b = (trues - model_b_preds) ** 2
    diff = err_a - err_b

    # Remove zero differences
    diff = diff[diff != 0]
    if len(diff) == 0:
        return {"test": "wilcoxon", "statistic": 0, "p_value": 1.0, "significant": False}

    w_stat, p_value = stats.wilcoxon(diff)

    return {
        "test": "wilcoxon",
        "statistic": float(w_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def cohens_d(
    model_a_preds: np.ndarray,
    model_b_preds: np.ndarray,
    trues: np.ndarray,
) -> Dict:
    """Cohen's d effect size"""
    err_a = np.abs(trues - model_a_preds)
    err_b = np.abs(trues - model_b_preds)

    mean_diff = np.mean(err_a) - np.mean(err_b)
    pooled_std = np.sqrt((np.var(err_a) + np.var(err_b)) / 2)

    if pooled_std == 0:
        d = 0.0
    else:
        d = mean_diff / pooled_std

    # Interpret
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "test": "cohens_d",
        "d": float(d),
        "interpretation": interpretation,
    }


def compare_models_on_subset(
    model_a_preds_all: np.ndarray,
    model_b_preds_all: np.ndarray,
    trues_all: np.ndarray,
    test_data: pd.DataFrame,
    activity_counts: Dict[int, int],
    tau: int,
    model_a_name: str = "Profile-Only",
    model_b_name: str = "Behavior-Only",
    dataset: str = "ml1m",
    split_type: str = "strict_cold",
) -> pd.DataFrame:
    """在 cold/warm/overall 子集上比较两个模型"""

    rows = []

    # Classify samples
    n_interactions = np.array([
        activity_counts.get(int(uid), 0)
        for uid in test_data["user_id"]
    ])

    subsets = {
        "cold": n_interactions <= tau,
        "warm": n_interactions > tau,
        "overall": np.ones(len(n_interactions), dtype=bool),
    }

    for subset_name, mask in subsets.items():
        if mask.sum() < 2:
            continue

        t = trues_all[mask]
        pa = model_a_preds_all[mask]
        pb = model_b_preds_all[mask]

        ttest = paired_ttest(pa, pb, t)
        wilcoxon = wilcoxon_test(pa, pb, t)
        cohen = cohens_d(pa, pb, t)

        rows.append({
            "dataset": dataset,
            "split_type": split_type,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "subset": subset_name,
            "n_samples": int(mask.sum()),
            "tau": tau,
            "t_test_p": ttest["p_value"],
            "t_test_significant": ttest["significant"],
            "wilcoxon_p": wilcoxon["p_value"],
            "wilcoxon_significant": wilcoxon["significant"],
            "cohens_d": cohen["d"],
            "cohens_d_interpretation": cohen["interpretation"],
        })

    return pd.DataFrame(rows)


def run_all_significance_tests(
    results_dict: Dict[str, Dict],  # {model_name: {subset_name: (trues, preds)}}
    tau: int,
    output_path: str,
) -> pd.DataFrame:
    """对所有模型在 cold/warm 子集上运行显著性检验"""
    import os

    all_rows = []
    model_names = list(results_dict.keys())

    for i, ma in enumerate(model_names):
        for j, mb in enumerate(model_names):
            if i >= j:
                continue

            ma_data = results_dict.get(ma, {})
            mb_data = results_dict.get(mb, {})

            for subset in ["cold", "warm", "overall"]:
                if subset not in ma_data or subset not in mb_data:
                    continue

                ta, pa = ma_data[subset]
                tb, pb = mb_data[subset]

                ttest = paired_ttest(pa, pb, ta)
                wilcoxon = wilcoxon_test(pa, pb, ta)
                cohen = cohens_d(pa, pb, ta)

                all_rows.append({
                    "model_a": ma,
                    "model_b": mb,
                    "subset": subset,
                    "t_test_p": ttest["p_value"],
                    "t_test_significant": ttest["significant"],
                    "wilcoxon_p": wilcoxon["p_value"],
                    "wilcoxon_significant": wilcoxon["significant"],
                    "cohens_d": cohen["d"],
                    "is_significant": ttest["significant"] or wilcoxon["significant"],
                })

    df = pd.DataFrame(all_rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
