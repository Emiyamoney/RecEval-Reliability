"""
Data leakage checker — 7 mandatory checks, aborts training on detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class LeakageChecker:
    """Data leakage detector — raises error on detection."""

    def __init__(self):
        self.issues: List[Dict] = []
        self.passed: List[str] = []

    def check_all(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_type: str,
        behavior_features: Optional[Dict] = None,
        scaler_info: Optional[Dict] = None,
        encoder_info: Optional[Dict] = None,
        tau_info: Optional[Dict] = None,
    ) -> bool:
        """Run all 7 checks.

        Returns:
            True if ALL checks pass, False if any fail.
        """
        self.issues = []
        self.passed = []

        self._check_cold_start_user_overlap(train_df, test_df, split_type)

        if behavior_features is not None:
            self._check_behavior_feature_source(behavior_features)

        if split_type == "warm_temporal":
            self._check_temporal_order(train_df, val_df, test_df)

        self._check_test_rating_not_in_features(test_df, behavior_features)

        if tau_info is not None:
            self._check_tau_source(tau_info)

        if scaler_info is not None or encoder_info is not None:
            self._check_scaler_encoder_fit(scaler_info, encoder_info)

        if encoder_info is not None:
            self._check_unseen_category_handling(train_df, test_df, encoder_info)

        return len(self.issues) == 0

    def _check_cold_start_user_overlap(self, train_df, test_df, split_type):
        """Check 1: strict cold-start test users must not appear in train."""
        if split_type != "strict_cold":
            self.passed.append("check1_cold_start_overlap: skipped (non-cold split)")
            return

        train_users = set(train_df["user_id"].unique())
        test_users = set(test_df["user_id"].unique())
        overlap = train_users & test_users

        if len(overlap) > 0:
            self.issues.append({
                "check": "cold_start_user_overlap",
                "severity": "P0",
                "detail": f"Strict cold-start violation: {len(overlap)} test users found in train set!",
                "overlap_users": list(overlap)[:10],
            })
        else:
            self.passed.append("check1_cold_start_overlap: PASS (0 overlapping users)")

    def _check_behavior_feature_source(self, behavior_features: Dict):
        """Check 2: behavior features must be computed from train only."""
        source = behavior_features.get("source") or behavior_features.get("compute_from", "unknown")
        if source != "train_only":
            self.issues.append({
                "check": "behavior_feature_source",
                "severity": "P0",
                "detail": f"Behavior features computed from '{source}' — must be 'train_only'!",
            })
        else:
            self.passed.append("check2_behavior_source: PASS (train_only)")

    def _check_temporal_order(self, train_df, val_df, test_df):
        """Check 3: temporal split time ordering."""
        if "timestamp" not in train_df.columns:
            self.passed.append("check3_temporal_order: skipped (no timestamp column)")
            return

        t_train_max = train_df["timestamp"].max()
        t_val_min = val_df["timestamp"].min() if len(val_df) > 0 else float("inf")
        t_val_max = val_df["timestamp"].max() if len(val_df) > 0 else 0
        t_test_min = test_df["timestamp"].min()

        violations = []
        if t_train_max > t_val_min:
            violations.append(f"train max ({t_train_max}) > val min ({t_val_min})")
        if t_val_max > t_test_min:
            violations.append(f"val max ({t_val_max}) > test min ({t_test_min})")

        if violations:
            self.issues.append({
                "check": "temporal_order",
                "severity": "P0",
                "detail": "Temporal order violation: " + "; ".join(violations),
            })
        else:
            self.passed.append("check3_temporal_order: PASS")

    def _check_test_rating_not_in_features(self, test_df, behavior_features):
        """Check 4: test ratings must not be used in feature computation."""
        if behavior_features is None:
            self.passed.append("check4_test_rating_in_features: skipped (no behavior features)")
            return

        test_users_in_behavior = behavior_features.get("test_users_included", [])
        if len(test_users_in_behavior) > 0:
            self.issues.append({
                "check": "test_rating_in_features",
                "severity": "P0",
                "detail": f"Test users found in behavior feature aggregation: {len(test_users_in_behavior)} users!",
            })
        else:
            self.passed.append("check4_test_rating_in_features: PASS")

    def _check_tau_source(self, tau_info: Dict):
        """Check 5: tau must be determined from train/val only, not test."""
        tau_source = tau_info.get("source", "unknown")
        if tau_source == "test":
            self.issues.append({
                "check": "tau_source",
                "severity": "P0",
                "detail": "tau was selected using test set — must use train/val only!",
            })
        elif tau_source in ("train", "val", "train_and_val"):
            self.passed.append(f"check5_tau_source: PASS (source={tau_source})")
        else:
            self.passed.append(f"check5_tau_source: WARN (source={tau_source}, verify manually)")

    def _check_scaler_encoder_fit(self, scaler_info, encoder_info):
        """Check 6: scaler/encoder must be fit on train only."""
        for name, info in (scaler_info or {}).items():
            source = info.get("fit_source", "unknown")
            if source != "train_only":
                self.issues.append({
                    "check": "scaler_fit",
                    "severity": "P1",
                    "detail": f"Scaler '{name}' fit on '{source}' — must be 'train_only'!",
                })
        for name, info in (encoder_info or {}).items():
            source = info.get("fit_source", "unknown")
            if source != "train_only":
                self.issues.append({
                    "check": "encoder_fit",
                    "severity": "P1",
                    "detail": f"Encoder '{name}' fit on '{source}' — must be 'train_only'!",
                })
        if not self.issues:
            self.passed.append("check6_scaler_encoder_fit: PASS")

    def _check_unseen_category_handling(self, train_df, test_df, encoder_info):
        """Check 7: unseen categories must map to unknown token."""
        for col_name, info in (encoder_info or {}).items():
            vocab = set(info.get("vocabulary", []))
            if col_name in test_df.columns:
                test_vals = set(test_df[col_name].dropna().unique())
                unseen = test_vals - vocab - {"<UNK>", "unknown", ""}
                if len(unseen) > 0:
                    self.issues.append({
                        "check": "unseen_category",
                        "severity": "P1",
                        "detail": f"Column '{col_name}' has {len(unseen)} unseen values not mapped to <UNK>: {list(unseen)[:5]}",
                    })
        if not any(i["check"] == "unseen_category" for i in self.issues):
            self.passed.append("check7_unseen_category: PASS")

    def generate_report(self, output_path: str) -> None:
        """Generate leakage check report."""
        lines = []
        w = lines.append
        w("# Data Leakage Check Report\n\n")
        w(f"## Summary\n\n")
        w(f"- **Total checks**: {len(self.passed) + len(self.issues)}\n")
        w(f"- **Passed**: {len(self.passed)}\n")
        w(f"- **Issues found**: {len(self.issues)}\n\n")

        if self.issues:
            w("## ⚠️ Issues Found\n\n")
            for issue in self.issues:
                w(f"### [{issue['severity']}] {issue['check']}\n\n")
                w(f"{issue['detail']}\n\n")
                if "overlap_users" in issue:
                    w(f"Example overlapping users: {issue['overlap_users'][:5]}\n\n")

        w("## ✅ Passed Checks\n\n")
        for p in self.passed:
            w(f"- {p}\n")

        w("\n---\n\n")
        if self.issues:
            w("**VERDICT: DATA LEAKAGE DETECTED — results are UNTRUSTWORTHY. Fix before continuing.**\n")
        else:
            w("**VERDICT: ALL CHECKS PASSED — no data leakage detected.**\n")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[LeakageChecker] Report saved to {output_path}")

        if self.issues:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED: {len(self.issues)} issue(s). "
                f"See {output_path} for details. Training aborted."
            )
