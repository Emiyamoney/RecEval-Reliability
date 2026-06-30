"""
早停模块
"""

import numpy as np
from typing import List, Optional


class EarlyStopping:
    """早停 — 监控验证集指标，patience 轮不提升则停止"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 mode: str = "min", verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss/rmse, "max" for ndcg
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch = 0
        self.should_stop = False
        self.history: List[float] = []

    def __call__(self, score: float, epoch: int) -> bool:
        """Returns True if should stop"""
        self.history.append(score)

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best_score:.4f} @ epoch {self.best_epoch+1})")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: STOPPED at epoch {epoch+1}")
                return True
        return False
