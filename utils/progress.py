"""
进度条模块 — tqdm 封装, 根据配置启用/禁用
支持非交互环境和日志重定向时自动降级
"""
import sys, os
from typing import Optional, Dict, Any, Iterator

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def is_interactive() -> bool:
    """检测是否为交互终端"""
    return sys.stdout.isatty() and sys.stderr.isatty()


def create_progress(iterable: Optional[Iterator] = None,
                    total: Optional[int] = None,
                    desc: str = "",
                    leave: bool = False,
                    position: Optional[int] = None,
                    runtime_cfg: Optional[Dict] = None,
                    **kwargs) -> Any:
    """
    统一进度条创建

    Args:
        runtime_cfg: {"show_progress": bool, "progress_leave": bool}
    """
    cfg = runtime_cfg or {}
    show = cfg.get("show_progress", is_interactive())
    leave_flag = cfg.get("progress_leave", False) or leave

    if not show or not HAS_TQDM:
        return DummyProgress(iterable)

    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave_flag,
        position=position,
        ncols=120,
        mininterval=0.5,
        **kwargs,
    )


class DummyProgress:
    """无 tqdm 或无交互环境下的静默替代"""
    def __init__(self, iterable=None):
        self.iterable = iterable
        self.n = 0
        self.total = None

    def __iter__(self):
        for item in self.iterable:
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        self.n += n

    def set_postfix(self, **kwargs):
        pass

    def set_description(self, desc):
        pass

    def close(self):
        pass


def make_epoch_pbar(epoch: int, total_epochs: int, n_batches: int,
                    runtime_cfg: Optional[Dict] = None,
                    desc_prefix: str = "",
                    leave: bool = False):
    """创建一个 epoch 级别的进度条"""
    desc = f"{desc_prefix}Epoch {epoch+1}/{total_epochs}"
    return create_progress(
        total=n_batches, desc=desc, leave=leave, runtime_cfg=runtime_cfg,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )


def make_model_pbar(total: int, desc: str, runtime_cfg: Optional[Dict] = None):
    """创建一个模型级别的进度条"""
    return create_progress(
        total=total, desc=desc, leave=False, runtime_cfg=runtime_cfg,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
