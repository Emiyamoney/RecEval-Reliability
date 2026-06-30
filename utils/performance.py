"""
性能优化模块 — 线程/CUDA/AMP/计时/内存统计

所有参数通过 runtime_cfg 控制, 不硬编码
"""
import os, sys, time
import torch
import numpy as np
from typing import Dict, Optional


# ================================================================
# 自动检测最优线程数
# ================================================================
def auto_num_workers() -> int:
    """根据 CPU 核心数自动推荐 DataLoader workers"""
    cpu_count = os.cpu_count() or 4
    # Windows 下过多 worker 容易卡, 建议 max 8
    if sys.platform == "win32":
        return min(cpu_count // 2, 8)
    return min(cpu_count - 2, 16)


def auto_torch_threads() -> int:
    """PyTorch 内部线程数"""
    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 16)


# ================================================================
# 通用设置入口
# ================================================================
def setup_runtime(runtime_cfg: Optional[Dict] = None):
    """
    根据配置设置 CPU/CUDA 环境变量和 PyTorch 参数。

    Args:
        runtime_cfg: {
            "device": "cuda"|"cpu",
            "num_workers": int|"auto",
            "torch_num_threads": int|"auto",
            "torch_interop_threads": int|"auto",
            "cudnn_benchmark": bool,
            "amp": bool,
            "torch_compile": bool,
        }
    """
    cfg = runtime_cfg or {}

    # CPU 线程
    n_threads = cfg.get("torch_num_threads", "auto")
    if n_threads == "auto":
        n_threads = auto_torch_threads()
    torch.set_num_threads(n_threads)

    n_interop = cfg.get("torch_interop_threads", "auto")
    if n_interop == "auto":
        n_interop = max(1, n_threads // 2)
    torch.set_num_interop_threads(n_interop)

    # 环境变量
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n_threads))

    # CUDA 设置
    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        if cfg.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if cfg.get("torch_compile", False):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    return {
        "num_workers": auto_num_workers(),
        "torch_num_threads": n_threads,
        "torch_interop_threads": n_interop,
    }


# ================================================================
# 解析 num_workers (支持 auto)
# ================================================================
def resolve_workers(runtime_cfg: Optional[Dict] = None) -> int:
    cfg = runtime_cfg or {}
    nw = cfg.get("num_workers", "auto")
    if nw == "auto":
        return auto_num_workers()
    return int(nw)


# ================================================================
# AMP 上下文管理器
# ================================================================
def get_amp_context(runtime_cfg: Optional[Dict] = None, device: torch.device = None):
    """
    返回 autocast 上下文 + GradScaler (如果启用 AMP)

    Returns:
        (autocast_fn, scaler_or_None)
    """
    cfg = runtime_cfg or {}
    if cfg.get("amp", True) and device and device.type == "cuda":
        return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler()
    return lambda enabled=False: null_context(), None


class null_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ================================================================
# GPU 内存统计
# ================================================================
def get_gpu_memory_mb() -> Optional[float]:
    """返回峰值 GPU 显存 (MB), CUDA 不可用返回 None"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ================================================================
# 计时工具
# ================================================================
class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def reset(self):
        self.start = time.time()


# ================================================================
# 性能日志
# ================================================================
def log_performance(model_name: str, split_type: str, device: torch.device,
                    train_time: float, eval_time: float,
                    n_samples: int, batch_size: int, num_workers: int,
                    amp_enabled: bool, torch_threads: int,
                    runtime_cfg: Optional[Dict] = None):
    total = train_time + eval_time
    samples_per_sec = n_samples / max(train_time, 0.001)
    gpu_mem = get_gpu_memory_mb()

    parts = [
        f"[Perf] model={model_name}",
        f"split={split_type}",
        f"device={device}",
        f"batch={batch_size}",
        f"workers={num_workers}",
        f"amp={amp_enabled}",
        f"train={train_time:.1f}s",
        f"eval={eval_time:.1f}s",
    ]
    if gpu_mem is not None:
        parts.append(f"gpu_mem={gpu_mem:.0f}MB")
    parts.append(f"samples/s={samples_per_sec:.0f}")
    print(" ".join(parts))

    return {
        "model": model_name, "split": split_type, "device": str(device),
        "train_time": round(train_time, 2), "eval_time": round(eval_time, 2),
        "total_time": round(total, 2), "batch_size": batch_size,
        "num_workers": num_workers, "amp_enabled": amp_enabled,
        "torch_num_threads": torch_threads,
        "peak_gpu_memory_mb": round(gpu_mem, 1) if gpu_mem else None,
        "samples_per_sec": round(samples_per_sec, 0),
        "n_samples": n_samples,
    }
