"""
统一 DataLoader 创建 — 所有深度模型共用

自动处理:
  - shuffle / batch_size / num_workers / pin_memory / persistent_workers / prefetch_factor / drop_last
  - Windows 安全: num_workers=0 时不传 persistent_workers / prefetch_factor
  - non_blocking tensor 搬运
"""
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 2048,
    shuffle: bool = True,
    runtime_cfg: Optional[Dict] = None,
    **kwargs,
) -> DataLoader:
    """
    创建 DataLoader, 根据 runtime_cfg 自动配置。

    runtime_cfg 可包含:
      - num_workers: int|"auto"
      - pin_memory: bool
      - non_blocking: bool
      - persistent_workers: bool
      - prefetch_factor: int
      - device: str
    """
    cfg = runtime_cfg or {}

    # num_workers
    nw = cfg.get("num_workers", "auto")
    if nw == "auto":
        from utils.performance import auto_num_workers
        nw = auto_num_workers()
    else:
        nw = int(nw)

    # pin_memory: 仅当有 GPU 且 num_workers > 0 时默认开启
    device_str = cfg.get("device", "cpu")
    pin = cfg.get("pin_memory", None)
    if pin is None:
        pin = (device_str == "cuda" and nw > 0)

    # persistent_workers / prefetch_factor: 仅 num_workers > 0
    persistent = cfg.get("persistent_workers", False)
    prefetch = cfg.get("prefetch_factor", None)

    if nw == 0:
        persistent = False
        prefetch = None

    # 构建 DataLoader 参数
    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": nw,
        "pin_memory": pin,
        "drop_last": kwargs.get("drop_last", False),
    }

    if nw > 0:
        dl_kwargs["persistent_workers"] = persistent
        if prefetch is not None:
            dl_kwargs["prefetch_factor"] = prefetch

    # 合并额外参数 (如 collate_fn)
    for k in ["collate_fn", "sampler", "batch_sampler", "timeout", "worker_init_fn"]:
        if k in kwargs and kwargs[k] is not None:
            dl_kwargs[k] = kwargs[k]

    return DataLoader(dataset, **dl_kwargs)


def to_device(batch, device: torch.device, non_blocking: bool = True):
    """递归将 batch (tensor/tuple/list/dict) 搬到 device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device, non_blocking) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k, v in batch.items()}
    return batch
