"""
数据集适配器注册表 — 统一入口
"""

from typing import Dict, Optional
from data.adapters.base import BaseDatasetAdapter
from data.adapters.ml1m_adapter import ML1MAdapter
from data.adapters.goodbooks_adapter import GoodbooksAdapter
from data.adapters.book_crossing_adapter import BookCrossingAdapter


ADAPTER_REGISTRY: Dict[str, type] = {
    "ml1m": ML1MAdapter,
    "goodbooks": GoodbooksAdapter,
    "book_crossing": BookCrossingAdapter,
}


def get_adapter(dataset_name: str, config: Optional[Dict] = None) -> BaseDatasetAdapter:
    """根据数据集名称获取适配器实例"""
    if dataset_name not in ADAPTER_REGISTRY:
        available = sorted(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Available: {available}")
    return ADAPTER_REGISTRY[dataset_name](config)


def load_dataset(dataset_name: str, config: Optional[Dict] = None,
                 sample_config: Optional[Dict] = None, raw_dir: str = ""):
    """统一数据集加载入口 — 返回 (DataFrame, adapter)"""
    adapter = get_adapter(dataset_name, config)
    df = adapter.load(raw_dir=raw_dir, sample_config=sample_config)
    return df, adapter
