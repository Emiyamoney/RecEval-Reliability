"""
Unified YAML configuration loader.

Supports: main config + per-dataset/per-model sub-config merging.
"""

import os
import yaml
from copy import deepcopy
from typing import Any, Dict, Optional


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base, override takes priority."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Load main config file, auto-merging dataset and model sub-configs."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = os.path.abspath(config_path)
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(config_path))
    cfg["_project_dir"] = os.path.dirname(cfg["_config_dir"])

    if "datasets" in cfg and isinstance(cfg["datasets"], list):
        resolved = []
        for ds_ref in cfg["datasets"]:
            if isinstance(ds_ref, str):
                ds_name = ds_ref
                ds_config_path = os.path.join(cfg["_config_dir"], "datasets", f"{ds_name}.yaml")
                if os.path.exists(ds_config_path):
                    with open(ds_config_path, "r", encoding="utf-8") as f:
                        ds_cfg = yaml.safe_load(f)
                    ds_cfg["name"] = ds_name
                    resolved.append(ds_cfg)
                else:
                    resolved.append({"name": ds_name})
            elif isinstance(ds_ref, dict):
                resolved.append(ds_ref)
        cfg["datasets"] = resolved

    if "models" in cfg and isinstance(cfg["models"], list):
        resolved = []
        for m_ref in cfg["models"]:
            if isinstance(m_ref, str):
                m_name = m_ref
                m_config_path = os.path.join(cfg["_config_dir"], "models", f"{m_name}.yaml")
                if os.path.exists(m_config_path):
                    with open(m_config_path, "r", encoding="utf-8") as f:
                        m_cfg = yaml.safe_load(f)
                    m_cfg["name"] = m_name
                    resolved.append(m_cfg)
                else:
                    resolved.append({"name": m_name})
            elif isinstance(m_ref, dict):
                resolved.append(m_ref)
        cfg["models"] = resolved

    return cfg


def resolve_path(cfg: Dict, key: str, default: str = "") -> str:
    """Resolve path relative to config file directory."""
    val = cfg.get(key, default)
    if not val:
        return default
    if os.path.isabs(val):
        return val
    return os.path.normpath(os.path.join(cfg.get("_config_dir", "."), val))


def resolve_project_path(cfg: Dict, key: str, default: str = "") -> str:
    """Resolve path relative to project root directory."""
    val = cfg.get(key, default)
    if not val:
        return default
    if os.path.isabs(val):
        return val
    return os.path.normpath(os.path.join(cfg.get("_project_dir", "."), val))


def get_dataset_config(cfg: Dict, dataset_name: str) -> Optional[Dict]:
    """Get configuration for a specific dataset."""
    for ds in cfg.get("datasets", []):
        if ds.get("name") == dataset_name:
            return ds
    return None


def get_model_config(cfg: Dict, model_name: str) -> Optional[Dict]:
    """Get configuration for a specific model."""
    for m in cfg.get("models", []):
        if isinstance(m, dict) and m.get("name") == model_name:
            return m
        elif isinstance(m, str) and m == model_name:
            return {"name": m}
    return None
