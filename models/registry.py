"""
Unified model registry — single entry point for model creation in paper-level experiments.

All models are registered via MODEL_REGISTRY, ensuring:
  - run_experiment.py / run_smoke.py use the same entry point
  - fail fast if model not found (no silent skip)
  - each model declares required_features, supported_tasks, etc.
"""

from typing import Dict, Any, Optional
from models.base_model import BaseModel

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "global_mean": {
        "cls": None,
        "module": "models.baselines",
        "class_name": "GlobalMean",
        "category": "baseline",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": [],
        "description": "Global mean baseline",
    },
    "user_bias": {
        "cls": None,
        "module": "models.baselines",
        "class_name": "UserBias",
        "category": "baseline",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": [],
        "description": "User bias model: r_ui = mu + b_u",
    },
    "item_bias": {
        "cls": None,
        "module": "models.baselines",
        "class_name": "ItemBias",
        "category": "baseline",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": [],
        "description": "Item bias model: r_ui = mu + b_i",
    },
    "user_item_bias": {
        "cls": None,
        "module": "models.baselines",
        "class_name": "UserItemBias",
        "category": "baseline",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": [],
        "description": "User+Item bias model: r_ui = mu + b_u + b_i",
    },
    "svd": {
        "cls": None,
        "module": "models.wrappers",
        "class_name": "SVDWrapper",
        "category": "mf",
        "supports_cold_start": False,
        "supports_warm_start": True,
        "required_features": [],
        "description": "SVD matrix factorization",
    },
    "neumf": {
        "cls": None,
        "module": "models.wrappers",
        "class_name": "NeuMFWrapper",
        "category": "neural_cf",
        "supports_cold_start": False,
        "supports_warm_start": True,
        "required_features": ["user_id", "item_id"],
        "description": "Neural Matrix Factorization (GMF + MLP)",
    },
    "deepfm": {
        "cls": None,
        "module": "models.wrappers",
        "class_name": "DeepFMWrapper",
        "category": "neural_ctr",
        "supports_cold_start": False,
        "supports_warm_start": True,
        "required_features": ["user_id", "item_id", "gender", "age", "occupation"],
        "description": "DeepFM: Factorization Machine + Deep Neural Network",
    },
    "lightgcn": {
        "cls": None,
        "module": "models.lightgcn",
        "class_name": "LightGCN",
        "category": "graph",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": ["user_id", "item_id"],
        "description": "LightGCN: Lightweight Graph Convolution Network",
    },
    "profile_mlp": {
        "cls": None,
        "module": "models.profile_mlp",
        "class_name": "ProfileOnlyModel",
        "category": "neural",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": ["profile"],
        "description": "Profile-only MLP (user + item features)",
    },
    "behavior_mlp": {
        "cls": None,
        "module": "models.profile_mlp",
        "class_name": "BehaviorOnlyModel",
        "category": "neural",
        "supports_cold_start": False,
        "supports_warm_start": True,
        "required_features": ["behavior"],
        "description": "Behavior-only MLP (interaction statistics)",
    },
    "hybrid": {
        "cls": None,
        "module": "models.hybrid_model",
        "class_name": "HybridModel",
        "category": "neural",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": ["profile", "behavior"],
        "description": "Hybrid MLP combining profile + behavior features",
    },
    "dual_hard_switch": {
        "cls": None,
        "module": "models.dual_scenario",
        "class_name": "DualScenarioModel",
        "category": "dual",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": ["profile", "behavior"],
        "description": "Dual-Scenario model with hard switch",
    },
    "dual_soft_gating": {
        "cls": None,
        "module": "models.dual_scenario",
        "class_name": "DualScenarioModel",
        "category": "dual",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": ["profile", "behavior"],
        "description": "Dual-Scenario model with soft gating",
    },
    "cf_userknn": {
        "cls": None,
        "module": "models.cf_model",
        "class_name": "UserCF",
        "category": "cf",
        "supports_cold_start": True,
        "supports_warm_start": True,
        "required_features": ["user_id", "item_id"],
        "description": "User-based KNN collaborative filtering",
    },
}


def _lazy_import(module_name: str, class_name: str):
    """Lazy import to avoid circular dependencies."""
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def create_model(model_name: str, model_config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """Create a model instance by name.

    Args:
        model_name: Model name (must be in MODEL_REGISTRY).
        model_config: Model configuration dict.

    Returns:
        BaseModel instance.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    config = model_config or {}

    if model_name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )

    entry = MODEL_REGISTRY[model_name]

    if model_name == "dual_hard_switch":
        config = {**config, "variant": "hard_switch"}
    elif model_name == "dual_soft_gating":
        config = {**config, "variant": "soft_gating"}

    if entry["cls"] is None:
        entry["cls"] = _lazy_import(entry["module"], entry["class_name"])

    cls = entry["cls"]
    return cls(config)


def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """Get model metadata (excluding class reference)."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    return {k: v for k, v in MODEL_REGISTRY[model_name].items() if k != "cls"}


def list_models(category: Optional[str] = None) -> list:
    """List all registered models, optionally filtered by category."""
    if category:
        return [name for name, entry in MODEL_REGISTRY.items()
                if entry.get("category") == category]
    return sorted(MODEL_REGISTRY.keys())


def get_model_count() -> int:
    """Return total number of registered models."""
    return len(MODEL_REGISTRY)
