"""Configuration loader and schema for UniProp dataset generation."""
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

DEFAULT_BASE_CONFIG = {
    "dataset_name": "agqa_scene_graphs",
    "version": "1.0",
    "scale": "100k",
    "paths": {
        "dataset_root": "data/dataset",
        "scene_graphs_train": "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl",
        "scene_graphs_test": "data/dataset/agqa_scene_graphs/AGQA_test_stsgs.pkl",
        "qa_train": "data/dataset/agqa_balanced/train_balanced.txt",
        "qa_test": "data/dataset/agqa_balanced/test_balanced.txt",
        "output_dir": "data/generated",
    },
    "generation": {
        "seed": 42,
        "shard_size": 50000,
    },
    "splits": {
        "video_split_ratio": {
            "train": 0.80,
            "val": 0.10,
            "test": 0.10,
        },
        "target_counts": {
            "train": 80000,
            "val": 10000,
            "test": 10000,
        },
    },
    "distributions": {
        "reasoning_families": {
            "action": 0.20,
            "temporal": 0.20,
            "compositional": 0.20,
            "object": 0.15,
            "spatial_contact": 0.15,
            "grounding": 0.10,
        },
        "task_modalities": {
            "single_choice": 0.45,
            "binary": 0.25,
            "multi_label": 0.20,
            "three_way": 0.10,
        },
        "option_counts": {
            "k_2": 0.25,
            "k_3": 0.10,
            "k_4_8": 0.25,
            "k_9_15": 0.20,
            "k_16_28": 0.15,
            "k_29_40": 0.05,
        },
        "reasoning_depth": {
            "hop_1": 0.20,
            "hop_2": 0.30,
            "hop_3": 0.30,
            "hop_4_plus": 0.20,
        },
        "negative_difficulty": {
            "easy": 0.20,
            "medium": 0.40,
            "hard": 0.40,
        },
    },
}

SCALE_TARGETS = {
    "100k": {"train": 80000, "val": 10000, "test": 10000},
    "1m": {"train": 800000, "val": 100000, "test": 100000},
    "10m": {"train": 8000000, "val": 1000000, "test": 1000000},
    "100m": {"train": 80000000, "val": 10000000, "test": 10000000},
}


class UniPropConfig:
    """UniProp pipeline configuration class."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        self.raw = self._deep_merge(DEFAULT_BASE_CONFIG, raw or {})
        self._validate()

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = UniPropConfig._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _validate(self):
        # Validate distribution totals
        dist = self.raw.get("distributions", {})
        for name, d in dist.items():
            if isinstance(d, dict):
                total = sum(d.values())
                if abs(total - 1.0) > 1e-4:
                    raise ValueError(f"Distribution '{name}' must sum to 1.0, got {total:.4f}")

        # Validate video split ratios sum to 1.0
        v_ratios = self.raw.get("splits", {}).get("video_split_ratio", {})
        if v_ratios and abs(sum(v_ratios.values()) - 1.0) > 1e-4:
            raise ValueError(f"Video split ratios must sum to 1.0, got {sum(v_ratios.values()):.4f}")

    @classmethod
    def from_yaml(cls, path: str) -> "UniPropConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @property
    def dataset_name(self) -> str:
        return self.raw.get("dataset_name", "agqa_scene_graphs")

    @property
    def version(self) -> str:
        return self.raw.get("version", "1.0")

    @property
    def scale(self) -> str:
        return str(self.raw.get("scale", "100k")).lower()

    @property
    def paths(self) -> Dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def output_dir(self) -> str:
        return self.paths.get("output_dir", "data/generated")

    @property
    def seed(self) -> int:
        return self.raw.get("generation", {}).get("seed", 42)

    @property
    def shard_size(self) -> int:
        return self.raw.get("generation", {}).get("shard_size", 50000)

    @property
    def video_split_ratios(self) -> Dict[str, float]:
        return self.raw.get("splits", {}).get("video_split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})

    @property
    def target_counts(self) -> Dict[str, int]:
        cfg_counts = self.raw.get("splits", {}).get("target_counts")
        if cfg_counts:
            return cfg_counts
        # Fallback to scale targets if present
        scale_key = self.scale
        if scale_key in SCALE_TARGETS:
            return SCALE_TARGETS[scale_key]
        return {"train": 80000, "val": 10000, "test": 10000}

    @property
    def total_target(self) -> int:
        return sum(self.target_counts.values())

    @property
    def reasoning_distribution(self) -> Dict[str, float]:
        return self.raw["distributions"]["reasoning_families"]

    @property
    def modality_distribution(self) -> Dict[str, float]:
        return self.raw["distributions"]["task_modalities"]

    @property
    def option_count_distribution(self) -> Dict[str, float]:
        return self.raw["distributions"]["option_counts"]

    @property
    def reasoning_depth_distribution(self) -> Dict[str, float]:
        return self.raw["distributions"]["reasoning_depth"]

    @property
    def negative_difficulty_distribution(self) -> Dict[str, float]:
        return self.raw["distributions"]["negative_difficulty"]

    def get_scale_output_dir(self, base_root: str = "") -> str:
        out_root = self.output_dir
        if base_root and not os.path.isabs(out_root):
            out_root = os.path.join(base_root, out_root)
        return os.path.join(out_root, self.scale)

    def get_split_output_dir(self, split: str, base_root: str = "") -> str:
        return os.path.join(self.get_scale_output_dir(base_root), split)


def load_config(config_path_or_dict=None) -> UniPropConfig:
    if config_path_or_dict is None:
        return UniPropConfig()
    if isinstance(config_path_or_dict, str):
        return UniPropConfig.from_yaml(config_path_or_dict)
    if isinstance(config_path_or_dict, dict):
        return UniPropConfig(config_path_or_dict)
    raise TypeError(f"Unsupported config type: {type(config_path_or_dict)}")
