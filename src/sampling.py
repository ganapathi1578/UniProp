"""Sampling controller and quota scheduler for UniProp generation."""
import random
from typing import Dict, Any, Optional, Tuple, NamedTuple
from collections import Counter
from src.config import UniPropConfig


class TargetSpec(NamedTuple):
    family: str
    task_type: str
    k: int
    hops: int
    difficulty: str


class SamplingController:
    """Enforces the 6 x 4 x 4 controlled sampling matrix and quota targets."""

    def __init__(self, config: UniPropConfig, split_target: int):
        self.config = config
        self.split_target = max(1, split_target)

        # Quotas
        self.family_quotas = {k: int(round(v * self.split_target)) for k, v in config.reasoning_distribution.items()}
        self.task_quotas = {k: int(round(v * self.split_target)) for k, v in config.modality_distribution.items()}
        self.hop_quotas = {
            1: int(round(config.reasoning_depth_distribution.get("hop_1", 0.20) * self.split_target)),
            2: int(round(config.reasoning_depth_distribution.get("hop_2", 0.30) * self.split_target)),
            3: int(round(config.reasoning_depth_distribution.get("hop_3", 0.30) * self.split_target)),
            4: int(round(config.reasoning_depth_distribution.get("hop_4_plus", 0.20) * self.split_target)),
        }
        self.diff_quotas = {k: int(round(v * self.split_target)) for k, v in config.negative_difficulty_distribution.items()}
        self.k_quotas = {k: int(round(v * self.split_target)) for k, v in config.option_count_distribution.items()}

        # Active counts
        self.family_counts = Counter()
        self.task_counts = Counter()
        self.hop_counts = Counter()
        self.diff_counts = Counter()
        self.k_counts = Counter()

    def _sample_from_deficit(self, quotas: Dict[Any, int], counts: Counter, fallback_probs: Dict[Any, float], rng: random.Random) -> Any:
        deficits = {k: max(0, quotas.get(k, 0) - counts.get(k, 0)) for k in quotas}
        total_deficit = sum(deficits.values())
        if total_deficit > 0:
            keys = list(deficits.keys())
            weights = [deficits[k] for k in keys]
            return rng.choices(keys, weights=weights, k=1)[0]
        # All quotas filled; use configured distribution
        keys = list(fallback_probs.keys())
        weights = [fallback_probs[k] for k in keys]
        return rng.choices(keys, weights=weights, k=1)[0]

    def sample_target(self, rng: random.Random) -> TargetSpec:
        """Sample a target specification for the next example."""
        # 1. Family (6 families)
        family = self._sample_from_deficit(
            self.family_quotas, self.family_counts, self.config.reasoning_distribution, rng
        )

        # 2. Task Modality (4 modalities)
        task_type = self._sample_from_deficit(
            self.task_quotas, self.task_counts, self.config.modality_distribution, rng
        )

        # 3. Option count (K)
        if task_type == "binary":
            k = 2
        elif task_type == "three_way":
            k = 3
        else:
            # single_choice or multi_label: choose range based on deficit among [k_4_8, k_9_15, k_16_28, k_29_40]
            k_sub_quotas = {
                "k_4_8": self.k_quotas.get("k_4_8", 0),
                "k_9_15": self.k_quotas.get("k_9_15", 0),
                "k_16_28": self.k_quotas.get("k_16_28", 0),
                "k_29_40": self.k_quotas.get("k_29_40", 0),
            }
            k_sub_fallback = {
                "k_4_8": self.config.option_count_distribution.get("k_4_8", 0.25),
                "k_9_15": self.config.option_count_distribution.get("k_9_15", 0.20),
                "k_16_28": self.config.option_count_distribution.get("k_16_28", 0.15),
                "k_29_40": self.config.option_count_distribution.get("k_29_40", 0.05),
            }
            bucket = self._sample_from_deficit(k_sub_quotas, self.k_counts, k_sub_fallback, rng)
            if bucket == "k_4_8":
                k = rng.randint(4, 8)
            elif bucket == "k_9_15":
                k = rng.randint(9, 15)
            elif bucket == "k_16_28":
                k = rng.randint(16, 28)
            else:
                k = rng.randint(29, 40)

        # 4. Reasoning depth (Hops: 1, 2, 3, 4+)
        hop_fallback = {
            1: self.config.reasoning_depth_distribution.get("hop_1", 0.20),
            2: self.config.reasoning_depth_distribution.get("hop_2", 0.30),
            3: self.config.reasoning_depth_distribution.get("hop_3", 0.30),
            4: self.config.reasoning_depth_distribution.get("hop_4_plus", 0.20),
        }
        hops = self._sample_from_deficit(self.hop_quotas, self.hop_counts, hop_fallback, rng)

        # 5. Negative difficulty (easy, medium, hard)
        difficulty = self._sample_from_deficit(
            self.diff_quotas, self.diff_counts, self.config.negative_difficulty_distribution, rng
        )

        return TargetSpec(family=family, task_type=task_type, k=k, hops=hops, difficulty=difficulty)

    def record_accepted(self, family: str, task_type: str, k: int, hops: int, difficulty: str = "medium"):
        """Record accepted example to update progress towards quotas."""
        # Normalize family for spatial/contact
        norm_family = family
        if family in ("spatial", "contact", "attention", "spatial_contact"):
            norm_family = "spatial_contact"

        self.family_counts[norm_family] += 1
        self.task_counts[task_type] += 1

        # K bucket
        if k == 2:
            self.k_counts["k_2"] += 1
        elif k == 3:
            self.k_counts["k_3"] += 1
        elif 4 <= k <= 8:
            self.k_counts["k_4_8"] += 1
        elif 9 <= k <= 15:
            self.k_counts["k_9_15"] += 1
        elif 16 <= k <= 28:
            self.k_counts["k_16_28"] += 1
        elif k >= 29:
            self.k_counts["k_29_40"] += 1

        # Hop bucket
        hop_key = min(max(1, hops), 4)
        self.hop_counts[hop_key] += 1

        # Difficulty bucket
        self.diff_counts[difficulty] += 1
