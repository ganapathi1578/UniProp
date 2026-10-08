import random
import pytest
from src.config import load_config
from src.sampling import SamplingController, TargetSpec

def test_sampling_controller_spec_generation():
    config = load_config()
    controller = SamplingController(config, split_target=1000)
    rng = random.Random(42)

    valid_families = set(config.reasoning_distribution.keys())
    valid_tasks = set(config.modality_distribution.keys())
    valid_diffs = set(config.negative_difficulty_distribution.keys())

    for _ in range(100):
        spec: TargetSpec = controller.sample_target(rng)
        assert spec.family in valid_families
        assert spec.task_type in valid_tasks
        assert spec.difficulty in valid_diffs
        assert 1 <= spec.hops <= 4

        if spec.task_type == "binary":
            assert spec.k == 2
        elif spec.task_type == "three_way":
            assert spec.k == 3
        else:
            assert 4 <= spec.k <= 40

        # Record accepted example
        controller.record_accepted(
            family=spec.family,
            task_type=spec.task_type,
            k=spec.k,
            hops=spec.hops,
            difficulty=spec.difficulty
        )

    assert sum(controller.family_counts.values()) == 100
    assert sum(controller.task_counts.values()) == 100
