import os
import pytest
from src.config import UniPropConfig, load_config

def test_default_config_loading():
    config = load_config()
    assert config.dataset_name == "agqa_scene_graphs"
    assert config.scale == "100k"
    assert config.total_target == 100000
    assert abs(sum(config.reasoning_distribution.values()) - 1.0) < 1e-4
    assert abs(sum(config.modality_distribution.values()) - 1.0) < 1e-4
    assert abs(sum(config.option_count_distribution.values()) - 1.0) < 1e-4
    assert abs(sum(config.reasoning_depth_distribution.values()) - 1.0) < 1e-4
    assert abs(sum(config.negative_difficulty_distribution.values()) - 1.0) < 1e-4

def test_yaml_config_loading():
    for scale in ["100k", "1m", "10m", "100m"]:
        path = os.path.join("config", "generation", f"{scale}.yaml")
        assert os.path.exists(path), f"Missing config for scale {scale}"
        cfg = load_config(path)
        assert cfg.scale == scale
        assert cfg.get_scale_output_dir().replace("\\", "/").endswith(f"data/generated/{scale}")
        assert cfg.get_split_output_dir("train").replace("\\", "/").endswith(f"data/generated/{scale}/train")

def test_invalid_distribution_error():
    invalid_data = {
        "distributions": {
            "reasoning_families": {
                "action": 0.50,
                "temporal": 0.20  # Sum is 0.70 != 1.0
            }
        }
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        UniPropConfig(invalid_data)
