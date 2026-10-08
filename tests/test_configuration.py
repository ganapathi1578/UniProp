import pytest
import os
import yaml
from unittest.mock import patch, MagicMock

from src.cli.main import main
from src.pipeline.dataset_pipeline import run_dataset_pipeline

def test_agqa_balanced_config():
    with open("config/datasets/agqa_balanced.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    assert config["selection"]["mode"] == "all"
    assert config["selection"]["max_records"] is None
    assert config["output"]["root_dir"] == "data/generated/agqa_balanced"
    
def test_agqa_scenegraphs_config():
    with open("config/datasets/agqa_scenegraphs.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    assert config["selection"]["mode"] == "target_size"
    assert config["selection"]["target_records"] == 1000000
    assert config["output"]["scale"] == "1m"
    assert config["output"]["root_dir"] == "data/generated/agqa_scenegraphs"

@patch('src.pipeline.dataset_pipeline.os.makedirs')
@patch('src.pipeline.dataset_pipeline.dataset_registry')
def test_pipeline_output_paths(mock_registry, mock_makedirs):
    # Mock adapter to return empty list
    mock_adapter = MagicMock()
    mock_adapter.load_records.return_value = []
    mock_registry.get_adapter.return_value = lambda cfg: mock_adapter
    
    # 1. Balanced output
    config_balanced = {
        "dataset": "agqa_balanced",
        "output": {"root_dir": "data/generated/agqa_balanced"}
    }
    run_dataset_pipeline(config_balanced, "train")
    mock_makedirs.assert_called_with(os.path.join("data/generated/agqa_balanced", "train"), exist_ok=True)
    
    # 2. Scenegraph output
    config_sg = {
        "dataset": "agqa_scenegraphs",
        "output": {"root_dir": "data/generated/agqa_scenegraphs", "scale": "1m"}
    }
    run_dataset_pipeline(config_sg, "train")
    mock_makedirs.assert_called_with(os.path.join("data/generated/agqa_scenegraphs", "1m", "train"), exist_ok=True)

    config_sg_10m = {
        "dataset": "agqa_scenegraphs",
        "output": {"root_dir": "data/generated/agqa_scenegraphs", "scale": "10m"}
    }
    run_dataset_pipeline(config_sg_10m, "train")
    mock_makedirs.assert_called_with(os.path.join("data/generated/agqa_scenegraphs", "10m", "train"), exist_ok=True)
    
    config_sg_100m = {
        "dataset": "agqa_scenegraphs",
        "output": {"root_dir": "data/generated/agqa_scenegraphs", "scale": "100m"}
    }
    run_dataset_pipeline(config_sg_100m, "train")
    mock_makedirs.assert_called_with(os.path.join("data/generated/agqa_scenegraphs", "100m", "train"), exist_ok=True)

def test_no_derived_subsets_in_codebase():
    # Grep codebase for 'agqa_balanced_subsets' to ensure it's not being created
    import subprocess
    result = subprocess.run(
        ["git", "grep", "agqa_balanced_subsets"], 
        cwd=".", capture_output=True, text=True
    )
    # The only acceptable place is in documentation or this test file
    for line in result.stdout.split('\n'):
        if line.strip() and not line.startswith('docs/') and not line.startswith('tests/'):
            pytest.fail(f"Found forbidden string 'agqa_balanced_subsets' in code: {line}")
