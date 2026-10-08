import sys
sys.path.insert(0, "c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp")

import yaml
from src.pipeline.dataset_pipeline import run_dataset_pipeline

with open("run_small.yaml", "r") as f:
    config = yaml.safe_load(f)

run_dataset_pipeline(config, "train", max_records=1000)
