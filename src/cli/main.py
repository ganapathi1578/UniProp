import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline.dataset_pipeline import run_dataset_pipeline

def main():
    parser = argparse.ArgumentParser(description="UniProp Dataset Engineering Framework")
    parser.add_argument('--config', type=str, default='config/datasets/agqa_balanced.yaml')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max-records', type=int, default=100, help="For smoke testing")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running UniProp with config: {config}")
    success, rejected = run_dataset_pipeline(config, args.split, args.max_records)
    print(f"Done! Success: {success}, Rejected: {rejected}")

if __name__ == '__main__':
    main()
