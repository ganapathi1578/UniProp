import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def main():
    parser = argparse.ArgumentParser(description="UniProp Dataset Engineering Framework")
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running dataset framework with config: {config}")

if __name__ == '__main__':
    main()
