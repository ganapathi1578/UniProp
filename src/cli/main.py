"""CLI entry point for the UniProp dataset engineering framework.

This module wires together the argument parser and the dataset pipeline so
that the framework can be driven from the command line.  It is the standard
way to launch a full pipeline run against a dataset configuration file.

**Usage**::

    python -m src.cli.main \\
        --config config/datasets/agqa_balanced.yaml \\
        --split  train \\
        --max-records 1000

Arguments
---------
--config : str, optional
    Path to a YAML configuration file describing the dataset, source paths,
    output settings, and generation parameters.
    Defaults to ``config/datasets/agqa_balanced.yaml``.

--split : str, optional
    Dataset split to process (e.g. ``"train"``, ``"val"``, ``"test"``).
    Must match a key under ``splits`` in the config file, or be ignored if
    the file has no ``splits`` block.
    Defaults to ``"train"``.

--max-records : int, optional
    If provided, stop after reading this many source records from the adapter.
    Useful for smoke-testing a configuration without processing the full
    dataset.  When omitted the entire split is processed.
    Defaults to ``None`` (no limit).

**Exit behaviour**
The script prints a one-line summary (``Done! Success: N, Rejected: M``) and
exits normally.  Any unhandled exception propagates to the shell.
"""

import argparse
import yaml
import sys
import os

# Ensure the project root is on sys.path when this script is invoked directly
# (e.g. `python src/cli/main.py`) rather than via `python -m`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline.dataset_pipeline import run_dataset_pipeline


def main():
    """Parse command-line arguments, load configuration, and run the pipeline.

    This function is the sole entry point for the CLI.  It performs three
    steps in sequence:

    1. **Parse arguments** – constructs an :class:`argparse.ArgumentParser`
       with the ``--config``, ``--split``, and ``--max-records`` flags and
       calls :meth:`~argparse.ArgumentParser.parse_args`.

    2. **Load configuration** – opens the YAML file at ``args.config`` with
       :func:`yaml.safe_load` and stores the result as a plain Python dict.

    3. **Run pipeline** – delegates to
       :func:`~src.pipeline.dataset_pipeline.run_dataset_pipeline` with the
       loaded config, the requested split, and the optional record cap.
       Prints a summary of ``success`` and ``rejected`` counts on completion.

    Args:
        None.  All input comes from ``sys.argv`` via :mod:`argparse`.

    Returns:
        None.  Results are reported to stdout; the function returns
        implicitly.

    Raises:
        FileNotFoundError: If the path given by ``--config`` does not exist.
        yaml.YAMLError: If the configuration file is not valid YAML.
        ValueError: Propagated from
            :func:`~src.pipeline.dataset_pipeline.run_dataset_pipeline` if
            the ``dataset`` key in the config is not registered.
    """
    # ------------------------------------------------------------------ #
    # 1. Argument parsing                                                  #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description="UniProp Dataset Engineering Framework")

    # Path to the YAML configuration file for the target dataset.
    parser.add_argument('--config', type=str, default='config/datasets/agqa_balanced.yaml')

    # Dataset split to process; must match a split key in the config file.
    parser.add_argument('--split', type=str, default='train')

    # Optional hard cap on source records; primarily for development / smoke testing.
    parser.add_argument('--max-records', type=int, default=None, help="Runtime override for development/smoke testing")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 2. Load YAML configuration                                           #
    # ------------------------------------------------------------------ #
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # 3. Run pipeline and report results                                   #
    # ------------------------------------------------------------------ #
    print(f"Running UniProp with config: {config}")
    success, rejected = run_dataset_pipeline(config, args.split, args.max_records)
    print(f"Done! Success: {success}, Rejected: {rejected}")


if __name__ == '__main__':
    main()
