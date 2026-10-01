# Development Guide

UniProp is a static dataset engineering pipeline.

## Adding a Generator
1. Create a new subclass in `src/generators/`.
2. Implement the `generate` generator yield method.
3. Yield `PropositionGroup` objects.
4. Register the class instance in the `generators` list inside `main_pipeline.py`.

## Changing Schema
If you change the PyArrow schema in `src/storage/parquet_writer.py`, you **must** update `DATA_SCHEMA.md` and bump the major version.

## Determinism
All randomness must use the local `rng = random.Random(SEED)` instance passed to the generators. Do not use global `random.*` or `numpy.random.*`.
