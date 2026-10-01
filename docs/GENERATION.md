# Generation Guide

## Running Generation

To run a production generation:
```bash
python main_pipeline.py
```

## Configurable Parameters (in main_pipeline.py)
- `SEED`: Fixed at `42` for deterministic reproducibility.
- `SHARD_SIZE`: Default `50000`. Controls how many records are stored in memory before flushing a chunk to disk.
- `generators`: The list of generator objects to execute.

## Behavior
- **Flushing**: Occurs when `SHARD_SIZE` is reached.
- **Output Directories**: Written natively to `data/generated/train`, `data/generated/val`, `data/generated/test`.
- **Checkpoint/Resume**: Checkpointing caches previously generated `example_id`s directly from the `.parquet` files on disk. If interrupted, the pipeline will reload the seen hashes and skip over them.
