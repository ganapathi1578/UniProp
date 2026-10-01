# Configuration

Configuration is currently handled natively in code inside `main_pipeline.py`.

## Core Settings
- `SEED`: Deterministic randomness lock (Default: 42).
- `SHARD_SIZE`: Number of examples to hold in memory before writing a `.parquet` chunk (Default: 50000).
- `DATA_ROOT`: Defines the base relative path for input/output resolution.
