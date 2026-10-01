# Resource Requirements

## General Constraints
- **CPU**: Multi-core highly recommended for scene graph parsing.
- **RAM**: Minimum 16GB required to hold the AGQA datasets in memory.
- **Disk Storage (AGQA Source)**: ~2.5 GB required.
- **Disk Storage (Generated Data)**: Highly efficient Parquet raw-string dictionary encoding.

## Expected Scaling (ESTIMATED from Parquet Dictionary Compression)
- **10K Examples**: ~0.35 MB
- **100K Examples**: ~3.95 MB
- **1M Examples**: ~40 MB
- **10M Examples**: ~400 MB
- **100M Examples**: ~4 GB

*Temporary disk usage during flushing scales linearly with `SHARD_SIZE` memory buffers.*
