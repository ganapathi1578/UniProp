# Contributing

Contributions to the pipeline logic must strictly adhere to the following conventions:
1. **No Data in Git**: Do not commit datasets, caches, or `.parquet` shards.
2. **Schema Lockdown**: Do not alter `src/storage/parquet_writer.py` without explicit review.
3. **Determinism**: Use the passed `rng` object for randomness.
4. **Documentation**: Update the relevant `docs/` Markdown files if you change behavior.
