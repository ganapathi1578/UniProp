# Reproducibility

UniProp guarantees identical byte-for-byte generation outputs across runs given the same source dataset and configuration.

## Determinism Chain
1. `PYTHONHASHSEED` is locked to `42` to freeze dictionary/set insertion order.
2. The global Python `random.seed` is locked to `42`.
3. An explicit localized `rng = random.Random(SEED)` is initialized and passed directly into the generator loops to prevent side-effect desynchronization.

## Verifying Reproducibility
To test, run a small generation, snapshot the `.parquet` outputs, delete the `data/generated/` folder, and run again. The outputs will be identical.
