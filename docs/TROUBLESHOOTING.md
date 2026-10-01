# Troubleshooting

### Missing Input Data
**Symptom**: `FileNotFoundError: AGQA_balanced/train_balanced.txt`
**Cause**: The AGQA datasets were not downloaded or placed in the root directory.
**Recovery**: Download AGQA source files and place them in `AGQA_balanced/` and `AGQA_scene_graphs/`.

### Checkpoint/Resume Issues
**Symptom**: `pyarrow.lib.ArrowInvalid: Parquet magic bytes not found`
**Cause**: An interrupted run left a corrupted `.parquet` chunk on disk.
**Recovery**: Delete the 0-byte or corrupted `.parquet` file in `data/generated/` and re-run. Checkpointing will recover the rest.

### Missing Video ID
**Symptom**: Assertion Error `Missing video_id in generated examples`
**Cause**: The generator failed to capture `media_id` or `video_id`.
**Recovery**: Inspect the traceback to find the offending generator and ensure it explicitly passes the AGQA `video_id` to the `PropositionGroup`.
