# Production Runbook

Follow these steps for a massive 10M+ generation run.

1. **Clean Environment Check**: Ensure no old corrupted `.parquet` files exist in `data/generated/`.
2. **Source-Data Check**: Verify `AGQA_balanced/` and `AGQA_scene_graphs/` are fully populated.
3. **Disk Space Check**: Ensure at least ~10GB of free disk space is available for PyArrow memory buffers and final storage.
4. **Production Launch**: Run `python main_pipeline.py`.
5. **Monitoring**: Watch the `tqdm` progress bars.
6. **Interruption Recovery**: If the script crashes or is terminated, simply run `python main_pipeline.py` again. It will automatically load the checkpoint hashes from the existing `.parquet` shards and resume where it left off.
