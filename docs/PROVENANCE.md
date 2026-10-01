# Provenance

Provenance ensures every generated example can be traced back to its raw origins.

## Source of Truth
- **Source Dataset**: AGQA.
- **video_id**: The mandatory, explicit source video ID (e.g., `46GP8`).

The `video_id` is passed directly from the `AGQA_balanced` and `AGQA_scene_graphs` loaders through the pipeline into the Parquet writer.

**example_id must not be the sole mechanism used to recover source-video provenance.** The `video_id` column is the canonical source-of-truth.
