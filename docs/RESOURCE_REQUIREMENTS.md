# RESOURCE REQUIREMENTS

> This document details the compute, memory, disk, and I/O requirements for
> running the UniProp dataset generation pipeline at various scales.

---

## Table of Contents

1. [Runtime Requirements](#runtime-requirements)
2. [Memory Breakdown](#memory-breakdown)
3. [Scaling Table](#scaling-table)
4. [Generation Speed](#generation-speed)
5. [Shard Flushing — Tuning `SHARD_SIZE`](#shard-flushing--tuning-shard_size)
6. [Disk I/O — Parquet Write Cost](#disk-io--parquet-write-cost)
7. [Monitoring](#monitoring)

---

## Runtime Requirements

### CPU

**Multi-core recommended.** The pipeline itself is single-threaded Python (no
`multiprocessing` or `concurrent.futures`), but multi-core processors benefit
the pipeline indirectly:

- The OS can schedule I/O-bound pickle loading and Parquet flushing on separate
  cores without blocking the main generation loop.
- `pyarrow` uses native threads internally during Parquet serialisation.
- The `tqdm` progress bar update overhead is negligible on faster CPUs.

A **4-core / 8-thread** modern CPU (e.g. Intel Core i5 / AMD Ryzen 5 or better)
provides adequate performance.  The scene-graph normalisation loop (Phase 4) is
the most CPU-intensive section, applying `normalize_video_sg()` to every video
in sequence.

### RAM

**Minimum 16 GB required.** The dominant memory consumers are the two AGQA
spatio-temporal scene-graph pickle files, which are fully loaded into memory
before generation begins.

| Reason | Minimum Requirement |
|---|---|
| AGQA train STSG pickle | ~2.0 GB in Python heap |
| AGQA test STSG pickle | ~500 MB in Python heap |
| Operating system + Python interpreter overhead | ~1.5 GB |
| **Total baseline (before generation)** | **~4 GB** |
| NegativePools per video, normalised SGs, seen_hashes | ~2–6 GB depending on scale |
| In-flight generation buffers (up to SHARD_SIZE × 3 splits) | ~0.5–2 GB |
| **Recommended headroom** | **16 GB total** |

Running with less than 16 GB may trigger OS swap activity during the pickle
loading phase, dramatically slowing the pipeline.  On systems with exactly
16 GB, close all non-essential applications before starting.

> [!CAUTION]
> Do NOT load both pickle files simultaneously if RAM is constrained.  The
> pipeline loads `train_stsgs.pkl` first, completes the train/val pass, and
> then loads `test_stsgs.pkl`.  If you modify the pipeline to keep both in
> memory simultaneously, peak usage can reach ~8–10 GB from pickles alone.

### Disk Storage

| Component | Size | Notes |
|---|---|---|
| `AGQA_train_stsgs.pkl` | ~2.0 GB | Required source file; do not delete |
| `AGQA_test_stsgs.pkl` | ~500 MB | Required source file; do not delete |
| `AGQA_balanced/train_balanced.txt` | ~50 MB | Streamed lazily; not held in memory |
| `AGQA_balanced/test_balanced.txt` | ~15 MB | Streamed lazily; not held in memory |
| **Total source data** | **~2.6 GB** | Must be present before pipeline runs |
| `data/generated/` output (see scaling table) | 0.35 MB – 4 GB | Depends on target example count |
| **Recommended free disk** | **≥ 15 GB** | Provides headroom for intermediate shards and OS temp files |

---

## Memory Breakdown

The following table decomposes peak memory usage during a full 1M-example run:

| Component | Approximate Size | When Active |
|---|---|---|
| `train_sgs_raw` (pickle) | ~2.0 GB | Phases 2–5 (train/val pass) |
| `test_sgs_raw` (pickle) | ~500 MB | Phase 5 (test pass only) |
| `sgs_norm` (all normalised SGs) | ~300–600 MB | Phases 4–5 (normalisation → generation) |
| `pools_cache` (NegativePools per video) | ~100–300 MB | Phases 4–5 (generation loop) |
| `seen_hashes` (Python `set` of ints) | ~50 MB per 1M hashes | Grows linearly with accepted examples |
| `buffers["train"]` (in-memory buffer) | up to ~200 MB at SHARD_SIZE=50 000 | Flushed when full |
| `buffers["val"]` | up to ~50 MB | Flushed when full |
| `buffers["test"]` | up to ~50 MB | Flushed when full |
| `by_rule` (sample groups dict) | < 10 MB | Entire run; at most 5 groups × 18 rules |

### `NegativePools` Memory Profile

Each `NegativePools` object pre-computes several auxiliary data structures for
hard-negative sampling:

- `present_objects`: set of object class tokens present in the video.
- `present_relations`: list of `(rel_type, rel_class, obj)` triples.
- Internal indices for mutually-exclusive and closed-world relation lookups.

For a typical Charades video (~20 frames, ~5 objects per frame), a single
`NegativePools` object consumes approximately **2–5 KB**.  Across all ~10 000
training videos, `pools_cache` totals approximately **20–50 MB** — well within
the 16 GB budget.

### `seen_hashes` Growth

The `seen_hashes` set grows by one Python `int` per accepted example.  Python
integers stored in a `set` occupy approximately **50–60 bytes** each (including
set overhead).

| Accepted Examples | `seen_hashes` Size |
|---|---|
| 100 000 | ~5 MB |
| 1 000 000 | ~50 MB |
| 10 000 000 | ~500 MB |

At 10M+ scale, consider switching to a probabilistic dedup structure (e.g.
`pybloom` Bloom filter) if `seen_hashes` memory becomes a constraint.

---

## Scaling Table

Output sizes use Parquet's dictionary encoding with per-column compression.
Proposition text columns compress extremely well because the vocabulary of
surface-form phrases is small relative to the number of rows.

| Target Examples | Est. Output Size | Est. `seen_hashes` RAM | Notes |
|---|---|---|---|
| 10 000 | ~0.35 MB | < 1 MB | Development / smoke test |
| 100 000 | ~3.95 MB | ~5 MB | Quick benchmark runs |
| 500 000 | ~20 MB | ~25 MB | Full development scale |
| 1 000 000 | ~40 MB | ~50 MB | Recommended training scale |
| 5 000 000 | ~200 MB | ~250 MB | Large-scale training |
| 10 000 000 | ~400 MB | ~500 MB | Near the practical limit of single-process generation |
| 100 000 000 | ~4 GB | ~5 GB | Requires distributed generation or Bloom filter for dedup |

> [!NOTE]
> These estimates assume the default `SHARD_SIZE=50 000` and Parquet
> dictionary + Snappy compression.  Enabling Zstandard (zstd) compression
> reduces output size by a further 15–30 % at the cost of slightly higher
> CPU usage during writes.

### Why Output is So Small Relative to Row Count

UniProp Parquet output is compact because:

1. **Dictionary encoding**: Proposition text columns (e.g. `"The person is
   holding the bag."`) use Parquet dictionary pages.  The same surface-form
   string is stored once in the dictionary and referenced by a small integer
   index in subsequent rows.
2. **Repeated schema values**: Fields like `task_type`, `reasoning.family`, and
   `provenance.generator_family` have very low cardinality (< 20 distinct values)
   — they compress almost perfectly.
3. **Struct nesting**: Nested structs (`propositions`, `reasoning`, `provenance`)
   are stored in columnar layout, grouping identical field values across rows for
   better run-length encoding.

---

## Generation Speed

### Approximate Throughput

Under typical hardware conditions (4-core CPU, 16 GB RAM, NVMe SSD):

| Phase | Approximate Speed |
|---|---|
| Pickle loading (`train_stsgs.pkl`, 2 GB) | 15–40 seconds |
| Scene-graph normalisation (all training videos) | 30–120 seconds |
| NegativePools construction (all training videos) | 20–60 seconds |
| **Active generation loop** | **200–800 examples / second** |
| Parquet shard flush (50 000 rows) | 0.5–3 seconds |

### What Affects Generation Speed

| Factor | Impact |
|---|---|
| **Validator rejection rate** | High rejection rates (> 50 %) mean many candidates are processed but not accepted, reducing effective throughput |
| **Generator yield rate** | Generators that yield 0 candidates (e.g. videos with no valid frames for grounding) consume a round-robin slot without producing output |
| **SHARD_SIZE** | Larger shards mean fewer but slower flushes; smaller shards mean more frequent but faster I/O |
| **Disk speed** | NVMe SSD preferred; HDD writes add ~2–5× more flush time |
| **Python object allocation** | Creating `Proposition`, `PropositionGroup`, and `Provenance` dataclasses dominates CPU time in the inner loop |
| **`seen_hashes` set size** | Python `in` lookups on large sets are O(1) average but become slower as hash collisions increase above ~10M entries |

### Progress Reporting

The pipeline logs a progress line approximately every 1 % of the target count
or every 10 seconds, whichever comes first:

```
[  1.0%] Generated 8000/800000 (412.3 records/sec) - 19.4s elapsed
[ 10.0%] Generated 80000/800000 (389.7 records/sec) - 205.2s elapsed
```

Use these logs to project total runtime:

```
estimated_total_seconds = (target_count / current_rate)
```

---

## Shard Flushing — Tuning `SHARD_SIZE`

`SHARD_SIZE = 50 000` is defined at the top of
[`main_pipeline.py`](../main_pipeline.py):

```python
SHARD_SIZE = 50000
```

### How Shard Flushing Works

The `flush_buffer(split, force=False)` inner function flushes the in-memory
buffer for a given split to a Parquet file:

```python
def flush_buffer(split, force=False):
    n = len(buffers[split])
    if not force and n < SHARD_SIZE:
        return          # Not enough rows yet — skip
    if n == 0:
        return          # Nothing to write
    ...
    write_shard(buffers[split], out_path)
    buffers[split] = []   # Clear after successful write
    shard_indices[split] += 1
```

After each accepted example is appended to `buffers[split]`, `flush_buffer` is
called immediately with `force=False`.  The buffer grows until it reaches
`SHARD_SIZE`, at which point the shard is written and the buffer is cleared.

At pipeline teardown, `flush_buffer(split, force=True)` is called for every
split to write any partial buffer that has not yet reached the threshold.

### Peak Buffer Memory

With `SHARD_SIZE=50 000` and typical `PropositionGroup` objects (~4 KB each
in Python heap), the peak buffer memory for a single split is approximately:

```
50 000 rows × ~4 KB/row ≈ 200 MB
```

Three splits (`train`, `val`, `test`) can simultaneously hold data, but in
practice `train` dominates: `val` and `test` receive ≈ 10 % and ≈ 20 % of
traffic respectively, so peak total buffer usage is approximately **220–240 MB**.

### Tuning Recommendations

| Scenario | Recommended `SHARD_SIZE` | Rationale |
|---|---|---|
| RAM-constrained (< 12 GB) | 10 000 – 20 000 | Reduces peak buffer footprint |
| Fast NVMe SSD, high throughput | 100 000 – 200 000 | Fewer I/O operations, better write efficiency |
| Slow HDD | 10 000 – 25 000 | Smaller shards reduce the time per flush, improving responsiveness |
| Distributed / multi-worker read | 50 000 (default) | Standard Parquet row-group size is compatible with most downstream tools |
| Debugging / development | 1 000 – 5 000 | Lets you inspect partial output quickly |

> [!TIP]
> When training downstream models with `datasets` (HuggingFace) or
> `spark.read.parquet()`, larger shard sizes (≥ 50 000 rows) yield better
> read throughput because metadata overhead per file is amortised over more rows.

---

## Disk I/O — Parquet Write Cost

### Write Path

Each shard is written by `src/storage/parquet_writer.py::write_shard()`, which
converts `PropositionGroup` objects to a PyArrow `Table` and calls
`pq.write_table()`:

```python
pq.write_table(
    table,
    out_path,
    compression="snappy",    # default; fast, moderate ratio
)
```

### Compression Options

| Option | Compression Level | Write Speed | Read Speed | Ratio vs. Uncompressed |
|---|---|---|---|---|
| `"snappy"` (default) | Low | Fast | Fast | ~40–60 % smaller |
| `"zstd"` | High | Moderate | Moderate | ~55–75 % smaller |
| `"gzip"` | High | Slow | Slow | ~55–70 % smaller |
| `None` (uncompressed) | None | Fastest | Fastest | Baseline |

To enable Zstandard compression, modify `write_shard()` in
`src/storage/parquet_writer.py`:

```python
pq.write_table(table, out_path, compression="zstd")
```

Zstandard is recommended for long-term archival or when disk space is at a
premium.  For active training workloads, `"snappy"` offers the best
read-speed / compression trade-off.

### Checkpoint Read Cost

During Phase 5 (checkpoint loading), the pipeline reads only the
`propositions` column from each existing shard (not the full table):

```python
table = pq.read_table(file_path, columns=["propositions"])
```

This column-pruning means checkpoint loading reads only ~30–40 % of each
shard's bytes, significantly reducing startup time for large resumed runs.

---

## Monitoring

### Watching Memory During a Long Run

**Windows (Task Manager)**:
1. Open Task Manager → Performance → Memory.
2. Monitor "In use" value.  If it exceeds 80 % of physical RAM, the pipeline
   may begin swapping.

**Windows (PowerShell)**:

```powershell
# Poll memory usage of the Python process every 10 seconds
while ($true) {
    $proc = Get-Process python -ErrorAction SilentlyContinue
    if ($proc) {
        $mb = [math]::Round($proc.WorkingSet64 / 1MB, 1)
        Write-Host "$(Get-Date -Format HH:mm:ss)  Python RSS: ${mb} MB"
    }
    Start-Sleep -Seconds 10
}
```

### Watching Disk Usage

```powershell
# Monitor generated data directory size every 60 seconds
while ($true) {
    $size = (Get-ChildItem -Recurse "data\generated" -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
    $mb = [math]::Round($size / 1MB, 2)
    Write-Host "$(Get-Date -Format HH:mm:ss)  Generated data: ${mb} MB"
    Start-Sleep -Seconds 60
}
```

### Interpreting Pipeline Logs

The pipeline emits structured log lines to stdout:

```
Loading AGQA Scene Graphs...
Loading checkpoints...
Loaded 0 previous examples into deduplication cache.
Normalizing Scene Graphs for TRAIN/VAL...
100%|██████████| 9848/9848 [01:42<00:00, 96.3it/s]
Generating TRAIN/VAL examples...
[  1.0%] Generated 8000/800000 (412.3 records/sec) - 19.4s elapsed
Flushed 50000 examples to data/generated/train/part-000000.parquet
[  7.3%] Generated 58000/800000 (398.1 records/sec) - 145.7s elapsed
```

Key indicators:

| Log Line | Healthy Value | Warning Sign |
|---|---|---|
| `records/sec` | 200–800 | < 50 suggests heavy swapping or I/O blocking |
| `Flushed N examples` message | Appears every ~50 000 examples | Long gap (> 5 min) may indicate a stalled flush |
| `Loaded X previous examples` | Equal to previously written count | Mismatch may indicate corrupted shards |

> [!WARNING]
> If the pipeline crashes mid-flush (`Failed to flush N examples to ...`), the
> partial Parquet file may be corrupt.  Delete the incomplete file before
> resuming so the checkpoint loader does not attempt to read it.
