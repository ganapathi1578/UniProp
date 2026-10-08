# UniProp — Data Layout Reference

> [!IMPORTANT]
> This document describes the **complete on-disk layout** of the UniProp project. Paths marked **NOT in git** must be obtained separately or generated locally. Never commit source data or generated output to version control (see [`GIT_POLICY.md`](./GIT_POLICY.md)).

---

## 1. Full Directory Tree

```
UniProp/
│
├── AGQA_balanced/                   # SOURCE INPUT — NOT in git (download separately)
│   └── AGQA_balanced/
│       ├── train_balanced.txt       # Training split QA pairs (flat JSON)
│       └── test_balanced.txt        # Test split QA pairs (flat JSON)
│
├── AGQA_scene_graphs/               # SOURCE INPUT — NOT in git (download separately)
│   ├── AGQA_train_stsgs.pkl         # Spatio-temporal scene graphs for train videos
│   └── AGQA_test_stsgs.pkl          # Spatio-temporal scene graphs for test videos
│
├── data/                            # GENERATED OUTPUT — NOT in git (re-generate locally)
│   └── generated/
│       ├── train/                   # Training split shards
│       │   ├── part-000000.parquet  # Shard 0 — up to 50,000 rows
│       │   ├── part-000001.parquet  # Shard 1 — up to 50,000 rows
│       │   └── ...                  # Additional shards as needed
│       ├── val/                     # Validation split shards
│       │   └── part-000000.parquet  # Typically a single shard for val
│       └── test/                    # Test split shards
│           └── part-000000.parquet  # Typically a single shard for test
│
├── src/                             # SOURCE CODE — in git
│   └── uniprop/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── scene_graph.py
│       ├── qa_loader.py
│       ├── writer.py
│       └── ...
│
├── tests/                           # TEST SUITE — in git
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_scene_graph.py
│   └── ...
│
├── docs/                            # DOCUMENTATION — in git
│   ├── DATA_LAYOUT.md               # ← this file
│   ├── GIT_POLICY.md
│   └── PRODUCTION_RUNBOOK.md
│
├── main_pipeline.py                 # ENTRY POINT — in git
├── requirements.txt                 # PYTHON DEPENDENCIES — in git
├── pyproject.toml                   # BUILD / TOOL CONFIGURATION — in git
├── .gitignore                       # Git exclusion rules — in git
├── CHANGELOG.md                     # Release history — in git
└── CONTRIBUTING.md                  # Contributor guide — in git
```

---

## 2. Source Data Formats

### 2.1 AGQA Balanced QA Files (`*.txt`)

Despite the `.txt` extension, each file contains a **flat JSON object** where every key is a unique Question ID (QID) and the value is a nested object with QA metadata.

**Top-level structure:**

```json
{
  "<QID>": {
    "video_id":  "<video_filename_without_extension>",
    "question":  "<natural language question>",
    "answer":    "<ground-truth answer string>",
    ...
  },
  ...
}
```

**Field descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | `str` | Identifier of the source video clip (matches scene graph keys) |
| `question` | `str` | Natural-language question about the video |
| `answer` | `str` | Ground-truth answer (free-form string, not an index) |

> [!NOTE]
> Additional fields may be present in the raw files (e.g., `"global"`, `"ans_type"`, `"steps"`). The pipeline reads only the fields it needs and ignores unknown keys, so forward-compatibility is maintained.

**Example record:**

```json
{
  "fLXHK7Tq3yw_15-30_000001": {
    "video_id":  "fLXHK7Tq3yw_15-30",
    "question":  "Before they hugged, what were they doing?",
    "answer":    "looking at each other"
  }
}
```

---

### 2.2 AGQA Scene Graph Files (`*.pkl`)

These are **Python pickle files** (protocol 2 or later) containing a nested dictionary keyed by `video_id`.

**Top-level structure (after `pickle.load`):**

```python
{
    "<video_id>": {
        "<frame_id>": {          # Integer or string frame index
            "objects":   [...],  # List of object dicts in this frame
            "relations": [...],  # List of relation triples
            "attributes":[...]   # List of attribute dicts
        },
        "<action_id>": {         # Action segment identifier
            "action":   "<label>",
            "start":    <frame_int>,
            "end":      <frame_int>
        },
        ...
    },
    ...
}
```

**Nested field descriptions:**

| Key | Type | Description |
|-----|------|-------------|
| `frame_id` | `str` or `int` | Zero-based frame index or timestamp key |
| `objects` | `list[dict]` | Per-frame object detections with class and bounding box |
| `relations` | `list[dict]` | Subject–predicate–object triples between detected objects |
| `attributes` | `list[dict]` | Object-level attributes (color, state, etc.) |
| `action_id` | `str` | Temporal segment key (e.g., `"0"`, `"1"`) |
| `action` | `str` | Action class label for that segment |
| `start` / `end` | `int` | Inclusive frame range of the action |

> [!NOTE]
> The same top-level dict key space contains both frame entries and action entries. The pipeline distinguishes them by checking for the presence of the `"objects"` key (frame) vs. the `"action"` key (action segment).

**Loading example:**

```python
import pickle

with open("AGQA_scene_graphs/AGQA_train_stsgs.pkl", "rb") as f:
    scene_graphs = pickle.load(f)

# Access a specific video's frame data
video_data = scene_graphs["fLXHK7Tq3yw_15-30"]
```

---

## 3. Generated Output Format

### 3.1 Parquet Shards

All generated data is written as **Apache Parquet** files using the Snappy compression codec. Each split is stored in its own subdirectory under `data/generated/`.

**Naming scheme:** `part-NNNNNN.parquet`

- `NNNNNN` is a zero-padded six-digit integer (e.g., `000000`, `000001`, …).
- Indices are assigned sequentially and never reused within a run.
- Each shard holds **up to 50,000 rows** (configurable via `SHARD_SIZE` in the pipeline).

**Column schema:**

| Column | Dtype | Description |
|--------|-------|-------------|
| `qid` | `utf8` | Unique Question ID from AGQA |
| `video_id` | `utf8` | Source video identifier |
| `question` | `utf8` | Natural-language question |
| `answer` | `utf8` | Ground-truth answer string |
| `split` | `utf8` | Dataset split: `"train"`, `"val"`, or `"test"` |
| `scene_graph_json` | `utf8` | JSON-serialised scene graph for the associated video |
| `num_frames` | `int32` | Number of frames present in the scene graph |
| `num_actions` | `int32` | Number of action segments in the scene graph |

> [!TIP]
> Parquet's columnar storage means that reading only a few columns (e.g., `qid`, `answer`) is very fast — unused columns are never decompressed.

---

## 4. Reading Generated Data

### 4.1 Polars (recommended for speed)

```python
import polars as pl

# Read a single shard
df = pl.read_parquet("data/generated/train/part-000000.parquet")

# Read the entire train split (all shards) via glob
df_train = pl.read_parquet("data/generated/train/*.parquet")

# Lazy scan — defers I/O until collect()
df_lazy = pl.scan_parquet("data/generated/train/*.parquet")
result = df_lazy.filter(pl.col("num_frames") > 10).collect()
```

### 4.2 DuckDB (recommended for SQL queries)

```python
import duckdb

con = duckdb.connect()

# Query across all train shards
con.execute("""
    SELECT split, COUNT(*) AS n_rows
    FROM read_parquet('data/generated/train/*.parquet')
    GROUP BY split
""").fetchdf()

# Join train and test splits
con.execute("""
    SELECT *
    FROM read_parquet('data/generated/**/*.parquet')
    LIMIT 5
""").fetchdf()
```

### 4.3 Pandas

```python
import pandas as pd
import glob

# Read all shards in train split
shards = glob.glob("data/generated/train/*.parquet")
df = pd.concat([pd.read_parquet(p) for p in sorted(shards)], ignore_index=True)
```

### 4.4 PyArrow

```python
import pyarrow.dataset as ds

# Partition-aware dataset read (reads all splits at once)
dataset = ds.dataset("data/generated/", format="parquet")
table = dataset.to_table(columns=["qid", "video_id", "answer"])

# Convert to pandas if needed
df = table.to_pandas()
```

---

## 5. Shard Indexing

The pipeline maintains a `shard_indices` dictionary in memory during execution to track which shard number to write next for each split.

```python
shard_indices: dict[str, int] = {
    "train": 0,
    "val":   0,
    "test":  0,
}
```

**How it works:**

1. When a buffer for split `S` reaches `SHARD_SIZE` rows, the writer flushes it to disk as `part-{shard_indices[S]:06d}.parquet`.
2. After a successful flush, `shard_indices[S]` is incremented by 1.
3. The next shard for that split starts fresh with an empty buffer.
4. At pipeline exit, any remaining rows in each buffer are flushed as a final (potentially partial) shard.

**Resume behavior:**

On resume after a crash, the pipeline scans existing `part-*.parquet` files in each split directory, determines the highest existing shard index, and initialises `shard_indices[S]` to `max_existing + 1`. This prevents overwriting previously written shards.

```python
import re, pathlib

def recover_shard_index(split_dir: str) -> int:
    """Returns the next shard index to use for a given split directory."""
    existing = list(pathlib.Path(split_dir).glob("part-*.parquet"))
    if not existing:
        return 0
    indices = [
        int(re.search(r"part-(\d+)\.parquet", p.name).group(1))
        for p in existing
    ]
    return max(indices) + 1
```

> [!WARNING]
> If a crash occurred mid-write, the last shard file may be partially written and corrupt. Always run the **Corruption Recovery** procedure (see [`PRODUCTION_RUNBOOK.md`](./PRODUCTION_RUNBOOK.md)) before resuming to delete any incomplete shards.
