# UniProp Configuration Reference

> [!NOTE]
> This document is the authoritative reference for all configuration parameters in the UniProp pipeline. The **primary pipeline** (`main_pipeline.py`) is configured entirely in code — there is no external YAML file for it. The **CLI pipeline** (`src/cli/main.py`) accepts a YAML configuration file as input and is documented separately in [§6](#6-cli-pipeline-configuration-srcclimaINpy)–[§7](#7-yaml-config-keys-cli-pipeline-only).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Constants](#2-core-constants)
3. [Train / Val Split](#3-train--val-split)
4. [Target Counts](#4-target-counts)
5. [Generator Registration](#5-generator-registration)
6. [CLI Pipeline Configuration](#6-cli-pipeline-configuration-srcclimaINpy)
7. [YAML Config Keys](#7-yaml-config-keys-cli-pipeline-only)
8. [PYTHONHASHSEED](#8-pythonhashseed)
9. [Modifying Configuration](#9-modifying-configuration)

---

## 1. Overview

UniProp has **two separate configuration surfaces**:

| Surface | File | Mechanism | When to use |
|---|---|---|---|
| **Primary pipeline** | [`main_pipeline.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/main_pipeline.py) | Python constants at module level | Full-scale AGQA dataset generation |
| **CLI pipeline** | [`src/cli/main.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/cli/main.py) | YAML file + CLI flags | Dataset-adapter experiments; smoke tests |

The primary pipeline is the canonical production path. The CLI pipeline is an experimental adapter layer that wraps the same underlying machinery but exposes configuration through YAML and command-line overrides.

> [!IMPORTANT]
> All constants in `main_pipeline.py` are at **module level** and are resolved at import time. There is no dynamic config loading, no environment-variable injection (except for `PYTHONHASHSEED`), and no CLI argument parser. Changes require editing the source file directly and re-running.

---

## 2. Core Constants

These constants are defined at the top of [`main_pipeline.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/main_pipeline.py#L92-L111) and govern global pipeline behaviour.

### `SEED`

| Property | Value |
|---|---|
| **Type** | `int` |
| **Default** | `42` |
| **Scope** | Global; used by all random operations in `main_pipeline.py` |

The master reproducibility seed. It is passed to two distinct RNG surfaces:

1. **`random.seed(SEED)`** — seeds the global Python `random` module state. This affects any code that calls `random.*` functions directly (not through the `rng` instance).
2. **`random.Random(SEED)`** — seeds the `rng` instance used for all shuffle operations inside `main()` (train/val split) and all stochastic decisions inside generators when called from `process_split()`.

Changing `SEED` will produce a different train/val split partition, a different video ordering during SG normalisation, and a different sequence of generated examples — a completely different (but still fully reproducible) dataset.

```python
# In main_pipeline.py — line 97–103
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
```

---

### `SHARD_SIZE`

| Property | Value |
|---|---|
| **Type** | `int` |
| **Default** | `50_000` |
| **Scope** | Per-split in-memory buffer threshold |

The maximum number of `PropositionGroup` rows accumulated in memory for a single split before the buffer is flushed to disk as a Parquet shard. When the buffer length reaches `SHARD_SIZE`, `flush_buffer(split, force=False)` writes a `part-NNNNNN.parquet` file under `data/generated/<split>/`.

**Tuning guidance:**

| `SHARD_SIZE` | Effect |
|---|---|
| Smaller (e.g. 5 000) | More shards, more I/O operations, smaller peak memory footprint |
| Larger (e.g. 200 000) | Fewer shards, less I/O overhead, larger peak memory footprint |
| Very large (> available RAM) | Risk of OOM on machines with limited memory; use smaller values |

For the development `target_count` of 10 000, the default 50 000 means no automatic mid-run flush occurs — only the final `force=True` flush at teardown writes shards. For production runs (1 000 000 examples), approximately 16 shards will be written per split.

```python
# In main_pipeline.py — line 111
SHARD_SIZE = 50000
```

---

### `DATA_ROOT`

| Property | Value |
|---|---|
| **Type** | `str` |
| **Default** | `os.path.dirname(os.path.abspath(__file__))` |
| **Scope** | All file-path resolution in `main_pipeline.py` |

The absolute path to the directory containing `main_pipeline.py`. Every asset path in the pipeline is constructed relative to `DATA_ROOT`, making the pipeline **portable** — it can be moved to any directory without changing any paths, as long as the standard directory layout is preserved.

Asset paths derived from `DATA_ROOT`:

| Asset | Resolved path |
|---|---|
| Train STSG pickle | `<DATA_ROOT>/AGQA_scene_graphs/AGQA_train_stsgs.pkl` |
| Test STSG pickle | `<DATA_ROOT>/AGQA_scene_graphs/AGQA_test_stsgs.pkl` |
| Train QA file | `<DATA_ROOT>/AGQA_balanced/AGQA_balanced/train_balanced.txt` |
| Test QA file | `<DATA_ROOT>/AGQA_balanced/AGQA_balanced/test_balanced.txt` |
| Output shards | `<DATA_ROOT>/data/generated/<split>/part-NNNNNN.parquet` |

`DATA_ROOT` is computed at import time and **cannot be overridden without editing the source**. There is no environment variable that changes it.

```python
# In main_pipeline.py — line 107
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
```

---

## 3. Train / Val Split

The 90/10 train/validation split is derived **deterministically** inside `main()`:

```python
rng = random.Random(SEED)          # Seeded RNG instance
train_vids = list(train_sgs_raw.keys())
rng.shuffle(train_vids)             # Reproducible shuffle
num_val        = int(len(train_vids) * 0.1)
val_vids       = set(train_vids[:num_val])       # First 10% → validation
train_vids_set = set(train_vids[num_val:])       # Remaining 90% → training
```

**Key properties:**

- The shuffle uses `random.Random(SEED)`, an **isolated RNG instance** that is not affected by prior calls to the global `random` module. This guarantees the split is identical regardless of what other random operations have occurred.
- The split is fixed for a given `SEED` and a given set of video IDs. Adding new videos to the training pickle would change the split because the list ordering changes.
- Validation videos are a strict subset of the training STSG pickle; they share the same scene-graph pool but are routed to `data/generated/val/` instead of `data/generated/train/`.
- Test videos come from a completely separate STSG pickle and are never mixed with train/val videos (enforced by an `assert` in `main()`).

To change the split ratio, edit the coefficient `0.1` in the `num_val` assignment.

---

## 4. Target Counts

`process_split()` accepts a `target_count` argument that specifies the total `generated_count` value at which the inner generation loop exits early.

```python
# Current development values — edit these for production
process_split(train_qs_path, train_sgs_raw, False, 8_000)    # train + val
process_split(test_qs_path,  test_sgs_raw,  True,  2_000)    # test
```

| Mode | Train/Val `target_count` | Test `target_count` | Total examples |
|---|---|---|---|
| **Smoke test** | `100` | `25` | `~125` |
| **Development (current)** | `8,000` | `2,000` | `~10,000` |
| **Production** | `800,000` | `200,000` | `~1,000,000` |

> [!IMPORTANT]
> `target_count` is a **global** threshold that accounts for checkpoint examples. If a previous run already wrote 5 000 examples and `target_count=8000`, the pipeline will generate only 3 000 new examples before stopping. This is by design — it allows interrupted runs to be resumed to the same total target without generating excess data.

> [!TIP]
> For a production run, also increase `SHARD_SIZE` to `100_000` or higher to reduce the number of Parquet files written and to lower I/O overhead per example.

---

## 5. Generator Registration

Generators are registered in [`main()`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/main_pipeline.py#L389-L398) as a list of `(name, instance)` tuples:

```python
generators = [
    ("orig_qa",       OriginalQAGenerator()),
    ("obj_exist",     ObjectExistenceGenerator()),
    ("rel_ver",       RelationVerificationGenerator()),
    ("rel_set",       RelationSetGenerator()),
    ("grounding",     GroundingGenerator()),
    ("temporal",      ActionTemporalGenerator()),
    ("compositional", CompositionalGenerator()),
    ("attributes",    AttributeGenerator()),
]
```

**How round-robin works:**  
A shared `gen_idx` counter is incremented modulo `len(generators)` after each attempt. For each question, up to `len(generators)` generators are tried in sequence from the current `gen_idx`. This means:

- Every generator is tried at most once per question.
- The cycling is **continuous** across all questions — it is not reset between questions.
- Generators that yield nothing for a given video (guard conditions) consume one round-robin slot but produce no output.

**To add a new generator:**

1. Implement the `GeneratorBase.generate()` interface.
2. Import the class at the top of `main_pipeline.py`.
3. Append a `(name, MyNewGenerator())` tuple to the `generators` list.
4. The pipeline picks it up automatically on the next run.

**To disable a generator:**  
Remove (or comment out) its tuple from the list. The round-robin index arithmetic adjusts automatically via `% len(generators)`.

---

## 6. CLI Pipeline Configuration (`src/cli/main.py`)

The CLI pipeline is an alternative entry point that wraps the dataset adapter layer (not the primary STSG pipeline). It is invoked with:

```bash
python -m src.cli.main \
    --config config/datasets/agqa_balanced.yaml \
    --split  train \
    --max-records 1000
```

### CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `config/datasets/agqa_balanced.yaml` | Path to the YAML configuration file. Resolved relative to the current working directory. |
| `--split` | `str` | `"train"` | Dataset split to process. Must match a top-level key under `splits:` in the config file, or be ignored if the file has no `splits:` block. |
| `--max-records` | `int` | `None` (no limit) | Hard cap on source records read from the adapter. Useful for smoke-testing without processing the full dataset. Passed directly to `run_dataset_pipeline()`. |

The CLI does not perform any generation itself — it delegates to `run_dataset_pipeline(config, split, max_records)` from [`src/pipeline/dataset_pipeline.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/pipeline/dataset_pipeline.py) and prints a one-line summary on completion:

```
Done! Success: 847, Rejected: 12
```

---

## 7. YAML Config Keys (CLI Pipeline Only)

The YAML file at `config/datasets/agqa_balanced.yaml` configures the CLI pipeline. The primary pipeline (`main_pipeline.py`) **does not read this file**.

### Full annotated example

```yaml
# Which dataset adapter to use.
# Must match a key registered in src/datasets/registry.py.
dataset: "agqa_balanced"

# Source data paths, keyed by split name.
# Each split can override source paths independently.
splits:
  train:
    source:
      qa_path: "data/dataset/agqa_balanced/train_balanced.txt"
      scenegraph_path: "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl"
  test:
    source:
      qa_path: "data/dataset/agqa_balanced/test_balanced.txt"
      scenegraph_path: "data/dataset/agqa_scene_graphs/AGQA_test_stsgs.pkl"

# Output configuration.
output:
  root_dir: "data/generated/agqa_balanced"   # Base output directory
  shard_size: 10000                           # Rows per Parquet shard
  format: "parquet"                           # Output format (only "parquet" is supported)

# Record selection strategy.
selection:
  mode: "all"           # "all" — process all records; "sample" — random subset
  max_records: null     # null = no limit; integer = hard cap

# Generation hyperparameters.
generation:
  seed: 42              # RNG seed for all stochastic generation decisions
  max_k: 40             # Maximum number of propositions per group (single_choice / multi_label)
```

### Key reference

| Key path | Type | Default | Description |
|---|---|---|---|
| `dataset` | `str` | — | Adapter name. Must be registered in `src/datasets/registry.py`. |
| `splits.<name>.source.qa_path` | `str` | — | Path to the balanced QA text file for this split. |
| `splits.<name>.source.scenegraph_path` | `str` | — | Path to the STSG pickle file for this split. |
| `output.root_dir` | `str` | — | Base directory for Parquet shard output. Created if it does not exist. |
| `output.shard_size` | `int` | `10000` | Rows per Parquet shard for the CLI pipeline (independent of `SHARD_SIZE` in `main_pipeline.py`). |
| `output.format` | `str` | `"parquet"` | Output file format. Only `"parquet"` is currently supported. |
| `selection.mode` | `str` | `"all"` | `"all"` processes all source records; `"sample"` selects a random subset. |
| `selection.target_records` | `int \| null` | `null` | Target number of records to select when `mode = "sample"`. |
| `generation.seed` | `int` | `42` | RNG seed for the CLI pipeline's generation decisions. Separate from the `SEED` constant in `main_pipeline.py`. |
| `generation.max_k` | `int` | `40` | Upper bound on K (proposition count) for variable-size task types. |

> [!NOTE]
> The CLI pipeline's `generation.seed` key is **independent** of the `SEED` constant in `main_pipeline.py`. Setting one does not affect the other. Ensure both are set consistently if you want full cross-pipeline reproducibility.

---

## 8. PYTHONHASHSEED

Python randomises the seed for its built-in `hash()` function at interpreter startup by default (since Python 3.3). Because the deduplication gate relies on `hash()` to fingerprint proposition groups, **the same set of propositions will produce different hashes across interpreter restarts unless `PYTHONHASHSEED` is fixed**.

`main_pipeline.py` calls:

```python
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
```

`setdefault` only sets the variable if it is not already present in the environment. However, `PYTHONHASHSEED` must be set **before the Python interpreter starts** — setting it inside a running Python process has no effect on that process's hash seed (the seed is read once at startup).

**Correct approach:**

```bash
# Linux / macOS (Bash)
PYTHONHASHSEED=42 python main_pipeline.py

# Windows PowerShell
$env:PYTHONHASHSEED = "42"
python main_pipeline.py

# Windows Command Prompt
set PYTHONHASHSEED=42
python main_pipeline.py
```

**Impact of not setting PYTHONHASHSEED:**

| Scenario | Consequence |
|---|---|
| Single run, no checkpoints | No impact — deduplication is consistent within a single process |
| Resume after interruption | Checkpoint hashes loaded from disk may not match re-computed hashes for the same propositions → **false-negative deduplication** (already-written examples may be regenerated) |
| Comparing runs across machines | Different hash values → different deduplication behaviour |

> [!CAUTION]
> On Windows, PowerShell sessions do not persist environment variables across windows. Set `$env:PYTHONHASHSEED` in the same shell session where you run the pipeline, or add it to your PowerShell profile / system environment variables.

---

## 9. Modifying Configuration

### Smoke Test (< 1 minute runtime)

Edit only `target_count` in the two `process_split()` calls:

```python
# main_pipeline.py — Phase 6, lines 875–877
process_split(train_qs_path, train_sgs_raw, False, 100)   # ← 100 examples
process_split(test_qs_path,  test_sgs_raw,  True,   25)   # ← 25 examples
```

Everything else can remain at default values.

### Development Run (current defaults)

```python
SEED       = 42          # Keep as-is
SHARD_SIZE = 50_000      # Keep as-is (no mid-run flush at 10K total)

process_split(train_qs_path, train_sgs_raw, False, 8_000)
process_split(test_qs_path,  test_sgs_raw,  True,  2_000)
```

### Production Run (1M examples)

Make the following changes to `main_pipeline.py`:

```python
SEED       = 42           # Keep for reproducibility
SHARD_SIZE = 100_000      # Larger shards → fewer files → lower I/O overhead

process_split(train_qs_path, train_sgs_raw, False, 800_000)
process_split(test_qs_path,  test_sgs_raw,  True,  200_000)
```

And launch with:

```bash
PYTHONHASHSEED=42 python main_pipeline.py
```

> [!TIP]
> For production runs, consider pre-computing and caching normalised scene graphs to avoid repeating `normalize_video_sg()` if the pipeline is restarted. The current checkpoint mechanism only deduplicates at the `PropositionGroup` level; SG normalisation is always repeated from scratch on each run.

### Changing the Train/Val Split Ratio

To hold out 20% as validation instead of 10%:

```python
# main_pipeline.py — Phase 3
num_val = int(len(train_vids) * 0.2)   # ← change 0.1 to 0.2
```

> [!WARNING]
> Changing `SEED`, `SHARD_SIZE`, or the split ratio **after** a partial run invalidates the checkpoint. Delete all existing shards under `data/generated/` before restarting, or accept that the new run will be a continuation of an inconsistently configured dataset.
