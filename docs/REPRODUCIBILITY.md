# REPRODUCIBILITY

> **UniProp guarantees byte-for-byte identical output across runs, given the same
> source dataset, Python version, and configuration.**

---

## Table of Contents

1. [Overview](#overview)
2. [Tri-Layer RNG Lock](#tri-layer-rng-lock)
3. [Why Three Layers?](#why-three-layers)
4. [Train / Val Split Determinism](#train--val-split-determinism)
5. [Generator Round-Robin Determinism](#generator-round-robin-determinism)
6. [Checkpoint Safety](#checkpoint-safety)
7. [Verifying Reproducibility](#verifying-reproducibility)
8. [Caveats](#caveats)

---

## Overview

Reproducibility in UniProp means that running the pipeline twice on the same
source data produces **identical Parquet shards**, bit-for-bit.  This guarantee
is essential for:

- **Ablation experiments** — swapping one generator and re-running produces a
  controlled diff rather than a random resample.
- **Checkpoint resume** — a run interrupted at shard 17 and resumed produces
  the same shards 18, 19, … as an uninterrupted run would have.
- **Cross-machine replication** — a colleague running the pipeline on a different
  machine (same OS / Python version) gets the same dataset.
- **Version control** — pinning `SEED=42` in source control ties a specific
  dataset version to a specific codebase commit.

The guarantee is achieved through a **Tri-Layer RNG Lock** that eliminates every
known source of non-determinism in the Python runtime.

---

## Tri-Layer RNG Lock

Three independent mechanisms must all be in place simultaneously.  They are set
up at the very top of [`main_pipeline.py`](../main_pipeline.py), before any
other import or computation:

```python
# ── Layer 1: Hash randomisation lock ─────────────────────────────────────────
import os
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))   # Must be set BEFORE import

# ── Layer 2: Global Python RNG ────────────────────────────────────────────────
import random
random.seed(SEED)

# ── Layer 3: Isolated local RNG ───────────────────────────────────────────────
# Created inside main() so it is passed explicitly to all generation loops.
rng = random.Random(SEED)
```

### Layer 1 — `os.environ['PYTHONHASHSEED'] = '42'`

**What it controls**: Python 3.3+ randomises the hash seed of `str`, `bytes`,
and `datetime` objects by default across interpreter invocations.  This affects
the iteration order of `dict`, `set`, and any data structure that uses hashing
internally.

**Why it matters**: Scene graphs are loaded as dicts keyed by `video_id`.  If
`dict` iteration order differs between runs, the order in which videos are
processed changes, and the round-robin generator assignment shifts — producing
a different sequence of examples even with the same seed.

**Critical constraint**: `PYTHONHASHSEED` **must be set before the Python process
starts**, not inside a running script.  Setting it inside `main.py` via
`os.environ["PYTHONHASHSEED"] = "42"` works *only if done before any `str.__hash__`
call*, which in practice means it must be the first statement in the module.
Alternatively — and most reliably — set it in the shell before invoking Python:

```bash
# Recommended: set in shell to guarantee pre-process application
PYTHONHASHSEED=42 python main_pipeline.py

# Also acceptable (set via os.environ.setdefault before any imports)
python main_pipeline.py   # PYTHONHASHSEED set at module top-level
```

### Layer 2 — `random.seed(42)`

**What it controls**: Python's global `random` module state, which is shared by
any code that calls `random.choice(...)`, `random.shuffle(...)`, etc. without
using an explicit `Random` instance.

**Why it matters**: Third-party libraries and utility functions may consume the
global RNG without the caller's knowledge.  Seeding the global RNG ensures these
calls are deterministic.

**Limitation**: The global RNG is a *shared* resource — any code that calls
`random.random()` advances it, potentially de-synchronising the state seen by
later calls.  This is why Layer 3 is also required.

### Layer 3 — `rng = random.Random(SEED)`

**What it controls**: An *isolated* `random.Random` instance that is independent
of the global RNG state.  This instance is created once in `main()` and passed
**explicitly** to every function that needs randomness:

```python
rng = random.Random(SEED)

# Passed explicitly to all callers:
rng.shuffle(train_vids)                         # Phase 3: train/val split
rng.shuffle(vids_needed)                        # Phase 5: SG normalisation order
for group in gen_obj.generate(..., rng, ...):   # All generators
    ...
```

**Why it matters**: The isolated RNG decouples the generation loop from any
side-effects in third-party code, logging libraries, or other modules that might
call the global `random` functions.  Even if the global RNG state is perturbed,
the generation sequence driven by `rng` remains identical.

**Critical design rule**: All `rng.randint(0, 2**32-1)` calls inside generators
(used to produce `generation_seed` values) must go through the *same* `rng`
instance that was passed in.  Creating a new `random.Random()` inside a generator
would break the deterministic sequence.

---

## Why Three Layers?

Each layer guards against a distinct failure mode:

| Layer | Failure Mode Guarded Against | Without This Layer… |
|---|---|---|
| **Layer 1** (`PYTHONHASHSEED`) | Hash randomisation changes `dict`/`set` iteration order | Videos processed in a different order → different round-robin assignments → different examples |
| **Layer 2** (`random.seed`) | Third-party code consuming global RNG silently | Global RNG drifts unpredictably → `random.choice` calls in utility code return different values |
| **Layer 3** (`rng = random.Random(SEED)`) | Inter-generator desync from shared global state | One generator's extra `random.random()` call shifts all subsequent generators' states |

All three layers together guarantee that the *full call sequence* of every
random operation is identical across runs.

---

## Train / Val Split Determinism

The 90 / 10 train / val split is computed in Phase 3 of `main()`:

```python
# Phase 3 — deterministic train/val split
train_vids = list(train_sgs_raw.keys())   # 1. Collect all training video IDs

# CRITICAL: sorted() is NOT called here because dict.keys() order is
# already deterministic once PYTHONHASHSEED is locked (Layer 1).
# The order of keys() in CPython 3.7+ is insertion order, which is
# the order in which pickle.load() populated the dict — identical
# across runs given the same .pkl file.

rng.shuffle(train_vids)                   # 2. Seeded shuffle (Layer 3)
num_val       = int(len(train_vids) * 0.1)
val_vids      = set(train_vids[:num_val]) # 3. First 10% → validation
train_vids_set = set(train_vids[num_val:]) # Remaining 90% → training
```

**Why this is deterministic**:

1. `train_sgs_raw.keys()` returns keys in insertion order (CPython 3.7+), which
   is the order `pickle.load` populated the dict.  Given the same `.pkl` file,
   this order is always the same.
2. `rng.shuffle(train_vids)` applies the seeded shuffle.  Because `rng` is
   seeded with `SEED=42` and has been advanced by exactly the same number of
   calls prior to this point (zero — it is the first `rng` call), the shuffle
   produces the same permutation every time.
3. The 10 % split boundary is computed by integer truncation — no floating-point
   rounding differences can affect it.

> [!IMPORTANT]
> Never replace `list(train_sgs_raw.keys())` with an `os.listdir()` call or any
> filesystem-ordered enumeration.  Filesystem order is OS-dependent and not
> guaranteed to be stable across machines.

---

## Generator Round-Robin Determinism

The eight generators are applied in a **fixed-order, continuous round-robin**:

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

gen_idx = 0   # Shared counter, not reset per question

for qid, qdata in stream_questions(qs_path):
    for _ in range(len(generators)):
        gen_name, gen_obj = generators[gen_idx]
        gen_idx = (gen_idx + 1) % len(generators)   # Always advance
        ...
```

**Why this is deterministic**:

- `gen_idx` starts at 0 and advances monotonically modulo 8.  Its value at any
  point in the loop is fully determined by how many generator attempts have been
  made — which is itself determined by how many questions have been processed and
  how many candidates each generator yielded.
- `stream_questions` is a lazy file reader that yields `(qid, qdata)` pairs in
  the order they appear in the balanced QA text file.  File content is stable.
- The `rng` instance passed to `gen_obj.generate(...)` is the same `rng` that
  drives the shuffle and split — so generator-internal random draws are part of
  the same deterministic sequence.

---

## Checkpoint Safety

The pipeline supports **safe resume** from checkpoints without re-generating
examples already written to disk.

### How Checkpoints Work (Phase 5)

At startup, the pipeline scans all existing Parquet shards:

```python
seen_hashes = set()

for split in ["train", "val", "test"]:
    split_dir = os.path.join(DATA_ROOT, "data", "generated", split)
    if os.path.exists(split_dir):
        for file in sorted(os.listdir(split_dir)):          # sorted → stable order
            if file.startswith("part-") and file.endswith(".parquet"):
                idx = int(file.replace("part-", "").replace(".parquet", ""))
                shard_indices[split] = max(shard_indices[split], idx + 1)

                table = pq.read_table(file_path, columns=["propositions"])
                for row in table["propositions"].to_pylist():
                    texts = [p["text"] for p in row]
                    seen_hashes.add(hash("".join(sorted(texts))))   # sorted → stable
```

### Why This Is Deterministic

1. **`sorted(os.listdir(split_dir))`** — Shard files are named `part-000000.parquet`,
   `part-000001.parquet`, etc.  Sorting by filename ensures they are loaded in
   numeric order, regardless of filesystem inode order.
2. **`sorted(texts)`** — Proposition texts within a group are sorted before
   hashing.  This ensures the dedup fingerprint is independent of the order in
   which propositions were appended to the list inside each generator.
3. **`shard_indices[split]`** — The next shard index is derived from the maximum
   existing index + 1.  New shards always receive filenames that are strictly
   greater than any existing shard, preventing overwrite collisions.

### Resumed Run Behaviour

When the pipeline resumes after a crash:

- All hashes from existing shards are loaded into `seen_hashes`.
- `generated_count` and `flushed_count` are pre-populated with the count of
  already-written examples.
- The generation loop immediately skips any candidate whose hash is already in
  `seen_hashes` — without calling the validator.
- New shards are written with indices that continue from where the previous run
  left off.

The net result: the resumed run produces **exactly the same examples** in the
same order as an uninterrupted run would have produced, minus the examples
already on disk.

> [!WARNING]
> Do **not** delete individual rows from a Parquet shard and expect the resumed
> run to regenerate only those rows.  The dedup gate operates at the group level;
> a partially-populated shard will prevent regeneration of the groups still
> present in it.

---

## Verifying Reproducibility

Follow these steps to verify that two runs produce identical output:

### Step 1 — Run a small generation

```bash
# Set a small target count for fast verification
PYTHONHASHSEED=42 python main_pipeline.py
# Let it run until at least one full shard is written (50 000 examples)
# then Ctrl-C
```

### Step 2 — Snapshot the output

```bash
# Hash every Parquet shard to capture the byte fingerprint
Get-FileHash data\generated\train\part-000000.parquet -Algorithm SHA256
# Record the SHA256 hash
```

Or using Python:

```python
import hashlib, pathlib

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

for p in sorted(pathlib.Path("data/generated").rglob("*.parquet")):
    print(p, sha256_of_file(p))
```

### Step 3 — Delete the generated data

```bash
Remove-Item -Recurse -Force data\generated\
```

### Step 4 — Run again with identical configuration

```bash
PYTHONHASHSEED=42 python main_pipeline.py
```

### Step 5 — Compare hashes

```python
# Re-run the sha256_of_file loop and compare against Step 2 output.
# Every shard should produce the exact same SHA256 hash.
```

If any hash differs, see [Caveats](#caveats) for known sources of
non-determinism.

---

## Caveats

The following situations can break the byte-for-byte reproducibility guarantee:

### Python Version Differences

| Component | Risk |
|---|---|
| `random.Random` algorithm | The Mersenne Twister implementation is stable across CPython versions, but the *seeding algorithm* changed between Python 3.1 and 3.2. Always use the **same minor Python version** (e.g. 3.10.x). |
| `hash()` for non-string types | `PYTHONHASHSEED` only fixes `str`, `bytes`, and `datetime` hashes. Integer hashes are always deterministic. |
| `dict` iteration order | Guaranteed insertion-order since CPython 3.7. Do not use Python < 3.7. |

### OS Differences in Float Representation

The `start_secs` field in action annotations is a `float` parsed from the raw
AGQA pickle.  On virtually all modern hardware (x86-64, ARM64) IEEE 754 double
precision is used uniformly, so float values are identical across platforms.
The risk is theoretical but worth noting if cross-architecture reproducibility
is required.

### File System Order

`os.listdir()` returns entries in filesystem-dependent order.  The pipeline
wraps all such calls in `sorted()`.  If you extend the pipeline with new file
discovery code, **always sort the result** before iterating.

### PyArrow / Parquet Encoder Version

Parquet encoding details (dictionary page boundaries, compression block
boundaries) can vary across `pyarrow` versions.  The schema and data values will
be identical, but the raw bytes may differ.  For byte-level reproducibility, pin
`pyarrow` to a specific version in `requirements.txt`.

### SHARD_SIZE Changes

Changing `SHARD_SIZE` changes where each shard ends and the next begins.  The
**content** of the full dataset remains identical, but individual shard files
will have different sizes and content.  For full byte reproducibility, keep
`SHARD_SIZE=50000` (the default).

> [!TIP]
> Add `pyarrow==X.Y.Z` and `python==3.10.x` to your environment lockfile
> (`requirements.txt` or `conda-lock.yml`) to guarantee full byte-level
> reproducibility across machines and CI systems.
