# UniProp — Troubleshooting Guide

This guide covers the most common errors and unexpected behaviours encountered
when setting up, running, or extending the UniProp dataset generation pipeline.
Each entry follows the **Symptom → Cause → Fix** pattern so you can jump
directly to the relevant section.

---

## Table of Contents

1. [FileNotFoundError: train\_balanced.txt](#1-filenotfounderror-train_balanced_txt)
2. [FileNotFoundError: AGQA\_train\_stsgs.pkl](#2-filenotfounderror-agqa_train_stsgs_pkl)
3. [ModuleNotFoundError: ijson](#3-modulenotfounderror-ijson)
4. [ModuleNotFoundError: pyarrow](#4-modulenotfounderror-pyarrow)
5. [ArrowInvalid: Parquet magic bytes not found](#5-arrowinvalid-parquet-magic-bytes-not-found)
6. [AssertionError: Missing video\_id](#6-assertionerror-missing-video_id)
7. [AssertionError: Test videos found in Train split](#7-assertionerror-test-videos-found-in-train-split)
8. [MemoryError during pickle load](#8-memoryerror-during-pickle-load)
9. [Generation appears stuck / no progress bar movement](#9-generation-appears-stuck--no-progress-bar-movement)
10. [All examples being rejected by the validator](#10-all-examples-being-rejected-by-the-validator)
11. [PYTHONHASHSEED warning at startup](#11-pythonhashseed-warning-at-startup)
12. [Duplicate proposition texts in output](#12-duplicate-proposition-texts-in-output)
13. [Unresolved object ID appearing in text (e.g. `'o12'`)](#13-unresolved-object-id-appearing-in-text-eg-o12)
14. [ImportError from src when running a script](#14-importerror-from-src-when-running-a-script)
15. [Parquet column type mismatch on read](#15-parquet-column-type-mismatch-on-read)
16. [KeyError in OBJECTS dict](#16-keyerror-in-objects-dict)
17. [RNG not producing variety across examples](#17-rng-not-producing-variety-across-examples)

---

## 1. FileNotFoundError: train\_balanced.txt

**Symptom**

```
FileNotFoundError: [Errno 2] No such file or directory:
  'AGQA_balanced/train_balanced.txt'
```

**Cause**

The AGQA balanced-split text files were not downloaded, or were placed in the
wrong directory relative to the UniProp project root.  The pipeline expects the
file at `<project_root>/AGQA_balanced/train_balanced.txt` (and similarly
`test_balanced.txt`).

**Fix**

1. Download the AGQA balanced-split files from the official AGQA release.
2. Place them under `<project_root>/AGQA_balanced/`:
   ```
   UniProp/
   └── AGQA_balanced/
       ├── train_balanced.txt
       └── test_balanced.txt
   ```
3. Confirm the path with:
   ```powershell
   Test-Path .\AGQA_balanced\train_balanced.txt   # should return True
   ```
4. If you stored the files elsewhere, update `DATA_ROOT` in
   `src/config/paths.py` (or your `.env`) to point to the correct directory.

---

## 2. FileNotFoundError: AGQA\_train\_stsgs.pkl

**Symptom**

```
FileNotFoundError: [Errno 2] No such file or directory:
  'AGQA_scene_graphs/AGQA_train_stsgs.pkl'
```

**Cause**

The AGQA scene-graph pickle files were not downloaded or were extracted to the
wrong location.  The pipeline loads spatio-temporal scene graphs from
`<project_root>/AGQA_scene_graphs/` at startup and will abort immediately if
either the train or test pickle is missing.

**Fix**

1. Download `AGQA_train_stsgs.pkl` and `AGQA_test_stsgs.pkl` from the AGQA
   dataset release page.
2. Place both files under `<project_root>/AGQA_scene_graphs/`:
   ```
   UniProp/
   └── AGQA_scene_graphs/
       ├── AGQA_train_stsgs.pkl
       └── AGQA_test_stsgs.pkl
   ```
3. Verify file sizes — each pickle is several GB; a truncated download will
   produce a `UnpicklingError` rather than a `FileNotFoundError`.
4. If the files are stored on a mounted network drive or external path, set
   `SCENE_GRAPH_ROOT` in `src/config/paths.py` accordingly.

---

## 3. ModuleNotFoundError: ijson

**Symptom**

```
ModuleNotFoundError: No module named 'ijson'
```

**Cause**

`ijson` is a streaming JSON parser used by the pipeline to read large AGQA JSON
files without loading them entirely into memory.  It is listed in
`requirements.txt` but may be absent in a fresh or partial virtual environment.

**Fix**

```powershell
pip install ijson
```

If you are using a conda environment:

```powershell
conda install -c conda-forge ijson
```

Confirm the install:

```powershell
python -c "import ijson; print(ijson.__version__)"
```

> [!NOTE]
> On Windows, `ijson` may fall back to a pure-Python backend if the `yajl`
> C library is not available.  This is significantly slower but fully correct.
> For large-scale generation runs, consider installing `yajl` via vcpkg or
> using WSL with `apt install libyajl-dev`.

---

## 4. ModuleNotFoundError: pyarrow

**Symptom**

```
ModuleNotFoundError: No module named 'pyarrow'
```

**Cause**

`pyarrow` provides the Parquet read/write backend used by the dataset writer.
It is a non-trivial install not always included in base Python environments.

**Fix**

```powershell
pip install "pyarrow>=12.0"
```

Pin the version to match what CI uses (see `requirements.txt`).  Mismatched
`pyarrow` versions can cause schema evolution issues — see
[Issue 15](#15-parquet-column-type-mismatch-on-read) for details.

Confirm:

```powershell
python -c "import pyarrow; print(pyarrow.__version__)"
```

---

## 5. ArrowInvalid: Parquet magic bytes not found

**Symptom**

```
pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer.
  Either the file is corrupted or this is not a parquet file.
```

**Cause**

An interrupted generation run (SIGKILL, power loss, OOM kill) left an
incomplete `.parquet` shard on disk.  Parquet files are only valid after the
footer is written; a file that was never closed is a 0-byte or partial binary
blob that cannot be read.

**Fix**

1. Identify the corrupted shard:
   ```powershell
   Get-ChildItem .\data\generated\ -Filter *.parquet |
       Where-Object { $_.Length -lt 4096 }
   ```
2. Delete the offending file(s):
   ```powershell
   Remove-Item .\data\generated\<shard_name>.parquet
   ```
3. Re-run the generator.  The checkpointing system reads existing valid shards
   and resumes from where the last complete shard ended — you will not lose
   already-written data.

> [!CAUTION]
> Do **not** delete the entire `data/generated/` directory unless you intend to
> start a completely fresh run.  Deleting valid shards forces redundant
> computation.

---

## 6. AssertionError: Missing video\_id

**Symptom**

```
AssertionError: Missing video_id in generated example from <GeneratorName>
```

or

```
AssertionError: PropositionGroup.media_id must not be None or empty
```

**Cause**

A generator constructed a `PropositionGroup` (or equivalent output object)
without populating `video_id` / `media_id` from the scene-graph entry.  This
is a generator-implementation bug — the field is required by the schema
validator before a group can be written.

**Fix**

1. Read the traceback carefully to identify the generator class name.
2. Open the corresponding file in `src/generators/`.
3. Locate the code that constructs `PropositionGroup` and confirm that
   `video_id` is read from the scene-graph node, e.g.:
   ```python
   group = PropositionGroup(
       video_id=sg_entry["video_id"],   # ← must be present
       ...
   )
   ```
4. If the key is missing from the scene-graph node at runtime, add a defensive
   check with a descriptive error message rather than silently defaulting to
   `None`.
5. Run the generator in isolation (see
   [Issue 10](#10-all-examples-being-rejected-by-the-validator)) to confirm the
   fix before a full run.

---

## 7. AssertionError: Test videos found in Train split

**Symptom**

```
AssertionError: Data contamination detected — N test video_ids present
  in the train split output.
```

**Cause**

The scene-graph pickle files (`AGQA_train_stsgs.pkl` /
`AGQA_test_stsgs.pkl`) or the balanced split text files
(`train_balanced.txt` / `test_balanced.txt`) do not correspond to the same
AGQA version, causing the split boundaries to mismatch.  It can also occur if
both pickles were accidentally passed to the train generator.

**Fix**

1. Verify that the pickle files and the balanced-split text files are from the
   **same AGQA release** (check MD5/SHA-256 hashes against the dataset
   release notes).
2. Confirm `src/config/paths.py` maps `TRAIN_PKL` → `AGQA_train_stsgs.pkl`
   and `TEST_PKL` → `AGQA_test_stsgs.pkl` — not swapped.
3. Re-download the files if hashes do not match.
4. After re-running, validate with:
   ```powershell
   python -m src.validate --check-split-contamination
   ```

---

## 8. MemoryError during pickle load

**Symptom**

```
MemoryError
```
raised during `pickle.load()` of `AGQA_train_stsgs.pkl`.

**Cause**

The AGQA scene-graph pickles hold the full spatio-temporal graph for every
training video in memory simultaneously.  Loading the train pickle requires at
least **12–16 GB of free RAM**.  Systems with less available memory will OOM
during deserialization.

**Fix**

- **Short term**: Close all other applications before running the pipeline to
  maximise available RAM.
- **Medium term**: Enable Windows virtual memory (pagefile) of at least 32 GB
  on an SSD — this prevents hard OOM at the cost of speed.
- **Long term**: The recommended approach is to run generation on a machine
  with ≥ 32 GB physical RAM.  See
  [`RESOURCE_REQUIREMENTS.md`](RESOURCE_REQUIREMENTS.md) for the full hardware
  specification.

> [!IMPORTANT]
> 16 GB RAM is the **minimum** for the train split.  The test split is smaller
> and typically loads in ~4 GB.

---

## 9. Generation appears stuck / no progress bar movement

**Symptom**

The `tqdm` progress bar is displayed but the counter does not advance for
several minutes, or advances extremely slowly (< 1 example/minute).

**Cause**

There are two common root causes:

- **Deduplication hash-set saturation**: The pipeline maintains a
  `seen_hashes` set and retries generation when a duplicate is produced.  If
  `len(seen_hashes)` is already close to `target_count`, the retry rate
  approaches 100 % and effective throughput collapses.
- **Heavy rejection by the validator**: Every generated candidate is being
  rejected and re-attempted (see
  [Issue 10](#10-all-examples-being-rejected-by-the-validator)).

**Diagnosis**

Add a temporary debug print inside the generation loop:

```python
logger.debug(
    "seen_hashes=%d  target=%d  retry_rate=%.1f%%",
    len(seen_hashes), target_count,
    100 * retries / (attempts or 1)
)
```

**Fix**

- If `seen_hashes` is near `target_count`, the generator has produced as many
  unique examples as it is capable of.  Reduce `target_count` or add more
  generator variation (new renderers, paraphrase templates).
- If the validator is the bottleneck, see
  [Issue 10](#10-all-examples-being-rejected-by-the-validator).

---

## 10. All examples being rejected by the validator

**Symptom**

The pipeline runs but writes zero (or near-zero) rows to the output Parquet
files.  Validator rejection counters climb but acceptance counters remain at 0.

**Cause**

A recently introduced change to a generator or renderer is producing outputs
that systematically violate one or more validation rules (e.g., empty
proposition text, missing required field, schema type mismatch).

**Fix**

1. Run a single generator in isolation to capture the raw output before
   validation:
   ```python
   from src.generators.my_generator import MyGenerator
   from src.loaders import load_scene_graphs
   import random

   sg = load_scene_graphs("train")
   entry = next(iter(sg.values()))
   gen = MyGenerator(rng=random.Random(42))
   groups = gen.generate(entry)
   for g in groups:
       print(g)
   ```
2. Inspect the printed `PropositionGroup` objects against the schema in
   [`DATA_SCHEMA.md`](DATA_SCHEMA.md).
3. Enable verbose validator logging:
   ```powershell
   $env:UNIPROP_LOG_LEVEL = "DEBUG"
   python -m src.generate --split train --limit 10
   ```
4. The debug log will name the specific validation rule that rejected each
   candidate, pinpointing the bug.

---

## 11. PYTHONHASHSEED warning at startup

**Symptom**

```
UserWarning: PYTHONHASHSEED is not set. Hash randomisation will make run
  results non-reproducible across processes.
```

**Cause**

Python randomises dictionary and set iteration order across processes unless
`PYTHONHASHSEED` is fixed.  The pipeline warns when this variable is absent
because it can cause non-deterministic shard boundaries or ordering.

**Fix**

Set `PYTHONHASHSEED` in your shell **before** invoking Python:

```powershell
# PowerShell
$env:PYTHONHASHSEED = "42"
python -m src.generate --split train
```

```bash
# bash / WSL
export PYTHONHASHSEED=42
python -m src.generate --split train
```

Add this to your `.env` file (loaded by `python-dotenv`) to make it permanent
for the project:

```
PYTHONHASHSEED=42
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full seed management
policy.

---

## 12. Duplicate proposition texts in output

**Symptom**

Post-generation analysis reveals that a significant fraction of proposition
`text` fields are byte-for-byte identical, despite the deduplication logic.

**Cause**

A renderer has a template set that is too small or too rigid, producing the
same surface string for different scene-graph inputs.  The hash-based
deduplication key may also be computed over a subset of fields that does not
include the full proposition text, allowing textually identical but
structurally distinct examples to slip through.

**Fix**

1. Inspect the suspect renderer directly:
   ```python
   from src.renderers.my_renderer import MyRenderer
   import random

   r = MyRenderer(rng=random.Random(0))
   texts = [r.render({"subject": "person", "object": "ball"}) for _ in range(20)]
   print(set(texts))   # should have high cardinality
   ```
2. If the set size is very small, add more paraphrase templates to the renderer.
3. Verify that the deduplication hash includes the full `text` field:
   ```python
   # In the generation loop
   hash_key = hashlib.md5(group.text.encode()).hexdigest()
   ```
4. Re-run generation after the fix; use `--no-resume` to start fresh if the
   existing checkpoint contains the duplicate-heavy data.

---

## 13. Unresolved object ID appearing in text (e.g. `'o12'`)

**Symptom**

Output proposition text contains raw object-ID tokens such as `'o12'`,
`'r5'`, or `'c3'` rather than human-readable labels like `'ball'` or
`'person'`.

**Cause**

A renderer called `OBJECTS.get(key)` (or equivalent) with a key that is not
present in the lookup dictionary, and then used the returned `None` or the
raw key as the label in the template string without a fallback guard.

**Fix**

1. Identify which renderer is producing the bad text (check the `generator`
   field in the offending row).
2. Open the renderer file and locate every `OBJECTS.get(...)` or
   `OBJECTS[...]` call.
3. Add an explicit fallback and log a warning when the key is missing:
   ```python
   label = OBJECTS.get(obj_id)
   if label is None:
       logger.warning("Unknown object ID '%s' — skipping proposition.", obj_id)
       return None   # caller must handle None → skip this candidate
   ```
4. Alternatively, ensure the object ID is always resolved through
   `src.constants.resolve_object_label()`, which centralises the fallback
   logic.
5. If the ID is genuinely absent from `constants.py`, see
   [Issue 16](#16-keyerror-in-objects-dict).

---

## 14. ImportError from src when running a script

**Symptom**

```
ImportError: attempted relative import with no known parent package
```
or
```
ModuleNotFoundError: No module named 'src'
```

**Cause**

The script was invoked from a subdirectory (e.g., `src/` or `src/generators/`)
rather than from the UniProp project root.  Python only resolves `src.*`
imports when the working directory is the root of the repository (or the root
is explicitly on `sys.path`).

**Fix**

Always run generation and utility scripts from the project root:

```powershell
# Correct — run from UniProp root
cd C:\Users\GANAPATHI\Desktop\NIT\Research\UniProp
python -m src.generate --split train
```

If you must run from another directory, prepend the root to `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "C:\Users\GANAPATHI\Desktop\NIT\Research\UniProp;$env:PYTHONPATH"
python -m src.generate --split train
```

> [!TIP]
> Using `python -m src.generate` (module syntax) is always preferred over
> `python src/generate.py` (path syntax) — it correctly sets `__package__`
> and avoids most relative-import issues.

---

## 15. Parquet column type mismatch on read

**Symptom**

```
pyarrow.lib.ArrowInvalid: Column N named 'X' expected type Y but got Z
```
raised when reading previously generated Parquet shards with the current
codebase.

**Cause**

The Parquet schema written by an older version of the pipeline used a different
Arrow type for one or more columns (e.g., `int32` vs `int64`, `string` vs
`large_string`).  A `pyarrow` version upgrade between runs can also silently
change the default type mapping.

**Fix**

1. Check the `pyarrow` version used during generation vs. the one currently
   installed:
   ```powershell
   python -c "import pyarrow; print(pyarrow.__version__)"
   ```
2. Compare against the version pinned in `requirements.txt`.
3. If the versions differ, either:
   - Downgrade `pyarrow` to match the pinned version, **or**
   - Convert the old shards using the migration script:
     ```powershell
     python -m src.tools.migrate_schema --input data/generated/ --output data/migrated/
     ```
4. After migration, validate the new shards:
   ```powershell
   python -m src.validate --input data/migrated/
   ```
5. Update `requirements.txt` to pin the new version so future runs are
   consistent.

---

## 16. KeyError in OBJECTS dict

**Symptom**

```
KeyError: 'c47'
```
(or any object-class ID) raised inside a generator or renderer when looking up
`OBJECTS['c47']`.

**Cause**

The scene graph contains an object-class ID that is not registered in
`src/constants.py`'s `OBJECTS` dictionary.  This can happen when a new AGQA
version adds object classes that were not present when `constants.py` was last
updated.

**Fix**

1. Note the missing key from the traceback (e.g., `'c47'`).
2. Cross-reference the AGQA object-class legend to find the human-readable
   label for that ID.
3. Add the entry to `OBJECTS` in `src/constants.py`:
   ```python
   OBJECTS = {
       ...
       "c47": "skateboard",   # ← add new entry
       ...
   }
   ```
4. Run the full validation suite to confirm no other IDs are missing:
   ```powershell
   python -m src.tools.audit_constants --split train
   ```
   This script iterates every scene graph and reports any ID not present in
   `constants.py`.
5. Commit the updated `constants.py` and note the AGQA version in the commit
   message.

---

## 17. RNG not producing variety across examples

**Symptom**

Generated examples from a single generator are nearly identical in structure,
even though the scene-graph inputs differ.  The renderer appears to always
choose the same template or the same attribute values.

**Cause**

The `rng` object was not passed through to the renderer or sub-sampler, so
those components are defaulting to Python's global `random` state (which may be
seeded to a fixed value) or always calling `random.Random()` with no seed
(which resets to the same state on each call within a process).

**Fix**

1. Confirm that every generator constructor accepts and stores the `rng`
   parameter:
   ```python
   class MyGenerator:
       def __init__(self, rng: random.Random):
           self.rng = rng   # ← must be stored, not ignored
   ```
2. Confirm that every renderer call passes `self.rng`:
   ```python
   text = self.renderer.render(context, rng=self.rng)  # ← pass through
   ```
3. Confirm that the renderer uses `rng.choice(...)` / `rng.shuffle(...)` etc.,
   not `random.choice(...)` (the module-level global):
   ```python
   # Wrong
   template = random.choice(self.templates)

   # Correct
   template = rng.choice(self.templates)
   ```
4. Write a unit test that generates 50 examples with two different `rng` seeds
   and asserts the text distributions differ:
   ```python
   gen_a = MyGenerator(rng=random.Random(1))
   gen_b = MyGenerator(rng=random.Random(2))
   texts_a = {g.text for g in gen_a.generate(entry)}
   texts_b = {g.text for g in gen_b.generate(entry)}
   assert texts_a != texts_b, "Generator is not using rng"
   ```

---

## Still stuck?

If none of the above resolves your issue:

1. Enable full debug logging:
   ```powershell
   $env:UNIPROP_LOG_LEVEL = "DEBUG"
   python -m src.generate --split train --limit 100 2>&1 | Tee-Object debug.log
   ```
2. Open `debug.log` and search for `ERROR` or `WARNING` lines near the point
   of failure.
3. File an issue in the repository and attach `debug.log` along with the output
   of:
   ```powershell
   python -m src.tools.env_report
   ```
   This prints Python version, all relevant package versions, OS details, and
   available RAM.
