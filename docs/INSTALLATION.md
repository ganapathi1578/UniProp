# UniProp — Installation Guide

This guide covers every step required to get a fully working UniProp environment,
from cloning the repository through verifying that the pipeline can run end-to-end.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Step 1 — Clone the Repository](#2-step-1--clone-the-repository)
3. [Step 2 — Create a Virtual Environment](#3-step-2--create-a-virtual-environment)
4. [Step 3 — Install Runtime Dependencies](#4-step-3--install-runtime-dependencies)
5. [Step 4 — Install Dev Dependencies](#5-step-4--install-dev-dependencies)
6. [Step 5 — Download Source Data](#6-step-5--download-source-data)
7. [Step 6 — Verify the Setup](#7-step-6--verify-the-setup)
8. [Troubleshooting](#8-troubleshooting)
9. [Updating](#9-updating)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.11 or 3.12 |
| **OS** | Windows 10, Ubuntu 20.04, macOS 12 | Ubuntu 22.04 / macOS 14 |
| **RAM** | 16 GB | 32 GB or more |
| **Free disk** | 15 GB | 50 GB (for full 1 M-example dataset) |
| **CPU** | 4 cores | 8+ cores |

> [!IMPORTANT]
> The pipeline loads two large pickle files (`AGQA_train_stsgs.pkl` and
> `AGQA_test_stsgs.pkl`) fully into RAM during the scene-graph normalisation
> phase.  With less than 16 GB of RAM the process may be killed by the OS.

---

## 2. Step 1 — Clone the Repository

```bash
git clone https://github.com/ganapathi1578/UniProp.git
cd UniProp
```

If you have SSH keys configured with GitHub:

```bash
git clone git@github.com:ganapathi1578/UniProp.git
cd UniProp
```

---

## 3. Step 2 — Create a Virtual Environment

Choose **one** of the two options below.

### Option A — Conda (recommended)

```bash
conda create -n uniprop python=3.11 -y
conda activate uniprop
```

### Option B — venv (standard library)

```bash
# Create the environment
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate — Windows (cmd.exe)
.\.venv\Scripts\activate.bat
```

> [!TIP]
> Always work inside a virtual environment.  Installing packages globally
> can introduce version conflicts with other projects.

---

## 4. Step 3 — Install Runtime Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` pins the following packages:

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `pyarrow` | ≥ 14.0 | Parquet read/write for sharded dataset I/O |
| `pyyaml` | ≥ 6.0 | Pipeline YAML configuration parsing |
| `tqdm` | ≥ 4.65 | Progress bars during generation |
| `zstandard` | ≥ 0.22 | Zstd compression for Parquet shards |
| `ijson` | ≥ 3.2.3 | Streaming JSON parser for AGQA QA files |

> [!NOTE]
> `ijson` is used in `src/io/streaming_json.py` to stream AGQA balanced-QA
> text files without loading them entirely into memory.  The default
> `ijson` backend on most platforms is the pure-Python one; installing the
> optional `ijson[yajl2_cffi]` extra or the `yajl2` C library will
> significantly increase throughput for very large QA files.

---

## 5. Step 4 — Install Dev Dependencies

Dev dependencies add the test runner (`pytest`), the fast DataFrame library
(`polars`), and the embedded SQL engine (`duckdb`) used to inspect generated
Parquet shards.

```bash
pip install -e '.[dev]'
```

> [!NOTE]
> On Windows PowerShell the single-quote syntax may need escaping.
> Use double quotes instead: `pip install -e ".[dev]"`.

This installs the package in **editable mode**, so changes to `src/` are
immediately reflected without reinstalling.

---

## 6. Step 5 — Download Source Data

The AGQA source files are **not included in the repository** (they are too
large and subject to the AGQA dataset licence).  You must download them
separately and place them in the exact directory layout shown below.

### Required files

| File | Location in repo | Description |
|------|-----------------|-------------|
| `train_balanced.txt` | `AGQA_balanced/` | Balanced training QA pairs (AGQA v2) |
| `test_balanced.txt` | `AGQA_balanced/` | Balanced test QA pairs (AGQA v2) |
| `AGQA_train_stsgs.pkl` | `AGQA_scene_graphs/` | Spatio-temporal scene graphs for training videos |
| `AGQA_test_stsgs.pkl` | `AGQA_scene_graphs/` | Spatio-temporal scene graphs for test videos |

### Expected directory layout

```text
UniProp/                          ← repository root
├── AGQA_balanced/
│   ├── train_balanced.txt        ← AGQA v2 balanced train QA file
│   └── test_balanced.txt         ← AGQA v2 balanced test QA file
├── AGQA_scene_graphs/
│   ├── AGQA_train_stsgs.pkl      ← Action Genome train STSGs
│   └── AGQA_test_stsgs.pkl       ← Action Genome test STSGs
├── src/
├── tests/
├── main_pipeline.py
└── requirements.txt
```

### Where to get the data

The AGQA v2 dataset and its scene-graph files are distributed by the AGQA
authors.  Request access at:

- **AGQA balanced QA files**: <https://cs.stanford.edu/people/ranjaykrishna/agqa/>
- **Action Genome STSGs**: <https://prior.allenai.org/projects/charades>

After downloading, verify file sizes match the expected values from the
dataset release notes before running the pipeline.

> [!CAUTION]
> **Never commit the raw data files to git.**  They are listed in `.gitignore`
> by default.  Accidentally pushing multi-GB pickle files will bloat the
> repository history permanently.

---

## 7. Step 6 — Verify the Setup

### 7.1 Smoke-test the schema import

```bash
python -c "from src.schema import PropositionGroup; print('Schema import: OK')"
```

Expected output:
```
Schema import: OK
```

### 7.2 Run the test suite

```bash
pytest tests/ -v
```

All tests should pass before you run the full pipeline.  A partial output
looks like:

```
tests/test_configuration.py::test_seed_constant        PASSED
tests/test_configuration.py::test_shard_size           PASSED
tests/test_optionizers.py::test_binary_options         PASSED
...
tests/test_validators.py::test_proposition_count_range PASSED
====== N passed in X.Xs ======
```

### 7.3 Run the pipeline (development mode)

The pipeline's `target_count` values in `main_pipeline.py` are set to small
numbers (8 000 train/val + 2 000 test) for fast development iteration:

```bash
python main_pipeline.py
```

On first run the pipeline will:

1. Load and normalise the scene graphs (takes 2–10 minutes depending on RAM).
2. Generate proposition groups with all 8 generators.
3. Write Parquet shards to `data/generated/{train,val,test}/`.
4. Print four Markdown audit reports.

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

You must run commands from the **repository root** (`UniProp/`), not from
inside `src/` or another subdirectory:

```bash
cd /path/to/UniProp
python -c "from src.schema import PropositionGroup; print('OK')"
```

Alternatively, the editable install (`pip install -e '.[dev]'`) adds the
package to `sys.path` automatically.

---

### Python version is wrong

```bash
python --version   # must be 3.10+
```

If the system Python is older, use the full path to your conda/venv Python
or activate the correct environment first.

---

### `pip install` fails with dependency conflicts

Try upgrading `pip` and `setuptools` first:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If conflicts persist, create a **fresh** virtual environment and reinstall
from scratch.

---

### `ijson` backend warning

```
ijson: backend 'yajl2_cffi' not found, falling back to 'python'
```

This warning is harmless but the pure-Python backend is slower.  Install the
optional CFFI backend:

```bash
pip install 'ijson[yajl2_cffi]'
```

Or, on Linux, install the system `yajl` library:

```bash
sudo apt-get install libyajl-dev   # Debian/Ubuntu
pip install ijson
```

---

### `FileNotFoundError` for scene-graph pickles

```
FileNotFoundError: [Errno 2] No such file or directory: '.../AGQA_scene_graphs/AGQA_train_stsgs.pkl'
```

The pipeline resolves paths relative to `main_pipeline.py`.  Ensure your
downloaded files match the directory layout in [Step 5](#6-step-5--download-source-data)
exactly (filenames are case-sensitive on Linux).

---

### Out-of-memory during scene-graph normalisation

Reduce the number of videos processed by uncommenting the limiting line in
`main_pipeline.py`:

```python
# vids_needed = vids_needed[:1500]  # Uncomment to limit for debugging
```

---

## 9. Updating

To pull the latest changes and refresh dependencies:

```bash
git pull
pip install -r requirements.txt
```

If the PyArrow schema in `src/storage/parquet_writer.py` has changed
(indicated by a major-version bump in `DATA_SCHEMA.md`), you must
**delete any existing Parquet shards** before re-running the pipeline,
as old shards will be incompatible:

```bash
rm -rf data/generated/
python main_pipeline.py
```

> [!WARNING]
> Deleting `data/generated/` is irreversible.  Back up your data first if
> the generation run took significant time.
