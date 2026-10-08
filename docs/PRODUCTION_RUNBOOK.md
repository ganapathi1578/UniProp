# UniProp — Production Runbook

> [!IMPORTANT]
> Follow every step in order. Do not skip the pre-flight checklist. A production run can take several hours and consume significant disk space. Mistakes mid-run may require a full restart.

---

## 1. Pre-flight Checklist

Verify **all 8 items** before issuing the pipeline command. Check them off in order.

- [ ] **1. Source data present** — Both pickle files and both QA files exist at the expected paths:
  ```
  AGQA_balanced/AGQA_balanced/train_balanced.txt   ✓
  AGQA_balanced/AGQA_balanced/test_balanced.txt    ✓
  AGQA_scene_graphs/AGQA_train_stsgs.pkl           ✓
  AGQA_scene_graphs/AGQA_test_stsgs.pkl            ✓
  ```

- [ ] **2. Source data readable** — Quick sanity check (run from the project root):
  ```powershell
  python -c "import pickle; d=pickle.load(open('AGQA_scene_graphs/AGQA_train_stsgs.pkl','rb')); print(f'Train SGs: {len(d)} videos')"
  python -c "import json; d=json.load(open('AGQA_balanced/AGQA_balanced/train_balanced.txt')); print(f'Train QAs: {len(d)} items')"
  ```

- [ ] **3. Output directory clean or resumed** — Either `data/generated/` does not exist (fresh run), or you have intentionally decided to resume (see §5). If in doubt, delete and start fresh:
  ```powershell
  Remove-Item -Recurse -Force data\generated
  ```

- [ ] **4. Sufficient disk space** — A full 1M-row production run requires approximately **15–25 GB** of free disk space. Check with:
  ```powershell
  Get-PSDrive C | Select-Object Used, Free
  ```

- [ ] **5. Sufficient RAM** — The pipeline loads both pickle files into memory simultaneously. Minimum **16 GB RAM** recommended; 32 GB preferred for production scale.

- [ ] **6. Python environment active and dependencies installed**:
  ```powershell
  python --version          # Must be 3.10+
  pip install -r requirements.txt
  python -m pytest tests/ -q  # Must show 0 failures
  ```

- [ ] **7. No stale lock files or partial shards** — If a previous run crashed, run the Corruption Recovery procedure (§6) before proceeding.

- [ ] **8. Terminal session is persistent** — Use Windows Terminal or a `tmux`/`screen` equivalent. A disconnected session will kill the run. For remote machines, use `Start-Process` or a background job.

---

## 2. Step-by-Step Production Run

### Step 1 — Activate your environment

```powershell
# If using venv:
.\.venv\Scripts\Activate.ps1

# If using conda:
conda activate uniprop
```

### Step 2 — Navigate to the project root

```powershell
Set-Location C:\Users\GANAPATHI\Desktop\NIT\Research\UniProp
```

### Step 3 — Verify the entry point

```powershell
python main_pipeline.py --help
```

Confirm the expected flags are shown (e.g., `--train-limit`, `--val-limit`, `--test-limit`, `--output-dir`).

### Step 4 — Launch the production run

```powershell
python main_pipeline.py `
    --train-limit 800000 `
    --val-limit   800000 `
    --test-limit  200000 `
    --output-dir  data/generated `
    2>&1 | Tee-Object -FilePath run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log
```

> [!NOTE]
> `Tee-Object` writes stdout/stderr to both the console and a timestamped log file simultaneously, enabling post-run analysis without losing real-time visibility.

### Step 5 — Monitor the run (see §4)

### Step 6 — Verify output on completion (see §7)

---

## 3. Target Counts

The following limits define a **standard production run**:

| Split | Target rows | Rationale |
|-------|-------------|-----------|
| `train` | **800,000** | Primary training corpus; maximises coverage |
| `val` | **800,000** | Large validation set for reliable metric estimation |
| `test` | **200,000** | Held-out evaluation; intentionally smaller to avoid test-set leakage |

**Total target:** 1,800,000 rows across all splits.

> [!TIP]
> For a **dry run** or **smoke test**, use `--train-limit 1000 --val-limit 500 --test-limit 500`. This completes in under a minute and verifies end-to-end correctness before committing to a full run.

---

## 4. Monitoring a Live Run

### 4.1 What to Watch in Stdout

The pipeline emits structured progress lines. Key patterns to observe:

```
[TRAIN]  Processed   50000 / 800000  (6.3%)   Rate: 2847 rows/s   ETA: 263s
[VAL]    Processed   50000 / 800000  (6.3%)   Rate: 2901 rows/s   ETA: 258s
[TRAIN]  Flushed shard part-000001.parquet  (50000 rows, 12.4 MB)
```

### 4.2 Expected Throughput

| Hardware | Expected rate |
|----------|---------------|
| Modern laptop (NVMe SSD) | 2,000 – 4,000 rows/s |
| Desktop with fast SSD | 4,000 – 8,000 rows/s |
| HDD-backed storage | 500 – 1,500 rows/s |

At 3,000 rows/s, a 1.8M-row run takes approximately **10 minutes**. Pickle loading at startup adds 30–90 seconds depending on file size and RAM speed.

### 4.3 How to Detect a Stall

A run has stalled if:

- No new progress line has been emitted for **>60 seconds** during active processing.
- The `Rate:` value drops below **100 rows/s** for a sustained period (not just briefly during a shard flush).
- The process shows 0% CPU in Task Manager for >30 seconds.

**What to do if stalled:**

1. Check available RAM (`Get-Process python | Select-Object WorkingSet`). If near system limit, the process is thrashing.
2. Check disk I/O — a slow flush can pause for 10–20 seconds on a cold HDD; this is normal.
3. If genuinely stuck (>5 minutes no output), perform an Emergency Stop (§9) and resume (§5).

---

## 5. Checkpoint / Resume

The pipeline supports crash-safe resume. Shards already written to disk are **never re-written** on resume.

### Procedure

1. **Do not delete any existing shard files.**

2. Run the Corruption Recovery procedure (§6) to remove any partially written shard from the crashed run.

3. Re-issue the **exact same command** as the original run. The pipeline automatically detects existing shards, computes the next shard index, skips already-processed QIDs, and continues from where it left off:

   ```powershell
   python main_pipeline.py `
       --train-limit 800000 `
       --val-limit   800000 `
       --test-limit  200000 `
       --output-dir  data/generated `
       2>&1 | Tee-Object -FilePath resume_$(Get-Date -Format 'yyyyMMdd_HHmmss').log
   ```

4. Verify the startup log confirms resume mode:
   ```
   [INFO]  Resume detected: train=8 existing shards (400000 rows), val=0 shards, test=0 shards
   [INFO]  Resuming from train shard index 8
   ```

> [!WARNING]
> Do **not** change `--train-limit`, `--val-limit`, or `--test-limit` between the original run and the resume. Changing limits on resume will produce an inconsistent dataset.

---

## 6. Corruption Recovery

A crash mid-write leaves the last shard partially written. A partial Parquet file cannot be read and will cause errors in downstream tools.

### Detection

```powershell
# Attempt to read every shard; print any that fail
python -c "
import polars as pl, pathlib, sys
errors = []
for p in sorted(pathlib.Path('data/generated').rglob('*.parquet')):
    try:
        pl.read_parquet(str(p))
    except Exception as e:
        errors.append((str(p), str(e)))
if errors:
    for path, err in errors:
        print(f'CORRUPT: {path}  ->  {err}')
    sys.exit(1)
else:
    print('All shards OK')
"
```

### Recovery

For each corrupted file reported:

```powershell
# Example — replace with actual corrupt path
Remove-Item -Force "data\generated\train\part-000008.parquet"
```

After deleting the corrupt shard(s), the pipeline will re-generate them cleanly on the next resume run.

> [!CAUTION]
> Only delete files confirmed corrupt by the detection script. Deleting a valid shard will cause rows to be lost from the final dataset.

---

## 7. Post-Generation Validation

Run these checks **immediately after** the pipeline reports completion.

### 7.1 Read Generated Parquet and Check Row Counts

```python
import polars as pl

splits = ["train", "val", "test"]
targets = {"train": 800_000, "val": 800_000, "test": 200_000}

for split in splits:
    df = pl.read_parquet(f"data/generated/{split}/*.parquet")
    actual = len(df)
    target = targets[split]
    status = "✓" if actual == target else f"✗ EXPECTED {target}"
    print(f"{split:6s}: {actual:>10,} rows  {status}")
```

**Expected output:**

```
train :    800,000 rows  ✓
val   :    800,000 rows  ✓
test  :    200,000 rows  ✓
```

### 7.2 Verify Schema

```python
df = pl.read_parquet("data/generated/train/part-000000.parquet")
print(df.schema)
# Expected:
# Schema({'qid': String, 'video_id': String, 'question': String,
#         'answer': String, 'split': String,
#         'scene_graph_json': String, 'num_frames': Int32, 'num_actions': Int32})
```

### 7.3 Check for Null Values

```python
for split in ["train", "val", "test"]:
    df = pl.read_parquet(f"data/generated/{split}/*.parquet")
    nulls = df.null_count()
    total_nulls = nulls.to_series().sum()
    print(f"{split}: {total_nulls} null values across all columns")
    # Expected: 0 for all splits
```

### 7.4 Verify Split Column Consistency

```python
for split in ["train", "val", "test"]:
    df = pl.read_parquet(f"data/generated/{split}/*.parquet")
    unique_splits = df["split"].unique().to_list()
    assert unique_splits == [split], f"Unexpected split values in {split}: {unique_splits}"
print("Split column consistency: ✓")
```

### 7.5 Check QID Uniqueness

```python
for split in ["train", "val", "test"]:
    df = pl.read_parquet(f"data/generated/{split}/*.parquet")
    dupes = len(df) - df["qid"].n_unique()
    status = "✓" if dupes == 0 else f"✗ {dupes} duplicate QIDs"
    print(f"{split}: QID uniqueness {status}")
```

---

## 8. Report Files

The pipeline generates the following markdown reports in the project root upon completion. These files are **not tracked by git** (see [`GIT_POLICY.md`](./GIT_POLICY.md)).

| File | Generated by | Contents |
|------|-------------|----------|
| `SAMPLE_REPORT.md` | Pipeline (post-run) | 20 randomly sampled rows per split showing QID, question, answer, and scene graph summary. Used for a quick human sanity check. |
| `VALIDATION_REPORT.md` | Pipeline (post-run) | Full row-count validation, schema check, null-value audit, and QID uniqueness results. Acts as the official sign-off document for a production run. |
| `SPLIT_REPORT.md` | Pipeline (post-run) | Per-split statistics: number of shards, rows per shard (min/mean/max), total size on disk, and shard size distribution histogram. |
| `DIVERSITY_REPORT.md` | Pipeline (post-run) | Answer vocabulary size, top-50 most frequent answers, unique video coverage, question-type breakdown (e.g., "what", "where", "how many"). Used to diagnose class imbalance before training. |

> [!TIP]
> Always archive the `VALIDATION_REPORT.md` alongside your model training logs. It provides an exact record of which dataset version was used and its quality metrics, enabling reproducibility audits months later.

---

## 9. Emergency Stop

**Ctrl+C is safe at any point during the run.**

The pipeline registers a SIGINT handler that:

1. Finishes writing the current row to the in-memory buffer (does not interrupt mid-row).
2. Flushes all non-empty in-memory buffers to disk as final partial shards.
3. Logs the exact QID at which processing stopped.
4. Exits with code 1.

Data already written to complete shards is **fully preserved**. The partial final shards written during shutdown are also valid Parquet files and will be retained on resume.

**After an emergency stop:**

1. Note the last QID logged (e.g., `[INFO] Stopped at QID: fLXHK7Tq3yw_15-30_047821`).
2. Run the Corruption Recovery procedure (§6) as a precaution.
3. Resume using the procedure in §5.

> [!WARNING]
> Hard-killing the process via Task Manager (`taskkill /F`) or a power interruption does **not** trigger the graceful shutdown handler. In this case, treat the last shard of each split as potentially corrupt and run the full Corruption Recovery procedure before resuming.

---

## 10. Scaling Beyond 1 Million Rows

### 10.1 Disk Requirements

| Total rows | Estimated disk usage |
|-----------|---------------------|
| 1,000,000 | ~12–15 GB |
| 2,000,000 | ~24–30 GB |
| 5,000,000 | ~60–75 GB |
| 10,000,000 | ~120–150 GB |

Estimates assume Snappy compression and average scene graph sizes. Scene graphs with many objects/relations will produce larger files.

### 10.2 Memory Tuning

At scales >5M rows, the in-memory buffer itself can become a bottleneck. Reduce the shard size to flush more frequently:

```python
# In src/uniprop/writer.py — reduce SHARD_SIZE
SHARD_SIZE = 10_000   # Default is 50,000; reduce for lower peak memory
```

Smaller shards mean more frequent I/O but lower peak RAM usage. The trade-off is more files on disk (each with a small fixed overhead) and slightly slower reads if not using glob-based parallel loading.

### 10.3 Parallel Processing

For datasets >5M rows, consider splitting source QA files by video ID range and running multiple pipeline instances in parallel, each writing to a separate output directory. Merge the directories afterwards:

```powershell
# Merge shard directories from parallel runs (re-index to avoid collisions)
python scripts/merge_shards.py `
    --inputs data/generated_part1/train data/generated_part2/train `
    --output data/generated/train
```

> [!NOTE]
> Parallel instances must write to **different** output directories. Never run two instances writing to the same directory simultaneously — shard index collisions will cause data overwrite.

### 10.4 Disk I/O Optimisation

- **SSD strongly recommended** for output directory. Parquet writes are sequential and benefit from low-latency storage.
- If using a network share (NFS/SMB), expect 5–10× slower write speeds and increased crash risk from network interruptions. Local storage is always preferred for production runs.
- Enable OS write-back caching on the output volume for better throughput on spinning disks.
