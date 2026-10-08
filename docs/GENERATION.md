# UniProp Dataset Generation Guide

> [!NOTE]
> This document describes the **primary generation pipeline** implemented in [`main_pipeline.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/main_pipeline.py). The pipeline produces the UniProp benchmark dataset from raw AGQA spatio-temporal scene-graph (STSG) annotations and balanced question-answer text files.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Run](#2-quick-run)
3. [What Happens During Generation — 7 Phases](#3-what-happens-during-generation--7-phases)
4. [All 8 Generators](#4-all-8-generators)
5. [Task-Type Distribution](#5-task-type-distribution)
6. [Hard-Negative Strategy](#6-hard-negative-strategy)
7. [NONE Proposition](#7-none-proposition)
8. [Deduplication](#8-deduplication)
9. [Progress Monitoring](#9-progress-monitoring)
10. [Output Reports](#10-output-reports)
11. [Checkpoint / Resume](#11-checkpoint--resume)
12. [Scaling](#12-scaling)

---

## 1. Overview

UniProp generation is a **streaming, multi-generator pipeline** that converts raw video scene-graph annotations into structured proposition groups — sets of textual claims that a vision-language model must evaluate.

The pipeline reads two input sources:

| Source file | Content |
|---|---|
| `AGQA_scene_graphs/AGQA_train_stsgs.pkl` | Per-video spatio-temporal scene graphs for training videos |
| `AGQA_scene_graphs/AGQA_test_stsgs.pkl` | Per-video spatio-temporal scene graphs for test videos |
| `AGQA_balanced/AGQA_balanced/train_balanced.txt` | Balanced question-answer pairs for train/val pass |
| `AGQA_balanced/AGQA_balanced/test_balanced.txt` | Balanced question-answer pairs for test pass |

Each raw scene graph is **normalised**, a **NegativePool** of hard-negative candidates is pre-computed per video, and then eight **proposition generators** are applied in round-robin order over every question entry. Accepted `PropositionGroup` objects are streamed directly to Parquet shards on disk, so the pipeline can be safely interrupted and resumed without reprocessing data that already exists.

End-to-end data flow:

```
STSG pickles + balanced QA text files
         │
         ▼
  Phase 1: Generator Registration
         │
         ▼
  Phase 2: Scene-Graph Loading  (pickle → dict: video_id → raw_sg)
         │
         ▼
  Phase 3: Train/Val Split      (90/10 seeded shuffle)
         │
         ▼
  Phase 4: Stats Initialisation (counters, sets)
         │
         ▼
  Phase 5: Checkpoint Loading   (scan existing .parquet shards → seen_hashes)
         │
         ▼
  Phase 6: Generation Passes    (train/val → test)
    │
    ├─ SG Normalisation         (normalize_video_sg + NegativePools per video)
    ├─ Round-Robin Generation   (8 generators × every question)
    ├─ Hash Deduplication       (sorted-text fingerprint → seen_hashes)
    ├─ Validation Gate          (validate_example)
    └─ Shard Flush              (buffers[split] → part-NNNNNN.parquet)
         │
         ▼
  Phase 7: Teardown & Reports   (force-flush + 5 consistency assertions + 4 Markdown reports)
```

---

## 2. Quick Run

```bash
python main_pipeline.py
```

No command-line arguments are required. All configuration is code-level — see [CONFIGURATION.md](CONFIGURATION.md) for the constants to change.

> [!IMPORTANT]
> Set `PYTHONHASHSEED` **before** starting the Python process to guarantee fully deterministic generation. The pipeline calls `os.environ.setdefault("PYTHONHASHSEED", "42")` at import time, but this only takes effect if the environment variable was not already set. The safest approach is to set it in your shell:
>
> ```bash
> # Linux / macOS
> PYTHONHASHSEED=42 python main_pipeline.py
>
> # Windows PowerShell
> $env:PYTHONHASHSEED = "42"; python main_pipeline.py
> ```

---

## 3. What Happens During Generation — 7 Phases

### Phase 1 — Generator Registration

Eight generator objects are instantiated and stored as `(name, generator_instance)` pairs in a fixed-order list. The round-robin scheduler cycles through this list continuously across all questions so that every generator gets equal opportunity. The registration order is:

```
orig_qa  →  obj_exist  →  rel_ver  →  rel_set
grounding  →  temporal  →  compositional  →  attributes
```

### Phase 2 — Scene-Graph Loading

Both STSG pickle files are loaded with `pickle.load()`. Each file maps `video_id → raw_scene_graph`. A sanity assertion verifies that no test video ID appears in the training set.

### Phase 3 — Split Assignment

The training video ID list is **seeded-shuffled** with `random.Random(SEED)` and the first **10 %** are held out as the validation set; the remaining **90 %** form the training set. Both sets are stored as Python `set` objects for O(1) membership lookup.

The inner helper `get_split(vid, is_test_set)` maps each video ID to `"train"`, `"val"`, `"test"`, or `None` (video does not belong to the current processing pass).

### Phase 4 — Stats Dictionary Initialisation

A single `stats` dictionary is created to accumulate all metrics:

| Key | Type | Tracks |
|---|---|---|
| `candidates` | `Counter` | Raw groups produced per generator |
| `accepted` | `Counter` | Groups that passed validation per generator |
| `rejected` | `Counter` | Groups that failed validation per generator |
| `rejection_reasons` | `Counter` | Validation error codes and frequencies |
| `task_types` | `Counter` | `group.task_type` values |
| `reasoning_families` | `Counter` | `group.reasoning.family` values |
| `generator_families` | `Counter` | `group.provenance.generator_family` values |
| `complexity` | `Counter` | `group.reasoning.complexity` values |
| `k_counts` | `Counter` | `group.num_propositions` (k-way choice size) |
| `truth_states` | `Counter` | `proposition.truth_state` across all propositions |
| `none_included` | `int` | Propositions with `canonical_type == "logical_none"` |
| `none_true` | `int` | NONE propositions where `label == 1` |
| `none_false` | `int` | NONE propositions where `label != 1` |
| `query_templates` | `set` | Distinct `query_template_id` values |
| `temporal_subtypes` | `Counter` | Temporal sub-categories (`"before"`, `"after"`, `"multihop"`, …) |
| `temporal_hops` | `Counter` | `group.reasoning.hops` for temporal groups |
| `grounding_subtypes` | `Counter` | `"identification"` vs `"verification"` |
| `source_videos` | `Counter` | Examples generated per source video |
| `split_counts` | `Counter` | Examples generated per split name |
| `examples_per_evidence` | `Counter` | Evidence-signature reuse counts |
| `unique_prop_signatures` | `set` | Hashed `(type, subj, pred, obj, trel, ea, eb, pol)` tuples |
| `unique_option_sets` | `set` | Hashed sorted-option-text tuples (group level) |
| `duplicate_count` | `int` | Groups discarded by hash-dedup gate |
| `total_candidates` | `int` | All raw groups produced before any filtering |

### Phase 5 — Checkpoint Loading

Existing Parquet shards under `data/generated/<split>/` are scanned at startup. For each shard:

1. The numeric index in the filename (`part-NNNNNN.parquet`) is used to advance `shard_indices[split]` so new shards receive non-colliding filenames.
2. Only the `propositions` column is read (avoiding full deserialization of large shards). The sorted-text hash of every existing `PropositionGroup` is added to `seen_hashes`.

This ensures the pipeline resumes exactly where it left off without re-generating any already-persisted data. `generated_count` is pre-loaded with the checkpoint count so `target_count` arithmetic remains correct.

### Phase 6 — Generation Passes

`process_split()` is called twice:

| Pass | QA file | SG source | `is_test_set` | Dev `target_count` | Production `target_count` |
|---|---|---|---|---|---|
| Train / Val | `train_balanced.txt` | `train_sgs_raw` | `False` | `8,000` | `800,000` |
| Test | `test_balanced.txt` | `test_sgs_raw` | `True` | `2,000` | `200,000` |

Inside `process_split()`:

**SG Normalisation sub-phase:**  
Every video in `sgs_raw` is passed through `normalize_video_sg(vid, raw_sg)` and a `NegativePools` object is constructed. Both are cached by `video_id`; normalisation is never repeated for the same video.

**Round-robin generation sub-phase:**  
For each `(qid, qdata)` yielded by `stream_questions(qs_path)`, up to `len(generators)` generators are tried in sequence starting from the current `gen_idx` position. After each generator attempt `gen_idx` is advanced modulo the list length. This guarantees no single generator dominates when `target_count` is reached early.

**Hash deduplication sub-phase:**  
The sorted proposition texts are joined and hashed (`hash("".join(sorted(p.text for p in group.propositions)))`). Any group whose fingerprint is already in `seen_hashes` is discarded. The hash is registered immediately upon acceptance to block within-run duplicates.

**Validation gate:**  
`validate_example(group)` enforces schema constraints (label counts, proposition counts, etc.). Failed groups are tallied in `stats["rejection_reasons"]`; error codes are recorded per-generator.

**Buffer accumulation:**  
Accepted groups are appended to `buffers[split]`. After each append, `flush_buffer(split, force=False)` is called — if the buffer has reached `SHARD_SIZE`, a new `part-NNNNNN.parquet` shard is written to `data/generated/<split>/`.

### Phase 7 — Teardown and Report Generation

After both passes:

1. **Force-flush** — any partial buffers (< `SHARD_SIZE` rows) are flushed to disk with `force=True`.
2. **Consistency assertions** — five invariants are verified:
   - `sum(task_types) == total_accepted`
   - `sum(reasoning_families) == total_accepted`
   - `sum(split_counts) == total_accepted`
   - `sum(k_counts) == total_accepted`
   - `sum(temporal_subtypes) == reasoning_families["temporal"]`
3. **Report generation** — four Markdown audit reports are written (see [§10](#10-output-reports)).

---

## 4. All 8 Generators

Each generator implements `GeneratorBase.generate(qid, qdata, sg, split, rng, pools) → Iterator[PropositionGroup]`.

### 4.1 `OriginalQAGenerator` (`orig_qa`)

**Source:** [`src/generators/original_qa.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/original_qa.py)

**Status: Intentional no-op (LLM dependency).**  
This generator is reserved for wrapping original AGQA question-answer pairs as propositions. Converting free-text QA pairs to structured `PropositionGroup` objects requires LLM-assisted reformatting that is not available in the offline pipeline. The `generate()` method yields nothing, acting as a placeholder in the round-robin registry.

| Property | Value |
|---|---|
| Task types produced | None |
| Yields | Nothing |
| Reason | LLM dependency for QA-to-proposition conversion |

---

### 4.2 `ObjectExistenceGenerator` (`obj_exist`)

**Source:** [`src/generators/scene_graph.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/scene_graph.py)

Generates propositions about whether specific objects appear in a video. Derives all data directly from `pools.present_objects` — no question text is required.

| Task type | Structure | K | Details |
|---|---|---|---|
| `binary` | 1 positive + 1 negative polarity for the same object | 2 | 50% positive-centred (present object), 50% negative-centred (absent object) |
| `single_choice` | 1 `TRUE` proposition + K-1 `FALSE` distractors | rand[10, 40] | Absent objects sampled from `OBJECTS.values() − present_objects` |
| `multi_label` | Mix of `TRUE` (present) + `FALSE` (absent) + optional NONE | rand[10, 40] | 20% chance `num_true=0`; 70% chance NONE sentinel appended |
| `three_way` | Fixed: `"Yes"` / `"No"` / `"Cannot say"` | 3 | Hand-crafted question string; object can be present or absent |

**Guard:** yields nothing if `pools` is `None` or `pools.present_objects` is empty.

---

### 4.3 `RelationVerificationGenerator` (`rel_ver`)

**Source:** [`src/generators/scene_graph.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/scene_graph.py)

Generates propositions about person-to-object spatial and contact relations drawn from `pools.present_relations`. Relations are triples of `(rel_type, rel_class, object_token)`.

| Task type | Structure | K | Details |
|---|---|---|---|
| `binary` | Positive-polarity + negative-polarity for one relation triple | 2 | Negative prefers mutually-exclusive relation; falls back to closed-world false |
| `single_choice` | 1 `TRUE` + K-1 `FALSE` compatible-but-absent relations for same object | rand[10, 40] | False candidates drawn from `ALL_RELATIONS` filtered by `is_compatible` |
| `multi_label` | All confirmed relations for one object + false distractors + optional NONE | rand[10, 40] | 20% chance `num_true=0`; 70% chance NONE sentinel appended; `complexity=2` |
| `three_way` | Fixed: `"Yes"` / `"No"` / `"Cannot say"` | 3 | Always anchors on a confirmed-present relation (so `"Yes"` is always correct) |

**Guard:** yields nothing if `pools` is `None` or `pools.present_relations` is empty. For the `binary` negative branch, if no suitable false relation can be found the method returns early.

---

### 4.4 `RelationSetGenerator` (`rel_set`)

**Source:** [`src/generators/scene_graph.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/scene_graph.py)

**Status: Delegated no-op.**  
All relation task types — including multi-label relation-set style tasks — are handled by `RelationVerificationGenerator`. This class is retained as an architectural placeholder for future separation of relation-set tasks into a dedicated generator, and to preserve the original design intent in the generator registry. `generate()` executes `yield from []` and produces nothing.

---

### 4.5 `GroundingGenerator` (`grounding`)

**Source:** [`src/generators/grounding.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/grounding.py)

Generates visual grounding tasks that link textual descriptions to spatial bounding-box evidence within a video frame.

| Sub-type | Task type | Structure |
|---|---|---|
| `box_object` (identification) | `single_choice` | Given a bounding box, which object does it contain? One `TRUE` object + K-1 distractors |
| `box_proposition` (verification) | `three_way` | Given a proposition about a box, is it `"Yes"` / `"No"` / `"Cannot say"`? |

`group.reasoning.type` contains `"identification"` for `box_object` tasks and `"verification"` for `box_proposition` tasks. The diversity report uses these substrings to populate `grounding_subtypes`.

---

### 4.6 `ActionTemporalGenerator` (`temporal`)

**Source:** [`src/generators/temporal.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/temporal.py)

Generates temporal reasoning tasks that probe a model's understanding of action ordering and duration within a video.

| Sub-type | Task type | Details |
|---|---|---|
| Action recognition | `single_choice` | Which action is the person performing? One `TRUE` action + distractors |
| Temporal ordering (1-hop) | `single_choice` / `binary` | Does action A occur before/after/during action B? |
| Temporal ordering (2-hop) | `single_choice` | Multi-hop chain involving two temporal relations (`"multihop"` in `reasoning.type`) |

The `temporal_subtypes` counter is populated from:
- `"multihop"` — when `"multihop"` appears in `group.reasoning.type`.
- The `temporal_relation` string of the first `TRUE` temporal proposition — for single-hop tasks.
- `"unknown"` — fallback when no `TRUE` temporal proposition is found.

`temporal_hops` is populated from `group.reasoning.hops`.

---

### 4.7 `CompositionalGenerator` (`compositional`)

**Source:** [`src/generators/compositional.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/compositional.py)

Generates compositional reasoning tasks by forming an AND-conjunction of two distinct scene-graph relations. A compound proposition is `TRUE` only if **both** component relations hold simultaneously.

| Task type | Structure |
|---|---|
| `binary` | Positive compound proposition (`rel_A AND rel_B holds`) vs. negative (`rel_A AND rel_B does not hold`) |
| `single_choice` | One `TRUE` compound proposition + K-1 `FALSE` distractors where at least one component relation fails |

`complexity` is set to `2` (two reasoning steps required).

---

### 4.8 `AttributeGenerator` (`attributes`)

**Source:** [`src/generators/attributes.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/generators/attributes.py)

**Status: Placeholder — no-op.**  
Reserved for future attribute-verification tasks (colour, size, state — e.g. *"Is the cup red?"*). AGQA scene-graph annotations do not include sufficient per-frame attribute data to support reliable automated attribute labelling at this time. `generate()` yields nothing.

---

## 5. Task-Type Distribution

Because task type is chosen uniformly at random within each active generator and generators are cycled in round-robin, the expected dataset distribution (excluding no-op generators) is approximately:

| Task type | Source generators | Expected share |
|---|---|---|
| `binary` | `obj_exist`, `rel_ver`, `compositional` | ~30% |
| `single_choice` | `obj_exist`, `rel_ver`, `grounding`, `temporal`, `compositional` | ~35% |
| `multi_label` | `obj_exist`, `rel_ver` | ~20% |
| `three_way` | `obj_exist`, `rel_ver`, `grounding` | ~15% |

> [!NOTE]
> The exact distribution will deviate from this estimate because (a) generators can yield nothing for a given video (guard conditions), (b) deduplication removes repeated examples, and (c) the `target_count` limit may be reached before all questions are processed. Consult `DIVERSITY_REPORT.md` after a run for the actual `task_types` breakdown.

---

## 6. Hard-Negative Strategy

The pipeline uses three tiers of hard-negative evidence, applied in preference order:

### Tier 1 — Mutually Exclusive Relations

For `RelationVerificationGenerator` binary tasks, the first choice for a negative proposition is a relation that is **logically incompatible** with the true relation for the same object. For example, if the true relation is `touching`, the mutually-exclusive negative would be `not_touching`.

Sourced via `NegativePools.get_mutually_exclusive_relation(rel_class, obj, rng)`.

### Tier 2 — Closed-World False Relations

If no mutually-exclusive relation is available, any relation in `ALL_RELATIONS` that (a) does not appear in the scene-graph for the chosen object **and** (b) passes the `is_compatible(rel, obj)` ontology filter is used as a false relation.

Sourced via `NegativePools.get_closed_world_false_relation(rel_class, obj, rng)` or by directly filtering `ALL_RELATIONS`.

### Tier 3 — Absent Objects

For `ObjectExistenceGenerator`, false propositions use objects drawn from the **complement of the present-object set**: `set(OBJECTS.values()) − set(present_objects)`. These are genuinely absent from the video's scene graph.

> [!TIP]
> Tier 1 negatives (mutually exclusive) produce the strongest training signal because they are semantically close to the true relation. Tier 3 negatives (absent objects) are the weakest but most reliable source since any object not in the scene graph is definitionally absent.

---

## 7. NONE Proposition

`multi_label` groups optionally include a special **NONE sentinel proposition** to teach models to recognise the all-false case rather than guessing randomly when no listed option is correct.

| Property | Value |
|---|---|
| Inclusion probability | **70%** (`rng.random() < 0.7`) |
| Text (object tasks) | `"None of these objects are present."` |
| Text (relation tasks) | `"None of these relationships hold with <object>."` |
| `canonical_type` | `"logical_none"` |
| `label` | `1` (TRUE) when `num_true == 0` (all other options are false) |
| `label` | `0` (FALSE) when at least one correct option exists |

**Forcing all-false pools:** With 20% probability `num_true` is forced to 0 before proposition construction, making the NONE proposition the only correct answer. This prevents NONE from being trivially uninformative.

The `stats` dictionary tracks all NONE propositions in:
- `none_included` — total count
- `none_true` — count where `label == 1`
- `none_false` — count where `label == 0`

---

## 8. Deduplication

Before any group reaches the validation gate, a **hash-based deduplication check** is applied:

```python
group_hash = hash("".join(sorted(p.text for p in group.propositions)))
if group_hash in seen_hashes:
    stats["duplicate_count"] += 1
    continue
seen_hashes.add(group_hash)
```

**Key properties:**

- The hash is computed from the **sorted** proposition texts — proposition ordering does not affect the fingerprint.
- `seen_hashes` is pre-populated from all existing Parquet shards during [Phase 5 checkpoint loading](#phase-5--checkpoint-loading), so cross-run duplicates are also blocked.
- The hash is registered **immediately** upon first encounter (before validation), so two identical groups generated in the same run will both be caught even if neither has been written to disk yet.
- Python's `hash()` is sensitive to `PYTHONHASHSEED`. Setting it consistently ensures deduplication behaviour is stable across interpreter restarts.

> [!WARNING]
> `hash()` is a 64-bit non-cryptographic hash and has a negligible but nonzero collision probability. For production runs targeting millions of examples, consider replacing it with `hashlib.sha256` if false-negative deduplication cannot be tolerated.

---

## 9. Progress Monitoring

### tqdm Bar

During SG normalisation, a `tqdm` progress bar tracks per-video normalisation. Nothing else is suppressed.

### Periodic Print-outs

During the main generation loop, a progress line is printed to stdout under two conditions:
- Every `target_count // 100` accepted examples (i.e. approximately every 1% of the target), **or**
- Whenever more than 10 seconds have elapsed since the last print.

Format:
```
[  5.2%] Generated 416/8000 (124.3 records/sec) - 3.4s elapsed
```

Fields:
- **%** — percentage of `target_count` reached by the current run (excludes checkpoint examples).
- **Generated N/target** — current `generated_count` vs `target_count`.
- **records/sec** — throughput since the start of the current `process_split()` call.
- **elapsed** — wall-clock seconds since `process_split()` started.

### Shard Flush Messages

Each successful shard write prints:
```
Flushed 50000 examples to /path/to/data/generated/train/part-000002.parquet
```

Failed flushes are logged without aborting the pipeline; the buffer is retained for the next flush attempt.

---

## 10. Output Reports

All four reports are written to the **working directory** (the directory from which `python main_pipeline.py` is run) at the end of Phase 7.

### `SAMPLE_REPORT.md`

Human-readable illustrations: up to **5 example groups per generation rule**. Each entry shows:
- `example_id`, split, task type, K
- Query text
- Full proposition list with `[label]`, text, and `(Truth: state)`

Useful for quickly verifying that each generation rule is producing semantically sensible outputs.

### `VALIDATION_REPORT.md`

Macro-level accept/reject/duplicate statistics:
- Total candidates produced
- Total accepted / rejected / duplicates caught
- Breakdown of the most common validation error codes and their frequencies

### `SPLIT_REPORT.md`

Video-universe and example-count summary per split:
- Train / val / test video universe sizes
- Actual number of generated examples per split

### `DIVERSITY_REPORT.md`

Production-readiness and diversity metrics:
- Unique source videos used; average examples per video
- Average examples per evidence signature
- Unique proposition signatures; unique option sets (group-level); unique query templates
- Duplicate catch ratio; evidence reuse ratio
- Truth-state distribution (`"TRUE"` vs `"FALSE"` vs `"UNKNOWN"`)
- Logical NONE proposition breakdown (`none_included`, `none_true`, `none_false`)
- Task-type, reasoning-family, k-count distributions
- Temporal subtype and hop-depth distributions

---

## 11. Checkpoint / Resume

The pipeline supports automatic resumption after interruption with **no manual intervention required**:

1. On startup, all existing `part-NNNNNN.parquet` shards under `data/generated/<split>/` are scanned.
2. The `propositions` column of each shard is read, and the sorted-text hash of every `PropositionGroup` is added to `seen_hashes`.
3. `generated_count` is pre-set to the total number of recovered examples so `target_count` arithmetic is correct.
4. New shards receive indices that continue after the last existing index.

**To resume after interruption:**
```bash
python main_pipeline.py
```

The pipeline will print:
```
Loading checkpoints...
Loaded 45123 previous examples into deduplication cache.
```

> [!IMPORTANT]
> Do **not** delete or move partial shards between runs. Orphaned shard files with intermediate indices will cause `shard_indices[split]` to be advanced past those indices, leaving gaps in the shard sequence. This is harmless for data integrity but may create non-contiguous shard numbering.

---

## 12. Scaling

The `target_count` parameter in `process_split()` controls how many accepted examples the pipeline attempts to generate per pass (train/val and test are separate calls).

| Mode | Train/Val target | Test target | Total |
|---|---|---|---|
| **Development (current)** | `8,000` | `2,000` | `10,000` |
| **Production** | `800,000` | `200,000` | `1,000,000` |

To switch to a production run, edit the two `process_split()` calls in [`main_pipeline.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/main_pipeline.py#L875-L877):

```python
# Development (current)
process_split(train_qs_path, train_sgs_raw, False, 8_000)
process_split(test_qs_path,  test_sgs_raw,  True,  2_000)

# Production
process_split(train_qs_path, train_sgs_raw, False, 800_000)
process_split(test_qs_path,  test_sgs_raw,  True,  200_000)
```

> [!TIP]
> For a smoke test (quick correctness check without writing large amounts of data), use `target_count=100`. The pipeline will stop early after accepting 100 examples and write a single partial shard.

> [!NOTE]
> Throughput is primarily bounded by Python-level `normalize_video_sg` and `NegativePools` construction during the SG normalisation phase. These operations are cached per video, so throughput scales with the ratio of questions to unique videos. On a modern multi-core workstation, expect 100–500 examples/sec in development mode.
