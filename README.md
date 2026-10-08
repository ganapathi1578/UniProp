# UniProp

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![PyArrow](https://img.shields.io/badge/PyArrow-%3E%3D14.0-e25a1c?logo=apache&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

UniProp is a **proposition-based multimodal dataset construction and labeling pipeline** built on top of AGQA (Action Genome Question Answering). It transforms raw AGQA spatio-temporal scene-graph data and balanced QA text files into a normalized, validated, deduplicated benchmark dataset expressed as **logical propositions**. Rather than asking "what is the answer?", UniProp frames every evaluation example as a structured set of candidate propositions paired with binary correctness labels — enabling richer, more diagnostically precise model evaluation across spatial, temporal, compositional, and relational reasoning skills. UniProp is **pure dataset engineering**: it produces raw-text string inputs only and contains no tokenizers, model formatters, embeddings, or training code.

---

## Key Features

- **Proposition-based schema** — every example is a `(query, options[], labels[])` triple; options are natural-language propositions and labels are binary (`1` = correct, `0` = incorrect).
- **8 generator families** — eight specialised generators cover original QA, object existence, relation verification, relation set, spatial grounding, temporal ordering, compositional reasoning, and attribute-based propositions.
- **4 task types** — `binary`, `single_choice`, `multi_label`, and `three_way` encoding schemes with well-defined label constraints.
- **Hard negative mining** — a per-video `NegativePools` object surfaces semantically plausible but factually wrong candidate propositions to prevent trivial label guessing.
- **Closed-world assumption** — every label is grounded exclusively in the scene-graph facts present in AGQA; no open-world inference is permitted.
- **Hash-based deduplication** — each candidate group is fingerprinted by hashing sorted proposition texts; cross-shard and cross-run duplicates are discarded before validation.
- **Shard-based streaming I/O** — accepted groups accumulate in per-split in-memory buffers and are flushed to zero-padded Parquet shards (`part-NNNNNN.parquet`) at configurable intervals, enabling safe interruption and resumption of long runs.
- **Tri-layer reproducibility** — `PYTHONHASHSEED`, `random.seed`, and per-generator `random.Random` instances are all seeded from a single global constant (`SEED=42`), guaranteeing bit-for-bit identical output across runs and Python versions.

---

## Core Task Representation

Every example in UniProp follows this schema:

```python
query:   str          # A natural-language question or probe
options: list[str]    # Ordered list of candidate proposition texts
labels:  list[int8]   # Parallel binary labels; labels[i]=1 iff options[i] is correct
```

**Invariant:** `len(options) == len(labels)` is enforced as a hard constraint at both generation and validation time.

**Label semantics:**
| Value | Meaning |
|-------|---------|
| `1`   | The proposition is **correct** given the scene graph (closed-world true) |
| `0`   | The proposition is **incorrect** given the scene graph |
| `-1`  | The proposition's truth value is **unknown** (three-way / abstain tasks) |

### Example Record

```json
{
  "example_id": "act_ml_46GP8_2548359539",
  "split": "train",
  "video_id": "46GP8",
  "query": "What is the person doing?",
  "options": [
    "The person is making some food.",
    "The person is typing on a laptop."
  ],
  "labels": [1, 0],
  "truth_state": "TRUE",
  "option_count": 2,
  "task_type": "multi_label",
  "reasoning_family": "action",
  "generator_family": "orig_qa"
}
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:ganapathi1578/UniProp.git
cd UniProp

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download source data (manual step)
#    Obtain AGQA_balanced/ and AGQA_scene_graphs/ from the AGQA authors.
#    Place both directories in the repository root:
#
#    UniProp/
#    ├── AGQA_balanced/
#    │   ├── AGQA_train_balanced.txt
#    │   └── AGQA_test_balanced.txt
#    └── AGQA_scene_graphs/
#        ├── AGQA_train_stsgs.pkl
#        └── AGQA_test_stsgs.pkl

# 5. Run the generation pipeline
python main_pipeline.py
```

> **Note:** Generated Parquet shards are written to `data/generated/` and are excluded from Git. See [Output & Data Policy](#output--data-policy).

---

## Project Structure

```
UniProp/
│
├── main_pipeline.py            # ← Pipeline entry point; run this to generate the dataset
│
├── requirements.txt            # Python dependency pins
├── .gitignore                  # Excludes generated data, source data, and caches
│
├── docs/                       # Comprehensive reference documentation (see index below)
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   ├── DATA_LAYOUT.md
│   ├── DATA_SCHEMA.md
│   ├── DEVELOPMENT.md
│   ├── GENERATION.md
│   ├── GIT_POLICY.md
│   ├── INSTALLATION.md
│   ├── LABEL_SEMANTICS.md
│   ├── PRODUCTION_RUNBOOK.md
│   ├── PROVENANCE.md
│   ├── README.md
│   ├── REPRODUCIBILITY.md
│   ├── RESOURCE_REQUIREMENTS.md
│   ├── SAMPLE_REPORT.md
│   ├── TROUBLESHOOTING.md
│   └── VALIDATION.md
│
├── src/                        # All source code lives here
│   ├── schema.py               # Core dataclasses: Proposition, PropositionGroup, Reasoning, Provenance
│   ├── constants.py            # Global constants: TASK_TYPES, SPLITS, SHARD_SIZE, etc.
│   ├── templates.py            # Query-template registry (template_id → query string)
│   ├── compatibility.py        # Cross-version Python/PyArrow shims
│   │
│   ├── generators/             # One module per generator family
│   │   ├── base.py             # Abstract BaseGenerator interface
│   │   ├── original_qa.py      # OriginalQAGenerator
│   │   ├── attributes.py       # AttributeGenerator
│   │   ├── compositional.py    # CompositionalGenerator
│   │   ├── grounding.py        # GroundingGenerator
│   │   ├── scene_graph.py      # SceneGraphGenerator (relation set / verification)
│   │   ├── temporal.py         # ActionTemporalGenerator
│   │   ├── renderer.py         # Proposition text renderer (template → surface form)
│   │   └── registry.py         # Generator registry & round-robin dispatcher
│   │
│   ├── candidates/             # Option-set construction (optionizers)
│   │   ├── base.py             # Abstract BaseOptionizer
│   │   ├── action_options.py   # Action-based candidate propositions
│   │   ├── binary_options.py   # Binary yes/no option builder
│   │   ├── comparison_options.py
│   │   ├── count_options.py
│   │   ├── logic_options.py
│   │   ├── more_options.py
│   │   ├── object_options.py
│   │   ├── superlative_options.py
│   │   ├── three_way_options.py
│   │   └── registry.py
│   │
│   ├── negatives/
│   │   └── hard_negatives.py   # NegativePools: per-video hard-negative surface pool
│   │
│   ├── normalization/
│   │   ├── scene_graph_normalizer.py  # Raw AGQA STSG → canonical SceneGraph
│   │   └── split_manager.py           # Reproducible train/val/test split assignment
│   │
│   ├── reasoning/              # Semantic and symbolic reasoning utilities
│   │   ├── semantic_types.py   # TruthState, SemanticSignature, relation taxonomy
│   │   ├── semantic_engine.py  # Truth-state evaluation against scene graphs
│   │   ├── canonicalization.py # Proposition text → canonical semantic key
│   │   ├── evidence.py         # Evidence extraction from scene graphs
│   │   ├── program_parser.py   # AGQA functional program → AST
│   │   └── program_evaluator.py# AST evaluation against normalized scene graph
│   │
│   ├── validation/
│   │   └── validators.py       # validate_example() — 10-check quality gate
│   │
│   ├── storage/
│   │   └── parquet_writer.py   # write_shard() — PropositionGroup list → Parquet file
│   │
│   ├── pipeline/
│   │   └── dataset_pipeline.py # Core pipeline orchestrator (called by main_pipeline.py)
│   │
│   ├── io/
│   │   └── streaming_json.py   # Streaming JSON reader for large AGQA QA files
│   │
│   ├── parsers/
│   │   └── agqa/               # AGQA-specific data loaders (scene graph + QA)
│   │
│   ├── datasets/               # Dataset manifest and split registry helpers
│   └── cli/
│       └── main.py             # Optional CLI wrapper around the pipeline
│
├── tests/                      # Pytest test suite
│   ├── test_configuration.py   # Configuration and constant validation tests
│   ├── test_optionizers.py     # Optionizer correctness and label-constraint tests
│   ├── test_new_optionizers.py # Tests for recently added optionizer families
│   └── test_validators.py      # Validator unit tests (all 10 checks)
│
└── data/                       # Runtime data directory (NOT committed to Git)
    └── generated/
        ├── train/              # Parquet shards for the training split
        ├── val/                # Parquet shards for the validation split
        └── test/               # Parquet shards for the test split
```

---

## Dataset Generation Pipeline

The pipeline executes **7 stages** in sequence:

1. **Scene-graph loading** — Two AGQA STSG pickle files (`AGQA_train_stsgs.pkl`, `AGQA_test_stsgs.pkl`) are deserialised into memory. The training graph set is shuffled with the global seed to derive reproducible train / val universe assignments.

2. **Scene-graph normalisation** — Each raw STSG is passed through `normalize_video_sg()`, which resolves entity aliases, canonicalises relation names, and emits a uniform internal `SceneGraph` representation. A per-video `NegativePools` object is built in the same pass for downstream hard-negative sampling.

3. **Split assignment** — `SplitManager` uses the seeded shuffle to partition training video IDs 90/10 into `train` and `val`. AGQA test video IDs map directly to the `test` split. Every QA record carries its split label throughout the pipeline.

4. **Proposition generation (round-robin)** — For each AGQA QA entry the eight registered generators are invoked in round-robin order. Each generator returns zero or more `PropositionGroup` candidates. Generators draw candidates from their respective optionizer and the video's `NegativePools`.

5. **Deduplication** — Every `PropositionGroup` is fingerprinted by SHA-256 hashing the sorted proposition texts. Hashes already present in the on-disk shard index or the current run's seen-set are discarded immediately.

6. **Validation** — Surviving candidates pass through `validate_example()`, which runs all 10 quality checks (see [Validation Gates](#validation-gates)). Groups that fail any check are tallied and discarded; the remainder are "accepted".

7. **Streaming shard flush** — Accepted groups accumulate in per-split in-memory buffers. When a buffer reaches `SHARD_SIZE` rows it is serialised by `write_shard()` to a zero-padded Parquet file (`part-NNNNNN.parquet`) under `data/generated/<split>/`. After all data are processed, four Markdown audit reports are written to the working directory.

---

## Generators

| Generator | Family key | Task types produced | Description |
|---|---|---|---|
| `OriginalQAGenerator` | `orig_qa` | `binary`, `single_choice` | Wraps the original AGQA question–answer pairs as propositions, preserving the original question as the query. |
| `ObjectExistenceGenerator` | `obj_exist` | `binary`, `multi_label` | Generates propositions asserting the presence or absence of specific objects detected in the scene graph. |
| `RelationVerificationGenerator` | `rel_ver` | `binary` | Produces single yes/no propositions testing whether a specific spatial or action relation holds between two entities. |
| `RelationSetGenerator` | `rel_set` | `single_choice`, `multi_label` | Constructs option sets across multiple candidate relations for the same subject–object pair. |
| `GroundingGenerator` | `grounding` | `binary`, `single_choice` | Builds spatially-grounded propositions that require linking textual descriptions to bounding-box regions. |
| `ActionTemporalGenerator` | `temporal` | `binary`, `single_choice`, `three_way` | Creates temporal ordering and duration propositions derived from action segment timestamps in the scene graph. |
| `CompositionalGenerator` | `compositional` | `single_choice`, `multi_label` | Assembles multi-hop reasoning propositions that chain together two or more scene-graph relations. |
| `AttributeGenerator` | `attributes` | `binary`, `single_choice` | Generates propositions about object attributes (colour, size, state) grounded in AGQA scene-graph attribute annotations. |

---

## Task Types

| Task type | Correct labels | Label constraint | Typical use |
|---|---|---|---|
| `binary` | Exactly **1** | `sum(labels) == 1`, `len(options) >= 2` | True/false single-proposition evaluation |
| `single_choice` | Exactly **1** | `sum(labels) == 1`, `len(options) >= 2` | Multiple-choice with one correct answer |
| `multi_label` | **≥ 1** | `sum(labels) >= 1`, no upper bound | Select-all-that-apply; models must output a set |
| `three_way` | Exactly **1** | `len(options) == 3`, `sum(labels) == 1` | True / False / Unknown trichotomy |

---

## Output Schema

Each row in the output Parquet files corresponds to one `PropositionGroup` and contains the following columns:

| Column | PyArrow type | Description |
|---|---|---|
| `example_id` | `string` | Globally unique example identifier (e.g. `act_ml_46GP8_2548359539`) |
| `split` | `string` | Dataset partition: `"train"`, `"val"`, or `"test"` |
| `video_id` | `string` | Source Charades video ID (non-null; used for provenance tracking) |
| `query` | `string` | Natural-language question or probe accompanying the propositions |
| `options` | `list<string>` | Ordered list of candidate proposition surface texts |
| `labels` | `list<int8>` | Binary labels parallel to `options`; `1` = correct, `0` = incorrect, `-1` = unknown |
| `truth_state` | `string` | Semantic truth-state of the primary proposition: `"TRUE"`, `"FALSE"`, or `"UNKNOWN"` |
| `option_count` | `int32` | Redundant count of `len(options)`; stored for fast columnar filtering |
| `task_type` | `string` | Evaluation protocol: `"binary"`, `"single_choice"`, `"multi_label"`, or `"three_way"` |
| `reasoning_family` | `string` | Reasoning-skill taxonomy label (e.g. `"spatial"`, `"temporal"`, `"compositional"`) |
| `generator_family` | `string` | Generator provenance key (e.g. `"orig_qa"`, `"grounding"`) |

---

## Validation Gates

`validate_example()` in [`src/validation/validators.py`](src/validation/validators.py) runs **10 checks** on every candidate `PropositionGroup`. All checks are run in sequence (no short-circuit) so that a complete diagnostic report is available. A group must pass all 10 to be accepted.

1. **Proposition count range** — non-`three_way` groups must have between 2 and 40 propositions.
2. **Three-way count** — `three_way` groups must have exactly 3 propositions.
3. **Label cardinality** — `binary`, `single_choice`, and `three_way` groups must each have exactly one `label == 1` proposition.
4. **Query-to-task compatibility** — the `query_template_id` prefix must be consistent with the declared `reasoning_type` and `task_type`.
5. **Semantic contradiction detection** — no two propositions may share the same canonical semantic signature while carrying opposite labels.
6. **Mutual exclusivity detection** — relation pairs known to be logically contradictory (e.g. `above` / `beneath`, `looking at` / `not looking at`) must not both appear as positive-label propositions.
7. **Temporal consistency** — detects cyclic or contradictory `before` / `after` orderings across the positive-label propositions.
8. **NONE proposition consistency** — if a `logical_none` proposition is present its label must be consistent with all other labels in the group.
9. **Duplicate text detection** — all proposition texts within a group must be unique (case-normalised).
10. **Surface-form quality** — detects unresolved placeholder IDs, known bad-grammar patterns, generic filler phrases, and double negations.

---

## Reproducibility

UniProp uses a **tri-layer RNG lock** to guarantee bit-for-bit identical datasets across machines, Python patch versions, and re-runs:

| Layer | Mechanism | Scope |
|---|---|---|
| **Layer 1** | `PYTHONHASHSEED=42` environment variable | Deterministic `str` / `bytes` hash seeds used by Python's built-in `hash()` |
| **Layer 2** | `random.seed(SEED)` at pipeline startup | Global `random` module state for any code using `random.*` directly |
| **Layer 3** | Per-generator `random.Random(SEED)` instances | Isolated RNG per generator so that adding/removing generators does not perturb other generators' sequences |

The global seed constant is `SEED = 42` and is defined in `main_pipeline.py`. The train/val split shuffle and all option-set sampling are derived exclusively from these three layers.

```bash
# Fully reproducible invocation (set PYTHONHASHSEED before launching):
PYTHONHASHSEED=42 python main_pipeline.py   # Linux / macOS
$env:PYTHONHASHSEED=42; python main_pipeline.py  # Windows PowerShell
```

---

## Output & Data Policy

| Artifact | Committed to Git? | Location |
|---|---|---|
| Source code (`src/`, `tests/`, `docs/`) | ✅ Yes | Repository root |
| `requirements.txt`, `.gitignore`, `README.md` | ✅ Yes | Repository root |
| Generated Parquet shards | ❌ **No** | `data/generated/` |
| AGQA source data (`AGQA_balanced/`, `AGQA_scene_graphs/`) | ❌ **No** | Repository root (manual download) |
| Audit / validation reports (`*_REPORT.md` at root) | ❌ **No** | Repository root (runtime artefacts) |
| Python bytecode caches (`__pycache__/`, `*.pyc`) | ❌ **No** | Various |

See [`docs/GIT_POLICY.md`](docs/GIT_POLICY.md) for the full, formal statement of what is and is not permitted in version control.

---

## Documentation Index

All reference documentation lives in [`docs/`](docs/):

| Document | Purpose |
|---|---|
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | End-to-end data-flow: source JSON → proposition logic → Parquet strings |
| [`CONFIGURATION.md`](docs/CONFIGURATION.md) | `main_pipeline.py` knobs explained (`SEED`, `SHARD_SIZE`, generator flags) |
| [`DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) | Directory layout of `data/generated/` and source datasets |
| [`DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | PyArrow schema definition with explicit invariants |
| [`DEVELOPMENT.md`](docs/DEVELOPMENT.md) | How to implement a new Generator or Reasoner |
| [`GENERATION.md`](docs/GENERATION.md) | CLI commands and runtime architecture (flushing, buffering) |
| [`GIT_POLICY.md`](docs/GIT_POLICY.md) | Formal definition of what must never enter version control |
| [`INSTALLATION.md`](docs/INSTALLATION.md) | Clean environment setup with `requirements.txt` |
| [`LABEL_SEMANTICS.md`](docs/LABEL_SEMANTICS.md) | Distinction between `truth_state` (`TRUE`/`FALSE`/`UNKNOWN`) and `label` (`1`/`0`/`-1`) |
| [`PRODUCTION_RUNBOOK.md`](docs/PRODUCTION_RUNBOOK.md) | Step-by-step checklist for initiating and recovering 10 M+ row runs |
| [`PROVENANCE.md`](docs/PROVENANCE.md) | `video_id` tracking mandate and audit-trail requirements |
| [`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Tri-layer RNG lock specification and verification procedure |
| [`RESOURCE_REQUIREMENTS.md`](docs/RESOURCE_REQUIREMENTS.md) | Memory and disk scaling extrapolations for large Parquet outputs |
| [`SAMPLE_REPORT.md`](docs/SAMPLE_REPORT.md) | Example of a generation sample report (up to 5 groups per rule) |
| [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Common failures, error codes, and filesystem recovery procedures |
| [`VALIDATION.md`](docs/VALIDATION.md) | Detailed specification of all 10 active quality gates |

---

## Requirements

| Package | Version constraint | Role |
|---|---|---|
| `pyarrow` | `>=14.0` | Parquet I/O and strongly-typed columnar schema |
| `pyyaml` | `>=6.0` | Configuration file parsing |
| `tqdm` | `>=4.65` | Progress bars for long-running pipeline stages |
| `zstandard` | `>=0.22` | Zstd compression for Parquet shards |
| `ijson` | `>=3.2.3` | Streaming JSON parser for large AGQA QA files |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Python **3.10 or later** is required. No GPU or CUDA dependency exists.

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Key points:

- Run the full test suite (`pytest tests/`) before submitting.
- New generators must include corresponding optionizer tests in `tests/`.
- Generated data files must **never** be committed (see [`docs/GIT_POLICY.md`](docs/GIT_POLICY.md)).
- Follow the Google-style docstring convention used throughout `src/`.

---

*UniProp is a dataset-engineering project. It does not contain model weights, tokenizers, or training code.*
