# UniProp — Development Guide

This guide explains how to extend UniProp: adding new generators, validators,
dataset adapters, text templates, and reasoning-engine tests.  Read the
[Installation Guide](INSTALLATION.md) first to ensure your environment is
configured correctly.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Adding a New Generator](#2-adding-a-new-generator)
3. [Adding a New Validator Check](#3-adding-a-new-validator-check)
4. [Adding a New Dataset Adapter](#4-adding-a-new-dataset-adapter)
5. [Rendering New Text Templates](#5-rendering-new-text-templates)
6. [Running Tests](#6-running-tests)
7. [Test Structure](#7-test-structure)
8. [Code Style](#8-code-style)
9. [Debugging Tips](#9-debugging-tips)
10. [Reasoning Engine](#10-reasoning-engine)

---

## 1. Overview

UniProp is a **static dataset engineering pipeline**.  Raw AGQA question-answer
pairs and Action Genome spatio-temporal scene graphs flow through a series of
Python modules and emerge as a sharded, schema-validated Parquet benchmark
dataset.

The key extension points are:

```
UniProp/
├── src/
│   ├── generators/          ← Add new task-family generators here
│   │   ├── base.py          ←   GeneratorBase ABC
│   │   ├── scene_graph.py   ←   ObjectExistence, RelationVerification, RelationSet
│   │   ├── temporal.py      ←   ActionTemporalGenerator
│   │   ├── compositional.py ←   CompositionalGenerator
│   │   ├── grounding.py     ←   GroundingGenerator
│   │   ├── original_qa.py   ←   OriginalQAGenerator
│   │   └── attributes.py    ←   AttributeGenerator
│   ├── validation/
│   │   └── validators.py    ← Extend validate_example() here
│   ├── datasets/
│   │   ├── base.py          ← BaseDatasetAdapter ABC
│   │   ├── registry.py      ← dataset_registry singleton
│   │   └── agqa/            ← Existing AGQA adapter
│   ├── generators/
│   │   └── renderer.py      ← Text template rendering engine
│   ├── templates.py         ← All query-text template strings
│   └── reasoning/
│       └── semantic_engine.py ← AGQA program evaluator
└── main_pipeline.py         ← Register new generators here
```

> [!NOTE]
> All randomness in generators **must** go through the `rng` argument
> (`random.Random` instance) passed into `generate()`.  Never use
> `random.random()`, `numpy.random`, or any global random state — doing so
> breaks reproducibility and makes test results non-deterministic.

---

## 2. Adding a New Generator

### 2a. Create a new file in `src/generators/`

Create `src/generators/my_generator.py`.  Name it after the task family it
covers (e.g. `causal.py`, `counting.py`).

### 2b. Inherit from `GeneratorBase`

```python
# src/generators/my_generator.py

"""MyGenerator — produces <describe task> proposition groups."""

import random
from typing import Iterator

from src.generators.base import GeneratorBase
from src.schema import (
    PropositionGroup, Proposition, PropositionSemantics,
    Reasoning, Provenance,
)
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools
```

### 2c. Implement `generate()`

The signature is fixed — match it exactly:

```python
class MyGenerator(GeneratorBase):
    """Generates <describe> proposition groups from a scene graph."""

    def generate(
        self,
        qid: str,
        qdata: dict,
        sg: NormalizedSceneGraph,
        split: str,
        rng: random.Random,
        pools: NegativePools,
    ) -> Iterator[PropositionGroup]:
        """Generate proposition groups for a single video.

        Args:
            qid: AGQA question ID (may be empty for SG-native generators).
            qdata: Raw question metadata dict from AGQA JSON.
            sg: Normalised scene graph for the current video.
            split: Dataset split — "train", "val", or "test".
            rng: Seeded random instance — use ONLY this for all sampling.
            pools: Pre-computed hard-negative pools for the current video.

        Yields:
            PropositionGroup: One benchmark example per yield.
        """
        # --- Guard: skip if the scene graph has no useful data ---
        if not sg.frames:
            return

        # --- Build your proposition(s) ---
        semantics = PropositionSemantics(
            canonical_type="object_existence",
            subject="person",
            predicate=None,
            object=None,
            polarity="positive",
        )
        prop_true = Proposition(
            text="The person appears in the video.",
            truth_state="TRUE",
            label=1,
            semantics=semantics,
        )
        prop_false = Proposition(
            text="A dog appears in the video.",
            truth_state="FALSE",
            label=0,
            semantics=PropositionSemantics(
                canonical_type="object_existence",
                subject="dog",
                polarity="positive",
            ),
        )

        propositions = [prop_true, prop_false]
        # Shuffle so the correct answer is not always first
        rng.shuffle(propositions)

        yield PropositionGroup(
            example_id=f"my_gen_{sg.video_id}_{qid}",
            source="agqa_v2",
            split=split,
            media_id=sg.video_id,
            query_text="Which of the following is true about this video?",
            query_template_id="obj_v_001",
            propositions=propositions,
            task_type="single_choice",
            num_propositions=len(propositions),
            reasoning=Reasoning(
                type="object_existence",
                complexity=1,
                family="object",
                hops=1,
            ),
            provenance=Provenance(
                source_question_id=qid,
                source_question_text=qdata.get("question"),
                source_program=qdata.get("program"),
                source_answer=qdata.get("answer"),
                scene_graph_id=sg.video_id,
                generation_rule="my_generator_object_existence",
                generation_seed=42,
                generator_family="scene_graph",
            ),
        )
```

### 2d. Build `PropositionGroup` with all required fields

Every `PropositionGroup` **must** supply:

| Field | Type | Notes |
|-------|------|-------|
| `example_id` | `str` | Globally unique; include video ID + generator tag |
| `source` | `str` | e.g. `"agqa_v2"` |
| `split` | `str` | Pass through the `split` argument verbatim |
| `media_id` | `str` | `sg.video_id` |
| `query_text` | `str \| None` | Natural-language prompt shown to the model |
| `query_template_id` | `str \| None` | Prefix must match reasoning type (see `validators.py`) |
| `propositions` | `List[Proposition]` | At least 2; exactly 3 for `three_way` tasks |
| `task_type` | `str` | One of `"binary"`, `"single_choice"`, `"three_way"`, `"multi_label"`, `"grounding"` |
| `num_propositions` | `int` | Must equal `len(propositions)` |
| `reasoning` | `Reasoning` | Complexity score (0–8), family, hops |
| `provenance` | `Provenance` | Requires at minimum `generation_rule` and `generation_seed` |

> [!IMPORTANT]
> For `"binary"`, `"single_choice"`, and `"three_way"` task types the
> validator (`validate_example`) enforces **exactly one** proposition with
> `label == 1`.  Build your proposition list accordingly.

### 2e. Register in `main_pipeline.py`

Open `main_pipeline.py` and import your generator at the top:

```python
from src.generators.my_generator import MyGenerator
```

Then add it to the `generators` list inside `main()`:

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
    ("my_gen",        MyGenerator()),   # ← Add your generator here
]
```

The pipeline's round-robin scheduler will automatically include your generator
with equal weighting alongside the existing eight.

---

## 3. Adding a New Validator Check

All validation logic lives in
[`src/validation/validators.py`](../src/validation/validators.py) inside the
`validate_example()` function.  The function runs a numbered series of checks
and appends human-readable error strings to an `errors` list.

To add a new check:

1. **Pick the next check number** (currently 1–10; your new check is 11).
2. **Add a comment block** following the existing style.
3. **Append an error string** to `errors` when the check fails.
4. **Never short-circuit** — always let all other checks run so the caller
   gets a complete diagnostic report.

### Example — Check 11: Maximum proposition text length

```python
# ------------------------------------------------------------------
# Check 11: Proposition text length.
# Each proposition text must be at most 512 characters.  Longer texts
# are almost certainly the result of a template rendering bug
# (e.g. an un-truncated list of objects concatenated into the text).
# ------------------------------------------------------------------
for p in group.propositions:
    if len(p.text) > 512:
        errors.append(f"Proposition text exceeds 512 characters ({len(p.text)})")
```

The function signature and return contract (`Tuple[bool, List[str]]`) must not
change — the pipeline expects exactly this interface.

---

## 4. Adding a New Dataset Adapter

### 4a. Inherit from `BaseDatasetAdapter`

```python
# src/datasets/my_dataset/adapter.py

from src.datasets.base import BaseDatasetAdapter
from src.datasets.normalized import NormalizedQA


class MyDatasetAdapter(BaseDatasetAdapter):
    """Adapter for MyDataset QA files."""

    def __init__(self, config: dict) -> None:
        """Initialise paths from the configuration block.

        Args:
            config: Pipeline YAML config dict.  Expected keys under
                ``config["source"]``: ``qa_path`` (str).
        """
        self.qa_path = config["source"]["qa_path"]

    def load_records(self):
        """Stream QA records from the source file.

        Yields:
            NormalizedQA: One normalised record per source QA pair.
        """
        with open(self.qa_path) as fh:
            for line in fh:
                row = parse_line(line)         # your parsing logic
                yield NormalizedQA(
                    question_id=row["id"],
                    video_id=row["video_id"],
                    question=row["question"],
                    answer=row["answer"],
                    program=row.get("program"),
                )
```

### 4b. Register with `dataset_registry`

Create `src/datasets/my_dataset/__init__.py`:

```python
from src.datasets.registry import dataset_registry
from src.datasets.my_dataset.adapter import MyDatasetAdapter

dataset_registry.register("my_dataset", MyDatasetAdapter)
```

### 4c. Use the adapter

Anywhere the pipeline resolves a dataset by name:

```python
from src.datasets.registry import dataset_registry

AdapterClass = dataset_registry.get_adapter("my_dataset")
adapter = AdapterClass(config)
for record in adapter.load_records():
    process(record)
```

---

## 5. Rendering New Text Templates

### Adding templates

All query-text template strings are stored in
[`src/templates.py`](../src/templates.py) as module-level dictionaries keyed
by `query_template_id`.  To add a new template:

```python
# src/templates.py  (add to the appropriate section)

OBJECT_VERIFICATION_TEMPLATES = {
    # ... existing templates ...
    "obj_v_015": "Does the video show a {subject}?",
    "obj_v_016": "Is there a {subject} present at any point in this video?",
}
```

### Rendering templates

The renderer in [`src/generators/renderer.py`](../src/generators/renderer.py)
fills template placeholders using standard Python `.format()` semantics.  Call
it from your generator:

```python
from src.generators.renderer import render_template

query_text = render_template(
    template_id="obj_v_015",
    subject="person",
)
# → "Does the video show a person?"
```

> [!TIP]
> Use the `query_template_id` prefix convention consistently so that the
> validator (Check 4) can confirm template–task-type compatibility:
>
> | Prefix | Reasoning family | Task type |
> |--------|-----------------|-----------|
> | `obj_v` | `object` | `binary`, `three_way` |
> | `obj_s` | `object` | `single_choice` |
> | `obj_m` | `object` | `multi_label` |
> | `rel_v` | `spatial / contact / attention` | `binary`, `three_way` |
> | `temp`  | `temporal` | any |
> | `grnd_s` | `grounding` | `single_choice` |
> | `grnd_v` | `grounding` | `binary`, `three_way` |

---

## 6. Running Tests

Run the full suite with verbose output:

```bash
pytest tests/ -v
```

Run a single test file:

```bash
pytest tests/test_validators.py -v
```

Run a single test function:

```bash
pytest tests/test_validators.py::test_duplicate_texts -v
```

Run only tests matching a keyword:

```bash
pytest tests/ -k "temporal" -v
```

Stop on the first failure:

```bash
pytest tests/ -x -v
```

---

## 7. Test Structure

| File | What it covers |
|------|---------------|
| [`tests/test_configuration.py`](../tests/test_configuration.py) | Asserts that constants in `src/constants.py` and `main_pipeline.py` have the expected values (e.g. `SEED == 42`, `SHARD_SIZE == 50000`).  Also validates that `TASK_TYPES`, `COMPLEXITY_LEVELS`, and other config enumerations are non-empty and contain the required entries. |
| [`tests/test_optionizers.py`](../tests/test_optionizers.py) | Unit tests for the candidate-generation helpers in `src/candidates/`.  Verifies that each optionizer module produces the correct number and type of distractors for a range of synthetic scene graphs. |
| [`tests/test_new_optionizers.py`](../tests/test_new_optionizers.py) | Extended optionizer tests added after the initial release.  Covers edge cases such as empty scene graphs, single-object videos, and videos with no relations, ensuring optionizers degrade gracefully instead of raising exceptions. |
| [`tests/test_validators.py`](../tests/test_validators.py) | Exercises all **10 validation checks** in `validate_example()`.  Each check has at least one *passing* test (valid input → no errors) and one *failing* test (deliberately broken input → specific error string).  The test file constructs minimal `PropositionGroup` fixtures inline rather than loading real data, so it runs without the AGQA source files. |

---

## 8. Code Style

UniProp uses **Google-style docstrings** throughout.

### Docstring format

```python
def my_function(arg1: str, arg2: int = 0) -> bool:
    """One-line summary in imperative mood.

    Optional extended description explaining non-obvious behaviour,
    design decisions, or important edge cases.

    Args:
        arg1: Description of the first argument.
        arg2: Description of the second argument.  Defaults to ``0``.

    Returns:
        ``True`` if the condition holds; ``False`` otherwise.

    Raises:
        ValueError: If ``arg1`` is an empty string.

    Example::

        result = my_function("hello", arg2=3)
        assert result is True
    """
```

### Type hints

All **public** functions, methods, and class attributes must carry full type
annotations.  Use `Optional[X]` instead of `X | None` for Python 3.10
compatibility.

### Determinism rule

```python
# ✅ Correct — use the rng argument
choice = rng.choice(candidates)

# ❌ Wrong — breaks determinism
choice = random.choice(candidates)
import numpy as np; choice = np.random.choice(candidates)
```

### Schema changes

If you change the PyArrow schema in `src/storage/parquet_writer.py`:

1. Update `DATA_SCHEMA.md` to document the new columns.
2. Bump the **major** version number in `DATA_SCHEMA.md`.
3. Delete existing Parquet shards before regenerating — old shards will be
   incompatible with the new schema.

---

## 9. Debugging Tips

### Run a single generator in isolation

You can instantiate and call any generator directly from a Python script or
interactive session — no pipeline harness required.

```python
import random, pickle
from src.normalization.scene_graph_normalizer import normalize_video_sg
from src.negatives.hard_negatives import NegativePools
from src.generators.scene_graph import RelationVerificationGenerator

# Load one video's raw scene graph
with open("AGQA_scene_graphs/AGQA_train_stsgs.pkl", "rb") as f:
    train_sgs = pickle.load(f)

video_id = list(train_sgs.keys())[0]
raw_sg   = train_sgs[video_id]
sg       = normalize_video_sg(video_id, raw_sg)
pools    = NegativePools(sg)
rng      = random.Random(42)

gen = RelationVerificationGenerator()
for group in gen.generate("qid_001", {}, sg, "train", rng, pools):
    print(group.example_id, group.task_type)
    for p in group.propositions:
        print(f"  [{p.label}] {p.text}")
    break   # inspect the first result only
```

### Inspect a `PropositionGroup` as a dictionary

`PropositionGroup.to_dict()` recursively serialises the entire object
(including all nested dataclasses) to a plain Python dictionary:

```python
import json
d = group.to_dict()
print(json.dumps(d, indent=2))
```

### Inspect generated Parquet shards with Polars

```python
import polars as pl

df = pl.read_parquet("data/generated/train/part-000000.parquet")
print(df.schema)
print(df.head(5))

# Filter to a specific reasoning family
print(df.filter(pl.col("reasoning").struct.field("family") == "temporal").shape)
```

### Inspect generated Parquet shards with DuckDB

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW train AS SELECT * FROM read_parquet('data/generated/train/*.parquet')")

# Count examples by task type
print(con.execute("SELECT task_type, COUNT(*) FROM train GROUP BY task_type").df())

# Inspect a single row's propositions
row = con.execute("SELECT propositions FROM train LIMIT 1").fetchone()[0]
print(row)
```

### Check a specific validation error

Run `validate_example` directly on any group to see exactly which checks fail:

```python
from src.validation.validators import validate_example

ok, errors = validate_example(group)
if not ok:
    for err in errors:
        print("VALIDATION ERROR:", err)
```

---

## 10. Reasoning Engine

The `SemanticEngine` in
[`src/reasoning/semantic_engine.py`](../src/reasoning/semantic_engine.py)
evaluates AGQA functional programs (AST tuples) against a
`NormalizedSceneGraph` and returns a typed `SemanticValue`.

### How the engine works

1. **Parse** the raw AGQA program string into an AST using
   `src.reasoning.program_parser.parse_agqa_program`.
2. **Dispatch** — `engine.evaluate(ast)` matches the top-level function name
   to an `eval_<funcname>` method (e.g. `eval_filter`, `eval_exists`,
   `eval_localize`).
3. **Query the scene graph** — each `eval_*` method reads from `engine.sg`
   (the `NormalizedSceneGraph`) to retrieve objects, relations, actions,
   and temporal intervals.
4. **Return** a `SemanticValue` whose `.type` is one of `BOOLEAN`,
   `OBJECT_SET`, `ACTION_SET`, `FRAME_SET`, or `UNKNOWN`.

### Testing the engine against a raw AGQA program

```python
import pickle
from src.normalization.scene_graph_normalizer import normalize_video_sg
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.semantic_types import SemanticType

# 1. Load a real scene graph
with open("AGQA_scene_graphs/AGQA_train_stsgs.pkl", "rb") as f:
    train_sgs = pickle.load(f)

video_id = list(train_sgs.keys())[0]
sg       = normalize_video_sg(video_id, train_sgs[video_id])

# 2. Parse a program string (copy from an AGQA QA record)
program_str = "Exists(person, Filter(video, [objects]))"
ast         = parse_agqa_program(program_str)

# 3. Evaluate
engine = SemanticEngine(scene_graph=sg)
result = engine.evaluate(ast)

print(f"Type  : {result.type}")   # e.g. SemanticType.BOOLEAN
print(f"Value : {result.value}")  # e.g. True

assert result.type == SemanticType.BOOLEAN
```

### Supported AGQA operators

| Operator | Category | Returns |
|----------|----------|---------|
| `And`, `Or`, `Xor`, `Not` | Boolean composition | `BOOLEAN` |
| `Subtract` | Numeric | `UNKNOWN` (float) |
| `Exists` | Membership | `BOOLEAN` |
| `Query` | Attribute retrieval | varies |
| `OnlyItem` | Pass-through | unchanged |
| `Filter` | Set construction | `OBJECT_SET` or `ACTION_SET` |
| `Iterate` | Temporal filter | `OBJECT_SET` or `ACTION_SET` |
| `Localize` | Temporal window | `FRAME_SET` |
| `Compare` | Comparison / selection | `UNKNOWN` (string) |
| `Superlative` | Ranking | `UNKNOWN` (string) |
| `Equals` | Equality test | `BOOLEAN` |
| `Choose` | Conditional selection | *(not yet implemented)* |

> [!NOTE]
> `EvaluatorUnsupportedError` is raised for unrecognised operators or
> mis-formed argument lists.  `MissingEvidenceError` is raised when the
> scene graph lacks the data needed to answer a sub-expression (e.g. a
> `Localize` for an action that does not appear in the video).  The
> `OriginalQAGenerator` catches both exceptions and silently skips the
> question rather than crashing the pipeline.
