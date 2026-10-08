# UniProp — Data Schema

> [!NOTE]
> All generated output is stored as **Apache Parquet** files using a fixed,
> strongly-typed **PyArrow schema**. Every shard produced by the pipeline is
> schema-identical. This document covers (1) the Parquet schema, (2) the
> internal Python dataclass hierarchy, and (3) invariants and example records.

---

## 1. Overview

Each row in a UniProp Parquet shard represents one **proposition group** —
a single evaluation example that pairs a video with a natural-language query
and a set of candidate propositions.

- **Schema enforcement**: applied at write time by
  [`src/storage/parquet_writer.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/storage/parquet_writer.py)
  via `pa.Table.from_pydict(data, schema=schema)`.
- **Column count**: 11 columns (no nested structs in the Parquet layer —
  all proposition details are flattened into parallel list columns).
- **Shard size**: up to `SHARD_SIZE = 50 000` rows per file.

---

## 2. Parquet Schema — Column Reference

The table below documents every column in the order it is declared in the
PyArrow schema.

| # | Column | PyArrow Type | Nullable | Description |
|---|---|---|---|---|
| 1 | `example_id` | `pa.string()` | No | Globally unique identifier for this example across all sources and splits (e.g. `"agqa_v2_train_00012345"`). Populated from `PropositionGroup.example_id`. |
| 2 | `split` | `pa.string()` | No | Dataset partition: `"train"`, `"val"`, or `"test"`. Derived from the source AGQA QA file and the 90/10 train/val split. |
| 3 | `video_id` | `pa.string()` | No | Charades video identifier (e.g. `"ABCDE"`). Non-null, non-empty — asserted before write. Populated from `PropositionGroup.media_id`. |
| 4 | `query` | `pa.string()` | No | Natural-language task prompt presented alongside the propositions (e.g. `"Which objects is the person interacting with?"`). Empty string `""` when the group has no query text. |
| 5 | `options` | `pa.list_(pa.string())` | No | Ordered list of candidate proposition surface-form strings. Each element corresponds to one `Proposition.text`. |
| 6 | `labels` | `pa.list_(pa.int8())` | No | Parallel list of integer labels aligned one-to-one with `options`. Values: `1` = TRUE, `0` = FALSE, `-1` = UNKNOWN/missing (sentinel for `None` labels). |
| 7 | `truth_state` | `pa.string()` | No | Coarse truth label of the **first** proposition in the group: `"TRUE"`, `"FALSE"`, or `"UNKNOWN"`. Defaults to `"UNKNOWN"` for empty groups (should not occur in valid data). |
| 8 | `option_count` | `pa.int32()` | No | Total number of candidate propositions. Equal to `len(options)` and `len(labels)`. Stored explicitly for fast filtering without loading list columns. |
| 9 | `task_type` | `pa.string()` | No | Evaluation protocol for this example. One of `"binary"`, `"single_choice"`, `"multi_label"`, `"open_answer_derived"`, `"grounding"`, `"temporal"`, `"three_way"`. |
| 10 | `reasoning_family` | `pa.string()` | No | Broad reasoning taxonomy family. One of `"object"`, `"attribute"`, `"spatial"`, `"attention"`, `"contact"`, `"action"`, `"temporal"`, `"compositional"`, `"grounding"`. Drawn from `PropositionGroup.reasoning.family`. |
| 11 | `generator_family` | `pa.string()` | No | Provenance tag for the generator that produced this example (e.g. `"scene_graph"`, `"question_derived"`, `"temporal_chain"`). Drawn from `PropositionGroup.provenance.generator_family`. |

### 2.1 Column Detail: `options` and `labels`

`options` and `labels` are **parallel lists**: `options[i]` is the surface form
of the proposition whose truth label is `labels[i]`.

```
options = ["The person is holding a book.",
           "The person is not holding a book.",
           "The person is carrying a book."]

labels  = [1, 0, 0]
```

The `-1` sentinel in `labels` encodes `Proposition.label = None`, which arises
for `UNKNOWN` propositions in task types that do not use a three-valued label
scheme.

### 2.2 Column Detail: `task_type`

| Value | Semantics |
|---|---|
| `"binary"` | Single yes/no proposition; exactly 1 correct label |
| `"single_choice"` | Multiple options; exactly 1 correct (mutually exclusive) |
| `"multi_label"` | Multiple options; multiple may be correct simultaneously |
| `"three_way"` | Exactly 3 options; exactly 1 correct |
| `"open_answer_derived"` | Open-ended answer derived from a source AGQA QA pair |
| `"grounding"` | Model must localise an entity in the video frame |
| `"temporal"` | Model must reason about event ordering over time |

### 2.3 Column Detail: `reasoning_family`

| Value | Generator(s) | Semantic domain |
|---|---|---|
| `"object"` | `obj_exist`, `orig_qa` | Object existence and identity |
| `"attribute"` | `attributes` | Object attributes (colour, size, state) |
| `"spatial"` | `rel_ver`, `rel_set` | Spatial relations (above, behind, …) |
| `"attention"` | `rel_ver`, `rel_set` | Gaze / attention relations |
| `"contact"` | `rel_ver`, `rel_set` | Physical contact relations (holding, sitting on, …) |
| `"action"` | `orig_qa`, `temporal` | Action recognition |
| `"temporal"` | `temporal` | Event ordering (before, after, during) |
| `"compositional"` | `compositional` | Multi-hop, multi-relation chains |
| `"grounding"` | `grounding` | Spatial localisation of an entity |

---

## 3. Internal Python Schema — `PropositionGroup` Dataclass Hierarchy

The Python layer uses a hierarchy of `dataclasses` defined in
[`src/schema.py`](file:///c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp/src/schema.py).
These are the **source of truth** for all pipeline stages; the Parquet columns
are a flattened projection of this hierarchy.

```
PropositionGroup
├── example_id:          str
├── source:              str
├── split:               str
├── media_id:            str
├── query_text:          Optional[str]
├── query_template_id:   Optional[str]
├── propositions:        List[Proposition]
│   └── Proposition
│       ├── text:          str
│       ├── truth_state:   str       ("TRUE" | "FALSE" | "UNKNOWN")
│       ├── label:         Optional[int]
│       └── semantics:     PropositionSemantics
│           ├── canonical_type:    str
│           ├── subject:           Optional[str]
│           ├── predicate:         Optional[str]
│           ├── object:            Optional[str]
│           ├── polarity:          str   ("positive" | "negative")
│           ├── temporal_relation: Optional[str]
│           ├── event_a:           Optional[str]
│           └── event_b:           Optional[str]
├── task_type:           str
├── num_propositions:    int
├── reasoning:           Reasoning
│   ├── type:       str
│   ├── complexity: int   (0–8)
│   ├── family:     str
│   └── hops:       int
├── provenance:          Provenance
│   ├── source_question_id:    Optional[str]
│   ├── source_question_text:  Optional[str]
│   ├── source_program:        Optional[str]
│   ├── source_answer:         Optional[str]
│   ├── scene_graph_id:        Optional[str]
│   ├── generation_rule:       str
│   ├── generation_seed:       int
│   ├── generator_family:      str
│   ├── evidence_object_ids:   List[str]
│   ├── evidence_relation_ids: List[str]
│   ├── frame_ids:             List[str]
│   ├── temporal_chain:        Optional[List[dict]]
│   ├── source_global:         Optional[str]
│   ├── source_local:          Optional[str]
│   ├── source_semantic:       Optional[str]
│   ├── source_structural:     Optional[str]
│   └── source_sg_grounding:   Optional[str]
└── grounding:           Optional[Grounding]
    ├── object_ids:            List[str]
    ├── boxes:                 Dict[str, Tuple[float,float,float,float]]
    ├── frame_ids:             List[str]
    ├── target_object_id:      Optional[str]
    ├── target_object_class:   Optional[str]
    └── target_bbox:           Optional[Tuple[float,float,float,float]]
```

---

## 4. Proposition Fields

### 4.1 `Proposition`

| Field | Type | Description | Constraints |
|---|---|---|---|
| `text` | `str` | Surface-form natural-language statement shown to the model (e.g. `"The person is holding a book."`). | Non-empty; unique within its group. |
| `truth_state` | `str` | Coarse gold label. | Must be `"TRUE"`, `"FALSE"`, or `"UNKNOWN"`. |
| `label` | `Optional[int]` | Integer encoding of `truth_state` for binary tasks. | `1` = TRUE, `0` = FALSE, `None` for UNKNOWN (stored as `-1` in Parquet). |
| `semantics` | `PropositionSemantics` | Structured decomposition; see §4.2. | Always present. |

### 4.2 `PropositionSemantics`

| Field | Type | Description | Example values |
|---|---|---|---|
| `canonical_type` | `str` | High-level semantic category. | `"object_existence"`, `"relation"`, `"action"`, `"temporal"`, `"grounding"`, `"compositional_and"`, `"logical_none"` |
| `subject` | `Optional[str]` | Primary entity (usually `"person"`). | `"person"`, `"bag"`, `None` |
| `predicate` | `Optional[str]` | Relation or action label. | `"holding"`, `"sitting on"`, `None` |
| `object` | `Optional[str]` | Secondary entity. | `"chair"`, `"book"`, `None` |
| `polarity` | `str` | Assertion polarity. | `"positive"` (default), `"negative"` |
| `temporal_relation` | `Optional[str]` | Ordering predicate for temporal propositions. | `"before"`, `"after"`, `"during"`, `None` |
| `event_a` | `Optional[str]` | First event in a temporal proposition. | `"holding a cup"`, `None` |
| `event_b` | `Optional[str]` | Second event in a temporal proposition. | `"sitting on a chair"`, `None` |

**`canonical_type` vocabulary:**

| Value | Meaning |
|---|---|
| `"object_existence"` | Asserts presence or absence of an object |
| `"relation"` | Asserts a spatial, contact, or attention relation |
| `"action"` | Asserts that a person performs an action |
| `"temporal"` | Asserts an event-ordering relation |
| `"grounding"` | Links a description to a spatial region or object ID |
| `"compositional_and"` | Conjunction of two or more relations (predicate = `"rel1 and rel2"`) |
| `"logical_none"` | Special "none of the above" distractor option |

---

## 5. Invariants

The following invariants are checked by `validate_example()` and are guaranteed
to hold for every row in the output dataset:

```
len(options) == len(labels) == option_count
```

> [!IMPORTANT]
> `options[i]` and `labels[i]` are always aligned by index. Never reorder one
> without reordering the other.

```
# Label value domain
labels[i] ∈ {-1, 0, 1}   (int8)
  -1  →  UNKNOWN / missing (Proposition.label == None)
   0  →  FALSE
   1  →  TRUE

# Label cardinality by task type
task_type == "binary"        → sum(l for l in labels if l != -1) == 1
task_type == "single_choice" → sum(l for l in labels if l != -1) == 1
task_type == "three_way"     → sum(l for l in labels if l != -1) == 1
                               AND option_count == 3
task_type == "multi_label"   → sum(l for l in labels if l != -1) >= 1
                               (no upper bound on correct count)

# Proposition count range
task_type != "three_way"  →  2 ≤ option_count ≤ 40
task_type == "three_way"  →  option_count == 3

# Option text uniqueness
len(set(options)) == len(options)

# truth_state domain
truth_state ∈ {"TRUE", "FALSE", "UNKNOWN"}

# video_id
video_id is not None and video_id.strip() != ""
```

---

## 6. Example Record

The following is a complete example record as it would appear in a Parquet row,
serialised to JSON for readability.

```json
{
  "example_id":        "agqa_v2_train_rel_ver_000042",
  "split":             "train",
  "video_id":          "ABCDE",
  "query":             "Is the person holding the book?",
  "options": [
    "The person is holding a book.",
    "The person is not looking at the book.",
    "The person is sitting on the book."
  ],
  "labels":            [1, 0, 0],
  "truth_state":       "TRUE",
  "option_count":      3,
  "task_type":         "single_choice",
  "reasoning_family":  "contact",
  "generator_family":  "scene_graph"
}
```

**Internal `PropositionGroup` (Python) that produced this row:**

```json
{
  "example_id":        "agqa_v2_train_rel_ver_000042",
  "source":            "agqa_v2",
  "split":             "train",
  "media_id":          "ABCDE",
  "query_text":        "Is the person holding the book?",
  "query_template_id": "rel_v_contact_01",
  "task_type":         "single_choice",
  "num_propositions":  3,
  "propositions": [
    {
      "text":        "The person is holding a book.",
      "truth_state": "TRUE",
      "label":       1,
      "semantics": {
        "canonical_type":    "relation",
        "subject":           "person",
        "predicate":         "holding",
        "object":            "book",
        "polarity":          "positive",
        "temporal_relation": null,
        "event_a":           null,
        "event_b":           null
      }
    },
    {
      "text":        "The person is not looking at the book.",
      "truth_state": "FALSE",
      "label":       0,
      "semantics": {
        "canonical_type":    "relation",
        "subject":           "person",
        "predicate":         "not looking at",
        "object":            "book",
        "polarity":          "negative",
        "temporal_relation": null,
        "event_a":           null,
        "event_b":           null
      }
    },
    {
      "text":        "The person is sitting on the book.",
      "truth_state": "FALSE",
      "label":       0,
      "semantics": {
        "canonical_type":    "relation",
        "subject":           "person",
        "predicate":         "sitting on",
        "object":            "book",
        "polarity":          "positive",
        "temporal_relation": null,
        "event_a":           null,
        "event_b":           null
      }
    }
  ],
  "reasoning": {
    "type":       "object_relation",
    "complexity": 2,
    "family":     "contact",
    "hops":       1
  },
  "provenance": {
    "source_question_id":    "Q00042",
    "source_question_text":  "What is the person holding?",
    "source_program":        "query(subject=person, relation=holding)",
    "source_answer":         "book",
    "scene_graph_id":        "ABCDE_frame_000023",
    "generation_rule":       "positive_relation_from_sg_edge",
    "generation_seed":       42,
    "generator_family":      "scene_graph",
    "evidence_object_ids":   ["o23"],
    "evidence_relation_ids": ["r15"],
    "frame_ids":             ["000023", "000031"],
    "temporal_chain":        null,
    "source_global":         null,
    "source_local":          "frame_000023",
    "source_semantic":       "agqa_compositional",
    "source_structural":     "scene_graph_edge",
    "source_sg_grounding":   null
  },
  "grounding": null
}
```

---

## 7. Provenance Fields

The `Provenance` dataclass (stored in `PropositionGroup.provenance`) provides
a complete **audit trail** for every generated example.

| Field | Type | Description |
|---|---|---|
| `source_question_id` | `Optional[str]` | Unique ID of the original AGQA question from which this group was derived. `None` for purely scene-graph-generated examples. |
| `source_question_text` | `Optional[str]` | Verbatim text of the original AGQA question. |
| `source_program` | `Optional[str]` | Functional program in AGQA's DSL associated with the source question. |
| `source_answer` | `Optional[str]` | Gold answer string to the original AGQA question (e.g. `"yes"`, `"holding"`). |
| `scene_graph_id` | `Optional[str]` | Identifier of the Action Genome scene graph frame or segment used as evidence. |
| `generation_rule` | `str` | Human-readable name of the rule/template used to generate this example (e.g. `"positive_relation_from_sg_edge"`). |
| `generation_seed` | `int` | Integer random seed (always `SEED = 42`) for full reproducibility of sampling steps. |
| `generator_family` | `str` | Broad generator category: `"scene_graph"`, `"question_derived"`, or `"temporal_chain"`. |
| `evidence_object_ids` | `List[str]` | Action Genome object node IDs that directly support the truth value (e.g. `["o23"]`). |
| `evidence_relation_ids` | `List[str]` | Action Genome relation/edge IDs that directly support the truth value (e.g. `["r15"]`). |
| `frame_ids` | `List[str]` | Frame IDs from which scene-graph evidence was drawn (e.g. `["000023", "000031"]`). |
| `temporal_chain` | `Optional[List[dict]]` | For temporally derived propositions, an ordered list of dicts describing each event (e.g. `[{"action": "holding bag", "frame": "23"}]`). `None` otherwise. |
| `source_global` | `Optional[str]` | Video-level source context identifier, if any. |
| `source_local` | `Optional[str]` | Clip- or frame-level source context identifier (e.g. `"frame_000023"`). |
| `source_semantic` | `Optional[str]` | Semantic source pathway tag (e.g. `"agqa_compositional"`). |
| `source_structural` | `Optional[str]` | Structural source pathway tag (e.g. `"scene_graph_edge"`). |
| `source_sg_grounding` | `Optional[str]` | Scene-graph grounding annotation ID, if applicable. |

---

## 8. Grounding Fields

The `Grounding` dataclass (stored in `PropositionGroup.grounding`) links a
proposition to its **spatial and temporal evidence** in the source video.
It is populated only for `reasoning_family == "grounding"` examples;
`PropositionGroup.grounding` is `None` for all other task types.

| Field | Type | Description |
|---|---|---|
| `object_ids` | `List[str]` | Ordered list of Action Genome object node IDs relevant to this proposition (e.g. `["o8", "o16"]`). |
| `boxes` | `Dict[str, Tuple[float,float,float,float]]` | Per-object bounding boxes. Key = object ID; value = `(x_min, y_min, x_max, y_max)`. |
| `frame_ids` | `List[str]` | Frame identifiers from which the grounding evidence is drawn. |
| `target_object_id` | `Optional[str]` | The single focal object ID for grounding tasks (the object the model must localise). `None` for non-grounding types. |
| `target_object_class` | `Optional[str]` | Human-readable class label of the focal object (e.g. `"chair"`). |
| `target_bbox` | `Optional[Tuple[float,float,float,float]]` | Bounding box of the focal object. `None` when no target is specified. |

### 8.1 Bounding Box Format

All bounding boxes are stored as 4-tuples `(x_min, y_min, x_max, y_max)`:

| Component | Symbol | Description |
|---|---|---|
| `x_min` | x₁ | Left edge of the bounding box, normalised to `[0, 1]` of frame width |
| `y_min` | y₁ | Top edge of the bounding box, normalised to `[0, 1]` of frame height |
| `x_max` | x₂ | Right edge of the bounding box, normalised to `[0, 1]` of frame width |
| `y_max` | y₂ | Bottom edge of the bounding box, normalised to `[0, 1]` of frame height |

> [!NOTE]
> Coordinates are in **pixel-space** as extracted from raw Action Genome
> annotations (not pre-normalised to `[0, 1]`). Applications that require
> normalised coordinates must divide `x` components by frame width and `y`
> components by frame height. Boxes with exactly 4 elements are stored as
> `Tuple[float, float, float, float]`; boxes with any other shape are stored
> as `None`.

**Example:**
```python
grounding = Grounding(
    object_ids=["o8", "o16"],
    boxes={
        "o8":  (142.0, 87.0, 310.0, 420.0),   # chair
        "o16": (0.0, 350.0, 640.0, 480.0),    # floor
    },
    frame_ids=["000023", "000031"],
    target_object_id="o8",
    target_object_class="chair",
    target_bbox=(142.0, 87.0, 310.0, 420.0),
)
```

---

## 9. Schema Mapping Summary

The table below shows how each Python dataclass field maps to its Parquet column.

| Parquet Column | Source | Python path |
|---|---|---|
| `example_id` | Direct | `PropositionGroup.example_id` |
| `split` | Direct | `PropositionGroup.split` |
| `video_id` | Direct | `PropositionGroup.media_id` (falls back to `.video_id`) |
| `query` | Direct | `PropositionGroup.query_text` or `""` |
| `options` | Derived | `[p.text for p in PropositionGroup.propositions]` |
| `labels` | Derived | `[p.label if p.label is not None else -1 for p in ...]` |
| `truth_state` | Derived | `PropositionGroup.propositions[0].truth_state` |
| `option_count` | Direct | `PropositionGroup.num_propositions` |
| `task_type` | Direct | `PropositionGroup.task_type` |
| `reasoning_family` | Derived | `PropositionGroup.reasoning.family` |
| `generator_family` | Derived | `PropositionGroup.provenance.generator_family` |

> [!CAUTION]
> `truth_state` in the Parquet layer reflects only the **first** proposition in
> the group. For groups with mixed truth states (e.g. multi-label tasks with
> both TRUE and FALSE propositions), use the `labels` column for per-option
> truth information. Do not use `truth_state` for multi-label evaluation logic.
