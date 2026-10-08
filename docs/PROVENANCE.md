# PROVENANCE

> **Every generated example is fully traceable to its raw source.**  
> The `Provenance` dataclass embedded in every `PropositionGroup` carries a complete
> audit trail — from the original AGQA question identifier down to the exact video
> frame that supplied the evidence.

---

## Table of Contents

1. [Overview](#overview)
2. [Provenance Fields Reference](#provenance-fields-reference)
3. [The `video_id` Mandate](#the-video_id-mandate)
4. [Evidence Signature — `get_signature()`](#evidence-signature--get_signature)
5. [Generation Rules Catalog](#generation-rules-catalog)
6. [Tracing an Example — Step-by-Step Walkthrough](#tracing-an-example--step-by-step-walkthrough)
7. [Provenance in the Schema Hierarchy](#provenance-in-the-schema-hierarchy)

---

## Overview

UniProp derives its benchmark examples from two raw AGQA artefacts:

| Artefact | File | Role |
|---|---|---|
| Balanced QA pairs | `AGQA_balanced/train_balanced.txt`, `test_balanced.txt` | Provides question text, functional programs, and answers |
| Spatio-temporal scene graphs (STSGs) | `AGQA_scene_graphs/AGQA_train_stsgs.pkl`, `AGQA_test_stsgs.pkl` | Provides object, relation, action, and bounding-box evidence |

Each `PropositionGroup` that the pipeline emits carries a `Provenance` instance
that records *which* raw source material was used, *how* it was transformed, and
*what random state* was active at the moment of generation.  This tri-layer audit
trail enables:

- **Reproducibility**: Given the same source files and seed, every example can be
  regenerated bit-for-bit identically.
- **Error analysis**: A mislabelled example can be traced directly back to the
  scene-graph frame or question that caused the error.
- **Ablation studies**: Researchers can filter examples by `generation_rule` or
  `generator_family` to evaluate models on specific sub-tasks.
- **Data versioning**: As AGQA releases update, `source_question_id` links each
  example to its exact upstream entry.

---

## Provenance Fields Reference

The `Provenance` dataclass is defined in
[`src/schema.py`](../src/schema.py).  Every field is documented below.

### Source Question Fields

These fields are populated when the example was derived from an AGQA QA pair.
For purely scene-graph-generated examples (e.g. grounding tasks), all four are
`None`.

---

#### `source_question_id: Optional[str]`

The unique question identifier from the AGQA balanced QA file — the string key
used to look up this question in `AGQA_balanced/train_balanced.txt` or
`test_balanced.txt`.

**Example**: `"7AAAA_1"`

**`None` when**: The example was generated entirely from scene-graph evidence
without referencing any specific AGQA question (typical for `ObjectExistenceGenerator`,
`RelationVerificationGenerator`, `GroundingGenerator`, `CompositionalGenerator`,
and `ActionTemporalGenerator`).

---

#### `source_question_text: Optional[str]`

Verbatim text of the original AGQA question, exactly as it appears in the source
file.

**Example**: `"Did the person wash dishes before taking a phone?"`

**`None` when**: `source_question_id` is `None`.

---

#### `source_program: Optional[str]`

The AGQA functional-program string (DSL expression) associated with the source
question.  AGQA programs encode the compositional reasoning steps needed to answer
the question — e.g. `"temporal_before(action_filter(wash dishes), action_filter(take phone))"`.

**Example**: `"temporal_before(action_filter(washing dishes), action_filter(taking a phone))"`

**`None` when**: `source_question_id` is `None`, or the source question record
does not include a program field.

---

#### `source_answer: Optional[str]`

The gold-standard string answer to the original AGQA question.

**Examples**: `"yes"`, `"no"`, `"holding"`, `"before"`

**`None` when**: `source_question_id` is `None`.

---

### Scene Graph Fields

#### `scene_graph_id: Optional[str]`

The `video_id` of the Action Genome / Charades video whose spatio-temporal scene
graph supplied the evidence for this proposition.  This is the **canonical
provenance anchor** (see [The `video_id` Mandate](#the-video_id-mandate) below).

**Example**: `"46GP8"`

**`None` when**: No scene-graph evidence was required to generate the example
(extremely rare; occurs only in `OriginalQAGenerator` when no scene-graph lookup
is needed).

> **Note**: In practice `scene_graph_id` is always set and equals `sg.video_id`.
> The pipeline asserts this equivalence before writing to Parquet.

---

### Generation Metadata Fields

#### `generation_rule: str`

A short, human-readable snake_case identifier naming the specific rule or branch
within a generator that produced this example.  Always set; never `None`.

See the [Generation Rules Catalog](#generation-rules-catalog) for the full list
of possible values.

**Example**: `"sg_obj_exist_binary"`

---

#### `generation_seed: int`

The integer value of `rng.randint(0, 2**32 - 1)` that was drawn immediately
before the `PropositionGroup` was constructed.  Storing this value makes the
exact sampling state recoverable — given `SEED=42`, the same seed will be drawn
at the same position in the RNG stream.

**Example**: `2947183621`

---

#### `generator_family: str`

A broad category label grouping related generators together.  Used by the
pipeline's `stats["generator_families"]` counter for high-level coverage
reporting.

**Default**: `"scene_graph"`

**All observed values**:

| Value | Generators that emit it |
|---|---|
| `"object_existence"` | `ObjectExistenceGenerator` |
| `"relation_verification"` | `RelationVerificationGenerator` (binary, sc, 3w) |
| `"relation_sets"` | `RelationVerificationGenerator` (multi-label branch) |
| `"action_temporal"` | `ActionTemporalGenerator` (both action and temporal sub-tasks) |
| `"compositional"` | `CompositionalGenerator` |
| `"grounding"` | `GroundingGenerator` |
| `"scene_graph"` | `OriginalQAGenerator` and attribute generators |

---

### Evidence Fields

These three fields together form the **evidence signature** used for deduplication
and coverage analysis.

#### `evidence_object_ids: List[str]`

Object class IDs (not instance IDs) from the Action Genome ontology that directly
support the truth value of the proposition.  For object-existence tasks this is
the set of objects verified.  For relation tasks it contains the single object
node involved in the verified relation.

**Example**: `["bag", "chair"]`

**Empty list `[]`** when: The generator did not associate specific object evidence
(e.g. some `act_bin` examples where only action phrases are relevant).

---

#### `evidence_relation_ids: List[str]`

Relation class IDs from the Action Genome ontology that directly support the
truth value.  For relation-verification tasks this holds the `rel_class` token
of the confirmed relation.  For compositional AND-tasks it holds both `r1` and `r2`.

**Example**: `["holding", "looking_at"]`

**Empty list `[]`** when: The generator did not use specific relation evidence
(e.g. pure object-existence or action tasks).

---

#### `frame_ids: List[str]`

Frame identifiers (video frame indices as strings) from which the scene-graph
evidence was drawn.  For grounding tasks this is the specific frame whose
bounding box was selected.  For non-grounding tasks this may be empty if the
evidence spans the full video rather than a single frame.

**Example**: `["023", "047"]`

**Empty list `[]`** when: Evidence was not tied to specific frames.

---

#### `temporal_chain: Optional[List[dict]]`

For **multihop temporal** propositions (`hops=2`), an ordered list of dicts
describing each explicit reasoning step in the chain.  Each dict has the form:

```python
{
    "event1": str,      # Description of the earlier event
    "relation": str,    # Always "before" in the current implementation
    "event2": str,      # Description of the later event
}
```

**Example** (2-hop chain: washing dishes → taking phone → watching TV):

```python
[
    {"event1": "washing dishes",  "relation": "before", "event2": "taking a phone"},
    {"event1": "taking a phone",  "relation": "before", "event2": "watching television"},
]
```

The **surface proposition** spans the outer pair (`washing dishes` before
`watching television`), while the chain records the intermediate event
(`taking a phone`) that makes the ordering derivable via transitivity.

**`None`** for all non-temporal propositions and for 1-hop temporal propositions.

---

### Optional Pathway Tags

These fields provide fine-grained source attribution for advanced analyses and
are `None` in most examples.

| Field | Type | Description |
|---|---|---|
| `source_global` | `Optional[str]` | Video-level source context identifier |
| `source_local` | `Optional[str]` | Clip- or frame-level source context |
| `source_semantic` | `Optional[str]` | Semantic pathway tag (e.g. `"agqa_compositional"`) |
| `source_structural` | `Optional[str]` | Structural pathway tag (e.g. `"scene_graph_edge"`) |
| `source_sg_grounding` | `Optional[str]` | Link to a specific scene-graph grounding annotation |

---

## The `video_id` Mandate

> **`video_id` is the canonical provenance anchor.  `example_id` alone is
> insufficient to recover the source video.**

### Why `example_id` Is Not Enough

`example_id` is a synthetic string assembled at generation time:

```
example_id = f"{rule_prefix}_{sg.video_id}_{seed}"
# e.g. "obj_ex_bin_46GP8_2947183621"
```

While the `video_id` is embedded in the `example_id` by convention, this
embedding is a convenience and **must not be relied upon** as a parsing target:

1. The format could change across pipeline versions without breaking the schema.
2. Parsing string IDs is fragile — a `video_id` that contains underscores
   would break a naive split on `"_"`.
3. Example IDs are not globally unique across dataset versions; two examples with
   the same `video_id` and `seed` could theoretically collide if the rule prefix
   changes.

### The Correct Way to Recover Source Video

Always use the **explicit `video_id` column** in the Parquet output:

```python
import pyarrow.parquet as pq

table = pq.read_table("data/generated/train/part-000000.parquet")
df = table.to_pandas()

# ✅ Correct: use the provenance field
df["video_id"] = df["provenance"].apply(lambda p: p["scene_graph_id"])

# ❌ Wrong: parsing example_id is fragile
df["video_id"] = df["example_id"].str.split("_").str[-2]  # Do NOT do this
```

The `scene_graph_id` field in `Provenance` is guaranteed to match the `video_id`
field in `PropositionGroup.media_id` — both are set to `sg.video_id` by every
generator in the pipeline.

---

## Evidence Signature — `get_signature()`

The `get_signature()` function in [`main_pipeline.py`](../main_pipeline.py)
computes a compact deduplication key from a `PropositionGroup`'s provenance
evidence fields:

```python
def get_signature(group: PropositionGroup) -> str:
    ev_obj = tuple(sorted(group.provenance.evidence_object_ids or []))
    ev_rel = tuple(sorted(group.provenance.evidence_relation_ids or []))
    ev_frm = tuple(sorted(group.provenance.frame_ids or []))
    return f"{ev_obj}_{ev_rel}_{ev_frm}"
```

### Purpose

The evidence signature is **not** the primary deduplication key (which is a hash
of sorted proposition texts).  It is a **secondary grouping key** stored in
`stats["examples_per_evidence"]` to measure how many distinct proposition groups
share the same underlying evidence.

### Why Sort?

Each evidence list is sorted independently before forming the tuple.  This
ensures that two `PropositionGroup` instances that reference the same set of
objects, relations, and frames — but in different orders — produce identical
signatures.  Without sorting, ordering differences in the source scene-graph
iteration order would produce spurious distinct signatures, inflating the
apparent diversity of evidence coverage.

### Signature Format

```
"(obj_ids,)_(rel_ids,)_(frame_ids,)"
```

**Example**:

```
"('bag', 'chair')_('holding',)_('023',)"
```

### Usage in Stats Reporting

```python
ev_sig = get_signature(group)
stats["examples_per_evidence"][ev_sig] += 1
```

The `DIVERSITY_REPORT.md` uses the length of
`stats["examples_per_evidence"]` to report the number of **distinct evidence
signatures** observed, and its value distribution to report **evidence reuse**.

---

## Generation Rules Catalog

The `generation_rule` field takes one of the following values.  Every value maps
to a unique code path in a specific generator class.

| `generation_rule` | Generator Class | Task Type | Description |
|---|---|---|---|
| `sg_obj_exist_binary` | `ObjectExistenceGenerator` | `binary` | Positive-polarity / negative-polarity pair for one object. 50 % true-centred, 50 % false-centred. |
| `sg_obj_exist_sc` | `ObjectExistenceGenerator` | `single_choice` | One TRUE present object + up to K-1 FALSE absent objects. K ∈ [10, 40]. |
| `sg_obj_exist_ml` | `ObjectExistenceGenerator` | `multi_label` | Mix of TRUE (present) and FALSE (absent) objects. Optional NONE sentinel (70 % probability). |
| `sg_obj_exist_3w` | `ObjectExistenceGenerator` | `three_way` | Fixed triple: "Yes" / "No" / "Cannot say" for one object. |
| `sg_rel_bin` | `RelationVerificationGenerator` | `binary` | Positive / negative pair for one person-object relation. Prefers mutually-exclusive negatives. |
| `sg_rel_sc` | `RelationVerificationGenerator` | `single_choice` | One TRUE relation + up to K-1 compatible-but-absent relations for the same object. |
| `sg_rel_ml` | `RelationVerificationGenerator` | `multi_label` | All confirmed relations for one object, plus compatible absent relations. Optional NONE sentinel. |
| `sg_rel_3w` | `RelationVerificationGenerator` | `three_way` | Fixed triple for a confirmed-present relation: always anchored on a TRUE relation. |
| `act_bin` | `ActionTemporalGenerator` | `binary` | Positive / negative pair for one action phrase. Negatives drawn from static `ALL_ACTIONS` pool. |
| `act_sc` | `ActionTemporalGenerator` | `single_choice` | One TRUE action + K-1 absent pool actions. |
| `act_ml` | `ActionTemporalGenerator` | `multi_label` | Mix of TRUE clip actions and FALSE pool actions. Optional NONE sentinel. |
| `act_3w` | `ActionTemporalGenerator` | `three_way` | Fixed triple for a polar action question. |
| `temp_bin` | `ActionTemporalGenerator` | `binary` | Temporal binary: "A before B" TRUE / FALSE, or reversed claim with negation TRUE. May be 1-hop or 2-hop. |
| `temp_sc` | `ActionTemporalGenerator` | `single_choice` | "Which event happened before B?" One TRUE answer; false options drawn from post-B events and pool. May be 1-hop or 2-hop. |
| `sg_comp_bin` | `CompositionalGenerator` | `binary` | AND-composition binary: one TRUE (r1 AND r2) vs. one FALSE (r1 AND r3) for the same object. |
| `sg_comp_sc` | `CompositionalGenerator` | `single_choice` | AND-composition single-choice: one TRUE conjunction + K-1 FALSE conjunctions differing in the second relation. |
| `sg_grounding_id` | `GroundingGenerator` | `single_choice` | Box-identification: K options for "Which object is in this bounding box?" |
| `sg_grounding_3w` | `GroundingGenerator` | `three_way` | Box-verification: "Is \<object\> present at this location?" Yes / No / Cannot say. |

> **Note on `rel_set`**: `RelationSetGenerator` is registered in the pipeline but
> yields nothing (delegated no-op).  All relation task types, including multi-label
> relation sets, are produced by `RelationVerificationGenerator` under the
> `sg_rel_ml` rule.

---

## Tracing an Example — Step-by-Step Walkthrough

The following procedure traces a generated `PropositionGroup` all the way back to
its raw AGQA source material.

### Step 1 — Read the Parquet shard

```python
import pyarrow.parquet as pq
import json

table = pq.read_table("data/generated/train/part-000000.parquet")
row = table.to_pydict()

# Pick row 42 as an example
example_id = row["example_id"][42]
provenance  = row["provenance"][42]   # dict with all Provenance fields

print(example_id)
# → "temp_bin_46GP8_2947183621"

print(json.dumps(provenance, indent=2))
```

### Step 2 — Extract the provenance anchor fields

```python
video_id         = provenance["scene_graph_id"]      # "46GP8"
generation_rule  = provenance["generation_rule"]     # "temp_bin"
generation_seed  = provenance["generation_seed"]     # 2947183621
generator_family = provenance["generator_family"]    # "action_temporal"
source_qid       = provenance["source_question_id"]  # None (SG-native example)
temporal_chain   = provenance["temporal_chain"]
# → [{"event1": "washing dishes", "relation": "before", "event2": "taking a phone"},
#    {"event1": "taking a phone",  "relation": "before", "event2": "watching television"}]
```

### Step 3 — Load the source scene graph

```python
import pickle

with open("AGQA_scene_graphs/AGQA_train_stsgs.pkl", "rb") as f:
    train_sgs = pickle.load(f)

raw_sg = train_sgs[video_id]   # raw STSG dict for video "46GP8"
```

### Step 4 — Confirm the evidence objects and relations

```python
evidence_objects   = provenance["evidence_object_ids"]   # []  (temporal tasks often have no objects)
evidence_relations = provenance["evidence_relation_ids"] # []
frame_ids          = provenance["frame_ids"]             # []

# For a temporal task, inspect the action annotations instead:
# (After normalisation)
from src.normalization.scene_graph_normalizer import normalize_video_sg
norm_sg = normalize_video_sg(video_id, raw_sg)

for action_id, action in norm_sg.actions.items():
    print(action.phrase, action.start_secs)
# → "washing dishes  4.2"
# → "taking a phone  9.1"
# → "watching television  15.7"
```

### Step 5 — Verify the temporal chain

```python
# The temporal_chain confirms the two-hop ordering:
#   washing dishes (t=4.2) → before → taking a phone (t=9.1) → before → watching television (t=15.7)
# Surface proposition: "washing dishes happened before watching television."
# Model must infer this via the intermediate action.
```

### Step 6 — Trace an AGQA-derived example (source_question_id is set)

For `OriginalQAGenerator` examples, `source_question_id` is non-`None`.  Trace
it back to the AGQA balanced QA file:

```python
from src.io.streaming_json import stream_questions

qa_path = "AGQA_balanced/train_balanced.txt"
target_qid = provenance["source_question_id"]   # e.g. "7AAAA_1"

for qid, qdata in stream_questions(qa_path):
    if qid == target_qid:
        print(qdata["question"])   # original question text
        print(qdata["answer"])     # original gold answer
        print(qdata["program"])    # AGQA DSL program
        break
```

---

## Provenance in the Schema Hierarchy

```
PropositionGroup
├── example_id         — synthetic unique ID (do NOT parse for video_id)
├── media_id           — equals sg.video_id (canonical anchor)
├── propositions[]
│   └── semantics      — SPO triple + polarity
├── reasoning          — type / complexity / family / hops
└── provenance         ← THIS DOCUMENT
    ├── source_question_id    — AGQA question key (None if SG-native)
    ├── source_question_text  — verbatim question text
    ├── source_program        — AGQA DSL program string
    ├── source_answer         — gold answer string
    ├── scene_graph_id        — video_id (CANONICAL ANCHOR)
    ├── generation_rule       — rule string (see catalog above)
    ├── generation_seed       — RNG value for exact replay
    ├── generator_family      — high-level family label
    ├── evidence_object_ids[] — object class IDs used as evidence
    ├── evidence_relation_ids[] — relation class IDs used as evidence
    ├── frame_ids[]           — frame IDs where evidence was found
    └── temporal_chain[]      — [{event1, relation, event2}] for multihop tasks
```
