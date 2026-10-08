# UniProp Label Semantics

> [!IMPORTANT]
> UniProp **explicitly decouples** `truth_state` (a semantic field encoding ground truth reality) from `label` (a structural integer used by training and evaluation code). Never conflate the two.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Truth State](#2-truth-state)
3. [Label](#3-label)
4. [Polarity](#4-polarity)
5. [Truth-State × Polarity → Label Matrix](#5-truth-state--polarity--label-matrix)
6. [Task Type × Label Constraints](#6-task-type--label-constraints)
7. [NONE Proposition](#7-none-proposition)
8. [Cannot Say (Three-Way Tasks)](#8-cannot-say-three-way-tasks)
9. [Concrete Examples](#9-concrete-examples)

---

## 1. Overview

Every `Proposition` in a `PropositionGroup` carries two distinct truth-related fields:

| Field | Type | Role |
|---|---|---|
| `truth_state` | `str` | **Semantic** — what is objectively true in the scene graph |
| `label` | `int \| None` | **Structural** — the integer signal used by classifiers and evaluators |

The `label` is **derived** from `truth_state` and `polarity` together (see §5). This two-field design preserves semantic provenance: downstream code can always reconstruct *why* a label has a given value, supporting auditing, loss masking, and error analysis.

```
truth_state   polarity
    ↓              ↓
  [mapping rule]
        ↓
      label
```

The mapping is deterministic and lossless — given `truth_state` and `polarity`, `label` is uniquely determined (with the exception of the `UNKNOWN` truth state, which requires task-type context; see §8).

---

## 2. Truth State

`truth_state` is a coarse string label encoding the **gold-standard truth value** of a proposition against the source scene graph or spatio-temporal evidence.

### Possible values

| Value | Meaning |
|---|---|
| `"TRUE"` | The proposition **holds** in the video according to the scene graph. |
| `"FALSE"` | The proposition **does not hold** in the video. |
| `"UNKNOWN"` | The truth value **cannot be determined** from available evidence. |

### Definitions

**`TRUE`**
The scene graph contains a direct edge, attribute, or event-ordering fact that confirms the proposition. For example, if an Action Genome relation edge `(person) --[holding]--> (book)` exists in the relevant frame, then *"The person is holding a book."* has `truth_state="TRUE"`.

**`FALSE`**
No scene-graph evidence supports the proposition, or a contradicting fact is present. For example, if no `holding` edge links the person to the book, then the proposition above has `truth_state="FALSE"`.

**`UNKNOWN`**
The relevant evidence is absent or ambiguous. Common causes:

- **Off-screen entities**: the object is referenced in the question but does not appear in the annotated frames for the segment in question.
- **Epistemic gaps in annotation**: Action Genome's frame-level annotations are sparse; not every relation is labelled in every frame.
- **Grounding three-way tasks**: the third option in a three-way question is always "Cannot say", which maps to `truth_state="UNKNOWN"` (see §8).
- **Compositional uncertainty**: one conjunct of a `compositional_and` proposition cannot be resolved.

> [!WARNING]
> `UNKNOWN` must **never** be silently coerced to `FALSE`. The `truth_state` is preserved verbatim in the output Parquet so that training code can apply loss masking (e.g. ignore UNKNOWN examples during cross-entropy computation) rather than treating them as confident negatives.

---

## 3. Label

`label` is an integer encoding of the proposition's correctness for classification tasks.

### Possible values

| Value | Meaning |
|---|---|
| `1` | The proposition is **correct** (the model should select it). |
| `0` | The proposition is **incorrect** (the model should not select it). |
| `-1` | Reserved; not used in current generation. |
| `None` | The proposition is UNKNOWN and the task does not require a label (rare). |

> [!NOTE]
> In practice `label=None` only occurs for UNKNOWN propositions outside three-way tasks. All propositions in three-way tasks always carry a label (see §8).

### Exact mapping rules

The label is computed from `truth_state` and `polarity` as follows (full matrix in §5):

1. **TRUE + positive polarity → `label = 1`**
   The proposition affirmatively states a fact that holds. It is the correct answer.

2. **FALSE + positive polarity → `label = 0`**
   The proposition affirmatively states a fact that does not hold. It is incorrect.

3. **FALSE + negative polarity → `label = 1`**
   The proposition negates a fact, and that negation is accurate (the fact indeed does not hold). The negation is therefore correct.

4. **TRUE + negative polarity → `label = 0`**
   The proposition negates a fact, but the fact actually holds. The negation is therefore incorrect.

5. **UNKNOWN (any polarity) → `label = 0` in three-way tasks; `label = None` otherwise.**
   See §8 for the special "Cannot say" treatment.

---

## 4. Polarity

`polarity` is a field on `PropositionSemantics` that records whether the proposition's surface text **affirms** or **negates** the underlying semantic claim.

| Value | Description |
|---|---|
| `"positive"` | The proposition states that something **is** the case. |
| `"negative"` | The proposition states that something **is not** the case. |

### Examples

| Surface text | Polarity |
|---|---|
| *"The chair is present."* | `positive` |
| *"The chair is not present."* | `negative` |
| *"The person is holding the book."* | `positive` |
| *"The person is not holding the book."* | `negative` |
| *"The person is not looking at the bag."* | `negative` |

### How polarity interacts with truth_state

Polarity does **not** change what fact is being tested — it only changes how the surface text frames it. The same underlying scene-graph fact (e.g., `person --[holding]--> book`) can be asserted positively or negatively. The `label` is determined by whether the surface claim is ultimately **correct**, which depends on both polarity and truth.

> [!TIP]
> Negated propositions (`polarity="negative"`) increase linguistic diversity and test whether models can handle surface-form negation. They are particularly important in binary and three-way tasks where the model must evaluate a single claim.

---

## 5. Truth-State × Polarity → Label Matrix

The full mapping from `(truth_state, polarity)` to `label`:

| `truth_state` | `polarity` | `label` | Rationale |
|---|---|---|---|
| `TRUE` | `positive` | **1** | Affirmative claim is correct. |
| `TRUE` | `negative` | **0** | Negation of a true fact is incorrect. |
| `FALSE` | `positive` | **0** | Affirmative claim of a non-existent fact is incorrect. |
| `FALSE` | `negative` | **1** | Negation of a false fact is correct. |
| `UNKNOWN` | `positive` | **0** *(three-way)* / `None` | Cannot confirm; treated as incorrect in three-way tasks. |
| `UNKNOWN` | `negative` | **0** *(three-way)* / `None` | Cannot confirm; treated as incorrect in three-way tasks. |

---

## 6. Task Type × Label Constraints

Each task type imposes strict structural invariants on the `label` values within a `PropositionGroup`. These invariants are enforced by `validate_example()` before any group is written to disk.

### `binary`

- **Proposition count**: exactly 2 (one positive proposition, one negative proposition, or one affirmative and one negated paraphrase).
- **Label constraint**: `sum(labels) == 1` — exactly one proposition must be correct.
- **Typical structure**: `[label=1, label=0]` or `[label=0, label=1]`.

```
PropositionGroup (task_type="binary")
├── Proposition: "The person is holding the book."  label=1 (TRUE, positive)
└── Proposition: "The person is not holding the book." label=0 (TRUE, negative)
```

### `single_choice`

- **Proposition count**: 2–40.
- **Label constraint**: `sum(labels) == 1` — exactly one proposition is the correct answer.
- **Semantics**: The propositions are mutually exclusive alternatives (e.g., different objects or different relations for the same subject).

```
PropositionGroup (task_type="single_choice")
├── Proposition: "The person is sitting on the chair."   label=1
├── Proposition: "The person is sitting on the sofa."    label=0
└── Proposition: "The person is sitting on the floor."   label=0
```

### `multi_label`

- **Proposition count**: 2–40.
- **Label constraint**: `sum(labels) ∈ [0, k]` — zero to all propositions may be correct simultaneously.
- **Note**: The NONE proposition is introduced specifically for `multi_label` tasks where zero propositions are correct (see §7).

```
PropositionGroup (task_type="multi_label")
├── Proposition: "The person is holding the book."     label=1
├── Proposition: "The person is holding the bag."      label=1
└── Proposition: "The person is holding the pillow."   label=0
```

### `three_way`

- **Proposition count**: **exactly 3** (enforced as a hard invariant, not a range).
- **Label constraint**: `sum(labels) == 1` — exactly one of the three propositions is correct.
- **Structure**: The three slots are always:
  1. A positive proposition (Yes / It is the case)
  2. A negative proposition (No / It is not the case)
  3. A "Cannot say" proposition (`truth_state="UNKNOWN"`, `label=0`; see §8)

```
PropositionGroup (task_type="three_way")
├── Proposition: "The person is holding the book."     label=1  (TRUE, positive)
├── Proposition: "The person is not holding the book." label=0  (TRUE, negative)
└── Proposition: "Cannot say."                         label=0  (UNKNOWN)
```

### Summary table

| Task type | Min props | Max props | `sum(labels)` |
|---|---|---|---|
| `binary` | 2 | 2 | exactly 1 |
| `single_choice` | 2 | 40 | exactly 1 |
| `multi_label` | 2 | 40 | 0 … k |
| `three_way` | 3 | 3 | exactly 1 |

---

## 7. NONE Proposition

When a `multi_label` query has **no valid correct answers** in the scene graph, a synthetic "None of the above" proposition is added to the group.

### Properties

| Property | Value |
|---|---|
| `canonical_type` | `"logical_none"` |
| `text` | A natural-language phrase such as *"None of the above."* |
| `truth_state` | `"TRUE"` when no other option is valid; `"FALSE"` otherwise |
| `label` | `1` when none of the other options are correct; `0` otherwise |

### Invariant

> **If `NONE.label == 1`, then all other `label` values in the group must be `0`.**
> **If `NONE.label == 0`, then at least one other `label` value in the group must be `1`.**

This invariant is mechanically checked by validation check #8 (NONE consistency). It mirrors the standard "None of the above" semantics in multiple-choice evaluation.

### When is NONE synthesised?

The NONE proposition is inserted by the generator when:

- The scene graph contains no positive evidence for any of the candidate propositions, **and**
- The task type is `multi_label`.

It is **never** inserted in `binary`, `single_choice`, or `three_way` tasks.

---

## 8. Cannot Say (Three-Way Tasks)

The third slot in every `three_way` group is a special **"Cannot say"** proposition.

### Semantics

"Cannot say" encodes **epistemic uncertainty**: the scene graph cannot confirm *or* deny the claim because the relevant evidence is absent, off-screen, or unannotated.

### Label convention

| Field | Value |
|---|---|
| `truth_state` | `"UNKNOWN"` |
| `label` | **always `0`** |
| `polarity` | `"positive"` (the proposition does not negate anything) |

The label is `0` even when "Cannot say" is the *correct* answer. Correctness is encoded by setting the other two options to `label=0` and "Cannot say" to `label=0` — and then checking `sum(labels) == 1` including the UNKNOWN slot.

> [!IMPORTANT]
> This is the **only** situation where `truth_state="UNKNOWN"` coexists with a non-`None` label. In all other task types, UNKNOWN propositions carry `label=None` and are excluded from cardinality checks.

---

## 9. Concrete Examples

### Example A — Binary task, positive proposition is TRUE

```json
{
  "task_type": "binary",
  "propositions": [
    {
      "text": "The person is holding the bag.",
      "truth_state": "TRUE",
      "label": 1,
      "semantics": { "canonical_type": "relation", "subject": "person",
                     "predicate": "holding", "object": "bag", "polarity": "positive" }
    },
    {
      "text": "The person is not holding the bag.",
      "truth_state": "TRUE",
      "label": 0,
      "semantics": { "canonical_type": "relation", "subject": "person",
                     "predicate": "holding", "object": "bag", "polarity": "negative" }
    }
  ]
}
```

`sum(labels) = 1` ✓

---

### Example B — Single-choice task

```json
{
  "task_type": "single_choice",
  "propositions": [
    { "text": "The person is sitting on the chair.",  "truth_state": "TRUE",  "label": 1 },
    { "text": "The person is sitting on the sofa.",   "truth_state": "FALSE", "label": 0 },
    { "text": "The person is sitting on the floor.",  "truth_state": "FALSE", "label": 0 }
  ]
}
```

`sum(labels) = 1` ✓

---

### Example C — Multi-label task with NONE

```json
{
  "task_type": "multi_label",
  "propositions": [
    { "text": "The person is carrying the bag.",    "truth_state": "FALSE", "label": 0 },
    { "text": "The person is holding the bag.",     "truth_state": "FALSE", "label": 0 },
    { "text": "None of the above.",
      "truth_state": "TRUE",  "label": 1,
      "semantics": { "canonical_type": "logical_none" } }
  ]
}
```

`NONE.label == 1` and all others are `0` ✓

---

### Example D — Three-way task, "Cannot say" is correct

```json
{
  "task_type": "three_way",
  "propositions": [
    { "text": "The person is holding the book.",      "truth_state": "FALSE",   "label": 0 },
    { "text": "The person is not holding the book.",  "truth_state": "FALSE",   "label": 1 },
    { "text": "Cannot say.",                          "truth_state": "UNKNOWN", "label": 0 }
  ]
}
```

`sum(labels) = 1` ✓  
The negative proposition is the correct one (the person is indeed *not* holding the book).

---

### Example E — Temporal task, negative polarity is FALSE

```json
{
  "task_type": "single_choice",
  "propositions": [
    {
      "text": "The person holding the bag happened before the person eating food.",
      "truth_state": "TRUE", "label": 1,
      "semantics": { "canonical_type": "temporal",
                     "event_a": "holding bag", "temporal_relation": "before",
                     "event_b": "eating food", "polarity": "positive" }
    },
    {
      "text": "The person holding the bag happened after the person eating food.",
      "truth_state": "FALSE", "label": 0,
      "semantics": { "canonical_type": "temporal",
                     "event_a": "holding bag", "temporal_relation": "after",
                     "event_b": "eating food", "polarity": "positive" }
    }
  ]
}
```

`sum(labels) = 1` ✓
