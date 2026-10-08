# UniProp Unified Proposition Architecture

## Core Design Principle
UniProp decouples AGQA's source classifications (`answer_type`, `semantic_type`, `structural_type`) from the final output format. 

AGQA's metadata describes *how the dataset was generated*, but for UniProp, what matters is:
**What kind of entity do we need to select/evaluate, and what should the final option look like?**

## Internal Evaluation vs. External Representation

Regardless of the reasoning operation (`Choose`, `Compare`, `Logic`, `Exists`), every AGQA program reduces to one of these core internal semantic families:

| Internal Answer Kind | Evaluated Entity | Example AGQA Operations |
| :--- | :--- | :--- |
| **BOOLEAN** | Truth of a proposition | `Exists`, `Verify`, `And`, `Xor` |
| **OBJECT** | An object satisfying the program | `Query(object)`, `Choose(object)` |
| **ACTION** | An action/event satisfying the program | `Query(action)` |
| **TEMPORAL** | Temporal relation/order | `Compare(temporal)`, `Localize` |
| **COMPARISON** | Comparison between values/events | `Compare(duration)` |
| **SUPERLATIVE** | Extreme/ordered candidate | `Superlative(longest)` |

### The "Full Sentence" Paradigm

Externally, **every** option is serialized as a complete, declarative natural-language proposition. 
UniProp options are never raw tokens like `window`, `before`, or `longer`. 

```text
                    answer_kind
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       BOOLEAN       OBJECT/ACTION   other
          │             │             │
          ▼             ▼             ▼
      proposition    proposition    proposition
```

**Example 1: Boolean / Logical**
*   **Source**: "Did they put down a blanket but not a broom?"
*   **Options**:
    *   "The person put down a blanket but not a broom." (1)
    *   "The person did not put down a blanket but not a broom." (0)

**Example 2: Object**
*   **Source**: "Which object did the person interact with after making some food?" (Answer: chair)
*   **Options**:
    *   "The person interacted with a chair after making some food." (1)
    *   "The person interacted with a window after making some food." (0)

**Example 3: Comparison**
*   **Source**: "Do they take a shorter amount of time running somewhere or holding some clothes?"
*   **Options**:
    *   "Holding some clothes takes less time." (1)
    *   "Running somewhere takes less time." (0)

## Handling Misleading AGQA Labels
*   **AGQA `binary` != Yes/No**: AGQA categorizes `Choose(food, window, ...)` as "binary" because there are two choices, but the answer is an `OBJECT`, not a boolean. UniProp strictly routes based on the semantic return type (`OBJECT`), not the AGQA dataset label.
*   **`Choose`, `Compare`, `Logic` are Operations**: These are AST nodes that determine *how* the candidate set is evaluated. They are not final answer types. A `Logic` operation evaluates to a `BOOLEAN`. A `Choose` operation evaluates to an `OBJECT` or `ACTION`.

## The Simplest Mental Model
Every AGQA record asks us to find a `BOOLEAN`, `OBJECT`, `ACTION`, `RELATION`, `COMPARISON`, or `SUPERLATIVE`.
We find it using the `SemanticEngine`.
We convert the candidates into a `FULL SENTENCE`.
The model decides `TRUE` or `FALSE` for each sentence.
