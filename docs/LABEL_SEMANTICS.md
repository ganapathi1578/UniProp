# Truth and Label Semantics

UniProp explicitly decouples the underlying semantic `truth_state` from the structural numeric `label`.

## Truth State
The `truth_state` field represents the objective reality of the proposition against the Scene Graph. It can be:
- `TRUE`
- `FALSE`
- `UNKNOWN`

**UNKNOWN must never be silently converted to FALSE.** It is preserved for auditing and loss-masking.

## Task Types and Labels

### Binary
A single logic proposition evaluated as True or False. `sum(labels)` is typically 1 (if True is positive) or 0 (if False is positive).

### Single-Choice
Multiple propositions where exactly one is correct.
- `sum(labels) == 1`

### Multi-Label
Multiple propositions where any number can be correct.
- Multiple `1` values are permitted.
- `sum(labels)` can range from `0` to `option_count`.

### Three-Way
A specific ternary logic evaluation.

## NONE Semantics
When no propositions are valid for a query, a logical `NONE` proposition is synthesized. If a `NONE` proposition is true, all other propositions must be false.
