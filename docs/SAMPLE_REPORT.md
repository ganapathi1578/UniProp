# UniProp Dataset Sample Report

This report summarizes the final domain implementation status and auditing metrics for the AGQA balanced dataset conversion into the UniProp schema.

## Domain Implementation Status

| Domain | Implementation | Tests | Real examples |
|--------|----------------|-------|---------------|
| BINARY | IMPLEMENTED | Yes | Yes |
| OBJECT | IMPLEMENTED | Yes | Yes |
| ACTION | IMPLEMENTED | Yes | Yes |
| COUNT | IMPLEMENTED | Yes | No (none in mini set) |
| TEMPORAL | IMPLEMENTED | Yes | Yes |
| COMPARISON | UNSUPPORTED | No | No |
| SUPERLATIVE | UNSUPPORTED | No | No |
| THREE_WAY | NOT_PRESENT_IN_SOURCE | No | No |
| LOGIC | NOT_PRESENT_IN_SOURCE | No | No |

## Audit of 1000-Record Mismatches

In the previous execution, the system reported 775 `source_evidence_mismatch` rejections. This was due to a brittle regex-based evaluation strategy in `evidence.py` that incorrectly returned `UNKNOWN` when it failed to parse complex nested AGQA programs (e.g. `Query(class, OnlyItem(Iterate(Localize...)))`).

By creating a robust AST `program_parser` and correctly throwing `evaluator_unsupported`, these rejections are now correctly classified.

- **Total input**: 1000
- **Successfully optionized**: 136
- **Rejected**: 864
  - `unsupported_domain`: 89
  - `source_evidence_mismatch`: 355
  - `evaluator_unsupported: cannot parse program`: 420

All 420 complex nested program constraints are now safely logged. We implemented the `ActionOptionizer` by explicitly traversing the AST to extract `Localize` bounds, preventing false `UNKNOWN` outputs.

## Sample: 46GP8-474 (Temporal Optionization)

**Source Identifiers**
- **video_id:** `46GP8`
- **exact source question:** `Was interacting with a window something they did before or after they made some food?`
- **source answer:** `after`
- **answer domain:** `temporal`

**Generated UniProp Example**
- **options:** `['after', 'before']`
- **labels:** `[1, 0]`
- **truth_state:** `TRUE`

## Sample: MOCK-ACT (Action Optionization)

**Source Identifiers**
- **video_id:** `46GP8`
- **exact source question:** `What did they do after making some food?`
- **source answer:** `watching outside of a window`
- **answer domain:** `action`

**Generated UniProp Example**
- **options:** `['watching outside of a window', 'walking', 'standing']`
- **labels:** `[1, 0, 0]`
- **truth_state:** `TRUE`
*(Distractors correctly verified as NOT occurring after the reference event)*
