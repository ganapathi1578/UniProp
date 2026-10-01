# Canonical Data Schema

UniProp generates structured `.parquet` files using PyArrow.

## Core Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `example_id` | `string` | Yes | Unique ID for the generated example. |
| `split` | `string` | Yes | The source dataset partition (`train`, `val`, `test`). |
| `video_id` | `string` | Yes | The explicit AGQA source video ID. |
| `query` | `string` | Yes | The natural language task prompt/question. |
| `options` | `list<string>` | Yes | A list of natural language propositions. |
| `labels` | `list<int8>` | Yes | Binary labels aligned with the `options` array. |
| `truth_state` | `string` | Yes | The core truth evaluation (`TRUE`, `FALSE`, `UNKNOWN`). |
| `option_count` | `int32` | Yes | The total length of the `options` array. |
| `task_type` | `string` | Yes | Task structure (`binary`, `single_choice`, `multi_label`, `three_way`). |
| `reasoning_family` | `string` | Yes | Semantic category (e.g., `object`, `relation`, `temporal`). |
| `generator_family` | `string` | Yes | The ID of the generator used. |

## Invariants
- `len(options) == len(labels) == option_count`
- `labels[i] == 1` -> `options[i]` is a correct answer.
- `labels[i] == 0` -> `options[i]` is not a correct answer.
