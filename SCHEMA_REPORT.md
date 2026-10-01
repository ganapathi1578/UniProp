# AGQA 2.0 Dataset Schema Report

This report documents the exact schema and data structures found in the downloaded AGQA 2.0 dataset files.

## 1. Balanced Questions (`train_balanced.txt`, `test_balanced.txt`)

Despite the `.txt` extension, these files contain a single, massive JSON object where the keys are question IDs and the values are objects containing the question details.

*   **Format:** JSON
*   **File Sizes:**
    *   `train_balanced.txt`: ~1.53 GB (~1,600,872 questions)
    *   `test_balanced.txt`: ~789.6 MB (~669,195 questions)

### Entry Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `question` | `string` | The natural language question text. |
| `answer` | `string` | The correct answer. |
| `video_id` | `string` | The 5-character ID of the associated video (e.g., "46GP8"). |
| `global` | `list[string]` | High-level reasoning categories (e.g., `superlative`, `obj-rel`, `exists`, `sequencing`). |
| `local` | `string` | Encoded local metadata (often a sequence of object/relation IDs). |
| `ans_type` | `string` | The type of answer expected: `"binary"` or `"open"`. |
| `steps` | `integer` | The number of compositional reasoning steps (typically 1-7). |
| `semantic` | `string` | The semantic focus: `"object"`, `"action"`, or `"relation"`. |
| `structural` | `string` | The structural question type: `"choose"`, `"compare"`, `"logic"`, `"query"`, or `"verify"`. |
| `novel_comp` | `integer` | Flag indicating novel composition (0 or 1). |
| `more_steps` | `integer` | Flag indicating if it has more steps than training examples (0 or 1). |
| `program` | `string` | A functional program representation of the reasoning (e.g., `Choose(...)`, `Exists(...)`). |
| `sg_grounding` | `dict` | Maps character ranges in the program string to specific scene graph entity IDs. |

---

## 2. Spatio-Temporal Scene Graphs (`AGQA_train_stsgs.pkl`, `AGQA_test_stsgs.pkl`)

These are Python pickle files containing deeply nested dictionaries representing the scene graphs.

*   **Format:** Python Pickle (`.pkl`)
*   **File Sizes:**
    *   `AGQA_train_stsgs.pkl`: ~443.5 MB (7,787 videos)
    *   `AGQA_test_stsgs.pkl`: ~167.7 MB (1,814 videos)
*   **Top-Level Structure:** `dict[video_id: str, video_data: dict]`

### Video Data Structure

A `video_data` dictionary contains various entries keyed by IDs (like frame numbers, object IDs, action IDs). Each entry has a `type` field defining its schema.

#### 2.1. Frame Entries (`type: "frame"`)
Key format: `"{frame_num}"` (e.g., `"000030"`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Frame ID/number. |
| `secs` | `float` | Timestamp in seconds. |
| `type` | `string` | `"frame"` |
| `metadata` | `string` | Split (`"train"` or `"test"`). |
| `objects` | `dict` | Contains `names` (list of object class IDs) and `vertices` (list of full object instance dicts). |
| `attention` | `list` | Attention relations active in this frame. |
| `contact` | `list` | Contact relations active in this frame. |
| `spatial` | `list` | Spatial relations active in this frame. |
| `verb` | `list` | Verbs active in this frame. |
| `actions` | `list` | Higher-level actions active in this frame. |

#### 2.2. Object Instance Entries (`type: "object"`)
Key format: `"{class_id}/{frame_num}"` (e.g., `"o16/000030"`)
Found inside `frame['objects']['vertices']` and sometimes as standalone entries.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Object instance ID. |
| `type` | `string` | `"object"` |
| `class` | `string` | Object class ID (e.g., `"o16"`). |
| `attention` | `list[dict]` | Attention relations involving this object. |
| `contact` | `list[dict]` | Contact relations involving this object. |
| `spatial` | `list[dict]` | Spatial relations involving this object. |
| `verb` | `list[dict]` | Verbs involving this object. |
| `visible` | `boolean` | Visibility flag. |
| `bbox` | `tuple \| None` | Bounding box `(x1, y1, x2, y2)` or `None`. |
| `metadata` | `dict` | Contains `tag` (e.g., `"VIDEO.mp4/OBJECT_NAME/FRAME"`) and `set` (`"train"`). |
| `frame_num` | `string` | Frame number. |
| `secs` | `float` | Timestamp. |
| `next` | `dict \| None` | A recursive reference to the object's state in the next frame. |

#### 2.3. Relation Entries (`type: "attention" | "contact" | "spatial" | "verb"`)
Key format: `"{relation_id}/{frame_num}"`

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Relation instance ID. |
| `type` | `string` | Category (`"attention"`, `"contact"`, etc.). |
| `class` | `string` | Relation class ID (e.g., `"r2"`, `"v026"`). |
| `objects` | `list` | IDs/references of the involved objects. |
| `metadata` | `string` | Split. |
| `frame` | `string` | Frame number. |
| `secs` | `float` | Timestamp. |
| `next` | `dict \| None` | Recursive reference to next instance. |

#### 2.4. Action Entries (`type: "action"`)
Key format: `"{charades_id}/1"` (e.g., `"c098/1"`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Action instance ID. |
| `type` | `string` | `"action"` |
| `charades` | `string` | Charades action class ID (e.g., `"c098"`). |
| `phrase` | `string` | Human-readable description (e.g., `"holding a broom"`). |
| `start` | `float` | Start time (seconds). |
| `end` | `float` | End time (seconds). |
| `length` | `float` | Duration (seconds). |
| `objects` | `list` | Involved objects. |
| `attention`, `contact`, `spatial`, `verb` | `list` | Relations during this action. |
| `all_f` | `list[string]` | List of all frame IDs encompassed by this action. |

---

## 3. Vocabularies & Taxonomies (Action Genome)

The dataset uses coded IDs which map to the Action Genome taxonomy.

*   **Objects (28 classes):** `o2` (bag), `o3` (sofa/couch), `o4` (towel), `o6` (bag), `o7` (broom), `o8` (chair), `o9` (closet/cabinet), `o10` (clothes), `o12` (dish), `o13` (door), `o14` (doorknob), `o15` (doorway), `o16` (floor), `o17` (food), `o19` (laptop), `o20` (light), `o21` (medicine), `o22` (mirror), `o23` (book), `o24` (phone/camera), `o25` (picture), `o26` (pillow), `o27` (refrigerator), `o30` (shoe), `o32` (shelf), `o33` (television), `o35` (vacuum), `o36` (window).
*   **Attention Relations (2 classes):** `r1`, `r2` (looking at, not looking at)
*   **Spatial Relations (6 classes):** `r4`, `r5`, `r6`, `r7`, `r8`, `r9` (above, beneath, in front of, behind, on the side of, in)
*   **Contact Relations (16 classes):** `r10` through `r18`, and `r20` through `r26` (carrying, covered by, drinking from, eating, have it on the back, holding, leaning on, lying on, not contacting, other relationship, sitting on, standing on, touching, twisting, wearing, wiping)
*   **Verbs (23 classes):** `v000` to `v032` (subset)
*   **Actions (Charades):** Mapped directly in the data via the `phrase` field (e.g., `c098` -> "holding a broom").

## 4. Key Takeaways & Constraints

1.  **Memory Management:** The 1.5GB `train_balanced.txt` JSON must be processed using a streaming parser (e.g., `ijson` or a custom chunked reader). Loading it entirely with `json.load()` will consume too much RAM.
2.  **Scene Graph Recursion:** The `next` and `prev` fields in object/relation entries create deep linked lists across frames. Naive recursive traversal hits Python's recursion limit. Normalization must use iterative approaches or ignore these links in favor of iterating over frames (`all_f`).
3.  **Missing Validation Split:** Only `train` and `test` splits exist. A synthetic validation split will need to be derived deterministically from the `train` set video IDs to prevent data leakage.
4.  **Opaque IDs:** We must map `oXX` and `rXX` IDs to human-readable text using the embedded taxonomy and metadata tags to generate natural language propositions.
