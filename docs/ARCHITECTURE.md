# UniProp Architecture

UniProp is a dataset engineering pipeline. It transforms raw AGQA dataset information into proposition-based schemas ready for model training, without containing any downstream model code itself.

## Data Flow

1. **AGQA Source Data**: Raw AGQA QA text and scene graphs.
2. **Source Loading**: The data is loaded into memory (e.g., `streaming_json.py`, `scene_graph_normalizer.py`).
3. **Scene-Graph/Event Interpretation**: Events and objects are extracted and resolved.
4. **Candidate Generation**: The generators in `src/generators` produce candidate propositions.
5. **Truth Evaluation**: The semantic truth states (`TRUE`, `FALSE`, `UNKNOWN`) are logically assigned.
6. **Query/Proposition Construction**: The query string and final proposition strings are templated.
7. **Validation / Quality Gates**: `src/validation/validators.py` ensures constraints (like mutual exclusivity, schema bounds, label correctness).
8. **Example Assembly**: Groups are wrapped into `PropositionGroup` schemas.
9. **Parquet Writer**: `src/storage/parquet_writer.py` maps the Python schema to PyArrow.
10. **Shards**: The data is written out to disk in chunked `part-XXXXXX.parquet` files.
11. **Final Dataset**: Output located in `data/generated/`.

## Ownership of Fields

- **provenance**: Passed natively from the raw AGQA dataset loaders.
- **labels**: Assigned by the generators based on `truth_state` and task structure.
- **truth_state**: Logically determined by the reasoning sub-generators evaluating the Scene Graph.
- **task_type**: Assigned by the generator logic (e.g., `binary`, `multi_label`).
- **reasoning_family**: Assigned by the generator logic representing the semantic space (e.g., `object_existence`).
- **generator_family**: Represents the specific Python class that generated it.
- **split**: Preserved natively from the AGQA dataset split partitions.
- **video_id**: Extracted directly from the AGQA provenance/media properties and preserved as the source-of-truth.
