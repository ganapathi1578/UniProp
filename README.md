# UniProp

UniProp is a proposition-based multimodal dataset construction and labeling pipeline. It transforms raw AGQA dataset information into normalized, validated, and structurally explicit logical propositions.

### Scope
- **UniProp = dataset engineering.**
- **Downstream model = separate project.** UniProp does NOT contain tokenizers, model formatters, or embeddings. It generates RAW-TEXT string inputs only.

### Core Task Representation
```python
query: string
options: list[string]
labels: list[int8]
```
- `labels[i] = 1` -> option i is a correct answer
- `labels[i] = 0` -> option i is not a correct answer

### Example
```json
{
    "example_id": "act_ml_46GP8_2548359539",
    "split": "train",
    "video_id": "46GP8",
    "query": "What is the person doing?",
    "options": [
        "The person is making some food.",
        "The person is typing on a laptop."
    ],
    "labels": [1, 0]
}
```

### Quick Start
```bash
# 1. Clone repository
git clone git@github.com:ganapathi1578/UniProp.git
cd UniProp

# 2. Setup environment
pip install -r requirements.txt

# 3. Source Data (Must be downloaded manually)
# Place AGQA_balanced/ and AGQA_scene_graphs/ in the root directory.

# 4. Generate Dataset
python main_pipeline.py
```

### Output & Data Policy
Data is output natively to `data/generated/`.
**Generated datasets and source datasets are strictly excluded from Git.**

### Documentation Index

**Comprehensive Documents (`docs/`):**
- [Architecture](docs/ARCHITECTURE.md): Data flow from source json -> proposition logic -> parquet strings.
- [Data Schema](docs/DATA_SCHEMA.md): PyArrow schema definition with explicit invariables (`len(options) == len(labels)`).
- [Label Semantics](docs/LABEL_SEMANTICS.md): Separation of semantic `truth_state` (`TRUE`/`FALSE`/`UNKNOWN`) vs labels (`1`/`0`).
- [Provenance](docs/PROVENANCE.md): `video_id` tracking mandate.
- [Installation](docs/INSTALLATION.md): Clean environment setup with `requirements.txt`.
- [Configuration](docs/CONFIGURATION.md): Explanation of `main_pipeline.py` defaults (e.g. `SEED=42`).
- [Generation](docs/GENERATION.md): Commands and runtime architecture details (flushing buffers).
- [Validation](docs/VALIDATION.md): Quality gates currently active (mutual exclusivity, range checks).
- [Data Layout](docs/DATA_LAYOUT.md): `data/generated/` and source datasets overview.
- [Resource Requirements](docs/RESOURCE_REQUIREMENTS.md): Scaling extrapolation for large parquet strings.
- [Reproducibility](docs/REPRODUCIBILITY.md): Tri-layer randomness locks (`PYTHONHASHSEED`, `random.seed`, `random.Random()`).
- [Production Runbook](docs/PRODUCTION_RUNBOOK.md): Sequential checklist for initiating and recovering large 10M+ runs.
- [Troubleshooting](docs/TROUBLESHOOTING.md): Common failures and file-system recoveries.
- [Development](docs/DEVELOPMENT.md): Creating new Generators and Reasoners.
- [Git Policy](docs/GIT_POLICY.md): Rigid definition of what never touches Git.
