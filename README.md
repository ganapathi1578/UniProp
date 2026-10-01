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
See [docs/README.md](docs/README.md) for full architecture, schema, generation, and troubleshooting guides.
