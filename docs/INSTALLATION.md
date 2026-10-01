# Environment & Installation

## Requirements
- Python 3.9+
- Pip / Conda

## Setup

```bash
# Clone the repository
git clone git@github.com:ganapathi1578/UniProp.git
cd UniProp

# Install requirements
pip install -r requirements.txt
```

## Directory Layout Expectation
You must provide the AGQA source data manually:
```text
UniProp/
├── AGQA_balanced/
│   ├── train_balanced.txt
│   └── test_balanced.txt
├── AGQA_scene_graphs/
│   ├── train/
│   └── test/
```
