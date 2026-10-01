# Git and Data Policy

## What is Version Controlled
- Source code (`src/`)
- Tests (`tests/`)
- Documentation (`docs/`, `README.md`)
- Configuration and Dependency lists

## What is NEVER Version Controlled
- Generated datasets (`data/generated/`, `*.parquet`)
- Source datasets (`AGQA_balanced/`, `AGQA_scene_graphs/`)
- Downloaded video media
- Checkpoints and caches
- Virtual environments

**Always check `git status` before committing to ensure no large data files are accidentally staged.**
