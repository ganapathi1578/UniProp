# UniProp — Git Policy

> [!IMPORTANT]
> This document is the authoritative policy for what goes into — and what stays **out of** — the UniProp git repository. Every contributor must read this before making their first commit.

---

## 1. What IS in Git

The following paths are **tracked by git** and must always be kept up to date:

| Path | Purpose |
|------|---------|
| `src/` | All Python source packages and modules |
| `tests/` | Full unit and integration test suite |
| `docs/` | Project documentation (this file, DATA_LAYOUT.md, PRODUCTION_RUNBOOK.md) |
| `main_pipeline.py` | Top-level pipeline entry point |
| `requirements.txt` | Pinned Python dependency list |
| `pyproject.toml` | Build system config, tool settings (black, ruff, mypy, pytest) |
| `.gitignore` | Git exclusion rules (must stay current) |
| `CHANGELOG.md` | Chronological record of all notable changes |
| `CONTRIBUTING.md` | Contributor guidelines and development setup |

> [!NOTE]
> If you add a new Python module, test file, or documentation page, it is your responsibility to `git add` it and include it in the same commit that introduces it. Never leave tracked files in a "forgotten" state.

---

## 2. What is NEVER in Git

The following categories of files must **never** be committed. They are excluded via `.gitignore` and must remain excluded.

### 2.1 Source Data

| Path / Pattern | Reason |
|----------------|--------|
| `AGQA_balanced/` | Raw AGQA QA data — licensed, large, not ours to redistribute |
| `AGQA_scene_graphs/` | Raw AGQA scene graph pickles — same license restrictions |
| `*.pkl` | Any pickle file (includes scene graph data) |

### 2.2 Generated / Derived Data

| Path / Pattern | Reason |
|----------------|--------|
| `data/` | All pipeline output — fully re-generatable from source |
| `*.parquet` | Binary columnar data — not diff-able, can reach GB scale |

### 2.3 Python Artifacts

| Path / Pattern | Reason |
|----------------|--------|
| `*.pyc` | Compiled Python bytecode — platform/version specific |
| `__pycache__/` | Python's bytecode cache directory |
| `*.egg-info/` | Setuptools build artefacts |
| `.mypy_cache/` | Type-checker cache |
| `.pytest_cache/` | pytest cache |
| `dist/` | Built distribution packages |

### 2.4 Generated Report Files

| Path / Pattern | Reason |
|----------------|--------|
| `*_REPORT.md` | Auto-generated markdown reports (SAMPLE_REPORT.md, VALIDATION_REPORT.md, etc.) |
| `*.log` | Runtime log files |

> [!CAUTION]
> If you accidentally stage any of these files with `git add`, remove them immediately with `git rm --cached <file>` before committing. If they have been committed in a previous commit, use `git filter-repo` (never `git filter-branch`) to purge them from history, then force-push and notify all collaborators to re-clone.

---

## 3. Why Source Data is Excluded

1. **License Restrictions** — The AGQA dataset is distributed under its own research license that prohibits redistribution. Committing these files to a (potentially public) repository would violate that license.
2. **Size** — The combined size of `AGQA_balanced/` and `AGQA_scene_graphs/` exceeds **2.5 GB**. Git (and especially GitHub) is not designed for large binary blobs; they would bloat clone times and make the repository unusable.
3. **Reproducibility via Instructions** — Data provenance is documented in [`PRODUCTION_RUNBOOK.md`](./PRODUCTION_RUNBOOK.md) and `CONTRIBUTING.md`. Any collaborator can obtain the exact same source files by following the documented download instructions, ensuring full reproducibility without violating the license.

---

## 4. Why Generated Data is Excluded

1. **Re-generatable** — The entire `data/generated/` tree is a deterministic output of running `main_pipeline.py` over the source data. There is no information in it that cannot be recovered by re-running the pipeline.
2. **Scale** — Depending on target row counts, the generated parquet files can reach **tens of gigabytes**. Storing them in git would make the repository permanently heavyweight.
3. **Binary, Not Diff-able** — Parquet files are binary. Git cannot produce meaningful diffs between versions, so storing them provides no code-review or audit value.
4. **Separation of Concerns** — Keeping code and data separate is a foundational ML engineering practice. It allows the pipeline code to be versioned independently of any particular dataset snapshot.

---

## 5. Commit Hygiene

### 5.1 What a Good Commit Looks Like

Every commit should be **atomic** (one logical change), **self-contained** (tests pass), and **clearly described**.

**Commit message format:**

```
<type>(<scope>): <short imperative summary>  ← 72 chars max

<body>
Explain *what* changed and *why*. Reference issue numbers.
Wrap body lines at 72 characters.

Closes #<issue-number>  ← if applicable
```

**Type prefixes:**

| Type | Use for |
|------|---------|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or fixing tests |
| `chore` | Dependency updates, CI config, tooling |
| `perf` | Performance improvements |

**Example good commit:**

```
feat(writer): add Snappy compression to parquet shards

Previously shards were written uncompressed, leading to ~3× larger
files on disk. Switched to Snappy codec which provides good compression
with minimal CPU overhead.

Closes #42
```

**Example bad commit:**

```
fix stuff
```

### 5.2 Branch Naming

| Pattern | Use for |
|---------|---------|
| `main` | Stable, production-ready code only |
| `feat/<short-description>` | New features (e.g., `feat/add-diversity-report`) |
| `fix/<short-description>` | Bug fixes (e.g., `fix/shard-resume-index`) |
| `docs/<short-description>` | Documentation updates |
| `chore/<short-description>` | Maintenance tasks |

- Branch names must be **lowercase with hyphens** — no underscores, no spaces.
- Delete branches after merging.
- Never commit directly to `main`; always use a pull request.

---

## 6. Pre-commit Checks

Run the following checks **before every commit**. A failed check means the commit must not proceed until fixed.

### 6.1 Mandatory Checks

```powershell
# 1. Run the full test suite — must pass with 0 failures
python -m pytest tests/ -v

# 2. Check code style (no auto-fix; fix manually if needed)
python -m ruff check src/ tests/ main_pipeline.py

# 3. Check type annotations
python -m mypy src/ --ignore-missing-imports

# 4. Verify no large files are staged (>1MB is suspicious; >10MB is forbidden)
git diff --cached --stat
```

### 6.2 Quick Checklist

Before running `git commit`, confirm:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No linting errors (`ruff check`)
- [ ] No type errors (`mypy src/`)
- [ ] No `*.pkl`, `*.parquet`, `data/`, or `AGQA_*/` files are staged
- [ ] No `*_REPORT.md` files are staged
- [ ] Commit message follows the format in §5.1
- [ ] Branch name follows the naming convention in §5.2

> [!TIP]
> Consider installing [pre-commit](https://pre-commit.com/) hooks to automate these checks. A `.pre-commit-config.yaml` can be added to the repo root to enforce ruff, mypy, and file-size limits on every `git commit` invocation automatically.
