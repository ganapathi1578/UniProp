# UniProp — Documentation Hub

Welcome to the **UniProp** dataset-engineering documentation.  UniProp is a
production pipeline that synthesises richly-labelled video-understanding
proposition datasets from AGQA spatio-temporal scene graphs.

This hub is the single entry point for all documentation.  Use the paths below
to find exactly what you need, whether you are running the pipeline for the
first time, debugging a failure, extending the system with a new generator, or
auditing a published dataset for reproducibility.

---

## Quick Navigation

| Document | Description | Who should read it |
|---|---|---|
| [INSTALLATION.md](INSTALLATION.md) | Environment setup, dependency installation, and first-run verification | Everyone — start here |
| [CONFIGURATION.md](CONFIGURATION.md) | All runtime knobs: paths, counts, seeds, and feature flags | Engineers running or tuning generation |
| [GENERATION.md](GENERATION.md) | End-to-end guide to running the pipeline from raw data to Parquet shards | Engineers running generation |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, component diagram, and data-flow overview | Engineers extending or reviewing the system |
| [DATA_SCHEMA.md](DATA_SCHEMA.md) | Parquet schema, column types, constraints, and example rows | Anyone consuming or validating the dataset |
| [LABEL_SEMANTICS.md](LABEL_SEMANTICS.md) | Meaning of every label field, proposition type taxonomy, and edge cases | Researchers and ML practitioners |
| [VALIDATION.md](VALIDATION.md) | Validation rules, how to run the validator, and how to interpret reports | Engineers and QA |
| [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md) | Step-by-step operational procedures for full production runs | Engineers running large-scale generation |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Symptom / Cause / Fix for 17 known failure modes | Anyone encountering errors |
| [RESOURCE_REQUIREMENTS.md](RESOURCE_REQUIREMENTS.md) | RAM, disk, CPU, and time estimates for train/test generation | Engineers planning infrastructure |
| [DEVELOPMENT.md](DEVELOPMENT.md) | How to add a new generator, renderer, or label type | Contributors and researchers |
| [PROVENANCE.md](PROVENANCE.md) | Dataset lineage, source-data citations, and derivation record | Researchers publishing results |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Seed policy, environment pinning, and exact-replay instructions | Researchers and auditors |
| [GIT_POLICY.md](GIT_POLICY.md) | Branch strategy, commit conventions, and PR requirements | All contributors |
| [DATA_LAYOUT.md](DATA_LAYOUT.md) | Directory structure of source data, generated data, and checkpoints | Anyone working with the file system |

---

## Learning Paths

### 🚀 Getting Started

Follow this path if you are setting up UniProp for the first time.

```
INSTALLATION → CONFIGURATION → GENERATION
```

1. **[INSTALLATION.md](INSTALLATION.md)** — Create a virtual environment,
   install all dependencies from `requirements.txt`, and confirm the
   environment is healthy with the built-in sanity check.
2. **[CONFIGURATION.md](CONFIGURATION.md)** — Understand which settings
   you must provide (data paths, target counts) and which have sensible
   defaults.  Copy `.env.example` to `.env` and populate it.
3. **[GENERATION.md](GENERATION.md)** — Run the pipeline end-to-end for both
   the train and test splits, monitor progress, and verify the output shards.

---

### 📖 Reference

Follow this path to understand the system's design and data model in depth.

```
ARCHITECTURE → DATA_SCHEMA → LABEL_SEMANTICS → VALIDATION
```

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — Understand how the loaders,
   generators, renderers, validators, and writers fit together.  Includes a
   component diagram and data-flow description.
2. **[DATA_SCHEMA.md](DATA_SCHEMA.md)** — The authoritative specification of
   every column in the output Parquet files: name, Arrow type, nullability, and
   semantic meaning.
3. **[LABEL_SEMANTICS.md](LABEL_SEMANTICS.md)** — Deep-dive into the
   proposition taxonomy (spatial, temporal, action, count, …), what each label
   value means, and known ambiguities.
4. **[VALIDATION.md](VALIDATION.md)** — The full list of validation rules
   enforced at generation time, how to run the standalone validator, and how to
   read the JSON validation report.

---

### ⚙️ Operations

Follow this path if you are running or maintaining a production generation job.

```
PRODUCTION_RUNBOOK → TROUBLESHOOTING → RESOURCE_REQUIREMENTS
```

1. **[PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md)** — Detailed step-by-step
   procedures for a full production run: pre-run checks, launch commands,
   monitoring, checkpoint management, and post-run validation.
2. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — Quick-reference guide for
   the 17 most common failure modes.  Each entry provides the exact error
   message, root cause, and actionable fix.
3. **[RESOURCE_REQUIREMENTS.md](RESOURCE_REQUIREMENTS.md)** — RAM, disk space,
   CPU core, and wall-clock time estimates for train and test generation at
   various scales.  Use this to size your machine before starting a run.

---

### 🔧 Extension

Follow this path if you are adding a new generator, renderer, or label type.

```
DEVELOPMENT → PROVENANCE → REPRODUCIBILITY
```

1. **[DEVELOPMENT.md](DEVELOPMENT.md)** — Walkthrough for writing a new
   generator class, registering it with the pipeline, implementing a renderer,
   and writing the required unit tests.  Includes a template generator you can
   copy.
2. **[PROVENANCE.md](PROVENANCE.md)** — Records the lineage of every dataset
   artefact: which AGQA version was used as source data, which pipeline version
   produced each shard, and how to cite the dataset correctly in publications.
3. **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** — Documents the seed policy,
   environment-pinning strategy, and the exact commands needed to produce a
   byte-for-byte identical dataset on any conforming machine.

---

### 📋 Policy

These documents govern how the repository and data assets are managed.

| Document | Summary |
|---|---|
| [GIT_POLICY.md](GIT_POLICY.md) | Branch naming, commit message format, PR review requirements, and release tagging conventions |
| [DATA_LAYOUT.md](DATA_LAYOUT.md) | Canonical directory structure for source data, checkpoints, and generated output; what is tracked in git vs. stored externally |

---

## Document Descriptions

### [INSTALLATION.md](INSTALLATION.md)
Covers all steps from a clean machine to a working UniProp environment: Python
version requirements, virtual-environment creation, `pip install -r
requirements.txt`, placement of AGQA source files, and the `python -m
src.tools.health_check` command that verifies every dependency and data path.
Read this document first before attempting any other step.

### [CONFIGURATION.md](CONFIGURATION.md)
Describes every configuration key in `.env` and `src/config/settings.py`,
including data paths, per-split target counts, global random seed, logging
verbosity, and shard size.  Explains precedence rules (environment variable >
`.env` file > code default) and provides a commented `.env.example` template.

### [GENERATION.md](GENERATION.md)
End-to-end walkthrough of a generation run: loading scene graphs, running
all registered generators, deduplication, validation, checkpointing, and final
Parquet shard writing.  Includes commands for both the train and test splits,
how to perform a dry-run, and how to limit output for testing.

### [ARCHITECTURE.md](ARCHITECTURE.md)
Describes the high-level architecture of UniProp: the loader layer (AGQA
scene-graph deserialisation), the generator layer (proposition group
production), the renderer layer (natural-language surface realisation), the
validator layer (schema and semantic checks), and the writer layer (Parquet
shard output with checkpointing).  Includes a Mermaid component diagram.

### [DATA_SCHEMA.md](DATA_SCHEMA.md)
The authoritative, versioned specification of the output Parquet schema.
Every column is documented with its Arrow data type, whether it is nullable,
its allowed values (for enum columns), and at least one concrete example value.
Also documents schema evolution history and backward-compatibility guarantees.

### [LABEL_SEMANTICS.md](LABEL_SEMANTICS.md)
Defines the meaning of every `proposition_type` value, the semantics of
`label` (True/False/Unknown), spatial and temporal relation taxonomies, and
the criteria used to assign each label.  Includes worked examples and known
edge cases where labelling is deliberately conservative.

### [VALIDATION.md](VALIDATION.md)
Documents all validation rules applied to each generated `PropositionGroup`
before it is written: required fields, type constraints, text-quality checks,
split-contamination detection, and label-consistency rules.  Explains how to
run the standalone validator script and how to interpret the JSON report it
produces.

### [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md)
A step-by-step operations manual for large-scale production runs.  Covers
pre-run environment checks, the exact launch commands for train and test splits,
real-time monitoring with `tqdm` and log tailing, checkpoint inspection, how to
safely resume an interrupted run, and post-run validation procedures.

### [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
A quick-reference guide to 17 known failure modes covering missing data files,
missing Python packages, corrupted Parquet shards, assertion failures, memory
errors, stuck progress bars, validator rejections, seed/reproducibility issues,
rendering bugs, and import errors.  Each entry follows Symptom → Cause → Fix
format with copy-paste commands.

### [RESOURCE_REQUIREMENTS.md](RESOURCE_REQUIREMENTS.md)
Provides RAM, disk, CPU, and wall-clock time estimates for generating the train
and test splits at full scale, as well as for small exploratory runs.  Includes
a hardware recommendation table and guidance on how to estimate resource needs
for custom target counts.

### [DEVELOPMENT.md](DEVELOPMENT.md)
The contributor guide for extending UniProp.  Explains the generator and
renderer interfaces, the registration mechanism, testing conventions (unit tests
with fixed seeds, integration tests against a mini scene-graph fixture), and the
code-review checklist that must be satisfied before a new generator is merged.

### [PROVENANCE.md](PROVENANCE.md)
Records the lineage of every dataset artefact: AGQA source version and download
URLs, pipeline git commit hash used for each release, per-shard generation
metadata embedded in Parquet file metadata, and the correct BibTeX citation for
both AGQA and UniProp.

### [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
Documents the full reproducibility policy: how `PYTHONHASHSEED`, per-generator
seeds, and the global seed interact; the `requirements.txt` pinning strategy;
and the exact shell commands needed to reproduce a specific dataset release
byte-for-byte on a conforming machine.

### [GIT_POLICY.md](GIT_POLICY.md)
Defines branch naming conventions (`feat/`, `fix/`, `data/`, `docs/`), commit
message format (Conventional Commits), PR size limits, required review
approvals, CI gate requirements, and the release tagging and changelog update
process.

### [DATA_LAYOUT.md](DATA_LAYOUT.md)
Documents the canonical directory tree expected by the pipeline: where AGQA
source files must be placed, where checkpoints and generated shards are written,
what is committed to git versus stored in external storage, and how to set up
symlinks for large external data volumes.

---

> [!NOTE]
> All paths in the documentation are relative to the UniProp project root
> (`c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp`).  When a document
> references `src/`, `data/`, or `AGQA_balanced/`, the path starts from there.
