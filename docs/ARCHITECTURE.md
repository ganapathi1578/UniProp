# UniProp Architecture

UniProp is a general dataset-engineering framework.

## Components
- **Datasets**: Handled via adapters in `src/datasets`.
- **Candidates**: Generators for different options in `src/candidates`.
- **Truth**: Evaluators in `src/truth`.
- **Storage**: Parquet writers in `src/storage`.
