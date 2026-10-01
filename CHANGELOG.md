# Changelog

## Current Release
- **Refactor**: Finalized UniProp dataset pipeline and documentation.
- **Removed**: Tokenized Storage V1 architecture, `MobileCLIP` dependencies, token dictionaries, and `[MASK]` logic completely removed.
- **Restored**: Raw-text Parquet storage schema locked in.
- **Provenance**: Explicit `video_id` tracking locked into schema and validation.
- **Semantics**: UNKNOWN and NONE truth-state semantics preserved natively without silent loss.
- **Documentation**: Added full production documentation suite.
