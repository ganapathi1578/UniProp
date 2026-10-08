"""Canonical intermediate representation for a single QA record.

This module defines :class:`NormalizedQA`, the single shared data structure
that every dataset adapter must produce.  By converging all source formats
into one schema here, downstream components (candidate generators, validators,
writers) remain completely decoupled from source-specific quirks.

Typical usage::

    from src.datasets.normalized import NormalizedQA

    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="q_001",
        video_id="vid_042",
        query="What did the person do before sitting?",
        source_answer="walk",
        answer_type="action",
        semantic_type="action",
        structural_type="query",
        reasoning_type="superlative-action",
    )
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class NormalizedQA:
    """Canonical representation of a single QA record after source normalisation.

    Every dataset adapter converts its native row/dict into a ``NormalizedQA``
    before handing it to the pipeline.  Fields are split into three groups:

    * **Identity** – uniquely locate the example in the source dataset.
    * **Content** – the question text and the ground-truth answer.
    * **Typing** – abstract and structural labels used to route the record to
      the correct candidate generator.
    * **Optional extras** – program text, scene-graph hooks, and raw metadata
      preserved for reproducibility and debugging.

    Attributes:
        source_dataset: Human-readable key that identifies the originating
            dataset (e.g. ``"agqa_balanced"``).  Must match the name used to
            register the adapter in :class:`~src.datasets.registry.DatasetRegistry`.
        source_question_id: Primary key for the question as assigned by the
            source dataset (e.g. a QID string from AGQA).  Used to build the
            ``example_id`` for the output :class:`~src.schema.PropositionGroup`.
        video_id: Identifier of the video (or image / clip) that this question
            refers to.  Propagated as ``media_id`` in the output schema.
        query: The natural-language question text shown to the model.
        source_answer: The ground-truth answer string exactly as it appears in
            the source dataset, before any normalisation by this pipeline.
        answer_type: Abstract answer category inferred (or read) by the adapter.
            The pipeline uses this value to look up a
            :class:`~src.candidates.base.CandidateGenerator` in
            :class:`~src.candidates.registry.CandidateRegistry`.
            Recognised values include ``"binary"``, ``"temporal"``,
            ``"count"``, ``"comparison"``, ``"superlative"``,
            ``"action"``, ``"object"``, and ``"open"``.
        semantic_type: Coarse semantic category of the question, as supplied by
            the source dataset (e.g. ``"action"``, ``"object"``).  Stored in
            :class:`~src.schema.PropositionSemantics` and
            :class:`~src.schema.Reasoning`.
        structural_type: Structural / syntactic category of the question as
            labelled by the source dataset (e.g. ``"query"``, ``"count"``,
            ``"compare"``).  Informs answer-type inference in the adapter.
        reasoning_type: Compound reasoning chain label, typically a
            hyphen-joined list of the ``global`` tags from AGQA
            (e.g. ``"superlative-action-object"``).  Stored in
            :class:`~src.schema.Reasoning`.
        program: Structured program string produced by this pipeline, if any.
            Defaults to an empty string when no program has been generated yet.
        sg_grounding: Mapping from program variables or object references to
            scene-graph node identifiers.  Populated during downstream
            grounding steps.  Defaults to an empty dict.
        source_program: The original functional programme / compositional
            expression shipped with the source dataset, if present.  Kept
            verbatim for provenance.  ``None`` when not available.
        scenegraph_reference: A reference to the parsed scene-graph object
            (or a lazy-load key) for this video.  Populated on demand by the
            :class:`~src.reasoning.evidence.TruthEvaluator`.  ``None`` by
            default; filled during the optionisation step.
        source_metadata: The complete raw dict (or object) from the source
            dataset row, preserved without modification for debugging and
            future feature extraction.  ``None`` when not retained.
    """

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #
    source_dataset: str
    """Key of the originating dataset (must match the registry name)."""

    source_question_id: str
    """Primary key assigned by the source dataset."""

    video_id: str
    """Identifier of the associated video / clip / image."""

    # ------------------------------------------------------------------ #
    # Content                                                              #
    # ------------------------------------------------------------------ #
    query: str
    """Natural-language question text presented to the model."""

    source_answer: str
    """Ground-truth answer string verbatim from the source dataset."""

    # ------------------------------------------------------------------ #
    # Typing / routing                                                     #
    # ------------------------------------------------------------------ #
    answer_type: str
    """Abstract answer category used to select a CandidateGenerator."""

    semantic_type: str
    """Coarse semantic label from the source dataset (e.g. 'action', 'object')."""

    structural_type: str
    """Structural / syntactic label from the source dataset (e.g. 'query', 'count')."""

    reasoning_type: str
    """Hyphen-joined chain of reasoning tags (e.g. 'superlative-action')."""

    # ------------------------------------------------------------------ #
    # Optional extras                                                      #
    # ------------------------------------------------------------------ #
    program: str = ""
    """Pipeline-generated structured program string (empty until produced)."""

    sg_grounding: dict = field(default_factory=dict)
    """Variable-to-scene-graph-node mapping populated during grounding."""

    source_program: Optional[str] = None
    """Original functional programme from the source dataset, if any."""

    scenegraph_reference: Optional[Any] = None
    """Lazy reference to the parsed scene-graph; filled by TruthEvaluator."""

    source_metadata: Optional[Dict[str, Any]] = None
    """Complete raw source row preserved for debugging and provenance."""
