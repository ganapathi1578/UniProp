"""Data schemas for UniProp dataset.

This module defines the core dataclasses that constitute the UniProp data model.
Every pipeline stage — generation, annotation, evaluation — produces and consumes
instances of these classes.  The hierarchy is:

    PropositionGroup
    ├── Proposition  (one or more)
    │   └── PropositionSemantics
    ├── Reasoning
    ├── Provenance
    └── Grounding  (optional)

Typical usage::

    from src.schema import PropositionGroup, Proposition, PropositionSemantics
    from src.schema import Grounding, Reasoning, Provenance

    semantics = PropositionSemantics(
        canonical_type="relation",
        subject="person",
        predicate="holding",
        object="book",
    )
    prop = Proposition(
        text="The person is holding a book.",
        truth_state="TRUE",
        label=1,
        semantics=semantics,
    )
"""

import dataclasses
from typing import List, Dict, Tuple, Optional, Any


@dataclasses.dataclass
class PropositionSemantics:
    """Structured semantic decomposition of a single proposition.

    Captures the logical form of a proposition — what type of claim it makes,
    which entities are involved, what the polarity is, and (for temporal
    propositions) how two events are ordered relative to each other.

    Attributes:
        canonical_type: The high-level semantic category of the proposition.
            One of:

            * ``"object_existence"`` – asserts whether an object is present.
            * ``"relation"`` – asserts a spatial, contact, or attention
              relationship between two entities.
            * ``"action"`` – asserts that a person performs an action.
            * ``"temporal"`` – asserts an ordering between two events.
            * ``"grounding"`` – asserts a link between a description and a
              spatial region / object identity.

        subject: The primary entity (e.g. ``"person"``, ``"bag"``).
            ``None`` when the proposition has no explicit subject (rare).
        predicate: The relation or action label connecting subject to object
            (e.g. ``"holding"``, ``"sitting on"``). ``None`` for
            object-existence propositions.
        object: The secondary entity acted upon or related to the subject
            (e.g. ``"book"``). ``None`` when there is no object.
        polarity: Indicates whether the proposition is an affirmation or
            negation. Either ``"positive"`` (default) or ``"negative"``.
        temporal_relation: For ``canonical_type="temporal"`` propositions,
            the ordering predicate between the two events (e.g.
            ``"before"``, ``"after"``, ``"during"``). ``None`` otherwise.
        event_a: The first event description in a temporal proposition.
            ``None`` for non-temporal proposition types.
        event_b: The second event description in a temporal proposition.
            ``None`` for non-temporal proposition types.
    """

    # High-level semantic type — drives downstream evaluation logic.
    canonical_type: str  # object_existence, relation, action, temporal, grounding

    # SPO triple components; some may be absent depending on canonical_type.
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None

    # Whether the claim is asserted positively or negated.
    polarity: str = "positive"  # "positive" or "negative"

    # Temporal-proposition-specific fields.
    temporal_relation: Optional[str] = None
    event_a: Optional[str] = None
    event_b: Optional[str] = None


@dataclasses.dataclass
class Proposition:
    """A single verifiable claim about a video, paired with its ground-truth label.

    A ``Proposition`` bundles the natural-language surface form of a claim
    (``text``) with the gold-standard truth assignment (``truth_state`` /
    ``label``) and a structured semantic analysis (``semantics``).

    Attributes:
        text: The natural-language surface form of the proposition, as it
            would be presented to a model or annotator.
            Example: ``"The person is sitting on the chair."``
        truth_state: Coarse string label indicating the gold truth value.
            One of:

            * ``"TRUE"``    – the proposition holds in the video.
            * ``"FALSE"``   – the proposition does not hold.
            * ``"UNKNOWN"`` – ground truth cannot be determined from the
              available evidence (e.g. the relevant segment is off-screen).

        label: Integer encoding of ``truth_state`` for binary classification
            tasks.  Conventionally ``1`` for ``"TRUE"`` and ``0`` for
            ``"FALSE"``.  Set to ``None`` when ``truth_state`` is
            ``"UNKNOWN"`` unless the enclosing task uses a three-way label
            scheme.
        semantics: Structured decomposition of the proposition's meaning;
            see :class:`PropositionSemantics`.
    """

    # Surface-form text shown to the model / annotator.
    text: str

    # Coarse gold label: "TRUE", "FALSE", or "UNKNOWN".
    truth_state: str  # "TRUE", "FALSE", "UNKNOWN"

    # Binary integer label (1=True, 0=False). None for UNKNOWN unless three_way task.
    label: Optional[int]

    # Structured semantic analysis of the proposition.
    semantics: PropositionSemantics


@dataclasses.dataclass
class Grounding:
    """Spatial and temporal grounding annotations that link a proposition to evidence.

    Stores the object identities, bounding boxes, and frame references that
    justify or falsify a proposition in the source video.  The fields are
    populated by the grounding-annotation stage of the pipeline and are
    optional for non-grounding task types.

    Attributes:
        object_ids: Ordered list of Action Genome object node IDs that are
            relevant to this proposition (e.g. ``["o8", "o16"]``).
        boxes: Mapping from object ID to its bounding box, expressed as a
            4-tuple ``(x_min, y_min, x_max, y_max)`` in normalised [0, 1]
            image coordinates.
        frame_ids: List of frame identifiers (e.g. video frame indices or
            timestamps as strings) from which the grounding evidence is drawn.
        target_object_id: The Action Genome object ID of the single focal
            object for grounding tasks (e.g. the object the model must
            locate).  ``None`` for non-grounding propositions.
        target_object_class: Human-readable class label of the target object
            (e.g. ``"chair"``).  ``None`` when ``target_object_id`` is
            ``None``.
        target_bbox: The bounding box of the target object as a 4-tuple
            ``(x_min, y_min, x_max, y_max)`` in normalised coordinates.
            ``None`` when no target object is specified.
    """

    # IDs of all Action Genome object nodes relevant to the proposition.
    object_ids: List[str] = dataclasses.field(default_factory=list)

    # Per-object bounding boxes in normalised (x_min, y_min, x_max, y_max) format.
    boxes: Dict[str, Tuple[float, float, float, float]] = dataclasses.field(
        default_factory=dict
    )

    # Frame indices / timestamps that contain the grounding evidence.
    frame_ids: List[str] = dataclasses.field(default_factory=list)

    # Focal object for grounding-type tasks; None for all other task types.
    target_object_id: Optional[str] = None

    # Human-readable class name of the focal object (e.g. "chair").
    target_object_class: Optional[str] = None

    # Bounding box of the focal object in normalised coordinates.
    target_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclasses.dataclass
class Reasoning:
    """Characterisation of the reasoning skill required to evaluate a proposition.

    Supports stratified evaluation: researchers can slice benchmark results by
    ``type``, ``family``, ``complexity``, or number of ``hops`` to diagnose
    model capabilities and failure modes.

    Attributes:
        type: Fine-grained reasoning type string that identifies the specific
            skill being tested (e.g. ``"object_relation"``,
            ``"action_sequence"``, ``"temporal_before"``).
        complexity: Integer score (typically 0–8; see
            :data:`~src.constants.COMPLEXITY_LEVELS`) that reflects how many
            logical steps or scene-graph traversal hops the proposition
            requires.  Higher values indicate harder examples.
        family: Broad reasoning family that groups related types together.
            One of:

            * ``"object"``        – object existence or attribute queries.
            * ``"attribute"``     – object attribute (colour, size, …).
            * ``"spatial"``       – spatial relations (above, in front of, …).
            * ``"attention"``     – gaze / attention relations.
            * ``"contact"``       – physical contact relations.
            * ``"action"``        – action-recognition reasoning.
            * ``"temporal"``      – event-ordering reasoning.
            * ``"compositional"`` – multi-step, multi-relation chains.
            * ``"grounding"``     – spatial localisation of an entity.

            Defaults to ``"object"``.
        hops: Number of scene-graph traversal hops required to verify the
            proposition.  Single-hop propositions (``hops=1``) involve one
            direct relation; multi-hop propositions chain several relations or
            actions.  Defaults to ``1``.
    """

    # Fine-grained label used for per-type breakdown in evaluation reports.
    type: str

    # Numeric difficulty level; see constants.COMPLEXITY_LEVELS for the range.
    complexity: int

    # Broad category used for high-level evaluation slicing.
    family: str = "object"  # object, attribute, spatial, attention, contact, action, temporal, compositional, grounding

    # Number of reasoning hops (scene-graph edges) needed to verify the claim.
    hops: int = 1


@dataclasses.dataclass
class Provenance:
    """Audit trail recording how a proposition or proposition group was generated.

    Every ``PropositionGroup`` carries a ``Provenance`` instance so that
    generated examples can be traced back to their source material (the
    original AGQA question, the scene graph, the generation rule, etc.).
    This supports reproducibility, error analysis, and re-generation.

    Attributes:
        source_question_id: Unique identifier of the original AGQA question
            from which this proposition was derived.  ``None`` if the
            proposition was not derived from a question (e.g. purely
            scene-graph generated).
        source_question_text: Verbatim text of the original AGQA question.
            ``None`` when ``source_question_id`` is ``None``.
        source_program: Functional program (in AGQA's DSL) associated with
            the source question.  ``None`` when not applicable.
        source_answer: Gold answer string to the original AGQA question
            (e.g. ``"yes"``, ``"holding"``).  ``None`` when not applicable.
        scene_graph_id: Identifier of the Action Genome scene graph frame or
            video segment used as the evidence source.  ``None`` when the
            proposition was not grounded in a specific scene graph.
        generation_rule: Human-readable name of the rule or template used to
            generate the proposition (e.g.
            ``"positive_relation_from_sg_edge"``).
        generation_seed: Integer random seed used during generation for full
            reproducibility of the sampling / perturbation steps.
        generator_family: Broad category of the generator that produced this
            proposition.  Defaults to ``"scene_graph"``; other values include
            ``"question_derived"`` and ``"temporal_chain"``.
        evidence_object_ids: Object node IDs from the scene graph that
            directly support the proposition's truth value.
        evidence_relation_ids: Relation / edge IDs from the scene graph that
            directly support the proposition's truth value.
        frame_ids: Frame identifiers from which the scene graph evidence was
            extracted.
        temporal_chain: For temporally derived propositions, an ordered list
            of dicts describing each event in the chain (e.g.
            ``[{"action": "holding bag", "frame": "23"}]``).  ``None`` for
            non-temporal propositions.
        source_global: Identifier or description of the global (video-level)
            source context, if any.  ``None`` otherwise.
        source_local: Identifier or description of the local (clip- or
            frame-level) source context, if any.  ``None`` otherwise.
        source_semantic: Tag for the semantic source pathway used during
            generation (e.g. ``"agqa_compositional"``).  ``None`` otherwise.
        source_structural: Tag for the structural source pathway (e.g.
            ``"scene_graph_edge"``).  ``None`` otherwise.
        source_sg_grounding: Identifier linking this provenance entry to a
            specific scene-graph grounding annotation.  ``None`` otherwise.
    """

    # --- Source question fields (populated when derived from an AGQA QA pair) ---

    # Original AGQA question ID; None for purely scene-graph-generated entries.
    source_question_id: Optional[str]

    # Verbatim source question text.
    source_question_text: Optional[str]

    # Functional program from the AGQA DSL that produced the source question.
    source_program: Optional[str]

    # Gold-standard answer to the source AGQA question.
    source_answer: Optional[str]

    # --- Scene graph fields ---

    # Action Genome scene graph ID (frame-level or video-level).
    scene_graph_id: Optional[str]

    # --- Generation metadata ---

    # Name of the rule/template that generated this proposition.
    generation_rule: str

    # Random seed for reproducible generation.
    generation_seed: int

    # Broad generator family (e.g. "scene_graph", "question_derived").
    generator_family: str = "scene_graph"

    # Object node IDs that directly evidence the proposition's truth value.
    evidence_object_ids: List[str] = dataclasses.field(default_factory=list)

    # Relation/edge IDs that directly evidence the proposition's truth value.
    evidence_relation_ids: List[str] = dataclasses.field(default_factory=list)

    # Frame IDs from which scene-graph evidence was drawn.
    frame_ids: List[str] = dataclasses.field(default_factory=list)

    # Ordered event chain for temporal propositions; None otherwise.
    temporal_chain: Optional[List[dict]] = None

    # Optional pathway / context tags for fine-grained provenance tracking.
    source_global: Optional[str] = None
    source_local: Optional[str] = None
    source_semantic: Optional[str] = None
    source_structural: Optional[str] = None
    source_sg_grounding: Optional[str] = None


@dataclasses.dataclass
class PropositionGroup:
    """A self-contained evaluation example consisting of one or more propositions.

    A ``PropositionGroup`` is the top-level unit of the UniProp dataset.  It
    wraps a set of related :class:`Proposition` objects together with metadata
    that describes where the example came from (``source``, ``provenance``),
    what reasoning skill it targets (``reasoning``), how the model is expected
    to respond (``task_type``), and (optionally) what spatial grounding is
    required (``grounding``).

    Attributes:
        example_id: Globally unique identifier for this example across all
            sources and splits (e.g. ``"agqa_v2_train_00012345"``).
        source: Name of the dataset or sub-corpus from which this example was
            derived (e.g. ``"agqa_v2"``, ``"action_genome"``).
        split: Dataset partition assignment.  One of ``"train"``,
            ``"val"``, or ``"test"``.
        media_id: Identifier of the source video or image (e.g. a Charades
            video ID such as ``"AAAAA"``).  Used to look up the raw media.
        query_text: Optional natural-language prompt or question presented to
            the model alongside the propositions.  ``None`` for task types
            that do not use a query (e.g. plain binary classification).
        query_template_id: Optional identifier of the template used to
            generate ``query_text``.  Useful for analysing template-level
            performance variation.  ``None`` when no template was used.
        propositions: Ordered list of :class:`Proposition` objects belonging
            to this group.  For binary tasks this typically has length 1; for
            multi-choice or multi-label tasks it may have several.
        task_type: The evaluation protocol applied to this example.  One of
            the values in :data:`~src.constants.TASK_TYPES`, e.g.
            ``"binary"``, ``"single_choice"``, ``"grounding"``.
        num_propositions: Redundant count of ``len(propositions)``, stored
            explicitly for convenience and fast filtering without loading the
            full proposition list.
        reasoning: Reasoning-skill characterisation; see :class:`Reasoning`.
        provenance: Audit trail for the example; see :class:`Provenance`.
        grounding: Optional spatial / temporal grounding annotations; see
            :class:`Grounding`.  ``None`` for non-grounding task types.
    """

    # Globally unique example identifier across all sources and splits.
    example_id: str

    # Source corpus name (e.g. "agqa_v2", "action_genome").
    source: str

    # Dataset split: "train", "val", or "test".
    split: str

    # Source video / image identifier (e.g. Charades video ID).
    media_id: str

    # Optional natural-language query accompanying the propositions.
    query_text: Optional[str]

    # Optional ID of the template used to produce query_text.
    query_template_id: Optional[str]

    # The propositions that constitute this evaluation example.
    propositions: List[Proposition]

    # Evaluation protocol for this example; see constants.TASK_TYPES.
    task_type: str

    # Cached count of propositions — equals len(self.propositions).
    num_propositions: int

    # Reasoning-skill metadata for stratified evaluation.
    reasoning: Reasoning

    # Audit trail recording how this example was generated.
    provenance: Provenance

    # Spatial/temporal grounding annotations (None for non-grounding tasks).
    grounding: Optional[Grounding] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the ``PropositionGroup`` to a plain Python dictionary.

        Recursively converts all nested dataclass instances (``Proposition``,
        ``PropositionSemantics``, ``Reasoning``, ``Provenance``, and
        optionally ``Grounding``) using :func:`dataclasses.asdict`, producing
        a structure that is directly JSON-serialisable via ``json.dumps``.

        Returns:
            A ``dict`` with the following top-level keys:

            * ``"example_id"`` (:class:`str`) – unique example identifier.
            * ``"source"`` (:class:`str`) – originating corpus name.
            * ``"split"`` (:class:`str`) – dataset partition.
            * ``"media_id"`` (:class:`str`) – source video / image ID.
            * ``"query_text"`` (:class:`str` | ``None``) – optional query.
            * ``"query_template_id"`` (:class:`str` | ``None``) – optional
              template ID used to generate ``query_text``.
            * ``"propositions"`` (:class:`list` of :class:`dict`) – serialised
              :class:`Proposition` objects, each containing ``"text"``,
              ``"truth_state"``, ``"label"``, and ``"semantics"``.
            * ``"task_type"`` (:class:`str`) – evaluation protocol.
            * ``"num_propositions"`` (:class:`int`) – number of propositions.
            * ``"reasoning"`` (:class:`dict`) – serialised :class:`Reasoning`.
            * ``"provenance"`` (:class:`dict`) – serialised :class:`Provenance`.
            * ``"grounding"`` (:class:`dict` | ``None``) – serialised
              :class:`Grounding`, or ``None`` when no grounding is attached.

        Example::

            group = PropositionGroup(...)
            record = group.to_dict()
            with open("output.json", "w") as f:
                json.dump(record, f, indent=2)
        """
        return {
            "example_id": self.example_id,
            "source": self.source,
            "split": self.split,
            "media_id": self.media_id,
            "query_text": self.query_text,
            "query_template_id": self.query_template_id,
            # Recursively convert each Proposition (and its nested
            # PropositionSemantics) to a plain dict via dataclasses.asdict.
            "propositions": [dataclasses.asdict(p) for p in self.propositions],
            "task_type": self.task_type,
            "num_propositions": self.num_propositions,
            # Flatten the Reasoning dataclass to a dict.
            "reasoning": dataclasses.asdict(self.reasoning),
            # Flatten the Provenance dataclass to a dict.
            "provenance": dataclasses.asdict(self.provenance),
            # Grounding is optional; guard with an explicit None check.
            "grounding": dataclasses.asdict(self.grounding) if self.grounding else None,
        }
