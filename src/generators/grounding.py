"""Bounding-box grounding proposition generator for the UniProp pipeline.

This module provides :class:`GroundingGenerator`, which produces
:class:`~src.schema.PropositionGroup` instances that require a model to
spatially ground its predictions to a specific **bounding box** within a
video frame.

Grounding tasks
    Unlike purely linguistic proposition tasks, grounding tasks attach a
    :class:`~src.schema.Grounding` record to every emitted group.  This record
    carries:

    * ``object_ids`` – the scene-graph object(s) whose bounding box is shown.
    * ``boxes`` – a mapping from object ID to the raw bounding-box coordinates
      in the selected frame.
    * ``frame_ids`` – the frame(s) from which the box was taken.
    * ``target_object_id`` / ``target_object_class`` / ``target_bbox`` – the
      specific object and box that the question refers to (``None`` for the
      negative branch of ``box_proposition``).

    Downstream evaluation code uses this metadata to crop the relevant region
    of the frame and present it to the model (or to score spatial predictions).

Frame filtering
    Only frames that contain **at least one object with a non-``None`` bounding
    box AND a non-empty name** are considered.  Frames without bounding-box
    annotations are skipped entirely because:

    1. No visual region can be presented to the model without a box.
    2. Grounding metadata would be incomplete, breaking downstream evaluation.

Task formats
    Two sub-tasks are supported, chosen uniformly at random:

    ``box_object`` (``single_choice``)
        *"Which object is in this bounding box?"*  The model receives a cropped
        region and must identify the correct object name from ``k`` options.

    ``box_proposition`` (``three_way``)
        *"Is <object> present at this location?"*  The model receives a
        bounding box and must choose among *Yes* / *No* / *Cannot say*.

Typical usage::

    gen = GroundingGenerator()
    for group in gen.generate(qid, qdata, sg, split, rng, pools):
        dataset.append(group)
"""

import random
from typing import Iterator
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance, Grounding
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS
from src.generators.renderer import render_noun_phrase
from src.templates import get_query


class GroundingGenerator(GeneratorBase):
    """Generator that produces bounding-box grounding proposition groups.

    For each clip, the generator:

    1. Scans every annotated frame and retains only those that contain at
       least one object with a valid bounding box and a non-empty name.
    2. Randomly selects a qualifying frame and one of its boxed objects.
    3. Randomly dispatches to either ``box_object`` or ``box_proposition``
       and constructs the corresponding propositions and
       :class:`~src.schema.Grounding` metadata.

    Inherits from :class:`~src.generators.base.GeneratorBase` and implements
    its :meth:`generate` abstract method.

    Attributes:
        This class holds no instance-level state; all generation is driven by
        the arguments supplied to :meth:`generate`.
    """

    def generate(
        self,
        qid: str,
        qdata: dict,
        sg: NormalizedSceneGraph,
        split: str,
        rng: random.Random,
        pools: NegativePools,
    ) -> Iterator[PropositionGroup]:
        """Generate a single bounding-box grounding :class:`PropositionGroup`.

        Execution flow:

        1. **Frame filtering** – iterates all frames in ``sg.frames`` and
           collects ``(frame_id, objs_with_bbox)`` tuples where at least one
           object carries a bounding box and a name.  Frames that lack any
           boxed object are silently skipped because a grounding task cannot
           be constructed without spatial evidence.
        2. **Guard** – returns immediately if no valid frame exists.
        3. **Object selection** – picks a random qualifying frame and then a
           random object within that frame.
        4. **Task dispatch** – uniformly samples ``"box_object"`` or
           ``"box_proposition"`` and delegates to the corresponding branch.

        Args:
            qid: Unique identifier for the source question (unused here but
                required by the base interface).
            qdata: Raw question metadata dict (unused here but required by the
                base interface).
            sg: Normalised scene graph providing ``frames`` (each frame
                carrying ``objects``, each object carrying ``bbox``, ``name``,
                ``id``, and ``class_id``) and ``video_id``.
            split: Dataset partition label (e.g. ``"train"``, ``"val"``,
                ``"test"``).
            rng: Seeded random number generator for reproducibility.
            pools: Pre-computed hard-negative pools (unused by this generator
                but required by the base interface).

        Yields:
            A single :class:`~src.schema.PropositionGroup` with an attached
            :class:`~src.schema.Grounding` object.  The group's
            ``reasoning.family`` is ``"grounding"`` and
            ``provenance.generator_family`` is ``"grounding"``.

        Returns:
            Nothing if no frame in the scene graph contains a boxed, named
            object.
        """
        # Collect frames that provide at least one named, boxed object.
        # Frames without bounding boxes are excluded because:
        #   (a) the grounding task requires a visual region to present, and
        #   (b) the Grounding schema fields would be undefined without a bbox.
        valid_frames = []
        for frame in sg.frames.values():
            objs_with_bbox = [o for o in frame.objects.values() if o.bbox is not None and o.name]
            if objs_with_bbox:
                valid_frames.append((frame.id, objs_with_bbox))

        if not valid_frames:
            # No spatial evidence available in this clip; skip generation.
            return

        # Choose the frame and the specific object whose box will be shown.
        frame_id, objs = rng.choice(valid_frames)
        obj = rng.choice(objs)
        obj_name = obj.name

        task = rng.choice(["box_object", "box_proposition"])

        if task == "box_object":
            # Which object is in this box? (K=10-40)
            # Single-choice identification: the model sees a cropped bounding
            # box and must select the correct object name from k options.
            k = rng.randint(10, 40)

            # True proposition: the object that actually resides in the box.
            propositions = [Proposition(text=obj_name, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=obj_name, polarity="positive"))]

            # False candidates: all known OBJECTS minus those already visible
            # in the selected frame (to avoid accidentally including the truth).
            all_absent = list(set(OBJECTS.values()) - {o.name for o in objs})
            falses = rng.sample(all_absent, min(len(all_absent), k - 1))

            for f in falses:
                propositions.append(Proposition(text=f, truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=f, polarity="positive")))

            rng.shuffle(propositions)

            tid, q_text = get_query("GROUNDING_SINGLE", rng)
            # Attach full spatial metadata so evaluation code can reconstruct
            # the exact region shown to the model.
            grounding = Grounding(object_ids=[obj.id], boxes={obj.id: obj.bbox}, frame_ids=[frame_id], target_object_id=obj.id, target_object_class=obj.class_id, target_bbox=obj.bbox)

            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"grnd_id_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="grounding_identification", complexity=1, family="grounding"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_id", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id], frame_ids=[frame_id]),
                grounding=grounding
            )

        elif task == "box_proposition":
            # Is <object> present at this location? Yes / No / Cannot say
            # Three-way verification: the model sees a bounding-box crop and
            # must decide whether a named object occupies that region.
            is_pos = rng.random() > 0.5
            # Positive branch: ask about the object that actually IS in the box.
            # Negative branch: ask about a different object that is NOT in the frame,
            # making the expected answer "No".
            target_obj_name = obj_name if is_pos else rng.choice(list(set(OBJECTS.values()) - {o.name for o in objs}))

            q_text = f"Is {render_noun_phrase(target_obj_name, False)} present at this location?"

            # Fixed three-way answer set: Yes / No / Cannot say.
            propositions = [
                Proposition(text="Yes", truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive")),
                Proposition(text="No", truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="negative")),
                Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive"))
            ]

            # For positive cases the target metadata points to the confirmed object;
            # for negative cases target_object_id and class are set to None because
            # the named object is not present in the frame.
            grounding = Grounding(object_ids=[obj.id], boxes={obj.id: obj.bbox}, frame_ids=[frame_id], target_object_id=obj.id if is_pos else None, target_object_class=obj.class_id if is_pos else None, target_bbox=obj.bbox)

            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"grnd_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="grounding_verification", complexity=1, family="grounding"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_3w", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id] if is_pos else [], frame_ids=[frame_id]),
                grounding=grounding
            )
