"""Compositional (AND-relation) proposition generator for the UniProp pipeline.

This module provides :class:`CompositionalGenerator`, which produces
:class:`~src.schema.PropositionGroup` instances that require a model to
evaluate a **conjunction of two relations** between a person and an object.

Compositional reasoning
    A *compositional AND* proposition has the surface form::

        "The person is <r1> and <r2> the <object>."

    Both relations ``r1`` and ``r2`` must hold simultaneously for the
    proposition to be TRUE.  A false proposition is constructed by replacing
    one of the true relations with a **compatible but absent** relation ``r3``,
    making the conjunction fail::

        "The person is <r1> and <r3> the <object>."  # FALSE (r3 not present)

    This formulation tests whether models can track multiple concurrent
    relational facts and correctly reject propositions where only one of the
    two relations is satisfied.

Negative generation strategy
    False relations are drawn from ``ALL_RELATIONS`` minus the object's
    confirmed relations, then filtered by :func:`~src.compatibility.is_compatible`
    to guarantee semantic plausibility (e.g. a person cannot *wear* a sofa).
    This prevents models from solving the task by detecting absurd combinations
    rather than by inspecting the visual evidence.

Typical usage::

    gen = CompositionalGenerator()
    for group in gen.generate(qid, qdata, sg, split, rng, pools):
        dataset.append(group)
"""

import random
from typing import Iterator
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS, ALL_RELATIONS
from src.compatibility import is_compatible


class CompositionalGenerator(GeneratorBase):
    """Generator that produces AND-compositional proposition groups.

    For each clip, the generator:

    1. Builds a mapping from each object to the set of relations the person
       holds with it, using the pre-computed ``pools.present_relations``.
    2. Selects a *target object* that participates in **at least two** distinct
       relations (otherwise AND-composition is impossible).
    3. Randomly picks two confirmed relations ``r1`` and ``r2`` as the true
       conjunction.
    4. Selects one or more **absent but compatible** relations as false
       substitutes to form distractor propositions.
    5. Emits a ``binary`` or ``single_choice`` task with the composed
       propositions.

    Inherits from :class:`~src.generators.base.GeneratorBase` and implements
    its :meth:`generate` abstract method.

    Note:
        Relation names retrieved from ``ALL_RELATIONS`` are normalised at
        generation time:

        * ``"have it on the back"`` → ``"carrying on their back"``
        * ``"other relationship"`` → ``"interacting with"``

        This ensures the surface text is grammatically natural and avoids
        exposing internal relation-label artefacts to downstream models.
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
        """Generate a single AND-compositional :class:`PropositionGroup`.

        Execution flow:

        1. **Guard** – returns immediately if ``pools`` is ``None`` or
           ``pools.present_relations`` is empty (no relational evidence to
           compose).
        2. **Index relations by object** – iterates ``present_relations`` tuples
           ``(r_type, r_class, obj)`` and groups ``r_class`` values under each
           ``obj`` key.
        3. **Filter valid objects** – retains only objects associated with
           ``>= 2`` distinct relation classes (required for AND-composition).
           Returns if none qualify.
        4. **Sample target** – picks a random valid object and two of its
           confirmed relations as ``r1`` / ``r2``.
        5. **Relation name normalisation** – maps internal class IDs to
           human-readable strings, applying two surface-form corrections.
        6. **Task dispatch** – randomly chooses ``binary`` or
           ``single_choice`` and builds the proposition list accordingly.

        Args:
            qid: Unique identifier for the source question (unused here but
                required by the base interface).
            qdata: Raw question metadata dict (unused here but required by the
                base interface).
            sg: Normalised scene graph providing ``video_id`` for example IDs
                and provenance.
            split: Dataset partition label (e.g. ``"train"``, ``"val"``,
                ``"test"``).
            rng: Seeded random number generator for reproducibility.
            pools: Pre-computed hard-negative pools; ``pools.present_relations``
                is the primary evidence source.

        Yields:
            A single :class:`~src.schema.PropositionGroup` whose
            ``reasoning.type`` is ``"compositional_and"``,
            ``reasoning.complexity`` is ``3``, and
            ``provenance.generator_family`` is ``"compositional"``.
            ``provenance.evidence_relation_ids`` carries the two confirmed
            relation IDs used to form the true conjunction.

        Returns:
            Nothing if any of the early-exit guards are triggered (empty
            pools, no multi-relation objects, no false candidates found, or
            missing relation name lookups).
        """
        # Guard: nothing to compose if no relations have been pre-computed.
        if not pools or not pools.present_relations:
            return

        # Build a per-object inventory of confirmed relation class IDs.
        obj_rels = {}
        for r_type, r_class, obj in pools.present_relations:
            if obj not in obj_rels:
                obj_rels[obj] = []
            obj_rels[obj].append(r_class)

        # Keep only objects with at least two distinct relations so that a
        # non-trivial AND conjunction can be formed.
        valid_objs = [o for o, rels in obj_rels.items() if len(rels) >= 2]
        if not valid_objs:
            return

        target_obj = rng.choice(valid_objs)
        true_rels = obj_rels[target_obj]

        # We'll just generate binary or single-choice for AND composition
        task = rng.choice(["binary", "single_choice"])
        # Distractor pool size: large for single_choice, exactly 2 for binary.
        k = rng.randint(10, 40) if task == "single_choice" else 2

        # Pick the two confirmed relations that will form the true conjunction.
        selected_true = rng.sample(true_rels, 2)
        r1, r2 = selected_true
        # Normalise internal label artefacts to grammatically correct phrases.
        r1_name = ALL_RELATIONS.get(r1).replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
        r2_name = ALL_RELATIONS.get(r2).replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
        if not r1_name or not r2_name:
            # Lookup failed; cannot produce valid surface text.
            return

        obj_name = OBJECTS.get(target_obj, target_obj)

        # Construct the one true proposition: person IS r1 AND r2 the object.
        true_text = f"The person is {r1_name} and {r2_name} the {obj_name}."
        propositions = [Proposition(text=true_text, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))]

        if task == "binary":
            # Just one negative: one true, one false relation
            # False candidate: a compatible relation NOT in the object's confirmed set.
            # Replacing r2 with r3 makes the conjunction (r1 AND r3) false because
            # the person does not hold relation r3 with target_obj.
            false_cands = [r for r in ALL_RELATIONS if r not in true_rels and is_compatible(r, target_obj)]
            if not false_cands:
                return
            r3 = rng.choice(false_cands)
            r3_name = ALL_RELATIONS.get(r3).replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')

            # They either do r1 and r2 OR r1 and r3. If they don't do r3, r1 AND r3 is false.
            false_text = f"The person is {r1_name} and {r3_name} the {obj_name}."
            propositions.append(Proposition(text=false_text, truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r3_name}", object=obj_name, polarity="positive")))

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"comp_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=None, query_template_id=None, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="compositional_and", complexity=3, family="compositional"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_comp_bin", generation_seed=seed, generator_family="compositional", evidence_object_ids=[target_obj], evidence_relation_ids=selected_true)
            )

        elif task == "single_choice":
            # Build up to k-1 false propositions, each substituting a different
            # absent-but-compatible relation for r2 while keeping r1 fixed.
            # This forces the model to identify the exact confirmed pair rather
            # than accepting any conjunction that includes r1.
            false_cands = [r for r in ALL_RELATIONS if r not in true_rels and is_compatible(r, target_obj)]
            rng.shuffle(false_cands)

            for r3 in false_cands[:k-1]:
                r3_name = ALL_RELATIONS.get(r3).replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
                false_text = f"The person is {r1_name} and {r3_name} the {obj_name}."
                propositions.append(Proposition(text=false_text, truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r3_name}", object=obj_name, polarity="positive")))

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"comp_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=None, query_template_id=None, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="compositional_and", complexity=3, family="compositional"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_comp_sc", generation_seed=seed, generator_family="compositional", evidence_object_ids=[target_obj], evidence_relation_ids=selected_true)
            )
