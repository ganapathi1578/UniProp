"""Scene-graph-native proposition generators for object existence and relations.

This module contains three concrete
:class:`~src.generators.base.GeneratorBase` subclasses that derive benchmark
propositions directly from Action Genome Spatio-Temporal Scene-Graph (STSG)
annotations, bypassing any dependence on free-text question parsing:

* :class:`ObjectExistenceGenerator` — *"Is object X present in the video?"*
* :class:`RelationVerificationGenerator` — *"Does the person hold/touch/…
  object Y?"*
* :class:`RelationSetGenerator` — delegated no-op; all relation task types
  are handled by :class:`RelationVerificationGenerator`.

Task-type taxonomy
------------------
Each active generator can produce one of four *task types* per call,
chosen uniformly at random:

``binary``
    Exactly two propositions: a positive and a negative rendering of a
    single claim.  One is ``TRUE``, the other is ``FALSE``.  The model
    must identify which is correct.

``single_choice``
    One ``TRUE`` proposition and ``K-1`` ``FALSE`` propositions
    (distractors drawn from hard negatives).  The model must select the
    single correct proposition.  ``K`` is sampled in ``[10, 40]``.

``multi_label``
    A mix of ``TRUE`` and ``FALSE`` propositions about a shared subject
    (object or relation target).  The model must label each proposition
    independently.  Optionally includes a ``NONE`` sentinel proposition
    (see *NONE proposition* below).  ``K`` is sampled in ``[10, 40]``.

``three_way``
    Exactly three fixed propositions: ``"Yes"`` (``TRUE``), ``"No"``
    (``FALSE``), and ``"Cannot say"`` (``UNKNOWN``).  Mirrors a standard
    three-way classification setup.  Always uses a hand-crafted question
    string rather than a template.

NONE proposition
----------------
In ``multi_label`` tasks a special *"None of these …"* proposition is
appended with probability **0.7**.  Its ``label`` is ``1`` (TRUE) when
*no* correct propositions were selected (i.e. ``num_true == 0``), and
``0`` (FALSE) otherwise.  This teaches models to recognise the all-false
case rather than guessing randomly.

Shuffling rationale
-------------------
Propositions within ``single_choice`` and ``multi_label`` groups are
shuffled with ``rng.shuffle(propositions)`` *after* construction.  This
prevents positional bias: without shuffling, the single correct answer
would always appear at index 0 (it is appended first), which would allow
a model to achieve non-trivial accuracy by always predicting the first
item.

Negative sampling strategy
--------------------------
Hard negatives are supplied via :class:`~src.negatives.hard_negatives.NegativePools`:

* **Object negatives**: absent objects are sampled from
  ``set(OBJECTS.values()) - set(present_objects)``.
* **Relation negatives**: closed-world false relations are relations in
  ``ALL_RELATIONS`` that (a) are not present in the scene-graph for the
  chosen object and (b) pass the ``is_compatible`` compatibility filter.
  For binary tasks, a mutually-exclusive relation is preferred; a
  closed-world false relation is used as a fallback.
"""

import random
from typing import Iterator

from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools
from src.generators.base import GeneratorBase
from src.generators.renderer import render_existence, render_relation, render_noun_phrase
from src.constants import OBJECTS, ALL_RELATIONS
from src.templates import get_query
from src.compatibility import is_compatible


class ObjectExistenceGenerator(GeneratorBase):
    """Generates propositions that verify whether specific objects are present.

    For each call to :meth:`generate`, a single task type is chosen uniformly
    at random from ``{binary, single_choice, multi_label, three_way}``.
    Positive examples are drawn from ``pools.present_objects``; negative
    examples (hard negatives) are drawn from the complement of the full
    ``OBJECTS`` vocabulary relative to the present set.

    Task types produced
    -------------------
    ``binary``
        Two propositions — one positive-polarity, one negative-polarity — for
        a single randomly-chosen object.  With 50 % probability the target is
        a truly *present* object (positive-centred); otherwise an *absent*
        object is chosen (negative-centred), keeping the dataset balanced.

    ``single_choice``
        One ``TRUE`` proposition (a present object rendered in positive
        polarity) plus up to ``K-1`` ``FALSE`` distractors (absent objects).
        Propositions are shuffled to remove positional bias.

    ``multi_label``
        A mix of ``TRUE`` (present) and ``FALSE`` (absent) propositions.
        With 20 % probability ``num_true`` is forced to 0, producing an
        all-false pool that makes the NONE proposition true.  With 70 %
        probability a ``"None of these objects are present."`` sentinel
        proposition is appended; its label reflects whether ``num_true == 0``.
        Propositions are shuffled to remove positional bias.

    ``three_way``
        A fixed set of three propositions — ``"Yes"``, ``"No"``,
        ``"Cannot say"`` — anchored to a single randomly-chosen object.
        Uses a hand-crafted question string rather than a template lookup.

    Note:
        The guard ``if not pools or not pools.present_objects: return``
        at the top of :meth:`generate` ensures the generator silently
        produces nothing for videos with no annotated objects, rather than
        raising an exception.
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
        """Generate object-existence proposition groups for one video.

        Randomly selects a task type then delegates to the appropriate
        branch.  Each branch yields exactly one :class:`~src.schema.PropositionGroup`.

        Args:
            qid (str): Source question ID (may be unused for scene-graph-
                native generators; passed through for provenance tracking).
            qdata (dict): Raw question metadata.  Not used by this generator
                but kept for interface uniformity.
            sg (NormalizedSceneGraph): Normalised scene-graph for the current
                video.  ``sg.video_id`` is used to construct unique example
                IDs and to populate provenance fields.
            split (str): Dataset split (``"train"``, ``"val"``, or
                ``"test"``).  Written verbatim into the emitted group.
            rng (random.Random): Seeded RNG.  All random decisions (task-type
                selection, object sampling, shuffling, seed generation) must
                go through this instance to guarantee reproducibility.
            pools (NegativePools): Pre-computed pools for the current video.
                ``pools.present_objects`` supplies the set of confirmed-present
                object class tokens.

        Yields:
            PropositionGroup: A single benchmark example whose ``task_type``
                is one of ``"binary"``, ``"single_choice"``,
                ``"multi_label"``, or ``"three_way"``.  Nothing is yielded
                if ``pools`` is ``None`` or ``pools.present_objects`` is empty.
        """
        if not pools or not pools.present_objects: return
        
        # We need to generate binary, single-choice, multi-label, three-way
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = rng.choice(task_types)
        
        # Decide K
        # K controls the total number of propositions in the group.
        # For single_choice and multi_label, K is randomised in [10, 40] to vary task difficulty.
        # For three_way it is always 3 (Yes / No / Cannot say).
        # For binary it is always 2 (positive / negative polarity pair).
        k = rng.randint(10, 40) if task in ["single_choice", "multi_label"] else (3 if task == "three_way" else 2)
        
        # Collect true and false
        # Map raw class tokens to their canonical display names via OBJECTS lookup.
        # all_absent is the closed-world complement: every canonical object name
        # that does NOT appear in the present set for this video.
        all_true = list(set([OBJECTS.get(t, t) for t in pools.present_objects]))
        all_absent = list(set(OBJECTS.values()) - set(all_true))
        
        if task == "binary":
            # 50/50 split between positive-centred (present object) and
            # negative-centred (absent object) examples to keep the dataset balanced.
            is_positive_centered = rng.random() > 0.5
            target_obj = rng.choice(all_true) if is_positive_centered else (rng.choice(all_absent) if all_absent else rng.choice(all_true))
            is_true = target_obj in all_true
            obj_name = target_obj
            
            # Construct a complementary polarity pair for the chosen object.
            # p_pos: "A <obj> is present." — TRUE iff the object is in the scene-graph.
            # p_neg: "No <obj> is present." — TRUE iff the object is NOT in the scene-graph.
            p_pos = Proposition(text=render_existence(obj_name, "positive"), truth_state="TRUE" if is_true else "FALSE", label=1 if is_true else 0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive"))
            p_neg = Proposition(text=render_existence(obj_name, "negative"), truth_state="FALSE" if is_true else "TRUE", label=0 if is_true else 1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="negative"))
            propositions = [p_pos, p_neg]
            
            tid, q_text = get_query("OBJECT_VERIFY" if task in ["binary", "three_way"] else ("OBJECT_SINGLE" if task == "single_choice" else "OBJECT_SET"), rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid,
                propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="object_existence", complexity=1, family="object"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_binary", generation_seed=seed, generator_family="object_existence", evidence_object_ids=[target_obj])
            )
            
        elif task == "single_choice":
            # 1 true, K-1 false
            # Pick one confirmed-present object as the sole correct answer.
            target_true = rng.choice(all_true)
            # Sample up to K-1 absent objects as distractors (hard negatives).
            falses = rng.sample(all_absent, min(len(all_absent), k - 1))
            
            propositions = []
            obj_name = OBJECTS.get(target_true, target_true)
            # The correct proposition is always appended first; the shuffle below removes positional bias.
            propositions.append(Proposition(text=render_existence(obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")))
            
            for f in falses:
                f_name = f
                propositions.append(Proposition(text=render_existence(f_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=f_name, polarity="positive")))
                
            # Shuffle to prevent the model from exploiting the fixed position of the correct answer.
            rng.shuffle(propositions)
            tid, q_text = get_query("OBJECT_VERIFY" if task in ["binary", "three_way"] else ("OBJECT_SINGLE" if task == "single_choice" else "OBJECT_SET"), rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="object_existence", complexity=1, family="object"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_sc", generation_seed=seed, generator_family="object_existence", evidence_object_ids=[target_true])
            )
            
        elif task == "multi_label":
            # Randomly decide how many TRUE propositions to include.
            # With 20% probability force num_true=0 so the NONE sentinel is correct.
            num_true = rng.randint(1, min(len(all_true), k - 1)) if all_true else 0
            if rng.random() < 0.2: num_true = 0 # 20% all-false for NONE=1
            num_false = k - num_true
            # With 20% probability reduce k by 1 to reserve a slot for the NONE proposition.
            if rng.random() > 0.8: k -= 1 # Leave room for NONE
            
            selected_true = rng.sample(all_true, min(len(all_true), num_true))
            selected_false = rng.sample(all_absent, min(len(all_absent), num_false))
            
            propositions = []
            for t in selected_true:
                t_name = t
                propositions.append(Proposition(text=render_existence(t_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=t_name, polarity="positive")))
            for f in selected_false:
                f_name = f
                propositions.append(Proposition(text=render_existence(f_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=f_name, polarity="positive")))
                
            # With 70% probability append a NONE sentinel proposition.
            # Its label is 1 (TRUE) only when no correct propositions were selected,
            # teaching the model to recognise the all-false case.
            none_included = rng.random() < 0.7
            if none_included:
                none_val = 1 if not selected_true else 0
                propositions.append(Proposition(text="None of these objects are present.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))
                
            # Shuffle to prevent positional bias toward TRUE propositions.
            rng.shuffle(propositions)
            tid, q_text = get_query("OBJECT_VERIFY" if task in ["binary", "three_way"] else ("OBJECT_SINGLE" if task == "single_choice" else "OBJECT_SET"), rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="object_existence", complexity=1, family="object"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_ml", generation_seed=seed, generator_family="object_existence", evidence_object_ids=selected_true)
            )

        elif task == "three_way":
            # Just generate a standard unknown if we want
            # Pick a present (is_pos=True) or absent (is_pos=False) object with equal probability.
            is_pos = rng.random() > 0.5
            target_obj = rng.choice(all_true) if is_pos else rng.choice(all_absent)
            obj_name = target_obj
            
            # Three-way tasks use a hand-crafted question string rather than a template
            # because the fixed "Yes / No / Cannot say" structure does not fit the
            # open-ended OBJECT_VERIFY template vocabulary.
            q_text = f"Is {render_noun_phrase(obj_name, False)} present in the video?"
            
            # Fixed three-proposition structure:
            #   "Yes"        — TRUE iff the object is actually present (is_pos).
            #   "No"         — TRUE iff the object is absent (not is_pos).
            #   "Cannot say" — always UNKNOWN; label 0 (no credit for hedging).
            propositions = [
                Proposition(text="Yes", truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")),
                Proposition(text="No", truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="negative")),
                Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive"))
            ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="object_existence", complexity=1, family="object"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_3w", generation_seed=seed, generator_family="object_existence", evidence_object_ids=[target_obj])
            )


class RelationVerificationGenerator(GeneratorBase):
    """Generates propositions that verify person-to-object relations.

    Handles **all four task types** for person-object relations drawn from
    ``pools.present_relations``.  Relations are triples of the form
    ``(rel_type, rel_class, object_token)`` where ``rel_type`` is the
    broad family (e.g. ``"spatial"``, ``"contact"``), ``rel_class`` is the
    specific predicate token, and ``object_token`` is the object class.

    Hard-negative strategy
    ----------------------
    False (absent) relations are sampled in two ways:

    1. **Mutually-exclusive relation** (preferred for ``binary`` negatives):
       a relation that is logically incompatible with the true relation for
       the same object (e.g. ``touching`` vs. ``not_touching``).
       Sourced via ``pools.get_mutually_exclusive_relation``.

    2. **Closed-world false relation** (fallback): any relation in
       ``ALL_RELATIONS`` that does not appear in the scene-graph for the
       chosen object *and* passes the ``is_compatible`` ontology filter.
       Sourced via ``pools.get_closed_world_false_relation`` or by
       directly filtering ``ALL_RELATIONS``.

    Task types produced
    -------------------
    ``binary``
        Two propositions — positive-polarity and negative-polarity — for a
        single relation triple.  With 50 % probability a confirmed-true
        relation is chosen and rendered as TRUE; otherwise a confirmed-false
        relation (mutually-exclusive or closed-world) is used and rendered
        as FALSE.

    ``single_choice``
        One TRUE proposition (a confirmed-present relation for a randomly-
        chosen object) and up to ``K-1`` FALSE propositions (compatible
        relations that are absent from the scene-graph for that object).
        False candidates are shuffled before slicing to prevent ordering
        artefacts.  Final proposition list is shuffled to remove positional
        bias.

    ``multi_label``
        Groups all confirmed-present relations for one randomly-chosen
        object.  Samples ``num_true`` (≥ 0) of them as correct propositions
        and ``num_false`` compatible-but-absent relations as distractors.
        With 20 % probability ``num_true`` is forced to 0 (all-false pool).
        With 70 % probability appends a
        ``"None of these relationships hold with <obj>."`` sentinel whose
        label is 1 iff ``num_true == 0``.  Propositions are shuffled.
        Emitted with ``reasoning.type = "relation_set"`` and
        ``complexity = 2`` to distinguish it from single-relation tasks.

    ``three_way``
        Three fixed propositions — ``"Yes"``, ``"No"``, ``"Cannot say"`` —
        for a randomly-chosen confirmed-present relation.  Always uses a
        hand-crafted question rather than a template, because the target
        relation is always TRUE (present in the scene-graph) for three-way
        tasks.

    Note:
        The guard ``if not pools or not pools.present_relations: return``
        silently skips videos with no annotated person-object relations.

        For the ``binary`` negative branch, if neither a mutually-exclusive
        nor a closed-world false relation can be found for the chosen object,
        the method returns early (yields nothing) to avoid emitting an
        ill-formed example.
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
        """Generate relation-verification proposition groups for one video.

        Randomly selects a task type then delegates to the corresponding
        branch.  Each branch yields at most one
        :class:`~src.schema.PropositionGroup`.

        Args:
            qid (str): Source question ID.  Not directly used by this
                generator but preserved for provenance.
            qdata (dict): Raw question metadata.  Not used.
            sg (NormalizedSceneGraph): Normalised scene-graph.
                ``sg.video_id`` populates example IDs and provenance fields.
            split (str): Dataset split identifier.
            rng (random.Random): Seeded RNG for all stochastic decisions.
            pools (NegativePools): Pre-computed pools for the current video.
                ``pools.present_relations`` is a list of
                ``(rel_type, rel_class, object_token)`` triples for
                confirmed-present person-object relations.
                ``pools.get_mutually_exclusive_relation`` and
                ``pools.get_closed_world_false_relation`` supply hard
                negatives.

        Yields:
            PropositionGroup: One benchmark example whose ``task_type`` is
                one of ``"binary"``, ``"single_choice"``, ``"multi_label"``,
                or ``"three_way"``.  Nothing is yielded if ``pools`` is
                ``None``, ``pools.present_relations`` is empty, or (for the
                ``binary`` negative branch) no suitable false relation is
                available.
        """
        if not pools or not pools.present_relations: return
        
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = rng.choice(task_types)
        # K: total proposition count.  Sampled in [10, 40] for multi-proposition tasks;
        # fixed at 3 for three_way and 2 for binary.
        k = rng.randint(10, 40) if task in ["single_choice", "multi_label"] else (3 if task == "three_way" else 2)
        
        true_rels = list(pools.present_relations)
        
        if task == "binary":
            # 50/50 chance of a positive-centred vs. negative-centred example.
            is_pos = rng.random() > 0.5
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            
            if is_pos:
                # Positive-centred: the chosen relation IS present → p_pos is TRUE.
                rel_name = ALL_RELATIONS.get(rel_class, rel_class)
                p_pos = Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                p_neg = Proposition(text=render_relation(rel_name, obj_name, "negative"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative"))
                propositions = [p_pos, p_neg]
            else:
                # Negative-centred: substitute a false relation for the same object.
                # Prefer a mutually-exclusive relation (e.g. touching → not_touching)
                # for stronger signal; fall back to any closed-world false relation.
                neg_rel = pools.get_mutually_exclusive_relation(rel_class, obj, rng)
                if not neg_rel: neg_rel = pools.get_closed_world_false_relation(rel_class, obj, rng)
                # If no valid false relation is available, skip this example entirely
                # to avoid emitting a proposition with an incorrect truth_state.
                if not neg_rel: return
                
                rel_name = ALL_RELATIONS.get(neg_rel, neg_rel)
                # The false relation is NOT present → p_pos is FALSE, p_neg is TRUE.
                p_pos = Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                p_neg = Proposition(text=render_relation(rel_name, obj_name, "negative"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative"))
                propositions = [p_pos, p_neg]
                
            tid, q_text = get_query("RELATION_VERIFY" if task in ["binary", "three_way"] else "RELATION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="relation_verification", complexity=1, family=rel_type),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_bin", generation_seed=seed, generator_family="relation_verification", evidence_object_ids=[obj], evidence_relation_ids=[rel_class])
            )
            
        elif task == "single_choice":
            # Anchor on a confirmed-present relation triple.
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            rel_name = ALL_RELATIONS.get(rel_class, rel_class)
            
            # The single correct proposition is appended first; shuffle below removes positional bias.
            propositions = [Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))]
            
            # Collect all compatible relations that are not present for this object.
            # These form the hard-negative distractor pool.
            false_cands = []
            for r in ALL_RELATIONS:
                if r != rel_class and is_compatible(r, obj) and (rel_type, r, obj) not in true_rels:
                    false_cands.append(r)
                    
            # Shuffle false candidates before slicing to avoid systematic ordering
            # (ALL_RELATIONS is an ordered mapping; without shuffling earlier relations
            # would always be chosen as distractors).
            rng.shuffle(false_cands)
            for f in false_cands[:k-1]:
                f_name = ALL_RELATIONS.get(f, f)
                propositions.append(Proposition(text=render_relation(f_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=f_name, object=obj_name, polarity="positive")))
                
            # Shuffle the combined list to remove positional bias on the correct answer.
            rng.shuffle(propositions)
            tid, q_text = get_query("RELATION_VERIFY" if task in ["binary", "three_way"] else "RELATION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="relation_verification", complexity=1, family=rel_type),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_sc", generation_seed=seed, generator_family="relation_verification", evidence_object_ids=[obj], evidence_relation_ids=[rel_class])
            )
            
        elif task == "multi_label":
            # Build an index from object token → list of confirmed-present relation classes.
            obj_to_rels = {}
            for r_type, r_class, o in true_rels:
                if o not in obj_to_rels: obj_to_rels[o] = []
                obj_to_rels[o].append(r_class)
                
            # Anchor on one randomly-chosen object that has at least one relation.
            obj = rng.choice(list(obj_to_rels.keys()))
            obj_name = OBJECTS.get(obj, obj)
            obj_true_rels = obj_to_rels[obj]
            
            # With 20% probability force num_true=0 to create an all-false pool
            # (makes the NONE proposition TRUE, providing negative-only examples).
            num_true = rng.randint(1, min(len(obj_true_rels), k - 1))
            if rng.random() < 0.2: num_true = 0
            
            selected_true = rng.sample(obj_true_rels, num_true)
            
            # Build the closed-world false pool: compatible relations not confirmed
            # present for the chosen object.
            false_cands = []
            for r in ALL_RELATIONS:
                if r not in obj_true_rels and is_compatible(r, obj):
                    false_cands.append(r)
                    
            num_false = k - num_true
            # With 20% probability reduce k by 1 to leave a slot for the NONE proposition.
            if rng.random() > 0.8: k -= 1 # NONE
            selected_false = rng.sample(false_cands, min(len(false_cands), num_false))
            
            propositions = []
            for r in selected_true:
                r_name = ALL_RELATIONS.get(r, r)
                propositions.append(Proposition(text=render_relation(r_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=r_name, object=obj_name, polarity="positive")))
            for r in selected_false:
                r_name = ALL_RELATIONS.get(r, r)
                propositions.append(Proposition(text=render_relation(r_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=r_name, object=obj_name, polarity="positive")))
                
            # With 70% probability append NONE sentinel.
            # none_val=1 (TRUE) iff no correct relations were selected (all-false scenario).
            if rng.random() < 0.7:
                none_val = 1 if num_true == 0 else 0
                propositions.append(Proposition(text=f"None of these relationships hold with {render_noun_phrase(obj_name, True)}.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))
                
            # Shuffle to prevent positional bias toward TRUE propositions.
            rng.shuffle(propositions)
            tid, q_text = get_query("RELATION_VERIFY" if task in ["binary", "three_way"] else "RELATION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="relation_set", complexity=2, family="spatial"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_ml", generation_seed=seed, generator_family="relation_sets", evidence_object_ids=[obj], evidence_relation_ids=selected_true)
            )
            
        elif task == "three_way":
            # Always anchor on a confirmed-present relation triple so "Yes" is the correct answer.
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            rel_name = ALL_RELATIONS.get(rel_class, rel_class)
            
            # Hand-crafted question string: three-way tasks don't fit the RELATION template
            # (which expects an open-ended relational query rather than a Yes/No prompt).
            q_text = f"Is the person {rel_name} {render_noun_phrase(obj_name, True)}?"
            
            # Fixed three-proposition structure:
            #   "Yes"        — TRUE (the relation is present).
            #   "No"         — FALSE (the relation is present, so denial is wrong).
            #   "Cannot say" — UNKNOWN; label 0 (no credit for epistemic hedging).
            propositions = [
                Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive")),
                Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative")),
                Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
            ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="relation_verification", complexity=1, family=rel_type),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_3w", generation_seed=seed, generator_family="relation_verification", evidence_object_ids=[obj], evidence_relation_ids=[rel_class])
            )
            

class RelationSetGenerator(GeneratorBase):
    """Delegated no-op — all relation task types are handled by RelationVerificationGenerator.

    This class is a stub that explicitly yields nothing.  It exists as an
    architectural artefact from an earlier design where relation-set tasks
    (multi-label / single-choice over the full relation vocabulary for one
    object) were intended to live in a separate generator class.

    Current behaviour
    -----------------
    :class:`RelationVerificationGenerator` was extended to handle *all four*
    task types — including ``multi_label`` (relation-set style) — in a
    unified, seamless way.  Splitting multi-label relation tasks into a
    separate generator would introduce duplication and complicate the
    pipeline's generator registry.

    This class is retained to:

    1. Signal the original design intent in the codebase.
    2. Allow future re-activation if relation-set tasks need to be separated
       (e.g. to apply a different sampling budget or difficulty weighting)
       without restructuring the pipeline.

    Note:
        ``yield from []`` is functionally identical to ``return`` inside a
        generator function.  It is used here to make the no-op intent
        self-documenting.
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
        """Yield nothing — all relation task types are delegated to RelationVerificationGenerator.

        :class:`RelationVerificationGenerator` handles binary, single_choice,
        multi_label, and three_way relation tasks in a single unified
        generator.  This method is intentionally empty so that registering
        :class:`RelationSetGenerator` in the pipeline has no effect.

        Args:
            qid (str): Unused.
            qdata (dict): Unused.
            sg (NormalizedSceneGraph): Unused.
            split (str): Unused.
            rng (random.Random): Unused.
            pools (NegativePools): Unused.

        Yields:
            Nothing.
        """
        yield from [] # Handled by the above RelationVerificationGenerator which now handles everything seamlessly
