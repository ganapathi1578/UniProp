"""Temporal and action proposition generators for the UniProp pipeline.

This module provides :class:`ActionTemporalGenerator`, which produces
:class:`~src.schema.PropositionGroup` instances that test a model's understanding
of **which actions occur** in a video (``generate_action``) and **when they
occur relative to each other** (``generate_temporal``).

The generator is intentionally split into two conceptually distinct sub-tasks:

Action recognition tasks
    Verify whether a specific human action phrase (e.g. *"washing dishes"*) is
    present or absent in the video clip.  Hard negatives are drawn from a
    curated static pool (``ALL_ACTIONS``) of plausible Charades-domain actions
    so that the model cannot trivially distinguish true from false by checking
    semantic plausibility.

Temporal ordering tasks
    Verify the chronological order of action pairs. A **1-hop** task directly
    compares two actions (A before B), while a **2-hop (multihop)** task
    requires the model to chain an intermediate action (A before B before C)
    and then reason about the implied ordering A → C.

The routing decision between the two sub-tasks is made probabilistically in
:meth:`ActionTemporalGenerator.generate` based on the number of distinct
actions annotated in the scene graph.

Typical usage::

    gen = ActionTemporalGenerator()
    for group in gen.generate(qid, qdata, sg, split, rng, pools):
        dataset.append(group)
"""

import random
from typing import Iterator
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.generators.renderer import render_action, render_temporal
from src.templates import get_query


class ActionTemporalGenerator(GeneratorBase):
    """Generator that produces action-recognition and temporal-ordering tasks.

    This generator handles two related but distinct proposition families:

    * **Action tasks** – Does a particular action occur in the video?  Supports
      ``binary``, ``single_choice``, ``multi_label``, and ``three_way`` task
      formats.
    * **Temporal tasks** – Does action A happen *before* action B?  Supports
      ``binary`` and ``single_choice`` task formats.  Tasks may be either
      1-hop (direct pair) or 2-hop (mediated by a middle action).

    The generator decides which family to emit by checking that at least two
    distinct actions are present and applying a 60 % probability towards
    temporal tasks when that condition is met.

    Inherits from :class:`~src.generators.base.GeneratorBase` and implements
    its :meth:`generate` abstract method.
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
        """Route generation to either action or temporal sub-generators.

        Filters the scene graph's action inventory to those entries that carry
        a non-empty phrase, then stochastically decides whether to emit an
        action task or a temporal task.

        Decision rule:
            * If ``len(actions) >= 2`` **and** ``rng.random() > 0.4`` →
              :meth:`generate_temporal` (≈ 60 % probability given ≥ 2 actions).
            * Otherwise → :meth:`generate_action`.

        Args:
            qid: Unique identifier for the source question (unused here but
                required by the base interface).
            qdata: Raw question metadata dict (unused here but required by the
                base interface).
            sg: Normalised scene graph containing ``actions``, ``frames``, and
                ``video_id`` for the current clip.
            split: Dataset partition label (e.g. ``"train"``, ``"val"``,
                ``"test"``).
            rng: Seeded random number generator for reproducibility.
            pools: Pre-computed hard-negative pools for the current clip
                (unused by this generator but required by the base interface).

        Yields:
            :class:`~src.schema.PropositionGroup` instances produced by the
            selected sub-generator.  Yields nothing if the scene graph
            contains no actions with a non-empty phrase.
        """
        # Collect only actions that have a usable natural-language phrase.
        actions = [a for a in sg.actions.values() if a.phrase]
        if not actions:
            return

        # Route to temporal generation when enough distinct actions exist and
        # the random draw exceeds the threshold (≈ 60 % chance).
        if len(actions) >= 2 and rng.random() > 0.4:
            yield from self.generate_temporal(sg, split, rng, actions)
        else:
            yield from self.generate_action(sg, split, rng, actions)

    def generate_action(self, sg, split, rng, actions):
        """Generate a single action-recognition :class:`PropositionGroup`.

        Randomly selects one of four task formats and builds the corresponding
        set of propositions:

        ``binary``
            One positive proposition ("person IS doing X") and one negative
            ("person is NOT doing X"), where X is either a true action or a
            hard-negative action from ``ALL_ACTIONS``.

        ``single_choice``
            One true action plus ``k - 1`` false actions drawn from
            ``ALL_ACTIONS``.  Exactly one proposition has ``label=1``.

        ``multi_label``
            A mix of true and false action phrases where zero, one, or more
            propositions may be correct.  An optional *"None of these actions
            occur."* sentinel proposition is appended with 70 % probability to
            handle the all-absent case.

        ``three_way``
            A fixed triple of propositions — *"Yes"* / *"No"* / *"Cannot say"*
            — for a polar question "Is the person <action>?".

        Hard-negative pool (``ALL_ACTIONS``):
            A static list of 20 plausible Charades-domain action phrases.
            Phrases that actually appear in the clip are excluded before
            sampling, ensuring that every false proposition is genuinely absent
            from the video.  A single fallback phrase (*"doing something
            completely different"*) is used when the entire pool overlaps with
            the clip's true phrases.

        Args:
            sg: Normalised scene graph for the current clip.
            split: Dataset partition label.
            rng: Seeded random number generator.
            actions: Non-empty list of action objects (each with a ``.phrase``
                and ``.start_secs`` attribute) extracted from ``sg``.

        Yields:
            A single :class:`~src.schema.PropositionGroup` whose
            ``reasoning.family`` is ``"action"`` and whose
            ``provenance.generator_family`` is ``"action_temporal"``.
        """
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = rng.choice(task_types)
        # Pool size k: larger for choice/label tasks; fixed at 3 or 2 otherwise.
        k = rng.randint(10, 40) if task in ["single_choice", "multi_label"] else (3 if task == "three_way" else 2)

        # Deduplicate true phrases so the same action cannot appear twice.
        true_phrases = list(set([a.phrase for a in actions]))

        # Static pool of Charades-domain actions used as hard negatives.
        # These are generic enough to be plausible in any indoor video but
        # were chosen to cover a wide range of activity categories so that
        # models cannot rely on semantic implausibility to filter them out.
        ALL_ACTIONS = [
            "holding a broom", "tidying a blanket", "opening a refrigerator", "running somewhere",
            "drinking from a cup", "closing a door", "watching television", "sitting in a chair",
            "writing something", "putting clothes somewhere", "smiling", "sneezing somewhere",
            "taking a phone", "making some food", "looking out a window", "washing dishes",
            "folding clothes", "reading a book", "typing on a laptop", "turning on a light"
        ]

        # Remove any pool actions that genuinely appear in the clip.
        all_absent = [a for a in ALL_ACTIONS if a not in true_phrases]
        if not all_absent:
            # Safety fallback: ensures we always have at least one false candidate.
            all_absent = ["doing something completely different"]

        if task == "binary":
            # Coin-flip decides whether the queried phrase is a true or false action.
            is_pos = rng.random() > 0.5
            target_phrase = rng.choice(true_phrases) if is_pos else rng.choice(all_absent)

            # Build the complementary positive/negative pair for the same phrase.
            p_pos = Proposition(text=render_action(target_phrase, "positive"), truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive"))
            p_neg = Proposition(text=render_action(target_phrase, "negative"), truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="negative"))

            tid, q_text = get_query("ACTION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=[p_pos, p_neg], task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="action", complexity=1, family="action"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_bin", generation_seed=seed, generator_family="action_temporal")
            )

        elif task == "single_choice":
            # One confirmed true action; the rest are plausible but absent.
            target_true = rng.choice(true_phrases)
            falses = rng.sample(all_absent, min(len(all_absent), k - 1))

            propositions = [Proposition(text=render_action(target_true, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_true, polarity="positive"))]
            for f in falses:
                propositions.append(Proposition(text=render_action(f, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=f, polarity="positive")))

            rng.shuffle(propositions)
            tid, q_text = get_query("ACTION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="action", complexity=1, family="action"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_sc", generation_seed=seed, generator_family="action_temporal")
            )

        elif task == "multi_label":
            # Randomly decide how many true actions to include (0 → all-absent).
            num_true = rng.randint(1, min(len(true_phrases), k - 1))
            if rng.random() < 0.2:
                # 20 % chance of an all-absent scenario (num_true = 0).
                num_true = 0

            num_false = k - num_true
            if rng.random() > 0.8:
                # Occasionally reduce the pool size by one (creates shorter lists).
                k -= 1

            selected_true = rng.sample(true_phrases, min(len(true_phrases), num_true))
            selected_false = rng.sample(all_absent, min(len(all_absent), num_false))

            propositions = []
            for t in selected_true:
                propositions.append(Proposition(text=render_action(t, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=t, polarity="positive")))
            for f in selected_false:
                propositions.append(Proposition(text=render_action(f, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=f, polarity="positive")))

            if rng.random() < 0.7:
                # Append a sentinel proposition that is TRUE only when no real
                # actions were selected, covering the "none of the above" case.
                none_val = 1 if num_true == 0 else 0
                propositions.append(Proposition(text="None of these actions occur.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))

            rng.shuffle(propositions)
            tid, q_text = get_query("ACTION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="action", complexity=1, family="action"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_ml", generation_seed=seed, generator_family="action_temporal")
            )

        elif task == "three_way":
            # Fixed-answer-set polar question: Yes / No / Cannot say.
            is_pos = rng.random() > 0.5
            target_phrase = rng.choice(true_phrases) if is_pos else rng.choice(all_absent)

            q_text = f"Is the person {target_phrase}?"
            propositions = [
                Proposition(text="Yes", truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive")),
                Proposition(text="No", truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="negative")),
                Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive"))
            ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="action", complexity=1, family="action"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_3w", generation_seed=seed, generator_family="action_temporal")
            )

    def generate_temporal(self, sg, split, rng, actions):
        """Generate a single temporal-ordering :class:`PropositionGroup`.

        Constructs a set of chronologically ordered action pairs
        (``valid_pairs``) and then optionally extends to a 2-hop (multihop)
        chain before emitting a ``binary`` or ``single_choice`` task.

        **Pair construction**:
            Actions are sorted ascending by ``start_secs``.  A pair
            ``(act1, act2)`` is *valid* when:

            * ``act1.start_secs < act2.start_secs`` – strict chronological
              ordering is guaranteed by start time.
            * ``act1.phrase != act2.phrase`` – the two actions are distinct so
              the proposition is non-trivially verifiable.

        **Multihop detection**:
            After selecting a valid pair ``(act1, act2)``, the generator
            searches for a third action ``act3`` satisfying:

            * ``act2.start_secs < act3.start_secs`` – act3 is strictly after
              act2 (and therefore also strictly after act1).
            * ``act3.phrase != act2.phrase`` and ``act3.phrase != act1.phrase``
              – all three events are distinct.

            If at least one such ``act3`` exists **and** ``rng.random() > 0.5``
            (≈ 50 % probability), the task becomes a **2-hop** task:
            the surface proposition spans ``(act1, act3)`` while the
            ``temporal_chain`` field in :class:`~src.schema.Provenance` records
            the full reasoning chain::

                [
                    {"event1": act1.phrase, "relation": "before", "event2": act2.phrase},
                    {"event1": act2.phrase, "relation": "before", "event2": act3.phrase},
                ]

            For a **1-hop** task ``chain`` is ``None``.

        **Task formats**:

        ``binary``
            Two propositions testing whether the stated ordering is correct.
            When ``is_pos=True`` the proposition "A before B" is TRUE; when
            ``is_pos=False`` the reversed claim "B before A" is posed and the
            *negative* polarity proposition ("B is NOT before A") becomes TRUE.

        ``single_choice``
            Query: *"Which event happened before <target_event_b>?"*.
            The correct answer is ``target_event_a``.  False candidates are
            events from the clip that started *after* ``target_event_b``
            (genuinely wrong orderings) padded with generic action phrases from
            a mini-pool so that the total reaches ``k - 1`` distractors.

        **Provenance** (``temporal_chain`` field):
            The ``provenance.temporal_chain`` attribute carries a structured
            list of ``{"event1", "relation", "event2"}`` dicts describing the
            explicit reasoning steps required to answer the question.  For
            1-hop tasks this is ``None``; for 2-hop tasks it contains two
            entries bridged by the intermediate action.  Downstream evaluation
            code uses this field to distinguish single-hop from multi-hop
            difficulty tiers.

        Args:
            sg: Normalised scene graph for the current clip.
            split: Dataset partition label.
            rng: Seeded random number generator.
            actions: Non-empty list of action objects (each carrying
                ``.phrase`` and ``.start_secs``) extracted from ``sg``.

        Yields:
            A single :class:`~src.schema.PropositionGroup` whose
            ``reasoning.family`` is ``"temporal"``, ``reasoning.hops`` is
            either ``1`` or ``2``, and ``provenance.generator_family`` is
            ``"action_temporal"``.  Yields nothing if no valid chronologically
            ordered pair can be found.
        """
        # Sort all actions by onset time to enable chronological comparisons.
        sorted_acts = sorted(actions, key=lambda x: x.start_secs)

        # Build every valid strictly-ordered pair of distinct actions.
        valid_pairs = []
        for i in range(len(sorted_acts)):
            for j in range(i+1, len(sorted_acts)):
                if sorted_acts[i].start_secs < sorted_acts[j].start_secs and sorted_acts[i].phrase != sorted_acts[j].phrase:
                    valid_pairs.append((sorted_acts[i], sorted_acts[j]))

        if not valid_pairs:
            # No strictly ordered pair exists (e.g. all actions share onset or phrase).
            return
        act1, act2 = rng.choice(valid_pairs)

        # --- Multihop detection -------------------------------------------
        # Look for a third action act3 that is strictly after act2 and is
        # distinct from both act1 and act2.  If found, we can form the chain
        # act1 → act2 → act3 and pose the long-range question act1 vs act3.
        multihops = []
        for k in range(len(sorted_acts)):
            act3 = sorted_acts[k]
            if act2.start_secs < act3.start_secs and act3.phrase != act2.phrase and act3.phrase != act1.phrase:
                multihops.append(act3)

        # Use the 2-hop form with ≈ 50 % probability when candidates exist.
        is_multihop = bool(multihops) and rng.random() > 0.5
        if is_multihop:
            act3 = rng.choice(multihops)
            # Surface proposition spans the outer events; the middle event is
            # implied by the reasoning chain stored in provenance.
            target_event_a = act1.phrase
            target_event_b = act3.phrase
            hops = 2
            # Provenance chain records each explicit "before" step so that
            # downstream evaluation can verify multi-hop reasoning ability.
            chain = [{"event1": act1.phrase, "relation": "before", "event2": act2.phrase}, {"event1": act2.phrase, "relation": "before", "event2": act3.phrase}]
        else:
            # 1-hop: the proposition directly reflects the observed ordering.
            target_event_a = act1.phrase
            target_event_b = act2.phrase
            hops = 1
            chain = None
        # -----------------------------------------------------------------------

        task_types = ["binary", "single_choice"]
        task = rng.choice(task_types)
        # k is only used by single_choice to set the distractor pool size.
        k = rng.randint(10, 40) if task == "single_choice" else 2

        if task == "binary":
            # Coin-flip: either verify the true ordering or the false reversed
            # ordering.  In the false branch, the *negative* proposition ("A is
            # NOT before B") becomes the true one because the real order is
            # B before A.
            is_pos = rng.random() > 0.5
            if is_pos:
                # True ordering presented: "A before B" is TRUE.
                p_pos = Proposition(text=render_temporal(target_event_a, target_event_b, "before", "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))
                p_neg = Proposition(text=render_temporal(target_event_a, target_event_b, "before", "negative"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="negative"))
                propositions = [p_pos, p_neg]
            else:
                # Reversed ordering presented: "B before A" is FALSE, so its
                # negation ("B is NOT before A") is TRUE.
                p_pos = Proposition(text=render_temporal(target_event_b, target_event_a, "before", "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_b, event_b=target_event_a, temporal_relation="before", polarity="positive"))
                p_neg = Proposition(text=render_temporal(target_event_b, target_event_a, "before", "negative"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_b, event_b=target_event_a, temporal_relation="before", polarity="negative"))
                propositions = [p_pos, p_neg]

            tid, q_text = get_query("TEMPORAL", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"temp_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="temporal_multihop" if is_multihop else "temporal_singlehop", complexity=2, family="temporal", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="temp_bin", generation_seed=seed, generator_family="action_temporal", temporal_chain=chain)
            )

        elif task == "single_choice":
            # Which event happened before target_event_b?
            q_text = f"Which event happened before {target_event_b}?"
            propositions = [Proposition(text=render_temporal(target_event_a, target_event_b, "before", "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))]

            # Create false options. We want other plausible events but NOT events that actually happened before target_event_b.
            # Easiest way: generate fake events or take events from the video that happened AFTER target_event_b.
            after_events = [a.phrase for a in sorted_acts if a.start_secs >= [x for x in sorted_acts if x.phrase == target_event_b][0].start_secs and a.phrase != target_event_b]

            # Fill with generic actions
            ALL_ACTIONS = [
                "holding a broom", "tidying a blanket", "opening a refrigerator", "running somewhere",
                "drinking from a cup", "closing a door", "watching television", "sitting in a chair"
            ]

            # Combine events that genuinely came after target_event_b with generic
            # pool entries (excluding any phrase actually present in the clip).
            falses = after_events + [a for a in ALL_ACTIONS if a not in [x.phrase for x in sorted_acts]]
            rng.shuffle(falses)

            for f in falses[:k-1]:
                propositions.append(Proposition(text=render_temporal(f, target_event_b, "before", "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=f, event_b=target_event_b, temporal_relation="before", polarity="positive")))

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"temp_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="temporal_multihop" if is_multihop else "temporal_singlehop", complexity=2, family="temporal", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="temp_sc", generation_seed=seed, generator_family="action_temporal", temporal_chain=chain)
            )
