import random
from typing import Iterator, Optional, List
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.generators.renderer import render_action, render_temporal
from src.templates import get_query, get_query_formatted
from src.constants import AGQA_ACTION_PHRASES, AGQA_ACTION_PHRASES_SET, ACTION_VERB_GROUPS


def _get_hard_negative_actions(true_phrase: str, rng: random.Random) -> List[str]:
    """Return distractor actions from the same verb group when possible.

    If the true phrase starts with a known verb prefix (e.g. "holding"),
    we pick distractors from the same verb group first, producing harder
    negatives.  Remaining slots are filled from the full AGQA vocabulary.
    """
    hard = []
    for _group_name, members in ACTION_VERB_GROUPS.items():
        if true_phrase in members:
            hard = [m for m in members if m != true_phrase]
            break
    return hard


class ActionTemporalGenerator(GeneratorBase):
    def generate(
        self,
        qid: str,
        qdata: dict,
        sg: NormalizedSceneGraph,
        split: str,
        rng: random.Random,
        pools: NegativePools,
        target_task: Optional[str] = None,
        target_k: Optional[int] = None,
        target_hops: Optional[int] = None,
        target_difficulty: Optional[str] = None,
        target_family: Optional[str] = None,
    ) -> Iterator[PropositionGroup]:
        actions = [a for a in sg.actions.values() if a.phrase]
        if not actions:
            return

        if target_family == "action":
            yield from self.generate_action(sg, split, rng, actions, target_task, target_k, target_hops, target_difficulty)
        elif target_family == "temporal":
            if len(actions) >= 2:
                yield from self.generate_temporal(sg, split, rng, actions, target_task, target_k, target_hops, target_difficulty)
        else:
            if len(actions) >= 2 and rng.random() > 0.4:
                yield from self.generate_temporal(sg, split, rng, actions, target_task, target_k, target_hops, target_difficulty)
            else:
                yield from self.generate_action(sg, split, rng, actions, target_task, target_k, target_hops, target_difficulty)

    def generate_action(self, sg, split, rng, actions, target_task=None, target_k=None, target_hops=None, target_difficulty=None):
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = target_task if target_task in task_types else rng.choice(task_types)

        if task == "binary":
            k = 2
        elif task == "three_way":
            k = 3
        else:
            k = target_k if target_k is not None and 2 <= target_k <= 40 else rng.randint(4, 25)

        hops = target_hops if target_hops is not None else 1
        difficulty = target_difficulty if target_difficulty in ("easy", "medium", "hard") else "medium"

        true_phrases = list(set([a.phrase for a in actions]))
        if not true_phrases:
            return

        # Build absent pool from the *real* AGQA action vocabulary
        all_absent = [a for a in AGQA_ACTION_PHRASES if a not in true_phrases]
        if not all_absent:
            # Extremely unlikely – every AGQA action appears in this video
            all_absent = [a for a in AGQA_ACTION_PHRASES if a != true_phrases[0]]

        # For hard difficulty, prefer same-verb-group distractors
        if difficulty == "hard":
            hard_cands = []
            for tp in true_phrases:
                hard_cands.extend(_get_hard_negative_actions(tp, rng))
            hard_absent = [h for h in hard_cands if h not in true_phrases]
            if hard_absent:
                # Merge hard negatives first, then fill with rest
                remaining = [a for a in all_absent if a not in hard_absent]
                all_absent = hard_absent + remaining

        if task == "binary":
            is_pos = rng.random() > 0.5
            target_phrase = rng.choice(true_phrases) if is_pos else rng.choice(all_absent)

            p_pos = Proposition(text=render_action(target_phrase, "positive"), truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive"))
            p_neg = Proposition(text=render_action(target_phrase, "negative"), truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="negative"))

            tid, q_text = get_query("ACTION_VERIFY", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=[p_pos, p_neg], task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="action", complexity=min(hops, 4), family="action", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_bin", generation_seed=seed, generator_family="action_temporal")
            )

        elif task == "single_choice":
            target_true = rng.choice(true_phrases)
            sample_size = min(len(all_absent), k - 1)
            falses = rng.sample(all_absent, sample_size) if sample_size > 0 else []

            propositions = [Proposition(text=render_action(target_true, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_true, polarity="positive"))]
            for f in falses:
                propositions.append(Proposition(text=render_action(f, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=f, polarity="positive")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("ACTION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="action", complexity=min(hops, 4), family="action", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_sc", generation_seed=seed, generator_family="action_temporal")
            )

        elif task == "multi_label":
            num_true = rng.randint(1, min(len(true_phrases), k - 1))
            if rng.random() < 0.2:
                num_true = 0

            num_false = max(0, k - num_true)
            if rng.random() > 0.8 and k > 2:
                num_false = max(0, num_false - 1)  # None

            selected_true = rng.sample(true_phrases, min(len(true_phrases), num_true))
            selected_false = rng.sample(all_absent, min(len(all_absent), num_false))

            propositions = []
            for t in selected_true:
                propositions.append(Proposition(text=render_action(t, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=t, polarity="positive")))
            for f in selected_false:
                propositions.append(Proposition(text=render_action(f, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=f, polarity="positive")))

            if rng.random() < 0.7:
                none_val = 1 if num_true == 0 else 0
                propositions.append(Proposition(text="None of these actions occur.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("ACTION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="action", complexity=min(hops, 4), family="action", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_ml", generation_seed=seed, generator_family="action_temporal")
            )

        elif task == "three_way":
            is_unknown = rng.random() < 0.20
            is_pos = (not is_unknown) and (rng.random() > 0.5)

            if is_unknown:
                truth_choice = "UNKNOWN"
                target_phrase = rng.choice(all_absent)
            elif is_pos:
                truth_choice = "TRUE"
                target_phrase = rng.choice(true_phrases)
            else:
                truth_choice = "FALSE"
                target_phrase = rng.choice(all_absent)

            q_text = f"Is the person {target_phrase}?"
            if truth_choice == "TRUE":
                propositions = [
                    Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive")),
                    Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive"))
                ]
            elif truth_choice == "FALSE":
                propositions = [
                    Proposition(text="Yes", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive")),
                    Proposition(text="No", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive"))
                ]
            else:
                propositions = [
                    Proposition(text="Yes", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive")),
                    Proposition(text="No", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=1, semantics=PropositionSemantics(canonical_type="action", subject="person", predicate=target_phrase, polarity="positive"))
                ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"act_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="action", complexity=min(hops, 4), family="action", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="act_3w", generation_seed=seed, generator_family="action_temporal")
            )

    def generate_temporal(self, sg, split, rng, actions, target_task=None, target_k=None, target_hops=None, target_difficulty=None):
        sorted_acts = sorted(actions, key=lambda x: x.start_secs)
        if len(sorted_acts) < 2:
            return

        valid_pairs = []
        for i in range(len(sorted_acts)):
            for j in range(i+1, len(sorted_acts)):
                if sorted_acts[i].start_secs < sorted_acts[j].start_secs and sorted_acts[i].phrase != sorted_acts[j].phrase:
                    valid_pairs.append((sorted_acts[i], sorted_acts[j]))

        if not valid_pairs:
            return

        act1, act2 = rng.choice(valid_pairs)

        # Multihop chaining
        hops = target_hops if target_hops is not None else (2 if rng.random() > 0.5 else 1)
        difficulty = target_difficulty if target_difficulty in ("easy", "medium", "hard") else "medium"

        chain = None
        target_event_a = act1.phrase
        target_event_b = act2.phrase

        if hops >= 2 and len(sorted_acts) >= 3:
            multihops = [a for a in sorted_acts if a.start_secs > act2.start_secs and a.phrase not in (act1.phrase, act2.phrase)]
            if multihops:
                act3 = rng.choice(multihops)
                target_event_b = act3.phrase
                chain = [
                    {"event1": act1.phrase, "relation": "before", "event2": act2.phrase},
                    {"event1": act2.phrase, "relation": "before", "event2": act3.phrase}
                ]
                if hops >= 3 and len(sorted_acts) >= 4:
                    multihops_3 = [a for a in sorted_acts if a.start_secs > act3.start_secs and a.phrase not in (act1.phrase, act2.phrase, act3.phrase)]
                    if multihops_3:
                        act4 = rng.choice(multihops_3)
                        target_event_b = act4.phrase
                        chain.append({"event1": act3.phrase, "relation": "before", "event2": act4.phrase})

        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = target_task if target_task in task_types else rng.choice(["binary", "single_choice"])

        if task == "binary":
            k = 2
        elif task == "three_way":
            k = 3
        else:
            k = target_k if target_k is not None and 2 <= target_k <= 40 else rng.randint(4, 25)

        is_multihop = hops > 1

        if task == "binary":
            is_pos = rng.random() > 0.5
            if is_pos:
                p_pos = Proposition(text=render_temporal(target_event_a, target_event_b, "before", "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))
                p_neg = Proposition(text=render_temporal(target_event_a, target_event_b, "before", "negative"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="negative"))
                propositions = [p_pos, p_neg]
            else:
                # Hard negative: swap temporal order (B before A is FALSE!)
                p_pos = Proposition(text=render_temporal(target_event_b, target_event_a, "before", "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_b, event_b=target_event_a, temporal_relation="before", polarity="positive"))
                p_neg = Proposition(text=render_temporal(target_event_b, target_event_a, "before", "negative"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_b, event_b=target_event_a, temporal_relation="before", polarity="negative"))
                propositions = [p_pos, p_neg]

            tid, q_text = get_query("TEMPORAL", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"temp_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="temporal_multihop" if is_multihop else "temporal_singlehop", complexity=min(hops, 4), family="temporal", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="temp_bin", generation_seed=seed, generator_family="action_temporal", temporal_chain=chain)
            )

        elif task == "single_choice":
            tid, q_text = get_query_formatted("TEMPORAL_BEFORE", rng, event=target_event_b)
            propositions = [Proposition(text=render_temporal(target_event_a, target_event_b, "before", "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))]

            falses = []
            if difficulty == "hard":
                # Hard negative: actions occurring strictly AFTER target_event_b
                after_events = [a.phrase for a in sorted_acts if a.start_secs > act2.start_secs and a.phrase != target_event_b]
                falses.extend(after_events)

            # Fill remaining with real AGQA actions not present in this video
            absent_acts = [a for a in AGQA_ACTION_PHRASES if a not in [x.phrase for x in sorted_acts]]
            falses.extend(absent_acts)
            rng.shuffle(falses)

            for f in falses[:k-1]:
                propositions.append(Proposition(text=render_temporal(f, target_event_b, "before", "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=f, event_b=target_event_b, temporal_relation="before", polarity="positive")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"temp_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="temporal_multihop" if is_multihop else "temporal_singlehop", complexity=min(hops, 4), family="temporal", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="temp_sc", generation_seed=seed, generator_family="action_temporal", temporal_chain=chain)
            )

        elif task == "multi_label":
            # Which events happened before target_event_b?
            tid, q_text = get_query_formatted("TEMPORAL_BEFORE", rng, event=target_event_b)
            target_b_act = [x for x in sorted_acts if x.phrase == target_event_b]
            if not target_b_act:
                return
            target_b_start = target_b_act[0].start_secs
            earlier_events = [a.phrase for a in sorted_acts if a.start_secs < target_b_start and a.phrase != target_event_b]
            later_events = [a.phrase for a in sorted_acts if a.start_secs >= target_b_start and a.phrase != target_event_b]
            absent_acts = [a for a in AGQA_ACTION_PHRASES if a not in [x.phrase for x in sorted_acts]]

            num_true = rng.randint(1, min(len(earlier_events), k - 1)) if earlier_events else 0
            if rng.random() < 0.2:
                num_true = 0

            selected_true = rng.sample(earlier_events, num_true) if num_true > 0 else []
            falses_pool = later_events + absent_acts
            num_false = max(0, k - num_true)
            if rng.random() > 0.8 and k > 2:
                num_false = max(0, num_false - 1)

            selected_false = rng.sample(falses_pool, min(len(falses_pool), num_false))

            propositions = []
            for t in selected_true:
                propositions.append(Proposition(text=render_temporal(t, target_event_b, "before", "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=t, event_b=target_event_b, temporal_relation="before", polarity="positive")))
            for f in selected_false:
                propositions.append(Proposition(text=render_temporal(f, target_event_b, "before", "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=f, event_b=target_event_b, temporal_relation="before", polarity="positive")))

            if rng.random() < 0.7:
                none_val = 1 if not selected_true else 0
                propositions.append(Proposition(text="None of these events occurred before.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"temp_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="temporal_multihop" if is_multihop else "temporal_singlehop", complexity=min(hops, 4), family="temporal", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="temp_ml", generation_seed=seed, generator_family="action_temporal", temporal_chain=chain)
            )

        elif task == "three_way":
            is_unknown = rng.random() < 0.20
            q_text = f"Did the person {target_event_a} before {target_event_b}?"

            if is_unknown:
                truth_choice = "UNKNOWN"
            else:
                truth_choice = "TRUE" if rng.random() > 0.5 else "FALSE"

            if truth_choice == "TRUE":
                propositions = [
                    Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive")),
                    Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))
                ]
            elif truth_choice == "FALSE":
                propositions = [
                    Proposition(text="Yes", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive")),
                    Proposition(text="No", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))
                ]
            else:
                propositions = [
                    Proposition(text="Yes", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive")),
                    Proposition(text="No", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=1, semantics=PropositionSemantics(canonical_type="temporal", event_a=target_event_a, event_b=target_event_b, temporal_relation="before", polarity="positive"))
                ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"temp_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="temporal_multihop" if is_multihop else "temporal_singlehop", complexity=min(hops, 4), family="temporal", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="temp_3w", generation_seed=seed, generator_family="action_temporal", temporal_chain=chain)
            )
