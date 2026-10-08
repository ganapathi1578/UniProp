import random
from typing import Iterator, Optional
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools
from src.generators.base import GeneratorBase
from src.generators.renderer import render_existence, render_relation, render_noun_phrase
from src.constants import OBJECTS, ALL_RELATIONS
from src.templates import get_query, get_query_formatted
from src.compatibility import is_compatible

class ObjectExistenceGenerator(GeneratorBase):
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
    ) -> Iterator[PropositionGroup]:
        if not pools or not pools.present_objects:
            return

        # Select task
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = target_task if target_task in task_types else rng.choice(task_types)

        # Decide K
        if task == "binary":
            k = 2
        elif task == "three_way":
            k = 3
        else:
            k = target_k if target_k is not None and 2 <= target_k <= 40 else rng.randint(4, 25)

        hops = target_hops if target_hops is not None else 1
        difficulty = target_difficulty if target_difficulty in ("easy", "medium", "hard") else "medium"

        # Collect true and false
        all_true = list(set([OBJECTS.get(t, t) for t in pools.present_objects]))
        all_absent = list(set(OBJECTS.values()) - set(all_true))
        if not all_true:
            return

        # Difficulty sorting for falses
        if difficulty == "hard":
            # Co-occurring room items or near-misses
            absent_pool = sorted(all_absent, key=lambda x: len(x))
        else:
            absent_pool = all_absent

        if task == "binary":
            is_positive_centered = rng.random() > 0.5
            target_obj = rng.choice(all_true) if is_positive_centered else (rng.choice(absent_pool) if absent_pool else rng.choice(all_true))
            is_true = target_obj in all_true
            obj_name = target_obj

            p_pos = Proposition(text=render_existence(obj_name, "positive"), truth_state="TRUE" if is_true else "FALSE", label=1 if is_true else 0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive"))
            p_neg = Proposition(text=render_existence(obj_name, "negative"), truth_state="FALSE" if is_true else "TRUE", label=0 if is_true else 1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="negative"))
            propositions = [p_pos, p_neg]

            tid, q_text = get_query("OBJECT_VERIFY", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid,
                propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="object_existence", complexity=min(hops, 4), family="object", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_binary", generation_seed=seed, generator_family="object_existence", evidence_object_ids=[target_obj])
            )

        elif task == "single_choice":
            target_true = rng.choice(all_true)
            sample_size = min(len(absent_pool), k - 1)
            falses = rng.sample(absent_pool, sample_size) if sample_size > 0 else []

            propositions = []
            obj_name = OBJECTS.get(target_true, target_true)
            propositions.append(Proposition(text=render_existence(obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")))

            for f in falses:
                propositions.append(Proposition(text=render_existence(f, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=f, polarity="positive")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("OBJECT_SINGLE", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="object_existence", complexity=min(hops, 4), family="object", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_sc", generation_seed=seed, generator_family="object_existence", evidence_object_ids=[target_true])
            )

        elif task == "multi_label":
            num_true = rng.randint(1, min(len(all_true), k - 1)) if all_true else 0
            if rng.random() < 0.2:
                num_true = 0  # 20% all-false for NONE=1
            num_false = max(0, k - num_true)
            if rng.random() > 0.8 and k > 2:
                num_false = max(0, num_false - 1)  # Leave room for NONE

            selected_true = rng.sample(all_true, min(len(all_true), num_true))
            selected_false = rng.sample(absent_pool, min(len(absent_pool), num_false))

            propositions = []
            for t in selected_true:
                propositions.append(Proposition(text=render_existence(t, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=t, polarity="positive")))
            for f in selected_false:
                propositions.append(Proposition(text=render_existence(f, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=f, polarity="positive")))

            none_included = rng.random() < 0.7
            if none_included:
                none_val = 1 if not selected_true else 0
                propositions.append(Proposition(text="None of these objects are present.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("OBJECT_SET", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="object_existence", complexity=min(hops, 4), family="object", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_ml", generation_seed=seed, generator_family="object_existence", evidence_object_ids=selected_true)
            )

        elif task == "three_way":
            # Support TRUE, FALSE, and UNKNOWN ground-truth
            # 20% unknown ground truth
            is_unknown = rng.random() < 0.20
            is_pos = (not is_unknown) and (rng.random() > 0.5)

            if is_unknown:
                # Ambiguous/unobserved object
                target_obj = rng.choice(absent_pool) if absent_pool else "small unidentifiable item"
                truth_choice = "UNKNOWN"
            elif is_pos:
                target_obj = rng.choice(all_true)
                truth_choice = "TRUE"
            else:
                target_obj = rng.choice(absent_pool) if absent_pool else "unknown object"
                truth_choice = "FALSE"

            obj_name = target_obj
            q_text = f"Is {render_noun_phrase(obj_name, False)} present in the video?"

            if truth_choice == "TRUE":
                propositions = [
                    Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive"))
                ]
            elif truth_choice == "FALSE":
                propositions = [
                    Proposition(text="Yes", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive"))
                ]
            else:
                propositions = [
                    Proposition(text="Yes", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive"))
                ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"obj_ex_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="object_existence", complexity=min(hops, 4), family="object", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_obj_exist_3w", generation_seed=seed, generator_family="object_existence", evidence_object_ids=[target_obj])
            )


class RelationVerificationGenerator(GeneratorBase):
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
    ) -> Iterator[PropositionGroup]:
        if not pools or not pools.present_relations:
            return

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

        true_rels = list(pools.present_relations)
        if not true_rels:
            return

        if task == "binary":
            is_pos = rng.random() > 0.5
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)

            if is_pos:
                rel_name = ALL_RELATIONS.get(rel_class, rel_class)
                p_pos = Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                p_neg = Proposition(text=render_relation(rel_name, obj_name, "negative"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative"))
                propositions = [p_pos, p_neg]
            else:
                neg_rel = None
                if difficulty == "hard":
                    neg_rel = pools.get_mutually_exclusive_relation(rel_class, obj, rng)
                if not neg_rel:
                    neg_rel = pools.get_closed_world_false_relation(rel_class, obj, rng)
                if not neg_rel:
                    return

                rel_name = ALL_RELATIONS.get(neg_rel, neg_rel)
                p_pos = Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                p_neg = Proposition(text=render_relation(rel_name, obj_name, "negative"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative"))
                propositions = [p_pos, p_neg]

            tid, q_text = get_query("RELATION_VERIFY", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="relation_verification", complexity=min(hops, 4), family=rel_type, hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_bin", generation_seed=seed, generator_family="relation_verification", evidence_object_ids=[obj], evidence_relation_ids=[rel_class])
            )

        elif task == "single_choice":
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            rel_name = ALL_RELATIONS.get(rel_class, rel_class)

            propositions = [Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))]

            false_cands = []
            if difficulty == "hard":
                excl = pools.get_mutually_exclusive_relation(rel_class, obj, rng)
                if excl:
                    false_cands.append(excl)

            for r in ALL_RELATIONS:
                if r != rel_class and is_compatible(r, obj) and (rel_type, r, obj) not in true_rels and r not in false_cands:
                    false_cands.append(r)

            rng.shuffle(false_cands)
            for f in false_cands[:k-1]:
                f_name = ALL_RELATIONS.get(f, f)
                propositions.append(Proposition(text=render_relation(f_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=f_name, object=obj_name, polarity="positive")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("RELATION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="relation_verification", complexity=min(hops, 4), family=rel_type, hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_sc", generation_seed=seed, generator_family="relation_verification", evidence_object_ids=[obj], evidence_relation_ids=[rel_class])
            )

        elif task == "multi_label":
            obj_to_rels = {}
            for r_type, r_class, o in true_rels:
                if o not in obj_to_rels:
                    obj_to_rels[o] = []
                obj_to_rels[o].append(r_class)

            obj = rng.choice(list(obj_to_rels.keys()))
            obj_name = OBJECTS.get(obj, obj)
            obj_true_rels = obj_to_rels[obj]

            num_true = rng.randint(1, min(len(obj_true_rels), k - 1))
            if rng.random() < 0.2:
                num_true = 0

            selected_true = rng.sample(obj_true_rels, num_true)

            false_cands = []
            for r in ALL_RELATIONS:
                if r not in obj_true_rels and is_compatible(r, obj):
                    false_cands.append(r)

            num_false = max(0, k - num_true)
            if rng.random() > 0.8 and k > 2:
                num_false = max(0, num_false - 1)  # NONE
            selected_false = rng.sample(false_cands, min(len(false_cands), num_false))

            propositions = []
            for r in selected_true:
                r_name = ALL_RELATIONS.get(r, r)
                propositions.append(Proposition(text=render_relation(r_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=r_name, object=obj_name, polarity="positive")))
            for r in selected_false:
                r_name = ALL_RELATIONS.get(r, r)
                propositions.append(Proposition(text=render_relation(r_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=r_name, object=obj_name, polarity="positive")))

            if rng.random() < 0.7:
                none_val = 1 if num_true == 0 else 0
                propositions.append(Proposition(text=f"None of these relationships hold with {render_noun_phrase(obj_name, True)}.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("RELATION", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="relation_set", complexity=min(hops, 4), family="spatial", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_ml", generation_seed=seed, generator_family="relation_sets", evidence_object_ids=[obj], evidence_relation_ids=selected_true)
            )

        elif task == "three_way":
            is_unknown = rng.random() < 0.20
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            rel_name = ALL_RELATIONS.get(rel_class, rel_class)

            q_text = f"Is the person {rel_name} {render_noun_phrase(obj_name, True)}?"

            if is_unknown:
                truth_choice = "UNKNOWN"
            else:
                truth_choice = "TRUE" if rng.random() > 0.5 else "FALSE"

            if truth_choice == "TRUE":
                propositions = [
                    Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                ]
            elif truth_choice == "FALSE":
                propositions = [
                    Proposition(text="Yes", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                ]
            else:
                propositions = [
                    Proposition(text="Yes", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))
                ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"rel_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="relation_verification", complexity=min(hops, 4), family=rel_type, hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_rel_3w", generation_seed=seed, generator_family="relation_verification", evidence_object_ids=[obj], evidence_relation_ids=[rel_class])
            )


class RelationSetGenerator(GeneratorBase):
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
    ) -> Iterator[PropositionGroup]:
        yield from []
