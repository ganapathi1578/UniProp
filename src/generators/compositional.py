import random
from typing import Iterator, Optional
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS, ALL_RELATIONS
from src.compatibility import is_compatible
from src.templates import get_query

class CompositionalGenerator(GeneratorBase):
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

        obj_rels = {}
        for r_type, r_class, obj in pools.present_relations:
            if obj not in obj_rels:
                obj_rels[obj] = []
            obj_rels[obj].append(r_class)

        valid_objs = [o for o, rels in obj_rels.items() if len(rels) >= 2]
        if not valid_objs:
            return

        target_obj = rng.choice(valid_objs)
        true_rels = obj_rels[target_obj]

        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = target_task if target_task in task_types else rng.choice(["binary", "single_choice"])

        if task == "binary":
            k = 2
        elif task == "three_way":
            k = 3
        else:
            k = target_k if target_k is not None and 2 <= target_k <= 40 else rng.randint(4, 25)

        hops = target_hops if target_hops is not None else 2
        difficulty = target_difficulty if target_difficulty in ("easy", "medium", "hard") else "medium"

        selected_true = rng.sample(true_rels, 2)
        r1, r2 = selected_true
        r1_name = ALL_RELATIONS.get(r1, "").replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
        r2_name = ALL_RELATIONS.get(r2, "").replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
        if not r1_name or not r2_name:
            return

        obj_name = OBJECTS.get(target_obj, target_obj)
        true_text = f"The person is {r1_name} and {r2_name} the {obj_name}."

        false_cands = [r for r in ALL_RELATIONS if r not in true_rels and is_compatible(r, target_obj)]
        if not false_cands:
            return

        if task == "binary":
            r3 = rng.choice(false_cands)
            r3_name = ALL_RELATIONS.get(r3, "").replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
            false_text = f"The person is {r1_name} and {r3_name} the {obj_name}."

            is_pos = rng.random() > 0.5
            propositions = [
                Proposition(text=true_text, truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive")),
                Proposition(text=false_text, truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r3_name}", object=obj_name, polarity="negative" if not is_pos else "positive"))
            ]
            if not is_pos:
                propositions.reverse()

            seed = rng.randint(0, 2**32-1)
            tid, comp_q_text = get_query("COMPOSITIONAL", rng)
            yield PropositionGroup(
                example_id=f"comp_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=comp_q_text, query_template_id=tid, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="compositional_and", complexity=min(hops, 4), family="compositional", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_comp_bin", generation_seed=seed, generator_family="compositional", evidence_object_ids=[target_obj], evidence_relation_ids=selected_true)
            )

        elif task == "single_choice":
            propositions = [Proposition(text=true_text, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))]

            rng.shuffle(false_cands)
            for f in false_cands[:k-1]:
                f_name = ALL_RELATIONS.get(f, "").replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
                propositions.append(Proposition(text=f"The person is {r1_name} and {f_name} the {obj_name}.", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {f_name}", object=obj_name, polarity="positive")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            tid, comp_q_text = get_query("COMPOSITIONAL", rng)
            yield PropositionGroup(
                example_id=f"comp_sc_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=comp_q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="compositional_and", complexity=min(hops, 4), family="compositional", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_comp_sc", generation_seed=seed, generator_family="compositional", evidence_object_ids=[target_obj], evidence_relation_ids=selected_true)
            )

        elif task == "multi_label":
            propositions = [Proposition(text=true_text, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))]

            rng.shuffle(false_cands)
            for f in false_cands[:k-2]:
                f_name = ALL_RELATIONS.get(f, "").replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
                propositions.append(Proposition(text=f"The person is {r1_name} and {f_name} the {obj_name}.", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {f_name}", object=obj_name, polarity="positive")))

            if rng.random() < 0.7:
                propositions.append(Proposition(text="None of these joint interactions hold.", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="logical_none")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            tid, comp_q_text = get_query("COMPOSITIONAL", rng)
            yield PropositionGroup(
                example_id=f"comp_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=comp_q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="compositional_and", complexity=min(hops, 4), family="compositional", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_comp_ml", generation_seed=seed, generator_family="compositional", evidence_object_ids=[target_obj], evidence_relation_ids=selected_true)
            )

        elif task == "three_way":
            is_unknown = rng.random() < 0.20
            q_text = f"Is the person {r1_name} and {r2_name} the {obj_name}?"

            if is_unknown:
                truth_choice = "UNKNOWN"
            else:
                truth_choice = "TRUE" if rng.random() > 0.5 else "FALSE"

            if truth_choice == "TRUE":
                propositions = [
                    Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))
                ]
            elif truth_choice == "FALSE":
                propositions = [
                    Proposition(text="Yes", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))
                ]
            else:
                propositions = [
                    Proposition(text="Yes", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))
                ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"comp_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="compositional_and", complexity=min(hops, 4), family="compositional", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_comp_3w", generation_seed=seed, generator_family="compositional", evidence_object_ids=[target_obj], evidence_relation_ids=selected_true)
            )
