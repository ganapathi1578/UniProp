import random
from typing import Iterator
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS, ALL_RELATIONS
from src.compatibility import is_compatible

class CompositionalGenerator(GeneratorBase):
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        if not pools or not pools.present_relations: return
        
        obj_rels = {}
        for r_type, r_class, obj in pools.present_relations:
            if obj not in obj_rels: obj_rels[obj] = []
            obj_rels[obj].append(r_class)
            
        valid_objs = [o for o, rels in obj_rels.items() if len(rels) >= 2]
        if not valid_objs: return
            
        target_obj = rng.choice(valid_objs)
        true_rels = obj_rels[target_obj]
        
        # We'll just generate binary or single-choice for AND composition
        task = rng.choice(["binary", "single_choice"])
        k = rng.randint(10, 40) if task == "single_choice" else 2
        
        selected_true = rng.sample(true_rels, 2)
        r1, r2 = selected_true
        r1_name = ALL_RELATIONS.get(r1).replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
        r2_name = ALL_RELATIONS.get(r2).replace('have it on the back', 'carrying on their back').replace('other relationship', 'interacting with')
        if not r1_name or not r2_name: return
        
        obj_name = OBJECTS.get(target_obj, target_obj)
        
        true_text = f"The person is {r1_name} and {r2_name} the {obj_name}."
        propositions = [Proposition(text=true_text, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="compositional_and", subject="person", predicate=f"{r1_name} and {r2_name}", object=obj_name, polarity="positive"))]
        
        if task == "binary":
            # Just one negative: one true, one false relation
            false_cands = [r for r in ALL_RELATIONS if r not in true_rels and is_compatible(r, target_obj)]
            if not false_cands: return
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

