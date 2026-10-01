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
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        if not pools or not pools.present_objects: return
        
        # We need to generate binary, single-choice, multi-label, three-way
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = rng.choice(task_types)
        
        # Decide K
        k = rng.randint(10, 40) if task in ["single_choice", "multi_label"] else (3 if task == "three_way" else 2)
        
        # Collect true and false
        all_true = list(set([OBJECTS.get(t, t) for t in pools.present_objects]))
        all_absent = list(set(OBJECTS.values()) - set(all_true))
        
        if task == "binary":
            is_positive_centered = rng.random() > 0.5
            target_obj = rng.choice(all_true) if is_positive_centered else (rng.choice(all_absent) if all_absent else rng.choice(all_true))
            is_true = target_obj in all_true
            obj_name = target_obj
            
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
            target_true = rng.choice(all_true)
            falses = rng.sample(all_absent, min(len(all_absent), k - 1))
            
            propositions = []
            obj_name = OBJECTS.get(target_true, target_true)
            propositions.append(Proposition(text=render_existence(obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="object_existence", subject=obj_name, polarity="positive")))
            
            for f in falses:
                f_name = f
                propositions.append(Proposition(text=render_existence(f_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="object_existence", subject=f_name, polarity="positive")))
                
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
            num_true = rng.randint(1, min(len(all_true), k - 1)) if all_true else 0
            if rng.random() < 0.2: num_true = 0 # 20% all-false for NONE=1
            num_false = k - num_true
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
                
            none_included = rng.random() < 0.7
            if none_included:
                none_val = 1 if not selected_true else 0
                propositions.append(Proposition(text="None of these objects are present.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))
                
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
            is_pos = rng.random() > 0.5
            target_obj = rng.choice(all_true) if is_pos else rng.choice(all_absent)
            obj_name = target_obj
            
            q_text = f"Is {render_noun_phrase(obj_name, False)} present in the video?"
            
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
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        if not pools or not pools.present_relations: return
        
        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = rng.choice(task_types)
        k = rng.randint(10, 40) if task in ["single_choice", "multi_label"] else (3 if task == "three_way" else 2)
        
        true_rels = list(pools.present_relations)
        
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
                neg_rel = pools.get_mutually_exclusive_relation(rel_class, obj, rng)
                if not neg_rel: neg_rel = pools.get_closed_world_false_relation(rel_class, obj, rng)
                if not neg_rel: return
                
                rel_name = ALL_RELATIONS.get(neg_rel, neg_rel)
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
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            rel_name = ALL_RELATIONS.get(rel_class, rel_class)
            
            propositions = [Proposition(text=render_relation(rel_name, obj_name, "positive"), truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=rel_name, object=obj_name, polarity="positive"))]
            
            false_cands = []
            for r in ALL_RELATIONS:
                if r != rel_class and is_compatible(r, obj) and (rel_type, r, obj) not in true_rels:
                    false_cands.append(r)
                    
            rng.shuffle(false_cands)
            for f in false_cands[:k-1]:
                f_name = ALL_RELATIONS.get(f, f)
                propositions.append(Proposition(text=render_relation(f_name, obj_name, "positive"), truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="relation", subject="person", predicate=f_name, object=obj_name, polarity="positive")))
                
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
            obj_to_rels = {}
            for r_type, r_class, o in true_rels:
                if o not in obj_to_rels: obj_to_rels[o] = []
                obj_to_rels[o].append(r_class)
                
            obj = rng.choice(list(obj_to_rels.keys()))
            obj_name = OBJECTS.get(obj, obj)
            obj_true_rels = obj_to_rels[obj]
            
            num_true = rng.randint(1, min(len(obj_true_rels), k - 1))
            if rng.random() < 0.2: num_true = 0
            
            selected_true = rng.sample(obj_true_rels, num_true)
            
            false_cands = []
            for r in ALL_RELATIONS:
                if r not in obj_true_rels and is_compatible(r, obj):
                    false_cands.append(r)
                    
            num_false = k - num_true
            if rng.random() > 0.8: k -= 1 # NONE
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
            rel_type, rel_class, obj = rng.choice(true_rels)
            obj_name = OBJECTS.get(obj, obj)
            rel_name = ALL_RELATIONS.get(rel_class, rel_class)
            
            q_text = f"Is the person {rel_name} {render_noun_phrase(obj_name, True)}?"
            
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
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        yield from [] # Handled by the above RelationVerificationGenerator which now handles everything seamlessly

