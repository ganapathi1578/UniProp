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
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        valid_frames = []
        for frame in sg.frames.values():
            objs_with_bbox = [o for o in frame.objects.values() if o.bbox is not None and o.name]
            if objs_with_bbox:
                valid_frames.append((frame.id, objs_with_bbox))
                
        if not valid_frames:
            return
            
        frame_id, objs = rng.choice(valid_frames)
        obj = rng.choice(objs)
        obj_name = obj.name
        
        task = rng.choice(["box_object", "box_proposition"])
        
        if task == "box_object":
            # Which object is in this box? (K=10-40)
            k = rng.randint(10, 40)
            
            propositions = [Proposition(text=obj_name, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=obj_name, polarity="positive"))]
            
            all_absent = list(set(OBJECTS.values()) - {o.name for o in objs})
            falses = rng.sample(all_absent, min(len(all_absent), k - 1))
            
            for f in falses:
                propositions.append(Proposition(text=f, truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=f, polarity="positive")))
                
            rng.shuffle(propositions)
            
            tid, q_text = get_query("GROUNDING_SINGLE", rng)
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
            is_pos = rng.random() > 0.5
            target_obj_name = obj_name if is_pos else rng.choice(list(set(OBJECTS.values()) - {o.name for o in objs}))
            
            q_text = f"Is {render_noun_phrase(target_obj_name, False)} present at this location?"
            
            propositions = [
                Proposition(text="Yes", truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive")),
                Proposition(text="No", truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="negative")),
                Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive"))
            ]
            
            grounding = Grounding(object_ids=[obj.id], boxes={obj.id: obj.bbox}, frame_ids=[frame_id], target_object_id=obj.id if is_pos else None, target_object_class=obj.class_id if is_pos else None, target_bbox=obj.bbox)
            
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"grnd_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="grounding_verification", complexity=1, family="grounding"),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_3w", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id] if is_pos else [], frame_ids=[frame_id]),
                grounding=grounding
            )

