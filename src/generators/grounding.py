import random
from typing import Iterator, Optional
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance, Grounding
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS
from src.generators.renderer import render_noun_phrase
from src.templates import get_query

class GroundingGenerator(GeneratorBase):
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

        task_types = ["binary", "single_choice", "multi_label", "three_way"]
        task = target_task if target_task in task_types else rng.choice(["single_choice", "three_way"])

        if task == "binary":
            k = 2
        elif task == "three_way":
            k = 3
        else:
            k = target_k if target_k is not None and 2 <= target_k <= 40 else rng.randint(4, 25)

        hops = target_hops if target_hops is not None else 1
        difficulty = target_difficulty if target_difficulty in ("easy", "medium", "hard") else "medium"

        all_absent = list(set(OBJECTS.values()) - {o.name for o in objs})
        if not all_absent:
            all_absent = ["lamp", "remote_control"]

        grounding = Grounding(object_ids=[obj.id], boxes={obj.id: obj.bbox}, frame_ids=[frame_id], target_object_id=obj.id, target_object_class=obj.class_id, target_bbox=obj.bbox)

        if task == "single_choice":
            propositions = [Proposition(text=obj_name, truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=obj_name, polarity="positive"))]

            sample_size = min(len(all_absent), k - 1)
            falses = rng.sample(all_absent, sample_size) if sample_size > 0 else []
            for f in falses:
                propositions.append(Proposition(text=f, truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=f, polarity="positive")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            tid, q_text = get_query("GROUNDING_SINGLE", rng)
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"grnd_id_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="single_choice", num_propositions=len(propositions),
                reasoning=Reasoning(type="grounding_identification", complexity=min(hops, 4), family="grounding", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_id", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id], frame_ids=[frame_id]),
                grounding=grounding
            )

        elif task == "binary":
            is_pos = rng.random() > 0.5
            target_name = obj_name if is_pos else rng.choice(all_absent)
            propositions = [
                Proposition(text=f"{target_name} is located in this bounding box.", truth_state="TRUE" if is_pos else "FALSE", label=1 if is_pos else 0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_name, polarity="positive")),
                Proposition(text=f"{target_name} is not located in this bounding box.", truth_state="FALSE" if is_pos else "TRUE", label=0 if is_pos else 1, semantics=PropositionSemantics(canonical_type="grounding", subject=target_name, polarity="negative"))
            ]
            seed = rng.randint(0, 2**32-1)
            tid, q_text = get_query("GROUNDING_VERIFY", rng)
            yield PropositionGroup(
                example_id=f"grnd_bin_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="binary", num_propositions=2,
                reasoning=Reasoning(type="grounding_verification", complexity=min(hops, 4), family="grounding", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_bin", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id], frame_ids=[frame_id]),
                grounding=grounding
            )

        elif task == "multi_label":
            frame_obj_names = list(set(o.name for o in objs))
            num_true = rng.randint(1, min(len(frame_obj_names), k - 1)) if frame_obj_names else 0
            selected_true = rng.sample(frame_obj_names, num_true)
            num_false = max(0, k - num_true)
            selected_false = rng.sample(all_absent, min(len(all_absent), num_false))

            propositions = []
            for t in selected_true:
                propositions.append(Proposition(text=f"{t} is present in this frame.", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=t, polarity="positive")))
            for f in selected_false:
                propositions.append(Proposition(text=f"{f} is present in this frame.", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=f, polarity="positive")))

            if rng.random() < 0.7:
                none_val = 1 if not selected_true else 0
                propositions.append(Proposition(text="None of these objects are present in this frame.", truth_state="TRUE" if none_val else "FALSE", label=none_val, semantics=PropositionSemantics(canonical_type="logical_none")))

            if len(propositions) < 2:
                return

            rng.shuffle(propositions)
            seed = rng.randint(0, 2**32-1)
            tid, q_text = get_query("GROUNDING_SET", rng)
            yield PropositionGroup(
                example_id=f"grnd_ml_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=tid, propositions=propositions, task_type="multi_label", num_propositions=len(propositions),
                reasoning=Reasoning(type="grounding_set", complexity=min(hops, 4), family="grounding", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_ml", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id], frame_ids=[frame_id]),
                grounding=grounding
            )

        elif task == "three_way":
            is_unknown = rng.random() < 0.20
            target_obj_name = obj_name if (not is_unknown and rng.random() > 0.5) else rng.choice(all_absent)
            q_text = f"Is {render_noun_phrase(target_obj_name, False)} present at this location?"

            if is_unknown:
                truth_choice = "UNKNOWN"
            else:
                truth_choice = "TRUE" if target_obj_name == obj_name else "FALSE"

            if truth_choice == "TRUE":
                propositions = [
                    Proposition(text="Yes", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive"))
                ]
            elif truth_choice == "FALSE":
                propositions = [
                    Proposition(text="Yes", truth_state="FALSE", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="TRUE", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive"))
                ]
            else:
                propositions = [
                    Proposition(text="Yes", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive")),
                    Proposition(text="No", truth_state="UNKNOWN", label=0, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="negative")),
                    Proposition(text="Cannot say", truth_state="UNKNOWN", label=1, semantics=PropositionSemantics(canonical_type="grounding", subject=target_obj_name, polarity="positive"))
                ]
            seed = rng.randint(0, 2**32-1)
            yield PropositionGroup(
                example_id=f"grnd_3w_{sg.video_id}_{seed}", source="scene_graph", split=split, media_id=sg.video_id,
                query_text=q_text, query_template_id=None, propositions=propositions, task_type="three_way", num_propositions=3,
                reasoning=Reasoning(type="grounding_verification", complexity=min(hops, 4), family="grounding", hops=hops),
                provenance=Provenance(source_question_id=None, source_question_text=None, source_program=None, source_answer=None, scene_graph_id=sg.video_id, generation_rule="sg_grounding_3w", generation_seed=seed, generator_family="grounding", evidence_object_ids=[obj.class_id], frame_ids=[frame_id]),
                grounding=grounding
            )
