"""Synthetic Scene Graph fixtures for deterministic, fast unit tests."""
from src.normalization.scene_graph_normalizer import (
    NormalizedSceneGraph,
    NormalizedFrame,
    NormalizedObject,
    NormalizedRelation,
    NormalizedAction
)
from src.negatives.hard_negatives import NegativePools

def create_synthetic_scenegraph(video_id: str = "SYNTH_VID_001") -> NormalizedSceneGraph:
    """Creates a deterministic in-memory scene graph with real Action Genome taxonomy keys."""
    objects = {
        "obj_1": NormalizedObject(id="obj_1", class_id="o8", name="chair", visible=True, bbox=(0.1, 0.1, 0.4, 0.5)),
        "obj_2": NormalizedObject(id="obj_2", class_id="o36", name="window", visible=True, bbox=(0.6, 0.0, 0.9, 0.5)),
        "obj_3": NormalizedObject(id="obj_3", class_id="o17", name="food", visible=True, bbox=(0.3, 0.2, 0.8, 0.6)),
    }
    
    relations = [
        NormalizedRelation(id="rel_1", type="contact", class_id="r21", name="sitting on", object_ids=["o8"]),
        NormalizedRelation(id="rel_2", type="spatial", class_id="r6", name="in front of", object_ids=["o36"]),
        NormalizedRelation(id="rel_3", type="attention", class_id="r1", name="looking at", object_ids=["o17"]),
        NormalizedRelation(id="rel_4", type="contact", class_id="r15", name="holding", object_ids=["o17"]),
    ]
    
    frames = {
        "f_001": NormalizedFrame(id="f_001", secs=1.0, objects=objects, relations=relations),
        "f_002": NormalizedFrame(id="f_002", secs=2.5, objects=objects, relations=relations),
        "f_003": NormalizedFrame(id="f_003", secs=5.0, objects=objects, relations=relations),
    }
    
    actions = {
        "act_1": NormalizedAction(
            id="act_1",
            charades_id="c010",
            phrase="sitting in a chair",
            start_secs=0.5,
            end_secs=3.0,
            frame_ids=["f_001", "f_002"],
            object_id="o8",
            verb_id="v001"
        ),
        "act_2": NormalizedAction(
            id="act_2",
            charades_id="c020",
            phrase="watching outside of a window",
            start_secs=3.5,
            end_secs=6.0,
            frame_ids=["f_003"],
            object_id="o36",
            verb_id="v002"
        ),
    }
    
    return NormalizedSceneGraph(
        video_id=video_id,
        split="train",
        frames=frames,
        actions=actions
    )

def create_synthetic_pools(sg: NormalizedSceneGraph) -> NegativePools:
    """Creates a negative pool backed by the synthetic scene graph."""
    return NegativePools(sg)
