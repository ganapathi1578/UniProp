from typing import Iterator
from src.normalization.scene_graph_normalizer import (
    NormalizedSceneGraph,
    NormalizedFrame,
    NormalizedAction,
    NormalizedObject,
    NormalizedRelation
)

def iter_frames(sg: NormalizedSceneGraph) -> Iterator[NormalizedFrame]:
    """Iterate over all frames in the scene graph."""
    return iter(sg.frames.values())

def iter_actions(sg: NormalizedSceneGraph) -> Iterator[NormalizedAction]:
    """Iterate over all actions in the scene graph."""
    return iter(sg.actions.values())

def iter_objects(sg: NormalizedSceneGraph) -> Iterator[NormalizedObject]:
    """Iterate over all distinct objects across all frames."""
    seen = set()
    for frame in iter_frames(sg):
        for obj in frame.objects.values():
            if obj.id not in seen:
                seen.add(obj.id)
                yield obj

def iter_relations(sg: NormalizedSceneGraph) -> Iterator[NormalizedRelation]:
    """Iterate over all relations across all frames."""
    for frame in iter_frames(sg):
        for rel in frame.relations:
            yield rel
