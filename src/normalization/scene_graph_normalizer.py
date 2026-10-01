"""Scene graph normalization."""
import dataclasses
from typing import List, Dict, Any, Optional, Tuple

@dataclasses.dataclass
class NormalizedObject:
    id: str
    class_id: str
    name: str
    visible: bool
    bbox: Optional[Tuple[float, float, float, float]]

@dataclasses.dataclass
class NormalizedRelation:
    id: str
    type: str # attention, contact, spatial, verb
    class_id: str
    name: str
    object_ids: List[str]

@dataclasses.dataclass
class NormalizedAction:
    id: str
    charades_id: str
    phrase: str
    start_secs: float
    end_secs: float
    frame_ids: List[str]
    object_id: str
    verb_id: str

@dataclasses.dataclass
class NormalizedFrame:
    id: str
    secs: float
    objects: Dict[str, NormalizedObject]
    relations: List[NormalizedRelation]

@dataclasses.dataclass
class NormalizedSceneGraph:
    video_id: str
    split: str
    frames: Dict[str, NormalizedFrame]
    actions: Dict[str, NormalizedAction]

def get_human_readable_name(class_id: str, objects_dict: dict) -> str:
    from src.constants import OBJECTS
    return OBJECTS.get(class_id, class_id)

def get_relation_name(class_id: str) -> str:
    from src.constants import ALL_RELATIONS
    return ALL_RELATIONS.get(class_id, class_id)

def normalize_video_sg(video_id: str, raw_sg: dict) -> NormalizedSceneGraph:
    """Normalize a raw scene graph dictionary for a single video."""
    frames = {}
    actions = {}
    split = "unknown"
    
    # Pre-pass to find split and collect actions
    for key, entry in raw_sg.items():
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get('type')
        if not entry_type:
            continue
            
        if entry_type == 'frame':
            meta = entry.get('metadata', '')
            if isinstance(meta, str):
                split = meta
            elif isinstance(meta, dict) and 'set' in meta:
                split = meta['set']
        elif entry_type == 'action':
            actions[key] = NormalizedAction(
                id=key,
                charades_id=entry.get('charades', ''),
                phrase=entry.get('phrase', ''),
                start_secs=entry.get('start', 0.0),
                end_secs=entry.get('end', 0.0),
                frame_ids=entry.get('all_f', []),
                object_id=entry.get('object_id', ''),
                verb_id=entry.get('verb_id', '')
            )

    # Main pass to build frames
    for key, entry in raw_sg.items():
        if not isinstance(entry, dict):
            continue
        if entry.get('type') == 'frame':
            frame_id = entry.get('id', key)
            secs = entry.get('secs', 0.0)
            
            norm_objects = {}
            raw_objects = entry.get('objects', {})
            if isinstance(raw_objects, dict):
                for vertex in raw_objects.get('vertices', []):
                    if isinstance(vertex, dict):
                        obj_id = vertex.get('id', '')
                        class_id = vertex.get('class', '')
                        name = get_human_readable_name(class_id, raw_objects)
                        bbox = vertex.get('bbox')
                        # Ensure bbox is a tuple of 4 floats if present
                        if bbox and len(bbox) == 4:
                            bbox = tuple(float(x) for x in bbox)
                        else:
                            bbox = None
                        
                        norm_objects[obj_id] = NormalizedObject(
                            id=obj_id,
                            class_id=class_id,
                            name=name,
                            visible=vertex.get('visible', True),
                            bbox=bbox
                        )
            
            norm_relations = []
            # Relations are embedded inside the object vertices
            if isinstance(raw_objects, dict):
                for vertex in raw_objects.get('vertices', []):
                    if isinstance(vertex, dict):
                        for rel_type in ['attention', 'contact', 'spatial', 'verb']:
                            for rel_data in vertex.get(rel_type, []):
                                if isinstance(rel_data, dict):
                                    rel_id = rel_data.get('id', '')
                                    class_id = rel_data.get('class', '')
                                    
                                    # Get target object IDs
                                    # The 'objects' list inside the relation usually contains the target object dicts
                                    target_obj_ids = []
                                    for target_obj in rel_data.get('objects', []):
                                        if isinstance(target_obj, dict):
                                            # Action Genome uses the object class as the 'class' field
                                            target_class = target_obj.get('class', '')
                                            if target_class:
                                                target_obj_ids.append(target_class)
                                                
                                    norm_relations.append(NormalizedRelation(
                                        id=rel_id,
                                        type=rel_type,
                                        class_id=class_id,
                                        name=get_relation_name(class_id),
                                        object_ids=target_obj_ids
                                    ))
            
            frames[frame_id] = NormalizedFrame(
                id=frame_id,
                secs=secs,
                objects=norm_objects,
                relations=norm_relations
            )

    return NormalizedSceneGraph(
        video_id=video_id,
        split=split,
        frames=frames,
        actions=actions
    )
