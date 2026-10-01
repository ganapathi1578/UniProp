"""Hard negative and UNKNOWN generation logic for scene graphs."""
import random
from typing import List, Set, Tuple, Optional
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.constants import ATTENTION_RELATIONS, SPATIAL_RELATIONS, CONTACT_RELATIONS, OBJECTS

RELATION_FAMILIES = {}
for r_id in ATTENTION_RELATIONS:
    RELATION_FAMILIES[r_id] = "attention"
for r_id in SPATIAL_RELATIONS:
    RELATION_FAMILIES[r_id] = "spatial"
for r_id in CONTACT_RELATIONS:
    RELATION_FAMILIES[r_id] = "contact"

MUTUALLY_EXCLUSIVE_RELATIONS = {
    "r1": ["r2"], # looking at -> not looking at
    "r2": ["r1"],
    "r4": ["r5"], # above -> beneath
    "r5": ["r4"],
}

class NegativePools:
    def __init__(self, sg: NormalizedSceneGraph):
        self.sg = sg
        self._build_pools()
        
    def _build_pools(self):
        self.present_objects = set()
        self.present_relations = set() # (type, class_id, obj_target)
        
        for frame in self.sg.frames.values():
            for obj in frame.objects.values():
                self.present_objects.add(obj.class_id)
            for rel in frame.relations:
                # ONLY use attention, spatial, contact. IGNORE verbs since we don't have names for v000.
                if rel.type in ['attention', 'spatial', 'contact']:
                    for obj_target in rel.object_ids:
                        self.present_relations.add((rel.type, rel.class_id, obj_target))

    def get_mutually_exclusive_relation(self, true_rel_class: str, target_obj: str, rng: random.Random) -> Optional[str]:
        """Get a relation that is logically FALSE because it contradicts a true relation."""
        # 1. Direct explicit contradiction
        if true_rel_class in MUTUALLY_EXCLUSIVE_RELATIONS:
            return rng.choice(MUTUALLY_EXCLUSIVE_RELATIONS[true_rel_class])
        return None

    def get_closed_world_false_relation(self, true_rel_class: str, target_obj: str, rng: random.Random) -> Optional[str]:
        """
        Get a relation that is false assuming Action Genome exhaustively labels relations for the same object.
        If a person is interacting with an object, and relationship X is not annotated, we assume X is false.
        """
        family = RELATION_FAMILIES.get(true_rel_class)
        if not family:
            return None
            
        possible_classes = []
        if family == "attention":
            possible_classes = list(ATTENTION_RELATIONS.keys())
        elif family == "spatial":
            possible_classes = list(SPATIAL_RELATIONS.keys())
        elif family == "contact":
            possible_classes = list(CONTACT_RELATIONS.keys())
            
        # Filter to those definitely not present for this object in the video
        # AND are typed correctly
        false_classes = [
            c for c in possible_classes 
            if c != true_rel_class and (family, c, target_obj) not in self.present_relations and self._is_compatible(c, target_obj)
        ]
        
        if false_classes:
            return rng.choice(false_classes)
        return None
        
    def _is_compatible(self, rel_class: str, target_obj: str) -> bool:
        """Type check predicate / object compatibility."""
        obj_name = OBJECTS.get(target_obj, "")
        if not obj_name: return False
        
        # We don't have human names mapped inside constants easily for all, 
        # but we know some bad ones from the prompt (eating window, drinking window, standing on window)
        # We will block the most obvious bad combos based on keywords
        from src.constants import ALL_RELATIONS
        rel_name = ALL_RELATIONS.get(rel_class, "")
        
        if "eat" in rel_name:
            return obj_name in {"food", "sandwich"}
        if "drink" in rel_name:
            return obj_name in {"cup_glass_bottle", "medicine", "water"}
        if "stand" in rel_name:
            return obj_name in {"floor", "bed", "chair", "table"}
        if "sit" in rel_name:
            return obj_name in {"chair", "sofa_couch", "bed", "floor", "table"}
        if "wear" in rel_name:
            return obj_name in {"clothes", "shoe", "bag"}
        if "wipe" in rel_name:
            return obj_name in {"table", "mirror", "window", "floor", "television", "refrigerator"}
            
        # Default allow (since explicit annotation overrides it, but this is for generating fake negatives)
        return True

    def get_false_object(self, rng: random.Random) -> Optional[str]:
        """Get an object class that is definitely NOT present in the video."""
        all_objects = set(OBJECTS.keys())
        absent_objects = list(all_objects - self.present_objects)
        if absent_objects:
            return rng.choice(absent_objects)
        return None
