from typing import List, Tuple
import random
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class ObjectOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        # True answer
        true_obj = str(record.source_answer).strip()
        
        # Valid candidate objects from evidence (fallback to some default objects if SG not loaded)
        sg_objects = context.get("sg_objects", [])
        if not sg_objects:
            # Fallback for testing
            sg_objects = ["window", "refrigerator", "dish", "chair", "laptop", "broom"]
            
        candidates = set(sg_objects)
        if true_obj in candidates:
            candidates.remove(true_obj)
            
        cand_list = list(candidates)
        # Sort to ensure determinism before possible sampling
        cand_list.sort()
        
        # Select K-1 distractors
        num_distractors = min(len(cand_list), max_k - 1)
        # Normally we'd use rng here, but for determinism without it passed yet, we just slice
        distractors = cand_list[:num_distractors]
        
        options = [true_obj] + distractors
        labels = [1] + [0] * len(distractors)
        
        return options, labels, "TRUE" if true_obj else "UNKNOWN"
