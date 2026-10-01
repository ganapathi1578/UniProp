from typing import List, Tuple
import random
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class ObjectOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("MISSING_EVALUATOR")
            
        sg = evaluator.get_scenegraph(record.video_id)
        if not sg:
            raise ValueError("MISSING_SCENEGRAPH")
            
        # Valid candidate objects from evidence
        # Gather all objects mentioned in the scene graph to form the pool
        sg_objects = set()
        for f in sg.frames.values():
            for obj in f.objects.values():
                sg_objects.add(obj.name)
                
        # If true answer is not in the pool for some reason, we should still include it
        true_obj = str(record.source_answer).strip()
        sg_objects.add(true_obj)
        
        cand_list = list(sg_objects)
        cand_list.sort()
        
        # Select K distractors (cap at max_k)
        options = cand_list[:max_k]
        if true_obj not in options:
            options[-1] = true_obj
            options.sort()
            
        labels = []
        has_true = False
        
        for opt in options:
            truth = evaluator.evaluate(record, opt, context)
            if truth == "TRUE":
                if opt != true_obj:
                    # Semantic bleed: an option evaluates to TRUE but isn't the source answer
                    raise ValueError("SOURCE_EVIDENCE_MISMATCH")
                labels.append(1)
                has_true = True
            elif truth == "FALSE":
                if opt == true_obj:
                    raise ValueError("SOURCE_EVIDENCE_MISMATCH")
                labels.append(0)
            else:
                # UNKNOWN
                labels.append(0)
                
        if not has_true:
            # Maybe the question is UNKNOWN?
            return options, labels, "UNKNOWN"
            
        return options, labels, "TRUE"
