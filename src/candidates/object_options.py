from typing import List, Tuple
from src.datasets.normalized import NormalizedQA
from src.candidates.base import CandidateGenerator
from src.reasoning.semantic_types import SemanticType
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.canonicalization import deduplicate_candidates, are_objects_equivalent, canonicalize_object

class ObjectOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 40) -> Tuple[List[str], List[int], str]:
        evaluator = context["evaluator"]
        sg = evaluator.get_scenegraph(record.video_id, getattr(record.source_metadata, "split", "train") if hasattr(record, "source_metadata") else "train")
        
        if not sg:
            raise ValueError("missing_evidence")
            
        true_ans = str(record.source_answer).lower().strip()
        
        ast = parse_agqa_program(record.source_program)
        engine = SemanticEngine(sg)
        
        result = engine.evaluate(ast)
        
        true_objects = set()
        
        if result.type == SemanticType.OBJECT_SET:
            true_objects = result.value
        elif result.type == SemanticType.OBJECT:
            true_objects.add(result.value)
        else:
            # Fallback to checking all objects in SG if we can't extract exactly the set (e.g., if evaluator is unsupported)
            # but wait, the prompt says "do not use all objects anywhere in the video"
            # If we don't have the explicit set, we should raise.
            raise ValueError("source_evidence_mismatch")
            
        # Distractors are other objects from the SG that are NOT true objects
        all_objects = set()
        for f in sg.frames.values():
            for obj in f.objects.values():
                all_objects.add(obj.name)
                
        all_canon = set(deduplicate_candidates(list(all_objects), "object"))
        true_canon = set(canonicalize_object(o) for o in true_objects)
        
        if canonicalize_object(true_ans) not in true_canon:
            raise ValueError("source_evidence_mismatch")
            
        options = []
        labels = []
        
        options.append(true_ans)
        labels.append(1)
        
        false_pool = sorted(list(all_canon - true_canon))
        for opt in false_pool:
            if not are_objects_equivalent(opt, true_ans):
                options.append(opt)
                labels.append(0)
            if len(options) >= max_k:
                break
                
        if len(options) < 2:
            raise ValueError("insufficient_valid_candidates")
            
        return options, labels, "TRUE"
