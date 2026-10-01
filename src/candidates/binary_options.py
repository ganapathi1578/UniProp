from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class BinaryOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 2) -> Tuple[List[str], List[int], str]:
        options = ["Yes", "No"]
        
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("MISSING_EVALUATOR")
            
        sg = evaluator.get_scenegraph(record.video_id)
        if not sg:
            raise ValueError("MISSING_SCENEGRAPH")
            
        # evaluate the YES option
        truth = evaluator.evaluate(record, "Yes", context)
        
        # Consistency check with source answer
        ans = str(record.source_answer).lower().strip()
        
        if truth == "TRUE":
            if ans == "no":
                raise ValueError("SOURCE_EVIDENCE_MISMATCH")
            labels = [1, 0]
            truth_state = "TRUE"
        elif truth == "FALSE":
            if ans == "yes":
                raise ValueError("SOURCE_EVIDENCE_MISMATCH")
            labels = [0, 1]
            truth_state = "FALSE"
        else:
            options.append("Cannot say")
            labels = [0, 0, 1]
            truth_state = "UNKNOWN"
            
        return options, labels, truth_state
