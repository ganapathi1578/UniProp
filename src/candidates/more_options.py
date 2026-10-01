from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class TemporalOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        ans = str(record.source_answer).strip().lower()
        if ans not in ["before", "after"]:
            raise ValueError(f"UNSUPPORTED_TEMPORAL_ANSWER: {ans}")
            
        options = ["before", "after"]
        
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("MISSING_EVALUATOR")
            
        sg = evaluator.get_scenegraph(record.video_id)
        if not sg:
            raise ValueError("MISSING_SCENEGRAPH")
            
        truth_before = evaluator.evaluate(record, "before", context)
        truth_after = evaluator.evaluate(record, "after", context)
        
        if truth_before == "TRUE" and truth_after == "TRUE":
            # Ambiguous/Contradictory
            raise ValueError("CONTRADICTORY_TEMPORAL_EVIDENCE")
            
        if truth_before == "UNKNOWN" and truth_after == "UNKNOWN":
            labels = [0, 0]
            truth_state = "UNKNOWN"
            # It's an UNKNOWN semantic state. But we don't return TRUE/FALSE.
            # AGQA temporal usually doesn't have "Cannot say" structurally but if it's genuinely unknown, we must respect the UNKNOWN label mechanism.
            # But if source says "before" and we found UNKNOWN, it's a mismatch if we assume strictness!
            # The prompt says: "UNKNOWN must mean genuine semantic uncertainty, not unsupported implementation. If they disagree: reject with SOURCE_EVIDENCE_MISMATCH"
            raise ValueError("SOURCE_EVIDENCE_MISMATCH")
            
        if truth_before == "TRUE":
            if ans != "before":
                raise ValueError("SOURCE_EVIDENCE_MISMATCH")
            labels = [1, 0]
            truth_state = "TRUE"
        elif truth_after == "TRUE":
            if ans != "after":
                raise ValueError("SOURCE_EVIDENCE_MISMATCH")
            labels = [0, 1]
            truth_state = "TRUE"
        else:
            raise ValueError("SOURCE_EVIDENCE_MISMATCH")
            
        return options, labels, truth_state
