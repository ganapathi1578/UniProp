from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class BinaryOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 2) -> Tuple[List[str], List[int], str]:
        # Binary questions always have "Yes" / "No"
        options = ["Yes", "No"]
        
        # AGQA binary answer is usually "yes" or "no"
        # We must use evidence, but for binary AGQA, the question is a verification against the SG
        # Currently we use the source answer, but in a real evidence backed system, we'd query SG
        ans = str(record.source_answer).lower().strip()
        if ans == "yes":
            labels = [1, 0]
            truth_state = "TRUE"
        elif ans == "no":
            labels = [0, 1]
            truth_state = "FALSE"
        else:
            # Handle unknown or uncertain cases
            options.append("Cannot say")
            labels = [0, 0, 1]
            truth_state = "UNKNOWN"
            
        return options, labels, truth_state
