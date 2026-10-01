from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class CountOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        try:
            true_count = int(str(record.source_answer).strip())
        except ValueError:
            return [str(record.source_answer)], [1], "UNKNOWN"
            
        options = []
        labels = []
        
        # Simple policy: true count + sequential numbers
        # e.g., if true=3, K=5 -> 1, 2, 3, 4, 5
        # If true=0, K=5 -> 0, 1, 2, 3, 4
        
        start_count = max(0, true_count - (max_k // 2))
        for i in range(max_k):
            cand = start_count + i
            options.append(str(cand))
            labels.append(1 if cand == true_count else 0)
            
        return options, labels, "TRUE"
