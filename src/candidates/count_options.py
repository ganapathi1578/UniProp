from typing import List, Tuple
from src.datasets.normalized import NormalizedQA
from src.candidates.base import CandidateGenerator

class CountOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 40) -> Tuple[List[str], List[int], str]:
        # We did not observe COUNT questions in the 1000-record sample or quick scans.
        # But if they occur, the true answer should be a number.
        true_ans = str(record.source_answer).lower().strip()
        
        if not true_ans.isdigit():
            raise ValueError("evaluator_unsupported: count answer is not numeric")
            
        true_count = int(true_ans)
        
        # Make configurable later, for now max out at true_count + max_k/2
        start = max(0, true_count - 5)
        options = []
        labels = []
        
        for i in range(start, start + min(max_k, 10)):
            opt = str(i)
            options.append(opt)
            labels.append(1 if opt == true_ans else 0)
            
        if sum(labels) == 0:
            raise ValueError("source_evidence_mismatch")
            
        return options, labels, "TRUE"
