from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class TemporalOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        ans = str(record.source_answer).strip().lower()
        if ans in ["before", "after"]:
            options = ["before", "after"]
            labels = [1 if opt == ans else 0 for opt in options]
            return options, labels, "TRUE"
        return [ans], [1], "UNKNOWN"
