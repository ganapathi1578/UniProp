from typing import List, Tuple
from src.datasets.normalized import NormalizedQA
from src.candidates.base import CandidateGenerator

class ThreeWayOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 40) -> Tuple[List[str], List[int], str]:
        raise ValueError("NOT_PRESENT_IN_SOURCE")
