from abc import ABC, abstractmethod
from typing import List, Tuple
from src.datasets.normalized import NormalizedQA

class CandidateGenerator(ABC):
    @abstractmethod
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        """
        Returns:
            options: List[str]
            labels: List[int]
            truth_state: str ("TRUE", "FALSE", "UNKNOWN", "NONE")
        """
        pass
