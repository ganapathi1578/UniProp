from abc import ABC, abstractmethod
from typing import List

class CandidateGenerator(ABC):
    @abstractmethod
    def generate(self, context, num_candidates: int) -> List[str]:
        pass
