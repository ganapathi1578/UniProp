from typing import Type, Dict
from src.candidates.base import CandidateGenerator

class CandidateRegistry:
    def __init__(self):
        self._generators: Dict[str, Type[CandidateGenerator]] = {}

    def register(self, answer_type: str, generator_class: Type[CandidateGenerator]):
        self._generators[answer_type] = generator_class

    def get_generator(self, answer_type: str) -> Type[CandidateGenerator]:
        return self._generators.get(answer_type)

candidate_registry = CandidateRegistry()
