from .interfaces import CandidateGenerator
from typing import List

class ObjectOptionGenerator(CandidateGenerator):
    def generate(self, context, num_candidates: int) -> List[str]:
        return ["object1", "object2"][:num_candidates]

class ActionOptionGenerator(CandidateGenerator):
    def generate(self, context, num_candidates: int) -> List[str]:
        return ["action1", "action2"][:num_candidates]

class CountOptionGenerator(CandidateGenerator):
    def generate(self, context, num_candidates: int) -> List[str]:
        return ["1", "2"][:num_candidates]

class TemporalOptionGenerator(CandidateGenerator):
    def generate(self, context, num_candidates: int) -> List[str]:
        return ["before", "after"][:num_candidates]
