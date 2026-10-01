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

class ComparisonOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        ans = str(record.source_answer).strip().lower()
        # Mock comparison for now
        options = [ans, "alternative_comparison"]
        labels = [1, 0]
        return options, labels, "TRUE"
        
class ActionOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        true_act = str(record.source_answer).strip()
        candidates = ["walk", "sit", "hold a phone", "open a door", "drink", "stand", "run"]
        if true_act not in candidates:
            candidates.append(true_act)
            
        cands = list(set(candidates))
        cands.remove(true_act)
        cands.sort()
        
        distractors = cands[:max_k-1]
        options = [true_act] + distractors
        labels = [1] + [0]*len(distractors)
        return options, labels, "TRUE"
        
class SuperlativeOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        # Same fallback as object/action depending on the domain
        true_ans = str(record.source_answer).strip()
        return [true_ans, "distractor_superlative"], [1, 0], "TRUE"
        
class LogicOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        true_ans = str(record.source_answer).strip()
        return [true_ans, "false_logic_distractor"], [1, 0], "TRUE"
