from typing import List, Tuple
from src.datasets.normalized import NormalizedQA
from src.candidates.base import CandidateGenerator
from src.reasoning.semantic_types import SemanticType
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.canonicalization import deduplicate_candidates, are_actions_equivalent, canonicalize_action

class ActionOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 40) -> Tuple[List[str], List[int], str]:
        evaluator = context["evaluator"]
        sg = evaluator.get_scenegraph(record.video_id, getattr(record.source_metadata, "split", "train") if hasattr(record, "source_metadata") else "train")
        if not sg:
            raise ValueError("missing_evidence")
            
        true_ans = str(record.source_answer).lower().strip()
        ast = parse_agqa_program(record.source_program)
        engine = SemanticEngine(sg)
        
        result = engine.evaluate(ast)
        
        true_actions = set()
        if result.type == SemanticType.ACTION_SET:
            true_actions = result.value
        elif result.type == SemanticType.ACTION:
            true_actions.add(result.value)
        else:
            raise ValueError("source_evidence_mismatch")
            
        all_actions = set(a.phrase for a in sg.actions.values())
        
        all_canon = set(deduplicate_candidates(list(all_actions), "action"))
        true_canon = set(canonicalize_action(a) for a in true_actions)
        
        if canonicalize_action(true_ans) not in true_canon:
            raise ValueError("source_evidence_mismatch")
            
        options = []
        labels = []
        
        options.append(true_ans)
        labels.append(1)
        
        false_pool = sorted(list(all_canon - true_canon))
        for opt in false_pool:
            if not are_actions_equivalent(opt, true_ans):
                options.append(opt)
                labels.append(0)
            if len(options) >= max_k:
                break
                
        if len(options) < 2:
            raise ValueError("insufficient_valid_candidates")
            
        return options, labels, "TRUE"
