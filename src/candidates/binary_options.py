from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA

class BinaryOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 2) -> Tuple[List[str], List[int], str]:
        options = ["Yes", "No"]
        
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("missing_evaluator")
            
        sg = evaluator.get_scenegraph(record.video_id, getattr(record.source_metadata, "split", "train") if hasattr(record, "source_metadata") else "train")
        if not sg:
            raise ValueError("missing_evidence")
            
        from src.reasoning.semantic_engine import SemanticEngine
        from src.reasoning.program_parser import parse_agqa_program
        from src.reasoning.semantic_types import SemanticType

        try:
            ast = parse_agqa_program(record.source_program)
            engine = SemanticEngine(sg)
            res = engine.evaluate(ast)
        except Exception as e:
            # Let the semantic engine's exceptions bubble up (e.g. EvaluatorUnsupportedError)
            raise

        ans = str(record.source_answer).lower().strip()
        
        if res.type == SemanticType.UNKNOWN:
            truth_state = "UNKNOWN"
            is_yes = (ans == "yes")
        elif res.type == SemanticType.BOOLEAN:
            truth_state = "TRUE" if res.value else "FALSE"
            is_yes = res.value
            if (is_yes and ans != "yes") or (not is_yes and ans != "no"):
                raise ValueError("source_evidence_mismatch")
        else:
            raise ValueError(f"evaluator_unsupported:non_boolean_result({res.type})")
            
        labels = [1, 0] if is_yes else [0, 1]
        return options, labels, truth_state
