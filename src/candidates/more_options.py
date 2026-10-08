from typing import List, Tuple
from src.candidates.base import CandidateGenerator
from src.datasets.normalized import NormalizedQA
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.semantic_types import SemanticType

class TemporalOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int) -> Tuple[List[str], List[int], str]:
        ans = str(record.source_answer).strip().lower()
        if ans not in ["before", "after"]:
            raise ValueError(f"UNSUPPORTED_TEMPORAL_ANSWER: {ans}")
            
        options = ["before", "after"]
        
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("MISSING_EVALUATOR")
            
        sg = evaluator.get_scenegraph(record.video_id, getattr(record.source_metadata, "split", "train") if hasattr(record, "source_metadata") else "train")
        if not sg:
            raise ValueError("missing_evidence")
            
        # Program: Compare([before, after], condition_with_temporal_tag)
        ast = parse_agqa_program(record.source_program)
        
        if ast[0] != "Compare" or len(ast[1]) != 2:
            raise ValueError("evaluator_unsupported:Compare")
            
        inner_ast = ast[1][1]
        
        # Replace 'temporal tag' with 'before' and 'after' in inner_ast
        def replace_tag(node, tag):
            if isinstance(node, tuple):
                return (node[0], [replace_tag(arg, tag) for arg in node[1]])
            elif isinstance(node, list):
                return [replace_tag(item, tag) for item in node]
            elif node == "temporal tag":
                return tag
            return node
            
        ast_before = replace_tag(inner_ast, "before")
        ast_after = replace_tag(inner_ast, "after")
        
        engine = SemanticEngine(sg)
        
        res_before = engine.evaluate(ast_before)
        res_after = engine.evaluate(ast_after)
        
        truth_before = "TRUE" if res_before.type == SemanticType.BOOLEAN and res_before.value else "FALSE"
        truth_after = "TRUE" if res_after.type == SemanticType.BOOLEAN and res_after.value else "FALSE"
        
        if truth_before == "TRUE" and truth_after == "TRUE":
            raise ValueError("contradictory_temporal_evidence")
            
        if truth_before == "FALSE" and truth_after == "FALSE":
            raise ValueError("source_evidence_mismatch")
            
        if truth_before == "TRUE":
            if ans != "before":
                raise ValueError("source_evidence_mismatch")
            labels = [1, 0]
        else:
            if ans != "after":
                raise ValueError("source_evidence_mismatch")
            labels = [0, 1]
            
        return options, labels, "TRUE"
