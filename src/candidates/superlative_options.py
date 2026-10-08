from typing import List, Tuple
from src.datasets.normalized import NormalizedQA
from src.candidates.base import CandidateGenerator
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.semantic_types import SemanticType
from src.reasoning.canonicalization import deduplicate_candidates, canonicalize_action, canonicalize_object

class SuperlativeOptionizer(CandidateGenerator):
    def generate(self, record: NormalizedQA, context: dict, max_k: int = 40) -> Tuple[List[str], List[int], str]:
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("missing_evaluator")
            
        sg = evaluator.get_scenegraph(record.video_id, getattr(record.source_metadata, "split", "train") if hasattr(record, "source_metadata") else "train")
        if not sg:
            raise ValueError("missing_evidence")
            
        ast = parse_agqa_program(record.source_program)
        if ast[0] != "Superlative" or len(ast[1]) != 3:
            raise ValueError("evaluator_unsupported:Superlative")
            
        engine = SemanticEngine(sg)
        res = engine.evaluate(ast)
        
        if res.type != SemanticType.UNKNOWN:
            raise ValueError("source_evidence_mismatch")
            
        true_ans = str(record.source_answer).strip().lower()
        
        # We need the candidate pools to construct options
        collections = ast[1][1]
        candidates = []
        if isinstance(collections, list):
            for c_ast in collections:
                c_val = engine.evaluate(c_ast)
                if c_val.type in (SemanticType.ACTION_SET, SemanticType.OBJECT_SET):
                    candidates.extend(list(c_val.value))
                else:
                    candidates.append(c_val.value)
        else:
            c_val = engine.evaluate(collections)
            if c_val.type in (SemanticType.ACTION_SET, SemanticType.OBJECT_SET):
                candidates.extend(list(c_val.value))
            else:
                candidates.append(c_val.value)
                
        # Actually AGQA Superlative usually compares actions.
        # Let's just use action deduplication.
        cands_canon = deduplicate_candidates(candidates, "action")
        
        # Superlative options should be the list of all candidates!
        options = cands_canon
        if len(options) < 2:
            raise ValueError("insufficient_valid_candidates")
            
        if canonicalize_action(true_ans) != canonicalize_action(res.value):
            raise ValueError("source_evidence_mismatch")
            
        labels = [1 if canonicalize_action(o) == canonicalize_action(true_ans) else 0 for o in options]
        
        return options, labels, "TRUE"
