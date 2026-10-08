from typing import List, Tuple
from src.datasets.normalized import NormalizedQA
from src.candidates.base import CandidateGenerator
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.semantic_types import SemanticType, SemanticValue
from src.candidates.verbalizer import PropositionVerbalizer
from src.reasoning.answer_spec import AnswerSpecification

class UniversalCandidateGenerator(CandidateGenerator):
    def __init__(self):
        self.verbalizer = PropositionVerbalizer()

    def generate(self, record: NormalizedQA, context: dict, max_k: int = 40) -> Tuple[List[str], List[int], str]:
        evaluator = context.get("evaluator")
        if not evaluator:
            raise ValueError("missing_evaluator")
            
        sg = evaluator.get_scenegraph(record.video_id, getattr(record.source_metadata, "split", "train") if hasattr(record, "source_metadata") else "train")
        if not sg:
            raise ValueError("missing_evidence")
            
        engine = SemanticEngine(sg)
        ast = parse_agqa_program(record.source_program)
        
        # 1. Semantic Execution
        res = engine.evaluate(ast)
        
        # 2. Answer Specification & Candidate Values
        spec = self._build_specification(ast, res, record, engine)
        
        # 3. Proposition Verbalizer & Label Assignment
        options = []
        labels = []
        
        for cand, truth_val in spec.candidate_evaluation.items():
            prop = self.verbalizer.verbalize(
                ast=ast,
                query=record.query, 
                answer_kind=spec.answer_kind, 
                candidate=cand, 
                verbalization_info=spec.verbalization_info
            )
            options.append(prop)
            labels.append(truth_val)
            
        # Optional validation
        if not options:
            raise ValueError("insufficient_valid_candidates")
            
        truth_state = "TRUE" if any(labels) else "FALSE"
        if res.type == SemanticType.UNKNOWN:
            truth_state = "UNKNOWN"
            
        # Verify source answer
        ans_lower = str(record.source_answer).lower().strip()
        
        # Only verify if we have a true label
        if truth_state == "TRUE" and not any(ans_lower == str(c).lower() for c, v in spec.candidate_evaluation.items() if v == 1):
            if spec.answer_kind == "boolean":
                is_yes_ans = (ans_lower == "yes")
                try:
                    is_yes_pred = (labels[options.index(self.verbalizer.verbalize(ast, record.query, spec.answer_kind, "yes", spec.verbalization_info))] == 1)
                except ValueError:
                    is_yes_pred = False
                if is_yes_ans != is_yes_pred:
                    raise ValueError("source_evidence_mismatch")
            else:
                raise ValueError("source_evidence_mismatch")
            
        return options, labels, truth_state

    def _build_specification(self, ast, res: SemanticValue, record: NormalizedQA, engine: SemanticEngine) -> AnswerSpecification:
        spec = AnswerSpecification(answer_kind=record.answer_type)
        ans_lower = str(record.source_answer).lower().strip()
        
        # Resolve sets
        res_values = res.value if isinstance(res.value, (set, list, tuple)) else [res.value]
        
        # BOOLEAN Domain
        if res.type == SemanticType.BOOLEAN:
            spec.answer_kind = "boolean"
            is_yes = (res.value is True)
            if res.type == SemanticType.UNKNOWN:
                is_yes = (ans_lower == "yes")
            spec.candidate_evaluation["yes"] = 1 if is_yes else 0
            spec.candidate_evaluation["no"] = 0 if is_yes else 1
            return spec
            
        # CHOOSE Operation
        if ast[0] == "Choose":
            try:
                c1 = engine.evaluate(ast[1][0], {}).value if isinstance(ast[1][0], tuple) else ast[1][0]
                c2 = engine.evaluate(ast[1][1], {}).value if isinstance(ast[1][1], tuple) else ast[1][1]
            except:
                c1 = str(ast[1][0])
                c2 = str(ast[1][1])
            
            c1_str = str(c1).lower().strip()
            c2_str = str(c2).lower().strip()
            
            if c1_str in ["before", "after"]: spec.answer_kind = "temporal"
            else: spec.answer_kind = "object"
            
            spec.candidate_evaluation[c1_str] = 1 if res.value == c1 else 0
            spec.candidate_evaluation[c2_str] = 1 if res.value == c2 else 0
            return spec
            
        # COMPARE Operation
        if ast[0] == "Compare":
            opts = ast[1]
            if isinstance(opts, list) and len(opts) >= 2:
                c1_str = str(opts[0]).lower().strip()
                c2_str = str(opts[1]).lower().strip()
                
                if c1_str in ["longer", "shorter"]: spec.answer_kind = "comparison"
                elif c1_str in ["before", "after"]: spec.answer_kind = "temporal"
                else: spec.answer_kind = "object"
                
                if res.type == SemanticType.BOOLEAN:
                    spec.candidate_evaluation[c1_str] = 1 if res.value is True else 0
                    spec.candidate_evaluation[c2_str] = 1 if res.value is False else 0
                else:
                    spec.candidate_evaluation[c1_str] = 0
                    spec.candidate_evaluation[c2_str] = 0
                return spec
                
        # OBJECT Domain
        if res.type == SemanticType.OBJECT or res.type == SemanticType.OBJECT_SET:
            spec.answer_kind = "object"
            objs = set()
            for frame in engine.sg.frames.values():
                for obj in frame.objects.values():
                    objs.add(obj.name)
            valid_objs = list(objs)[:40]
            if ans_lower not in valid_objs and res.value and not isinstance(res.value, (set, list)):
                valid_objs.append(str(res.value).lower().strip())
            for o in valid_objs:
                spec.candidate_evaluation[o] = 1 if o in res_values else 0
            return spec
            
        # ACTION Domain
        if res.type == SemanticType.ACTION or res.type == SemanticType.ACTION_SET:
            spec.answer_kind = "action"
            actions = set(a.phrase for a in engine.sg.actions.values())
            valid_acts = list(actions)[:40]
            if ans_lower not in valid_acts and res.value and not isinstance(res.value, (set, list)):
                valid_acts.append(str(res.value).lower().strip())
            for a in valid_acts:
                spec.candidate_evaluation[a] = 1 if a in res_values else 0
            return spec

        # Fallbacks
        if record.answer_type == "count":
            true_val = res.value if (res.type != SemanticType.UNKNOWN and res.value is not None) else ans_lower
            try: 
                true_val_int = int(true_val)
                valid_counts = [str(max(0, true_val_int - 1)), str(true_val_int), str(true_val_int + 1), str(true_val_int + 2)]
                valid_counts = list(dict.fromkeys(valid_counts))
            except:
                valid_counts = [str(true_val), "1", "2", "3"]
            for c in valid_counts:
                spec.candidate_evaluation[c] = 1 if str(c) == str(true_val) else 0
            return spec
            
        if record.answer_type == "temporal":
            for o in ["before", "after"]:
                spec.candidate_evaluation[o] = 1 if o in res_values else 0
            return spec

        # Last Resort
        spec.candidate_evaluation[ans_lower] = 1
        return spec
