import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.reasoning.evidence import TruthEvaluator
from src.datasets.normalized import NormalizedQA
from src.candidates.universal_generator import UniversalCandidateGenerator
import ijson

def find_specific_samples():
    print("Loading Truth Evaluator...")
    evaluator = TruthEvaluator(scenegraph_dir="data/dataset/agqa_scene_graphs")
    generator = UniversalCandidateGenerator()

    found_boolean = False
    found_object = False
    found_comparison = False

    with open("data/dataset/agqa_balanced/train_balanced.txt", "rb") as f:
        parser = ijson.kvitems(f, '')
        
        for qid, qdata in parser:
            if found_boolean and found_object and found_comparison:
                break
                
            prog = qdata.get("program", "")
            
            is_bool = "Exists" in prog or "Verify" in prog or "And" in prog or "Xor" in prog
            is_obj = "Query" in prog and ("object" in prog or "class" in prog)
            is_comp = "Compare" in prog and ("longer" in prog or "shorter" in prog)
            
            if is_bool and found_boolean: continue
            if is_obj and found_object: continue
            if is_comp and found_comparison: continue
            if not is_bool and not is_obj and not is_comp: continue
            
            record = NormalizedQA(
                source_dataset="agqa_balanced",
                source_question_id=qid,
                video_id=qdata["video_id"],
                query=qdata["question"],
                semantic_type=qdata.get("semantic", ""),
                structural_type=qdata.get("structural", ""),
                answer_type=qdata.get("answer_type", "unknown"),
                reasoning_type="",
                source_answer=qdata["answer"],
                source_program=qdata["program"]
            )
            
            try:
                options, labels, truth_state = generator.generate(record, {"evaluator": evaluator}, max_k=4)
                
                print(f"--- SAMPLE ---")
                print(f"QID: {qid}")
                print(f"Query: {record.query}")
                print(f"Program: {record.source_program}")
                print(f"Truth State: {truth_state}")
                for opt, lbl in zip(options, labels):
                    mark = "[Y]" if lbl == 1 else "[N]"
                    print(f"  {mark} {opt}")
                print()
                
                if is_bool: found_boolean = True
                if is_obj: found_object = True
                if is_comp: found_comparison = True
                
            except Exception as e:
                pass
                
find_specific_samples()
