import os
import json
from collections import Counter
from src.reasoning.program_parser import parse_agqa_program

def scan_dataset():
    path = "data/dataset/agqa_balanced/train_balanced.txt"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    print("Loading JSON...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records.")
    
    domain_counter = Counter()
    op_counter = Counter()
    root_op_counter = Counter()
    
    count_programs = []
    three_way_programs = []
    comparison_programs = []
    superlative_programs = []
    temporal_programs = []
    
    def extract_ops(ast):
        if isinstance(ast, tuple):
            op = ast[0]
            op_counter[op] += 1
            for arg in ast[1]:
                extract_ops(arg)
        elif isinstance(ast, list):
            for item in ast:
                extract_ops(item)
                
    for i, (q_id, q_data) in enumerate(data.items()):
        prog_str = q_data.get("program", "")
        if not prog_str:
            continue
            
        try:
            ast = parse_agqa_program(prog_str)
            if isinstance(ast, tuple):
                root_op = ast[0]
                root_op_counter[root_op] += 1
                extract_ops(ast)
                
                # Semantic domain determination:
                # We can look at the semantic string or infer from root op.
                # The prompt asks to "identify answer domain from program semantics"
                semantic = q_data.get("semantic", "")
                ans = str(q_data.get("answer", "")).lower().strip()
                
                # Map semantic / root to domains
                domain = "OTHER"
                if "compare" in semantic or root_op == "Compare":
                    domain = "COMPARISON"
                    if len(comparison_programs) < 5:
                        comparison_programs.append((q_id, prog_str, semantic, ans))
                elif "superlative" in semantic or root_op == "Superlative":
                    domain = "SUPERLATIVE"
                    if len(superlative_programs) < 5:
                        superlative_programs.append((q_id, prog_str, semantic, ans))
                elif "temporal" in semantic and root_op == "Compare" and ans in ["before", "after"]:
                    domain = "TEMPORAL"
                elif root_op == "Exists" or root_op in ("And", "Or", "Xor", "Not") or ans in ["yes", "no"]:
                    domain = "BINARY"
                elif root_op == "Query":
                    # Query(action, ...) or Query(object, ...)
                    if "action" in str(ast[1]):
                        domain = "ACTION"
                    else:
                        domain = "OBJECT"
                        
                domain_counter[domain] += 1
                
                if "count" in semantic or "count" in prog_str.lower():
                    count_programs.append((q_id, prog_str, semantic, ans))
                if ans in ["yes", "no", "unknown"] and root_op not in ("Exists", "And", "Or", "Xor", "Not") and domain != "BINARY":
                    # Potentially three-way?
                    pass
        except Exception as e:
            pass
            
        if (i+1) % 100000 == 0:
            print(f"Processed {i+1} records...")
            
    print("\n--- Domains ---")
    for d, c in domain_counter.most_common():
        print(f"{d}: {c}")
        
    print("\n--- Root Ops ---")
    for r, c in root_op_counter.most_common(10):
        print(f"{r}: {c}")
        
    print("\n--- All Ops ---")
    for o, c in op_counter.most_common(20):
        print(f"{o}: {c}")
        
    print("\n--- Sample Comparison Programs ---")
    for p in comparison_programs:
        print(p)
        
    print("\n--- Sample Superlative Programs ---")
    for p in superlative_programs:
        print(p)
        
    print("\n--- Count Programs ---")
    print(f"Count of count programs: {len(count_programs)}")

scan_dataset()
