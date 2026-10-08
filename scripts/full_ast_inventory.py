import os
import json
from collections import Counter, defaultdict
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
    
    op_counts = Counter()
    root_counts = Counter()
    nested_counts = Counter()
    parent_ops = defaultdict(Counter)
    sample_programs = defaultdict(list)
    
    def extract_ops(ast, parent=None, is_root=False):
        if isinstance(ast, tuple):
            op = ast[0]
            op_counts[op] += 1
            if is_root:
                root_counts[op] += 1
            else:
                nested_counts[op] += 1
                if parent:
                    parent_ops[op][parent] += 1
            
            # Keep a few samples
            if len(sample_programs[op]) < 3:
                sample_programs[op].append(prog_str)
                
            for arg in ast[1]:
                extract_ops(arg, parent=op, is_root=False)
        elif isinstance(ast, list):
            for item in ast:
                extract_ops(item, parent=parent, is_root=False)
                
    for i, (q_id, q_data) in enumerate(data.items()):
        prog_str = q_data.get("program", "")
        if not prog_str:
            continue
            
        try:
            ast = parse_agqa_program(prog_str)
            if isinstance(ast, tuple):
                extract_ops(ast, parent=None, is_root=True)
        except Exception as e:
            pass
            
        if (i+1) % 200000 == 0:
            print(f"Processed {i+1} records...")
            
    print("\n--- FULL AST OPERATION INVENTORY ---")
    print(f"{'Operation':<20} | {'Total':<10} | {'Root':<10} | {'Nested':<10} | {'Top Parents'}")
    print("-" * 90)
    for op, total in op_counts.most_common():
        root = root_counts[op]
        nested = nested_counts[op]
        tops = parent_ops[op].most_common(3)
        tops_str = ", ".join([f"{p}({c})" for p, c in tops])
        print(f"{op:<20} | {total:<10} | {root:<10} | {nested:<10} | {tops_str}")
        
    print("\n--- SAMPLE PROGRAMS ---")
    for op in ["Choose", "Equals", "ToAction", "Compare", "Superlative", "AND", "XOR", "Exists", "Query"]:
        print(f"\n[{op}]")
        for s in sample_programs[op]:
            print(f"  {s}")

if __name__ == "__main__":
    scan_dataset()
