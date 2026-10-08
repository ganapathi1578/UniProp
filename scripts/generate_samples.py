import sys
import os
sys.path.insert(0, os.path.abspath("."))
import traceback
from collections import defaultdict
from src.reasoning.evidence import TruthEvaluator
from src.datasets.normalized import NormalizedQA
from src.candidates.universal_generator import UniversalCandidateGenerator
import ijson

def run_sampling():
    print("Loading Truth Evaluator (this takes a minute)...")
    evaluator = TruthEvaluator(scenegraph_dir="data/dataset/agqa_scene_graphs")
    generator = UniversalCandidateGenerator()

    targets = {
        "boolean": 4,
        "object": 4,
        "action": 4,
        "temporal": 4,
        "comparison": 4
    }
    collected = defaultdict(list)

    print("Generating samples (streaming JSON)...")
    with open("data/dataset/agqa_balanced/train_balanced.txt", "rb") as f:
        parser = ijson.kvitems(f, '')
        
        for qid, qdata in parser:
            if all(len(collected[k]) >= v for k, v in targets.items()):
                break

            ans_type = qdata.get("answer_type", "unknown")
            
            prog = qdata.get("program", "")
            if "Superlative" in prog:
                domain_key = "action"
            elif "Compare" in prog:
                if "longer" in prog or "shorter" in prog or "more" in prog or "less" in prog:
                    domain_key = "comparison"
                elif "before" in prog or "after" in prog:
                    domain_key = "temporal"
                else:
                    domain_key = "object"
            else:
                domain_key = ans_type

            if len(collected.get(domain_key, [])) >= targets.get(domain_key, 4):
                continue

            record = NormalizedQA(
                source_dataset="agqa_balanced",
                source_question_id=qid,
                video_id=qdata["video_id"],
                query=qdata["question"],
                semantic_type=qdata.get("semantic", ""),
                structural_type=qdata.get("structural", ""),
                answer_type=ans_type,
                reasoning_type="",
                source_answer=qdata["answer"],
                source_program=qdata["program"]
            )

            try:
                options, labels, truth_state = generator.generate(record, {"evaluator": evaluator}, max_k=4)
                
                if len(collected[domain_key]) < targets.get(domain_key, 4):
                    collected[domain_key].append({
                        "qid": qid,
                        "query": record.query,
                        "program": record.source_program,
                        "truth_state": truth_state,
                        "options": options,
                        "labels": labels
                    })
                    print(f"Collected 1 sample for {domain_key}")
                    
            except Exception as e:
                pass

    with open("SAMPLE_OUTPUT.md", "w", encoding="utf-8") as f:
        f.write("# UniProp Optionization Samples\n\n")
        for domain, samples in collected.items():
            f.write(f"## Domain: {domain.upper()}\n\n")
            for i, s in enumerate(samples):
                f.write(f"**Sample {i+1}** ({s['qid']})\n")
                f.write(f"- **Query**: {s['query']}\n")
                f.write(f"- **Program**: `{s['program']}`\n")
                f.write(f"- **Truth State**: {s['truth_state']}\n")
                f.write("- **Options**:\n")
                for opt, lbl in zip(s['options'], s['labels']):
                    mark = "✅" if lbl == 1 else "❌"
                    f.write(f"  - [{mark}] {opt}\n")
                f.write("\n")
                
    print("Done! Wrote SAMPLE_OUTPUT.md")

if __name__ == "__main__":
    run_sampling()
