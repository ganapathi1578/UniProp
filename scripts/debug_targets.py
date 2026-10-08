import sys
sys.path.insert(0, "c:/Users/GANAPATHI/Desktop/NIT/Research/UniProp")

import json
from src.datasets.agqa.adapter import AGQABalancedAdapter
from src.candidates.registry import candidate_registry
from src.reasoning.evidence import TruthEvaluator
import traceback

with open("data/dataset/agqa_balanced/train_balanced.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

adapter = AGQABalancedAdapter({"source": {"qa_path": "data/dataset/agqa_balanced/train_balanced.txt", "scenegraph_path": "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl"}})
evaluator = TruthEvaluator("data/dataset/agqa_scene_graphs")

target_ids = {
    "MJO7C-129", "MJO7C-114",
    "S6MPZ-4804", "S6MPZ-4805", "S6MPZ-4806", "S6MPZ-4807", "S6MPZ-4808",
    "S6MPZ-4809", "S6MPZ-4810", "S6MPZ-4811", "S6MPZ-4812", "S6MPZ-4813",
    "S6MPZ-4814", "S6MPZ-4815", "S6MPZ-4816", "S6MPZ-4817", "S6MPZ-4821",
    "S6MPZ-4830", "S6MPZ-4831", "S6MPZ-4832", "S6MPZ-4833", "S6MPZ-4834",
    "S6MPZ-4840", "S6MPZ-4841", "S6MPZ-4842", "S6MPZ-4843", "S6MPZ-4844",
    "S6MPZ-4847", "S6MPZ-4853", "S6MPZ-4854", "S6MPZ-4856", "S6MPZ-4857",
    "S6MPZ-4858", "S6MPZ-4860", "S6MPZ-4861", "S6MPZ-4862",
    "MCQO5-2624", "MCQO5-6005", "MCQO5-2880", "MCQO5-6154", "7HVU8-1612"
}

for record in adapter.load_records():
    if record.source_question_id in target_ids:
        print("="*60)
        print(f"SOURCE QUESTION: {record.query}")
        print(f"SOURCE ANSWER: {record.source_answer}")
        print(f"SEMANTIC: {record.semantic_type}")
        print(f"STRUCTURAL: {record.structural_type}")
        print(f"PROGRAM: {record.source_program}")
        print(f"VIDEO ID: {record.video_id}")
        generator = candidate_registry.get_generator(record.answer_type)
        if generator:
            try:
                gen = generator()
                gen.generate(record, {"evaluator": evaluator}, 40)
                print("-> SUCCESS")
            except Exception as e:
                print(f"-> FAILURE: {repr(e)}")
                traceback.print_exc()
