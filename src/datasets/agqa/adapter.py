import json
from typing import Iterator, Dict
from src.datasets.base import BaseDatasetAdapter
from src.datasets.normalized import NormalizedQA

class AGQABalancedAdapter(BaseDatasetAdapter):
    def __init__(self, config: dict):
        self.config = config
        self.balanced_path = config.get("balanced_path", "data/dataset/agqa_balanced/train_balanced.txt")
        self.scenegraph_path = config.get("scenegraph_path", "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl")
        self._qa_data = None
        self._sg_data = None

    def _load_data(self):
        if self._qa_data is None:
            # We assume it's a JSON file mapping QID to data
            with open(self.balanced_path, 'r', encoding='utf-8') as f:
                self._qa_data = json.load(f)
            # In a real implementation we would also load the scene graph via pickle
            # but to save memory during testing we might mock or load selectively.

    def load_records(self) -> Iterator[NormalizedQA]:
        self._load_data()
        
        for qid, qdata in self._qa_data.items():
            video_id = qdata.get("video_id")
            query = qdata.get("question")
            answer = qdata.get("answer")
            ans_type = qdata.get("ans_type", "open")
            semantic = qdata.get("semantic", "")
            structural = qdata.get("structural", "")
            program = qdata.get("program", "")
            
            # Identify abstract answer type (e.g., binary, count, object, action, temporal, etc.)
            ans_lower = str(answer).lower().strip()
            
            if ans_lower in ["yes", "no"]:
                a_type = "binary"
            elif ans_lower in ["before", "after"]:
                a_type = "temporal"
            elif structural == "count" or str(answer).isdigit():
                a_type = "count"
            elif "compare" in structural.lower():
                a_type = "comparison"
            elif "superlative" in qdata.get("global", []):
                a_type = "superlative"
            elif semantic == "action":
                a_type = "action"
            elif semantic == "object":
                a_type = "object"
            else:
                a_type = "open"

            yield NormalizedQA(
                source_dataset="agqa_balanced",
                source_question_id=qid,
                video_id=video_id,
                query=query,
                source_answer=answer,
                answer_type=a_type,
                semantic_type=semantic,
                structural_type=structural,
                reasoning_type="-".join(qdata.get("global", [])),
                source_program=program,
                scenegraph_reference=None, # Loaded from SG dataset during optionization
                source_metadata=qdata
            )
