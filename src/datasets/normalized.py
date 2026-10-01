from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class NormalizedQA:
    question_id: str
    question_text: str
    answer: str
    candidates: List[str]
    metadata: Dict[str, Any]
