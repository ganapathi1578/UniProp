from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass
class NormalizedQA:
    source_dataset: str
    source_question_id: str
    video_id: str
    query: str
    source_answer: str
    answer_type: str
    semantic_type: str
    structural_type: str
    reasoning_type: str
    source_program: Optional[str] = None
    scenegraph_reference: Optional[Any] = None
    source_metadata: Optional[Dict[str, Any]] = None
