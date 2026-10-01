"""Data schemas for UniProp dataset."""
import dataclasses
from typing import List, Dict, Tuple, Optional, Any

@dataclasses.dataclass
class PropositionSemantics:
    canonical_type: str  # object_existence, relation, action, temporal, grounding
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    polarity: str = "positive" # "positive" or "negative"
    temporal_relation: Optional[str] = None
    event_a: Optional[str] = None
    event_b: Optional[str] = None


@dataclasses.dataclass
class Proposition:
    text: str
    truth_state: str # "TRUE", "FALSE", "UNKNOWN"
    label: Optional[int] # 1 for True, 0 for False. None for UNKNOWN unless three_way task
    semantics: PropositionSemantics

@dataclasses.dataclass
class Grounding:
    object_ids: List[str] = dataclasses.field(default_factory=list)
    boxes: Dict[str, Tuple[float, float, float, float]] = dataclasses.field(default_factory=dict)
    frame_ids: List[str] = dataclasses.field(default_factory=list)
    target_object_id: Optional[str] = None
    target_object_class: Optional[str] = None
    target_bbox: Optional[Tuple[float, float, float, float]] = None

@dataclasses.dataclass
class Reasoning:
    type: str
    complexity: int
    family: str = "object" # object, attribute, spatial, attention, contact, action, temporal, compositional, grounding
    hops: int = 1

@dataclasses.dataclass
class Provenance:
    source_question_id: Optional[str]
    source_question_text: Optional[str]
    source_program: Optional[str]
    source_answer: Optional[str]
    scene_graph_id: Optional[str]
    generation_rule: str
    generation_seed: int
    generator_family: str = "scene_graph"
    evidence_object_ids: List[str] = dataclasses.field(default_factory=list)
    evidence_relation_ids: List[str] = dataclasses.field(default_factory=list)
    frame_ids: List[str] = dataclasses.field(default_factory=list)
    temporal_chain: Optional[List[dict]] = None
    source_global: Optional[str] = None
    source_local: Optional[str] = None
    source_semantic: Optional[str] = None
    source_structural: Optional[str] = None
    source_sg_grounding: Optional[str] = None

@dataclasses.dataclass
class PropositionGroup:
    example_id: str
    source: str
    split: str
    media_id: str
    query_text: Optional[str]
    query_template_id: Optional[str]
    propositions: List[Proposition]
    task_type: str
    num_propositions: int
    reasoning: Reasoning
    provenance: Provenance
    grounding: Optional[Grounding] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source": self.source,
            "split": self.split,
            "media_id": self.media_id,
            "query_text": self.query_text,
            "query_template_id": self.query_template_id,
            "propositions": [dataclasses.asdict(p) for p in self.propositions],
            "task_type": self.task_type,
            "num_propositions": self.num_propositions,
            "reasoning": dataclasses.asdict(self.reasoning),
            "provenance": dataclasses.asdict(self.provenance),
            "grounding": dataclasses.asdict(self.grounding) if self.grounding else None
        }
