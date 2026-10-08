from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

@dataclass
class AnswerSpecification:
    """
    Intermediate representation separating semantic AST execution from 
    proposition verbalization and UniProp output generation.
    """
    answer_kind: str  # e.g., 'object', 'action', 'boolean', 'temporal', 'comparison', 'superlative'
    candidate_values: List[Any] = field(default_factory=list)
    predicate_context: Optional[str] = None
    temporal_constraints: Optional[Dict[str, Any]] = None
    comparison_relation: Optional[str] = None
    superlative_attribute: Optional[str] = None
    evidence_scope: Optional[Dict[str, Any]] = None
    
    # Mapping of candidate values to their semantic truth value (1 for TRUE, 0 for FALSE)
    candidate_evaluation: Dict[Any, int] = field(default_factory=dict)
    
    # Information needed to verbalize this specification into a declarative proposition
    verbalization_info: Dict[str, Any] = field(default_factory=dict)
