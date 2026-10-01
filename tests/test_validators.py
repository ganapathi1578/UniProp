import pytest
from src.schema import PropositionGroup, Provenance, Reasoning, Proposition, PropositionSemantics
from src.validation.validators import validate_example

def create_mock_group(task_type="binary", props_info=None):
    if props_info is None:
        props_info = [("Yes", "TRUE", 1), ("No", "FALSE", 0)]
    
    propositions = []
    for i, (text, ts, lbl) in enumerate(props_info):
        propositions.append(Proposition(
            text=text,
            truth_state=ts,
            label=lbl,
            semantics=PropositionSemantics(canonical_type="object_existence", subject=f"obj{i}")
        ))
        
    return PropositionGroup(
        example_id="1",
        source="test",
        split="train",
        media_id="v1",
        query_text="Is this valid?",
        query_template_id=None,
        propositions=propositions,
        task_type=task_type,
        num_propositions=len(propositions),
        reasoning=Reasoning(type="test", complexity=1, family="object"),
        provenance=Provenance(source_question_id="q1", source_question_text="q", source_program="p", source_answer="a", scene_graph_id="s1", generation_rule="r", generation_seed=42)
    )

def test_validator_valid():
    group = create_mock_group(task_type="binary")
    is_valid, errs = validate_example(group)
    assert is_valid
    assert len(errs) == 0

def test_validator_invalid_options():
    group = create_mock_group(task_type="binary", props_info=[("Yes", "TRUE", 1)])
    is_valid, errs = validate_example(group)
    assert not is_valid
    assert any("Proposition count" in e for e in errs)

def test_validator_invalid_labels():
    group = create_mock_group(task_type="single_choice", props_info=[("Yes", "TRUE", 1), ("No", "TRUE", 1)])
    is_valid, errs = validate_example(group)
    assert not is_valid
    assert any("exactly 1 correct" in e for e in errs)
