import pytest
from src.schema import PropositionExample, Option, Reasoning, Provenance
from src.validation.validators import validate_example

def test_validator_valid():
    ex = PropositionExample(
        example_id="1", source="orig", split="train", media_id="v1",
        query_text="Is it a window?", query_type="verify",
        options=[Option(text="Yes.", label=1), Option(text="No.", label=0)],
        task_type="binary", num_options=2,
        reasoning=Reasoning(type="orig", complexity=1),
        provenance=Provenance("q1", None, "rule1", 0)
    )
    is_valid, errs = validate_example(ex)
    assert is_valid
    assert len(errs) == 0

def test_validator_invalid_options():
    ex = PropositionExample(
        example_id="1", source="orig", split="train", media_id="v1",
        query_text="Is it a window?", query_type="verify",
        options=[Option(text="Yes.", label=1)], # only 1 option
        task_type="binary", num_options=1,
        reasoning=Reasoning(type="orig", complexity=1),
        provenance=Provenance("q1", None, "rule1", 0)
    )
    is_valid, errs = validate_example(ex)
    assert not is_valid
    assert any("Option count" in e for e in errs)

def test_validator_invalid_labels():
    ex = PropositionExample(
        example_id="1", source="orig", split="train", media_id="v1",
        query_text="Is it a window?", query_type="verify",
        options=[Option(text="Yes.", label=1), Option(text="No.", label=1)],
        task_type="binary", num_options=2,
        reasoning=Reasoning(type="orig", complexity=1),
        provenance=Provenance("q1", None, "rule1", 0)
    )
    is_valid, errs = validate_example(ex)
    assert not is_valid
    assert any("exactly 1 correct" in e for e in errs)
