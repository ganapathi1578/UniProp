import os
import random
import tempfile
import pytest
import pyarrow.parquet as pq

from tests.fixtures.mock_scenegraph import create_synthetic_scenegraph, create_synthetic_pools
from src.generators.scene_graph import ObjectExistenceGenerator, RelationVerificationGenerator
from src.generators.temporal import ActionTemporalGenerator
from src.generators.compositional import CompositionalGenerator
from src.generators.grounding import GroundingGenerator
from src.validation.validators import validate_example
from src.storage.parquet_writer import write_shard

@pytest.fixture
def sg_env():
    sg = create_synthetic_scenegraph()
    pools = create_synthetic_pools(sg)
    rng = random.Random(42)
    return sg, pools, rng

def test_object_existence_generator(sg_env):
    sg, pools, rng = sg_env
    gen = ObjectExistenceGenerator()
    
    generated = list(gen.generate(qid="q1", qdata={}, sg=sg, split="train", rng=rng, pools=pools))
    assert len(generated) >= 1
    
    for group in generated:
        assert group.media_id == sg.video_id
        assert group.task_type in ["binary", "single_choice", "multi_label", "three_way"]
        assert len(group.propositions) == group.num_propositions
        assert len(group.propositions) >= 2
        
        # Verify valid proposition strings
        for prop in group.propositions:
            assert isinstance(prop.text, str) and len(prop.text) > 0
            assert prop.label in (0, 1)
            assert prop.truth_state in ("TRUE", "FALSE", "UNKNOWN")
            
        # Quality gates validation pass
        is_valid, errors = validate_example(group)
        assert is_valid, f"Validation failed: {errors}"

def test_relation_verification_generator(sg_env):
    sg, pools, rng = sg_env
    gen = RelationVerificationGenerator()
    
    generated = list(gen.generate(qid="q2", qdata={}, sg=sg, split="train", rng=rng, pools=pools))
    assert len(generated) >= 1
    
    for group in generated:
        assert group.media_id == sg.video_id
        assert group.task_type in ["binary", "single_choice", "multi_label"]
        
        labels = [p.label for p in group.propositions]
        if group.task_type == "single_choice":
            assert sum(labels) == 1
        elif group.task_type == "binary":
            assert sum(labels) == 1
            assert len(labels) == 2
            
        is_valid, errors = validate_example(group)
        assert is_valid, f"Validation failed: {errors}"

def test_action_temporal_generator(sg_env):
    sg, pools, rng = sg_env
    gen = ActionTemporalGenerator()
    
    generated = list(gen.generate(qid="q3", qdata={}, sg=sg, split="train", rng=rng, pools=pools))
    assert len(generated) >= 1
    
    for group in generated:
        assert group.media_id == sg.video_id
        assert group.reasoning.family in ("temporal", "action")
        
        is_valid, errors = validate_example(group)
        assert is_valid, f"Validation failed: {errors}"

def test_compositional_generator(sg_env):
    sg, pools, rng = sg_env
    gen = CompositionalGenerator()
    
    generated = list(gen.generate(qid="q4", qdata={}, sg=sg, split="train", rng=rng, pools=pools))
    assert len(generated) >= 1
    
    for group in generated:
        assert group.reasoning.family == "compositional"
        is_valid, errors = validate_example(group)
        assert is_valid, f"Validation failed: {errors}"

def test_grounding_generator(sg_env):
    sg, pools, rng = sg_env
    gen = GroundingGenerator()
    
    generated = list(gen.generate(qid="q5", qdata={}, sg=sg, split="train", rng=rng, pools=pools))
    assert len(generated) >= 1
    
    for group in generated:
        assert group.reasoning.family == "grounding"
        is_valid, errors = validate_example(group)
        assert is_valid, f"Validation failed: {errors}"

def test_parquet_storage_roundtrip(sg_env):
    sg, pools, rng = sg_env
    gen = ObjectExistenceGenerator()
    
    groups = list(gen.generate(qid="q6", qdata={}, sg=sg, split="train", rng=rng, pools=pools))
    assert len(groups) > 0
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_path = os.path.join(tmp_dir, "test_shard.parquet")
        write_shard(groups, parquet_path)
        assert os.path.exists(parquet_path)
        
        # Readback verification
        table = pq.read_table(parquet_path)
        assert table.num_rows == len(groups)
        
        col_names = table.column_names
        expected_cols = [
            "example_id", "split", "video_id", "query", "options",
            "labels", "truth_state", "option_count", "task_type",
            "reasoning_family", "generator_family"
        ]
        for col in expected_cols:
            assert col in col_names, f"Missing column {col} in parquet output"
            
        row_0 = table.to_pylist()[0]
        assert row_0["video_id"] == sg.video_id
        assert len(row_0["options"]) == len(row_0["labels"])
        assert row_0["option_count"] == len(row_0["options"])
