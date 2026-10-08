import pytest
import os
from src.reasoning.evidence import TruthEvaluator
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.program_parser import parse_agqa_program

@pytest.fixture(scope="module")
def evaluator():
    sg_dir = "data/dataset/agqa_scene_graphs"
    if not os.path.exists(os.path.join(sg_dir, "AGQA_train_stsgs.pkl")):
        pytest.skip("Real dataset not found")
    return TruthEvaluator(sg_dir)

def test_s6mpz_object_access(evaluator):
    # Test an S6MPZ object example that previously accessed sg.objects
    # We just need to load S6MPZ and parse a query
    sg = evaluator.get_scenegraph("S6MPZ", "train")
    assert sg is not None
    engine = SemanticEngine(sg)
    
    # Query(object, Filter(video, [objects])) -> should return an OBJECT_SET
    ast = parse_agqa_program("Query(object, Filter(video, [objects]))")
    res = engine.evaluate(ast)
    assert res.type.value == "OBJECT_SET"
    assert len(res.value) > 0 # should contain some objects

def test_mjo7c_filter_relations(evaluator):
    # Test MJO7C Filter([relations]) example
    sg = evaluator.get_scenegraph("MJO7C", "train")
    assert sg is not None
    engine = SemanticEngine(sg)
    
    # Filter(frame, [relations, behind, objects])
    ast = parse_agqa_program("Filter(frame, [relations, behind, objects])")
    res = engine.evaluate(ast)
    assert res.type.value == "OBJECT_SET"

def test_s6mpz_none_value(evaluator):
    # S6MPZ example containing the None value
    sg = evaluator.get_scenegraph("S6MPZ", "train")
    assert sg is not None
    engine = SemanticEngine(sg)
    
    # Something that used to return None.
    # Like Query(start, action_that_does_not_exist)
    ast = parse_agqa_program("Query(start, Filter(video, [actions, flying]))")
    import src.reasoning.semantic_types as st
    with pytest.raises(st.MissingEvidenceError):
        engine.evaluate(ast)

def test_nested_filter_iterate_query(evaluator):
    sg = evaluator.get_scenegraph("MJO7C", "train")
    engine = SemanticEngine(sg)
    import src.reasoning.semantic_types as st
    ast = parse_agqa_program("Query(action, Iterate(Localize(after, Filter(video, [actions, standing up])), Filter(video, [actions])))")
    try:
        res = engine.evaluate(ast)
        assert res.type.value in ["ACTION_SET", "ACTION"]
    except st.MissingEvidenceError:
        pass
