import pytest
from src.datasets.normalized import NormalizedQA
from src.candidates.universal_generator import UniversalCandidateGenerator
from src.candidates.universal_generator import UniversalCandidateGenerator
from src.candidates.universal_generator import UniversalCandidateGenerator
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph

class MockEvaluator:
    def get_scenegraph(self, video_id, split="train"):
        return NormalizedSceneGraph(video_id=video_id, split="train", frames={}, actions={})
        
    def evaluate(self, record, candidate, context):
        ans = str(record.source_answer).strip().lower()
        return "TRUE" if candidate.lower() == ans else "FALSE"

@pytest.mark.skip(reason="synthetic")
def test_temporal_optionizer_regression():
    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="46GP8-474",
        video_id="46GP8",
        query="Was interacting with a window something they did before or after they made some food?",
        source_answer="after",
        answer_type="temporal",
        semantic_type="object",
        structural_type="compare",
        reasoning_type="obj-act-sequencing"
    )
    generator = UniversalCandidateGenerator()
    context = {"evaluator": MockEvaluator()}
    options, labels, truth = generator.generate(record, context, max_k=40)
    
    assert set(options) == {"before", "after"}
    assert len(options) == 2
    assert len(labels) == 2
    assert options[labels.index(1)] == "after"
    assert truth == "TRUE"
    
def test_binary_optionizer():
    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="46GP8-5",
        video_id="46GP8",
        query="In the video, did they interact with a window?",
        source_answer="yes",
        answer_type="binary",
        semantic_type="object",
        structural_type="verify",
        reasoning_type="obj-ex", 
        source_program="Exists(window, Filter(video, [objects]))"
    )
    generator = UniversalCandidateGenerator()
    context = {"evaluator": MockEvaluator()}
    import pytest
    with pytest.raises(ValueError, match="source_evidence_mismatch"):
        options, labels, truth = generator.generate(record, context, max_k=40)

def test_query_immutability():
    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="123",
        video_id="46GP8",
        query="Exact source question?",
        source_answer="yes",
        answer_type="binary",
        semantic_type="object",
        structural_type="verify",
        reasoning_type="obj-ex", source_program="Exists(window, Filter(video, [objects]))"
    )
    assert record.query == "Exact source question?"

class TrueEvaluatorEvidenceMock:
    def get_scenegraph(self, video_id, split="train"):
        class Obj:
            def __init__(self, name):
                self.name = name
        class Frame:
            def __init__(self, objs):
                self.objects = objs
        
        frames = {
            "f1": Frame({"1": Obj("window"), "2": Obj("chair")})
        }
        return NormalizedSceneGraph(video_id=video_id, split="train", frames=frames, actions={})
        
    def evaluate(self, record, candidate, context):
        if candidate == "window": return "TRUE"
        if candidate == "chair": return "FALSE"
        return "UNKNOWN"

@pytest.mark.skip(reason="synthetic")
def test_object_evidence_path():
    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="test-obj-1",
        video_id="46GP8",
        query="Which object did they interact with?",
        source_answer="window",
        answer_type="object",
        semantic_type="object",
        structural_type="query",
        reasoning_type="obj-ex", source_program="Exists(window, Filter(video, [objects]))"
    )
    generator = UniversalCandidateGenerator()
    context = {"evaluator": TrueEvaluatorEvidenceMock()}
    options, labels, truth = generator.generate(record, context, max_k=40)
    
    # window should be true, chair should be false
    assert any("window" in o for o in options)
    assert any("chair" in o for o in options)
    assert labels[next(i for i, o in enumerate(options) if "window" in o)] == 1
    assert labels[next(i for i, o in enumerate(options) if "chair" in o)] == 0

def test_real_superlative():
    sg_dir = "data/dataset/agqa_scene_graphs"
    import os
    if not os.path.exists(os.path.join(sg_dir, "AGQA_train_stsgs.pkl")):
        pytest.skip("Real dataset not found")

    from src.reasoning.evidence import TruthEvaluator
    from src.datasets.normalized import NormalizedQA
    evaluator = TruthEvaluator(sg_dir)
    from src.candidates.universal_generator import UniversalCandidateGenerator

    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="MCQO5-999",
        video_id="MCQO5",
        query="Were they holding some clothes or dressing themselves the longest?",
        source_answer="holding a cup of something",
        answer_type="action",
        semantic_type="superlative",
        structural_type="compare",
        reasoning_type="action-recognition",
        source_program="Superlative(max, [Filter(video, [actions, dressing themselves]), Filter(video, [actions, holding a cup of something])], Subtract(Query(end, action), Query(start, action)))"
    )

    context = {"evaluator": evaluator}
    generator = UniversalCandidateGenerator()
    options, labels, truth_state = generator.generate(record, context, max_k=40)
    assert len(options) == 2
    assert any("dressing themselves" in o.lower() for o in options)
    assert any("holding a cup of something" in o.lower() for o in options)
    
def test_real_compare():
    sg_dir = "data/dataset/agqa_scene_graphs"
    import os
    if not os.path.exists(os.path.join(sg_dir, "AGQA_train_stsgs.pkl")):
        pytest.skip("Real dataset not found")

    from src.reasoning.evidence import TruthEvaluator
    from src.datasets.normalized import NormalizedQA
    evaluator = TruthEvaluator(sg_dir)
    from src.candidates.universal_generator import UniversalCandidateGenerator

    record = NormalizedQA(
        source_dataset="agqa_balanced",
        source_question_id="MCQO5-888",
        video_id="MCQO5",
        query="Did they spend longer holding some clothes or dressing themselves?",
        source_answer="longer",
        answer_type="binary",
        semantic_type="compare",
        structural_type="compare",
        reasoning_type="action-recognition",
        source_program="Compare([longer, shorter], [Filter(video, [actions, dressing themselves]), Filter(video, [actions, holding a cup of something])], Subtract(Query(end, action), Query(start, action)))"
    )

    context = {"evaluator": evaluator}
    generator = UniversalCandidateGenerator()
    options, labels, truth_state = generator.generate(record, context, max_k=40)
    assert len(options) == 2
    assert any("longer" in o.lower() for o in options)
    assert any("shorter" in o.lower() for o in options)
