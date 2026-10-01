import pytest
from src.datasets.normalized import NormalizedQA
from src.candidates.more_options import TemporalOptionizer
from src.candidates.object_options import ObjectOptionizer
from src.candidates.binary_options import BinaryOptionizer
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph

class MockEvaluator:
    def get_scenegraph(self, video_id):
        return NormalizedSceneGraph(video_id=video_id, split="train", frames={}, actions={})
        
    def evaluate(self, record, candidate, context):
        ans = str(record.source_answer).strip().lower()
        return "TRUE" if candidate.lower() == ans else "FALSE"

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
    generator = TemporalOptionizer()
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
        reasoning_type="obj-ex"
    )
    generator = BinaryOptionizer()
    context = {"evaluator": MockEvaluator()}
    options, labels, truth = generator.generate(record, context, max_k=40)
    assert set(options) == {"Yes", "No"}
    assert options[labels.index(1)] == "Yes"

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
        reasoning_type="obj-ex"
    )
    assert record.query == "Exact source question?"

class TrueEvaluatorEvidenceMock:
    def get_scenegraph(self, video_id):
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
        reasoning_type="obj-ex"
    )
    generator = ObjectOptionizer()
    context = {"evaluator": TrueEvaluatorEvidenceMock()}
    options, labels, truth = generator.generate(record, context, max_k=40)
    
    # window should be true, chair should be false
    assert "window" in options
    assert "chair" in options
    assert labels[options.index("window")] == 1
    assert labels[options.index("chair")] == 0
