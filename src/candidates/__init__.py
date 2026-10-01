from src.candidates.registry import candidate_registry
from src.candidates.binary_options import BinaryOptionizer
from src.candidates.object_options import ObjectOptionizer
from src.candidates.count_options import CountOptionizer
from src.candidates.more_options import TemporalOptionizer

candidate_registry.register("binary", BinaryOptionizer)
candidate_registry.register("object", ObjectOptionizer)
candidate_registry.register("count", CountOptionizer)
candidate_registry.register("temporal", TemporalOptionizer)
