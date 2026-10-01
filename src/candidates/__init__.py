from src.candidates.registry import candidate_registry
from src.candidates.binary_options import BinaryOptionizer
from src.candidates.object_options import ObjectOptionizer
from src.candidates.count_options import CountOptionizer

candidate_registry.register("binary", BinaryOptionizer)
candidate_registry.register("object", ObjectOptionizer)
candidate_registry.register("count", CountOptionizer)
# We can register more as they are built (action, temporal, comparison, superlative)
from src.candidates.more_options import TemporalOptionizer, ComparisonOptionizer, ActionOptionizer, SuperlativeOptionizer, LogicOptionizer

candidate_registry.register("temporal", TemporalOptionizer)
candidate_registry.register("comparison", ComparisonOptionizer)
candidate_registry.register("action", ActionOptionizer)
candidate_registry.register("superlative", SuperlativeOptionizer)
candidate_registry.register("logic", LogicOptionizer)
candidate_registry.register("open", ObjectOptionizer) # Fallback open to object for now
