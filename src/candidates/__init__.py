from src.candidates.registry import candidate_registry
from src.candidates.universal_generator import UniversalCandidateGenerator

candidate_registry.register("binary", UniversalCandidateGenerator)
candidate_registry.register("object", UniversalCandidateGenerator)
candidate_registry.register("count", UniversalCandidateGenerator)
candidate_registry.register("temporal", UniversalCandidateGenerator)
candidate_registry.register('action', UniversalCandidateGenerator)
candidate_registry.register('comparison', UniversalCandidateGenerator)
candidate_registry.register('superlative', UniversalCandidateGenerator)
candidate_registry.register('three_way', UniversalCandidateGenerator)
candidate_registry.register('logic', UniversalCandidateGenerator)
