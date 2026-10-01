"""Original QA based generators."""
import random
import re
from typing import Iterator, Optional
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS

class OriginalQAGenerator(GeneratorBase):
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        # Generating declarative propositions safely from free-text questions requires an LLM or strict semantic parser.
        # User requirement: "If the program cannot be converted safely... DO NOT generate a model-facing proposition from it."
        # We now rely exclusively on the Scene Graph Native generators (temporal, actions, relations, grounding)
        # to produce the dataset, ensuring 100% grammar validity and provenance.
        return
        yield
