"""Attribute generators."""
import random
from typing import Iterator
from src.schema import PropositionGroup
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools

class AttributeGenerator(GeneratorBase):
    """Generates object attribute propositions. 
    Note: Action Genome baseline does not include rich attributes (like color, material).
    This acts as a placeholder that yields 0 examples for now unless AGQA raw attributes are provided.
    """
    
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        # No attributes in Action Genome STSG pickles.
        # Future extension: if we extract attributes from the text questions, we could generate them here.
        return
        yield
