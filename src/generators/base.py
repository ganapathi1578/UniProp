"""Base generator class."""
import random
from typing import Iterator
from src.schema import PropositionGroup
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools

class GeneratorBase:
    def generate(self, qid: str, qdata: dict, sg: NormalizedSceneGraph, split: str, rng: random.Random, pools: NegativePools) -> Iterator[PropositionGroup]:
        raise NotImplementedError
