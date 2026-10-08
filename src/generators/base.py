"""Base generator class."""
import random
from typing import Iterator, Optional
from src.schema import PropositionGroup
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools


class GeneratorBase:
    def generate(
        self,
        qid: str,
        qdata: dict,
        sg: NormalizedSceneGraph,
        split: str,
        rng: random.Random,
        pools: NegativePools,
        target_task: Optional[str] = None,
        target_k: Optional[int] = None,
        target_hops: Optional[int] = None,
        target_difficulty: Optional[str] = None,
    ) -> Iterator[PropositionGroup]:
        raise NotImplementedError
