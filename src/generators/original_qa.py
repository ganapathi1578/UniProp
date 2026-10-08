"""Original-QA generator (intentional no-op stub).

This module houses :class:`OriginalQAGenerator`, a concrete but permanently
inert subclass of :class:`~src.generators.base.GeneratorBase`.

Why this class yields nothing
------------------------------
The AGQA dataset ships *free-text* natural-language questions such as
"What did the person do before sitting on the couch?".  Converting these
questions into well-formed, grammar-valid declarative propositions
(e.g. "The person sat on the couch.") requires either:

1. A large language model (LLM) with a carefully engineered prompt, or
2. A strict, hand-authored semantic parser that covers the full question
   grammar.

Neither approach is currently integrated into the UniProp pipeline.  The
original project requirement states explicitly:

    *"If the program cannot be converted safely … DO NOT generate a
    model-facing proposition from it."*

Attempting heuristic conversions (e.g. regex-based question-to-statement
transformations) risks producing malformed or semantically incorrect
propositions that would silently corrupt the benchmark.

Resolution
----------
The dataset is generated **exclusively** from scene-graph-native generators
(:class:`~src.generators.scene_graph.ObjectExistenceGenerator`,
:class:`~src.generators.scene_graph.RelationVerificationGenerator`, temporal
generators, etc.).  These generators produce propositions directly from
structured scene-graph annotations, guaranteeing 100 % grammatical validity
and full provenance traceability without any natural-language parsing step.

:class:`OriginalQAGenerator` is retained in the codebase as an explicit
architectural marker — its presence documents the conscious decision *not* to
use free-text questions, and its no-op implementation prevents any accidental
re-activation without a proper LLM integration.
"""

import random
import re
from typing import Iterator, Optional

from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools
from src.constants import OBJECTS


class OriginalQAGenerator(GeneratorBase):
    """No-op generator that intentionally produces zero proposition groups.

    This class is a deliberate stub.  It fulfils the :class:`GeneratorBase`
    interface so that it can be registered in the generator pipeline without
    causing errors, but it *never* yields any
    :class:`~src.schema.PropositionGroup` instances.

    Rationale
    ---------
    Converting AGQA free-text questions into declarative propositions safely
    requires an LLM or a fully-specified semantic parser — neither of which is
    currently part of the UniProp pipeline.  Producing propositions via
    heuristics (e.g. simple regex) would introduce semantic errors and break
    benchmark integrity.  Therefore, the pipeline relies solely on the
    scene-graph-native generators for all output propositions.

    See the module-level docstring for a detailed explanation of the design
    decision.

    Note:
        The ``return`` / ``yield`` pattern below (``return`` before any
        ``yield``) is the idiomatic Python way to create a generator function
        that immediately terminates.  The trailing ``yield`` makes the
        function's return type an :class:`~typing.Iterator` at the bytecode
        level without emitting any values.
    """

    def generate(
        self,
        qid: str,
        qdata: dict,
        sg: NormalizedSceneGraph,
        split: str,
        rng: random.Random,
        pools: NegativePools,
    ) -> Iterator[PropositionGroup]:
        """Yield nothing — original free-text QA conversion is disabled.

        This method satisfies the :class:`GeneratorBase` contract but is a
        permanent no-op.  Safe conversion of AGQA free-text questions to
        declarative propositions requires an LLM; the project relies
        exclusively on scene-graph-native generators instead.

        Args:
            qid (str): Unused.  Present only to satisfy the base interface.
            qdata (dict): Unused.  Present only to satisfy the base interface.
            sg (NormalizedSceneGraph): Unused.  Present only to satisfy the
                base interface.
            split (str): Unused.  Present only to satisfy the base interface.
            rng (random.Random): Unused.  Present only to satisfy the base
                interface.
            pools (NegativePools): Unused.  Present only to satisfy the base
                interface.

        Yields:
            Nothing.  This generator is permanently inert.
        """
        # Generating declarative propositions safely from free-text questions requires an LLM or strict semantic parser.
        # User requirement: "If the program cannot be converted safely... DO NOT generate a model-facing proposition from it."
        # We now rely exclusively on the Scene Graph Native generators (temporal, actions, relations, grounding)
        # to produce the dataset, ensuring 100% grammar validity and provenance.
        return
        yield
