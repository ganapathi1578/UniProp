"""Abstract base class for all UniProp proposition generators.

This module defines the :class:`GeneratorBase` interface that every concrete
generator in the ``src.generators`` package must implement.  The contract is
intentionally minimal: given a normalised scene-graph and its associated
metadata, produce zero or more :class:`~src.schema.PropositionGroup` objects
that can be written directly to the benchmark dataset.

Design principles
-----------------
* **Reproducibility** – every generator receives a seeded
  :class:`random.Random` instance (``rng``) so that the full dataset can be
  re-created deterministically from the same seed.
* **Pluggability** – new task families (e.g. temporal, causal) are added by
  subclassing :class:`GeneratorBase` and registering the class in the
  pipeline; no other files need to change.
* **Separation of concerns** – generators are *pure* in the sense that they
  only read from their arguments and ``pools``; they never mutate shared
  state.
"""

import random
from typing import Iterator

from src.schema import PropositionGroup
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools


class GeneratorBase:
    """Abstract base class that all proposition generators must subclass.

    A *generator* is responsible for converting one video's
    :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
    (and its companion :class:`~src.negatives.hard_negatives.NegativePools`)
    into a stream of :class:`~src.schema.PropositionGroup` benchmark examples.

    Concrete subclasses implement :meth:`generate` and ``yield`` one
    :class:`~src.schema.PropositionGroup` per logical evaluation unit.
    Yielding *nothing* is valid (see :class:`~src.generators.original_qa.OriginalQAGenerator`
    and :class:`~src.generators.attributes.AttributeGenerator`) and simply
    means the generator contributes no examples for that video.

    Usage example::

        class MyGenerator(GeneratorBase):
            def generate(self, qid, qdata, sg, split, rng, pools):
                yield PropositionGroup(...)

    Note:
        Subclasses **must not** mutate ``sg``, ``pools``, or any other shared
        argument; doing so would break reproducibility across parallel workers.
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
        """Generate proposition groups for a single video scene-graph.

        This method **must** be overridden by every concrete subclass.  The
        base implementation raises :exc:`NotImplementedError` to enforce the
        contract at runtime.

        The caller is responsible for advancing ``rng`` between generator
        invocations so that each generator operates on an independent random
        stream.

        Args:
            qid (str): Unique identifier of the source question (e.g. an
                AGQA question ID).  May be ``None`` or an empty string when
                the generator is scene-graph-native and not derived from a
                question.
            qdata (dict): Raw question metadata dictionary as loaded from the
                AGQA dataset JSON/pickle.  Keys typically include
                ``"question"``, ``"answer"``, ``"program"``, etc.  Generators
                that are scene-graph-native may ignore this argument entirely.
            sg (NormalizedSceneGraph): The normalised scene-graph for the
                current video.  Provides ``video_id``, the set of present
                objects, frame-level relation tuples, and other derived
                attributes used to construct propositions.
            split (str): Dataset split identifier, e.g. ``"train"``,
                ``"val"``, or ``"test"``.  Propagated verbatim into every
                emitted :class:`~src.schema.PropositionGroup`.
            rng (random.Random): A seeded random number generator shared
                across all generators for a given video.  Subclasses must
                use *only* this instance (never ``random.random()`` or
                ``numpy.random``) to ensure full determinism.
            pools (NegativePools): Pre-computed pools of absent objects and
                closed-world false relations for the current video.  Used to
                sample high-quality hard negatives without re-scanning the
                scene-graph on every call.

        Yields:
            PropositionGroup: One benchmark example per logical evaluation
                unit (e.g. one binary verification task, one single-choice
                set, etc.).  A generator may yield any number of groups,
                including zero.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError
