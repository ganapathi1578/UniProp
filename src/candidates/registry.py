"""Candidate generator registry.

This module provides :class:`CandidateRegistry`, a mapping from abstract
answer-type strings (e.g. ``"binary"``, ``"temporal"``) to the concrete
:class:`~src.candidates.base.CandidateGenerator` subclass that knows how to
produce candidate answers of that type.

A module-level singleton :data:`candidate_registry` is exported so that all
pipeline components share one registry instance.  Generator modules register
themselves by importing this singleton at module level::

    # src/candidates/binary/__init__.py
    from src.candidates.registry import candidate_registry
    from src.candidates.binary.generator import BinaryCandidateGenerator

    candidate_registry.register("binary", BinaryCandidateGenerator)

The pipeline resolves a generator at runtime based on the
:attr:`~src.datasets.normalized.NormalizedQA.answer_type` field of each
record::

    generator_class = candidate_registry.get_generator(record.answer_type)
    if generator_class:
        generator = generator_class()
        options, labels, truth_state = generator.generate(record, context)
"""

from typing import Type, Dict
from src.candidates.base import CandidateGenerator


class CandidateRegistry:
    """Name-to-generator-class registry for candidate answer generators.

    Maintains an internal mapping from answer-type strings to
    :class:`~src.candidates.base.CandidateGenerator` *classes*.  The pipeline
    calls :meth:`get_generator` to obtain the class (or ``None`` for
    unimplemented types), then instantiates it with no arguments before
    calling ``generate``.

    Unlike :class:`~src.datasets.registry.DatasetRegistry`,
    :meth:`get_generator` returns ``None`` rather than raising on a miss, so
    the pipeline can log an ``"unsupported_domain"`` rejection and continue
    processing other records.

    Attributes:
        _generators: Internal dict mapping answer-type strings to
            ``CandidateGenerator`` subclasses.  Not intended to be accessed
            directly — use :meth:`register` and :meth:`get_generator`.
    """

    def __init__(self):
        """Initialise an empty registry.

        Creates the internal ``_generators`` dictionary.  In normal usage only
        one instance of this class is ever created (the module-level
        :data:`candidate_registry` singleton).
        """
        self._generators: Dict[str, Type[CandidateGenerator]] = {}

    def register(self, answer_type: str, generator_class: Type[CandidateGenerator]):
        """Register a candidate generator class for an answer type.

        If a class is already registered under *answer_type*, it will be
        silently overwritten, allowing tests or plug-ins to swap generators
        at runtime.

        Args:
            answer_type: The abstract answer-category key (e.g. ``"binary"``,
                ``"temporal"``, ``"count"``).  Must match the value that the
                dataset adapter assigns to
                :attr:`~src.datasets.normalized.NormalizedQA.answer_type`.
            generator_class: The :class:`~src.candidates.base.CandidateGenerator`
                subclass responsible for producing candidate options for this
                answer type.  The class itself is stored; the pipeline
                instantiates it with no constructor arguments via
                ``generator_class()``.

        Example::

            from src.candidates.registry import candidate_registry
            from src.candidates.binary.generator import BinaryCandidateGenerator

            candidate_registry.register("binary", BinaryCandidateGenerator)
        """
        self._generators[answer_type] = generator_class

    def get_generator(self, answer_type: str) -> Type[CandidateGenerator]:
        """Return the generator class for *answer_type*, or ``None`` if absent.

        Unlike :meth:`~src.datasets.registry.DatasetRegistry.get_adapter`,
        this method returns ``None`` on a miss rather than raising.  The
        pipeline interprets a ``None`` return as ``"unsupported_domain"`` and
        rejects the record, allowing partial coverage of answer types without
        crashing the entire run.

        Args:
            answer_type: The abstract answer-category string to look up
                (e.g. ``"binary"``, ``"action"``).

        Returns:
            The :class:`~src.candidates.base.CandidateGenerator` subclass
            registered under *answer_type*, or ``None`` if no generator has
            been registered for that type.

        Example::

            generator_class = candidate_registry.get_generator("temporal")
            if generator_class is None:
                # answer type not yet implemented — skip record
                ...
        """
        return self._generators.get(answer_type)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared :class:`CandidateRegistry` instance used by the entire codebase.
#: Generator modules import this object and call
#: :meth:`~CandidateRegistry.register` at import time to make themselves
#: discoverable by the pipeline.
candidate_registry = CandidateRegistry()
