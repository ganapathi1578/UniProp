"""Abstract base class for all dataset adapters.

This module defines :class:`BaseDatasetAdapter`, the interface that every
source-specific adapter must implement.  The contract is intentionally
minimal: an adapter receives a configuration dictionary at construction time
and exposes a single generator method, :meth:`~BaseDatasetAdapter.load_records`,
that yields :class:`~src.datasets.normalized.NormalizedQA` instances.

New dataset adapters should:

1. Subclass :class:`BaseDatasetAdapter`.
2. Implement ``__init__`` to parse any source-specific keys from *config*.
3. Implement ``load_records`` to open the source file(s), iterate rows, and
   ``yield`` one :class:`~src.datasets.normalized.NormalizedQA` per record.
4. Register the adapter with :data:`~src.datasets.registry.dataset_registry`
   so that the pipeline can discover it by name.

Example::

    from src.datasets.base import BaseDatasetAdapter
    from src.datasets.normalized import NormalizedQA
    from src.datasets.registry import dataset_registry

    class MyDatasetAdapter(BaseDatasetAdapter):
        def __init__(self, config: dict):
            self.path = config["source"]["qa_path"]

        def load_records(self):
            for row in open(self.path):
                yield NormalizedQA(...)

    dataset_registry.register("my_dataset", MyDatasetAdapter)
"""

from abc import ABC, abstractmethod
from typing import Iterator
from src.datasets.normalized import NormalizedQA


class BaseDatasetAdapter(ABC):
    """Abstract interface that every dataset adapter must satisfy.

    A dataset adapter is responsible for reading a single source dataset in
    its native on-disk format and translating each record into the canonical
    :class:`~src.datasets.normalized.NormalizedQA` schema.  The pipeline only
    ever calls :meth:`load_records`; all I/O, parsing, and field-mapping logic
    live inside the concrete subclass.

    Subclasses must override both abstract methods.  The ``__init__`` signature
    is fixed so that :class:`~src.datasets.registry.DatasetRegistry` can
    instantiate adapters uniformly via ``AdapterClass(config)``.
    """

    @abstractmethod
    def __init__(self, config: dict):
        """Initialise the adapter with a dataset configuration block.

        Concrete implementations should extract all source-specific settings
        from *config* here (file paths, split keys, field name overrides, etc.)
        and store them as instance attributes.  Heavy I/O (opening files,
        loading pickles) should be deferred to :meth:`load_records` or a
        lazy-loading helper so that constructing an adapter is always cheap.

        Args:
            config: Nested configuration dictionary, typically loaded from a
                YAML file.  Expected top-level keys vary by adapter but
                commonly include ``"source"`` (paths), ``"generation"``
                (seed / max_k), and ``"output"`` (shard size / root dir).
        """
        pass

    @abstractmethod
    def load_records(self) -> Iterator[NormalizedQA]:
        """Load source records and yield normalised QA representations.

        This method is the sole data-access entry point called by the
        pipeline.  Implementations must:

        * Open and parse the underlying source file(s).
        * Map every source field to the corresponding field of
          :class:`~src.datasets.normalized.NormalizedQA`.
        * ``yield`` one :class:`~src.datasets.normalized.NormalizedQA`
          per question-answer pair, in the order they appear in the source.

        The method should be implemented as a generator (using ``yield``) to
        allow the pipeline to stream records without loading the entire dataset
        into memory at once.

        Yields:
            NormalizedQA: A fully populated normalised record for each
                question-answer pair found in the source dataset.

        Raises:
            FileNotFoundError: If a required source file cannot be found at
                the configured path.
            ValueError: If a source record is malformed and cannot be mapped
                to :class:`~src.datasets.normalized.NormalizedQA`.
        """
        pass
