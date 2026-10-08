"""Dataset adapter registry.

This module provides :class:`DatasetRegistry`, a simple name-to-class mapping
that allows the pipeline to resolve a dataset name string (e.g.
``"agqa_balanced"``) to the concrete :class:`~src.datasets.base.BaseDatasetAdapter`
subclass that handles it.

A module-level singleton :data:`dataset_registry` is exported so that all
parts of the codebase share one registry instance.  Adapter modules register
themselves by importing this singleton at module level and calling
:meth:`~DatasetRegistry.register`::

    # src/datasets/agqa/__init__.py
    from src.datasets.registry import dataset_registry
    from src.datasets.agqa.adapter import AGQABalancedAdapter

    dataset_registry.register("agqa_balanced", AGQABalancedAdapter)

The pipeline then resolves the adapter at runtime::

    from src.datasets.registry import dataset_registry

    AdapterClass = dataset_registry.get_adapter("agqa_balanced")
    adapter = AdapterClass(config)
"""

from typing import Type, Dict
from src.datasets.base import BaseDatasetAdapter


class DatasetRegistry:
    """Name-to-adapter-class registry for dataset adapters.

    Maintains an internal mapping from human-readable dataset name strings to
    :class:`~src.datasets.base.BaseDatasetAdapter` *classes* (not instances).
    The pipeline calls :meth:`get_adapter` to obtain the class, then
    instantiates it with the relevant config block.

    This design separates adapter discovery from adapter construction, keeping
    each adapter's ``__init__`` arguments flexible.

    Attributes:
        _adapters: Internal dict mapping ``name`` strings to
            ``BaseDatasetAdapter`` subclasses.  Not intended to be accessed
            directly — use :meth:`register` and :meth:`get_adapter` instead.
    """

    def __init__(self):
        """Initialise an empty registry.

        Creates the internal ``_adapters`` dictionary.  In normal usage only
        one instance of this class is ever created (the module-level
        :data:`dataset_registry` singleton).
        """
        self._adapters: Dict[str, Type[BaseDatasetAdapter]] = {}

    def register(self, name: str, adapter_class: Type[BaseDatasetAdapter]):
        """Register a dataset adapter class under a given name.

        If a class is already registered under *name*, it will be silently
        overwritten.  This allows test code or plug-ins to replace adapters
        at runtime.

        Args:
            name: The dataset name string used as the lookup key (e.g.
                ``"agqa_balanced"``).  Must match the value of the
                ``"dataset"`` key in the pipeline configuration YAML.
            adapter_class: The :class:`~src.datasets.base.BaseDatasetAdapter`
                subclass to associate with *name*.  The class itself is stored
                (not an instance) so the pipeline can instantiate it with the
                correct split-specific config.

        Example::

            from src.datasets.registry import dataset_registry
            from src.datasets.agqa.adapter import AGQABalancedAdapter

            dataset_registry.register("agqa_balanced", AGQABalancedAdapter)
        """
        self._adapters[name] = adapter_class

    def get_adapter(self, name: str) -> Type[BaseDatasetAdapter]:
        """Return the adapter class registered under *name*.

        Args:
            name: The dataset name string to look up (e.g.
                ``"agqa_balanced"``).

        Returns:
            The :class:`~src.datasets.base.BaseDatasetAdapter` subclass that
            was previously registered under *name*.

        Raises:
            ValueError: If no adapter has been registered under *name*.  The
                error message includes *name* to aid debugging.

        Example::

            AdapterClass = dataset_registry.get_adapter("agqa_balanced")
            adapter = AdapterClass(config)
        """
        if name not in self._adapters:
            raise ValueError(f"Dataset adapter '{name}' not found in registry.")
        return self._adapters[name]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared :class:`DatasetRegistry` instance used by the entire codebase.
#: Adapter modules import this object and call :meth:`~DatasetRegistry.register`
#: to make themselves discoverable by the pipeline.
dataset_registry = DatasetRegistry()
