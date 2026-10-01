from typing import Type, Dict
from src.datasets.base import BaseDatasetAdapter

class DatasetRegistry:
    def __init__(self):
        self._adapters: Dict[str, Type[BaseDatasetAdapter]] = {}

    def register(self, name: str, adapter_class: Type[BaseDatasetAdapter]):
        self._adapters[name] = adapter_class

    def get_adapter(self, name: str) -> Type[BaseDatasetAdapter]:
        if name not in self._adapters:
            raise ValueError(f"Dataset adapter '{name}' not found in registry.")
        return self._adapters[name]

dataset_registry = DatasetRegistry()
