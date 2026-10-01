from abc import ABC, abstractmethod
from typing import Iterator
from src.datasets.normalized import NormalizedQA

class BaseDatasetAdapter(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def load_records(self) -> Iterator[NormalizedQA]:
        """Loads source records and yields normalized QA representations."""
        pass
