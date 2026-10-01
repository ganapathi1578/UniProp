"""Streaming JSON reader for large AGQA balanced files."""
import ijson
from typing import Iterator, Tuple, Dict, Any

def stream_questions(file_path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Stream questions from a large JSON file.
    The AGQA text files are actually a single JSON object:
    {
      "QID1": {...},
      "QID2": {...}
    }
    """
    with open(file_path, 'rb') as f:
        # ijson.kvitems streams key-value pairs from the top-level object
        for qid, qdata in ijson.kvitems(f, ''):
            yield qid, qdata
