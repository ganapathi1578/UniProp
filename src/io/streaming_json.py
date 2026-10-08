"""Streaming JSON reader for large AGQA balanced files.

This module provides a memory-efficient mechanism for reading the AGQA
(Atomic Question Question Answering) dataset annotation files, which are
stored as a single, monolithic JSON object whose top-level keys are question
IDs (QIDs) and whose values are the corresponding question/answer metadata
dictionaries.

Because these files can be several gigabytes in size, loading the entire
document into memory with the standard ``json`` module is impractical.
This module wraps the ``ijson`` library, which performs *incremental*,
event-driven JSON parsing, to expose a simple generator interface that yields
one QID/question-data pair at a time.

Typical usage example::

    from src.io.streaming_json import stream_questions

    for qid, qdata in stream_questions("data/agqa_balanced_train.json"):
        # Process each question without loading the full file into RAM.
        process(qid, qdata)

Dependencies:
    ijson: Third-party incremental JSON parser.  Install with
        ``pip install ijson``.  The C backend (``ijson[yajl2_cffi]``) is
        recommended for large files.
"""

import ijson
from typing import Any, Dict, Iterator, Tuple


def stream_questions(
    file_path: str,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Stream question entries from a large AGQA-format JSON file.

    The AGQA balanced annotation files are structured as a single flat JSON
    object where every top-level key is a unique question ID (QID) and its
    associated value is a dictionary containing the question text, answer,
    video identifier, and related metadata::

        {
            "QID1": {"question": "...", "answer": "...", "video_id": "...", ...},
            "QID2": {"question": "...", "answer": "...", "video_id": "...", ...},
            ...
        }

    Rather than reading the whole file into memory, this function uses
    ``ijson.kvitems`` to parse the top-level key-value pairs *lazily*,
    yielding each pair as it is encountered in the byte stream.  This keeps
    peak memory consumption proportional to the size of a **single** entry
    rather than the size of the full file.

    Args:
        file_path: Absolute or relative path to the AGQA JSON annotation
            file.  The file is opened in binary mode (``'rb'``), as required
            by ``ijson``.

    Yields:
        A 2-tuple ``(qid, qdata)`` for each top-level entry in the JSON
        object:

        - **qid** (``str``): The unique question identifier that serves as
          the top-level key in the JSON object (e.g. ``"Q1234"``).
        - **qdata** (``Dict[str, Any]``): A dictionary containing all
          metadata associated with the question, such as the question text,
          expected answer, video ID, and compositional program annotations.
          The exact schema depends on the AGQA dataset version being parsed.

    Raises:
        FileNotFoundError: If *file_path* does not point to an existing file.
        ijson.JSONError: If the file content is not valid JSON or does not
            conform to the expected top-level object structure.
        OSError: For other low-level I/O errors encountered while opening or
            reading the file.

    Notes:
        - The file handle is closed automatically when the generator is
          exhausted or garbage-collected, because it is managed by a
          ``with`` statement inside the generator body.
        - Prefer the ``yajl2`` or ``yajl2_cffi`` ijson backend for
          significantly better throughput on files larger than ~1 GB.
          Set the backend via the ``IJSON_BACKEND`` environment variable or
          by importing the desired backend explicitly.

    Example::

        from src.io.streaming_json import stream_questions

        total = 0
        for qid, qdata in stream_questions("agqa_balanced_test.json"):
            assert "question" in qdata
            total += 1
        print(f"Streamed {total} questions.")
    """
    with open(file_path, 'rb') as f:
        # ijson.kvitems streams key-value pairs from the top-level object
        for qid, qdata in ijson.kvitems(f, ''):
            yield qid, qdata
