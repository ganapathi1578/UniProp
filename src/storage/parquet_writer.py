"""Parquet shard writer for UniProp proposition groups.

This module provides a single public function, :func:`write_shard`, which
serialises a batch of :class:`PropositionGroup` objects into a flat Apache
Parquet file using a fixed, strongly-typed PyArrow schema.

Each row in the output file represents one *example* — a video-query pair
together with all its candidate propositions (options), their binary labels,
and the metadata needed to reconstruct the full evaluation context.

Typical usage::

    from src.storage.parquet_writer import write_shard

    write_shard(groups=my_groups, out_path="data/shards/train_000.parquet")

Schema summary:
    +-----------------+--------------------+----------------------------------+
    | Column          | PyArrow type       | Description                      |
    +=================+====================+==================================+
    | example_id      | string             | Unique example identifier        |
    | split           | string             | Dataset split (train/val/test)   |
    | video_id        | string             | Source video identifier          |
    | query           | string             | Natural-language query string    |
    | options         | list<string>       | Candidate proposition texts      |
    | labels          | list<int8>         | Binary labels per option (0/1/-1)|
    | truth_state     | string             | Truth-state of the first option  |
    | option_count    | int32              | Number of candidate propositions |
    | task_type       | string             | High-level task category         |
    | reasoning_family| string             | Reasoning taxonomy family        |
    | generator_family| string             | Generator provenance family      |
    +-----------------+--------------------+----------------------------------+

Dependencies:
    pyarrow: Apache Arrow in-memory columnar format and Parquet I/O.
"""

import pyarrow as pa
import pyarrow.parquet as pq


def write_shard(groups, out_path):
    """Serialise a list of PropositionGroup objects to a Parquet file.

    Iterates over every :class:`PropositionGroup` in *groups*, flattens each
    group's propositions into parallel list columns (``options`` and
    ``labels``), enforces a fixed PyArrow schema, and writes the resulting
    :class:`pyarrow.Table` to *out_path* as a single-file Parquet shard.

    The function performs a non-null assertion on ``video_id`` before writing
    to catch upstream data issues early and prevent silent corruption of the
    output file.

    PyArrow schema applied to the output table:

    .. code-block:: text

        example_id      : pa.string()          — Unique per-example UUID / key.
        split           : pa.string()          — "train", "val", or "test".
        video_id        : pa.string()          — Source video identifier; must
                                                  be non-null and non-empty for
                                                  every row.
        query           : pa.string()          — Natural-language query string;
                                                  falls back to "" when absent.
        options         : pa.list_(pa.string()) — Ordered list of candidate
                                                  proposition surface forms.
        labels          : pa.list_(pa.int8())  — Parallel list of binary labels
                                                  (0 = false, 1 = true,
                                                  -1 = unknown/missing).
        truth_state     : pa.string()          — Truth-state tag derived from
                                                  the first proposition; defaults
                                                  to "UNKNOWN" for empty groups.
        option_count    : pa.int32()           — Total number of candidate
                                                  propositions in the group.
        task_type       : pa.string()          — High-level task category (e.g.
                                                  "existence", "relation").
        reasoning_family: pa.string()          — Reasoning taxonomy family drawn
                                                  from ``g.reasoning.family``.
        generator_family: pa.string()          — Generator provenance family
                                                  drawn from
                                                  ``g.provenance.generator_family``.

    Args:
        groups (Iterable[PropositionGroup]): An iterable of
            :class:`PropositionGroup` instances to serialise.  Each group must
            expose the following attributes:

            * ``example_id`` (str): Unique identifier for this example.
            * ``split`` (str): Dataset split label.
            * ``media_id`` or ``video_id`` (str): Source video identifier.
              ``media_id`` is preferred when present (checked via
              :func:`hasattr`).
            * ``query_text`` (str | None): Natural-language query; ``None``
              is coerced to ``""``.
            * ``propositions`` (Sequence[Proposition]): Ordered candidate
              propositions.  Each :class:`Proposition` must expose:

              - ``text`` (str): Surface form of the proposition.
              - ``label`` (int | None): Binary label; ``None`` is stored as
                ``-1``.
              - ``truth_state`` (str): Truth-state tag.

            * ``num_propositions`` (int): Expected to equal
              ``len(group.propositions)``.
            * ``task_type`` (str): Task category string.
            * ``reasoning`` (object): Object with a ``family`` (str) attribute.
            * ``provenance`` (object): Object with a ``generator_family`` (str)
              attribute.

        out_path (str | os.PathLike): Destination file path for the Parquet
            shard.  The parent directory must already exist; this function does
            not create directories.

    Raises:
        AssertionError: If any row has a ``video_id`` that is ``None``,
            evaluates to ``False``, or is a blank/whitespace-only string.
            The message is ``"Missing video_id in generated examples"``.
        pyarrow.lib.ArrowInvalid: If the collected Python data cannot be cast
            to the declared PyArrow schema (e.g. a ``label`` value outside the
            int8 range).
        OSError: If *out_path* is not writable.

    Returns:
        None: The function writes data to disk and returns nothing.

    Example::

        from src.storage.parquet_writer import write_shard

        # Assume `train_groups` is already built by a generator pipeline.
        write_shard(groups=train_groups, out_path="shards/train_000.parquet")
        # => Parquet file created at the specified path.
    """
    # Initialise the columnar accumulator with one empty list per output column.
    # Keys match the final Parquet column names exactly.
    data = {
        "example_id": [], "split": [], "video_id": [], "query": [], "options": [],
        "labels": [], "truth_state": [], "option_count": [], "task_type": [],
        "reasoning_family": [], "generator_family": []
    }

    for g in groups:
        # --- Scalar metadata columns ---
        data["example_id"].append(g.example_id)
        data["split"].append(g.split)

        # Prefer `media_id` (newer attribute name) over the legacy `video_id`.
        data["video_id"].append(g.media_id if hasattr(g, 'media_id') else g.video_id)

        # Coerce absent query text to an empty string to satisfy pa.string().
        data["query"].append(g.query_text or "")

        # --- List columns derived from the propositions sequence ---

        # Collect the surface-form text of every candidate proposition.
        data["options"].append([p.text for p in g.propositions])

        # Collect binary labels; replace None with -1 (sentinel for missing).
        data["labels"].append([p.label if p.label is not None else -1 for p in g.propositions])

        # Use the first proposition's truth_state; fall back when list is empty.
        data["truth_state"].append(g.propositions[0].truth_state if g.propositions else "UNKNOWN")

        # --- Additional metadata columns ---
        data["option_count"].append(g.num_propositions)
        data["task_type"].append(g.task_type)
        data["reasoning_family"].append(g.reasoning.family)
        data["generator_family"].append(g.provenance.generator_family)

    # Guard: every example must have a non-null, non-empty video_id.
    # Catching this early prevents silent row-level data corruption downstream.
    assert all(v is not None and str(v).strip() != "" for v in data["video_id"]), \
        "Missing video_id in generated examples"

    # Define the strict PyArrow schema so that column types are explicit and
    # reproducible regardless of the Python runtime types collected above.
    schema = pa.schema([
        ("example_id",       pa.string()),          # Unique example key
        ("split",            pa.string()),          # train / val / test
        ("video_id",         pa.string()),          # Source video identifier
        ("query",            pa.string()),          # Natural-language query
        ("options",          pa.list_(pa.string())),# Candidate proposition texts
        ("labels",           pa.list_(pa.int8())), # Binary labels (0/1/-1)
        ("truth_state",      pa.string()),          # Truth-state of first option
        ("option_count",     pa.int32()),           # Number of options per group
        ("task_type",        pa.string()),          # High-level task category
        ("reasoning_family", pa.string()),          # Reasoning taxonomy family
        ("generator_family", pa.string()),          # Generator provenance family
    ])

    # Build an Arrow Table from the dict and enforce the schema.
    table = pa.Table.from_pydict(data, schema=schema)

    # Write to Parquet using default snappy compression (PyArrow default).
    pq.write_table(table, out_path)
