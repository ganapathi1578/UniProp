"""Dataset split management for the UniProp training pipeline.

This module provides utilities for deterministically partitioning a pool of
video IDs into disjoint training and validation subsets.  Determinism is
guaranteed by seeding Python's ``random.Random`` instance with a fixed
integer, ensuring that repeated runs — or runs on different machines — always
produce identical splits given the same inputs.

The typical workflow is:

1. Load the full set of *training* video IDs from the AGQA metadata (test
   IDs are kept completely separate and are never passed here).
2. Call ``create_splits`` to divide those IDs into a *train* set and a
   smaller *validation* set.
3. Use the returned sets to filter downstream data loading.

Typical usage::

    from src.normalization.split_manager import create_splits

    all_train_ids = load_train_video_ids()          # set or list of str
    train_ids, val_ids = create_splits(all_train_ids, val_ratio=0.1, seed=42)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

Notes:
    The sort-then-shuffle strategy (``sorted`` followed by ``rng.shuffle``)
    is critical for reproducibility: ``sorted`` imposes a canonical order
    regardless of the insertion order of the source collection, and the
    seeded ``rng`` then applies a deterministic permutation on top of that.
"""

import random


def create_splits(
    train_video_ids: list[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split training video IDs into disjoint train and validation sets.

    Produces a reproducible partition of *train_video_ids* into a larger
    training subset and a smaller validation subset.  The split is fully
    deterministic: identical inputs (same IDs, same *val_ratio*, same *seed*)
    always yield identical outputs, regardless of execution environment or
    Python version (within the same minor version series).

    Algorithm:
        1. Sort the input IDs lexicographically to remove any dependence on
           the source collection's insertion order.
        2. Shuffle the sorted list using a seeded ``random.Random`` instance
           for a reproducible permutation.
        3. Take the first ``floor(N * val_ratio)`` IDs as the validation set
           and the remainder as the training set.

    Args:
        train_video_ids: Collection of video identifier strings that make up
            the *full* training pool (i.e. excluding any held-out test IDs).
            May be any iterable of strings; duplicates are not explicitly
            removed, so callers should deduplicate if necessary.
        val_ratio: Fraction of *train_video_ids* to allocate to the
            validation set.  Must be in the range ``[0.0, 1.0)``.  Defaults
            to ``0.1`` (10 % validation split).  The exact number of
            validation IDs is computed as ``int(N * val_ratio)``, so for
            small datasets the resulting ratio may differ slightly from the
            requested one due to integer truncation.
        seed: Integer seed for the ``random.Random`` instance used to shuffle
            the sorted IDs.  Changing this value produces a different but
            equally reproducible partition.  Defaults to ``42``.

    Returns:
        A 2-tuple ``(train_ids, val_ids)`` where:

        - **train_ids** (``set[str]``): The larger subset of IDs reserved for
          model training, containing ``N - floor(N * val_ratio)`` elements.
        - **val_ids** (``set[str]``): The smaller subset of IDs reserved for
          validation, containing ``floor(N * val_ratio)`` elements.

        The two sets are guaranteed to be disjoint and their union equals the
        set of all unique IDs in *train_video_ids*.

    Raises:
        TypeError: If *train_video_ids* is not iterable or contains non-string
            elements that cannot be sorted.

    Notes:
        - The function converts the result to Python ``set`` objects, so
          membership tests on the returned collections are O(1).
        - Because ``int()`` truncates towards zero, a *val_ratio* of exactly
          ``0.0`` always produces an empty validation set.
        - The original *train_video_ids* collection is not modified; an
          independent copy is sorted and shuffled internally.

    Example::

        from src.normalization.split_manager import create_splits

        video_ids = [f"video_{i:04d}" for i in range(1000)]
        train_ids, val_ids = create_splits(video_ids, val_ratio=0.15, seed=0)

        assert len(train_ids) == 850
        assert len(val_ids)   == 150
        assert train_ids.isdisjoint(val_ids)
    """
    rng = random.Random(seed)
    # Sort first to guarantee a canonical order independent of input ordering.
    shuffled = sorted(list(train_video_ids))
    # Deterministically permute using the seeded RNG.
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids
