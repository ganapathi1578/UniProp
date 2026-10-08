"""UniProp Dataset Generation Pipeline — Main Entry Point.

This module orchestrates the end-to-end construction of the UniProp benchmark
dataset from raw AGQA scene-graph and question-answer data.  At a high level,
the pipeline executes the following stages:

Data loading
    Two AGQA spatio-temporal scene-graph (STSG) pickle files are read:
    ``AGQA_train_stsgs.pkl`` (training videos) and
    ``AGQA_test_stsgs.pkl`` (test videos).  A 90 / 10 train / validation
    split is derived from the training videos using a seeded shuffle so that
    results are exactly reproducible.

Scene-graph normalisation
    Each raw scene-graph is passed through
    :func:`src.normalization.scene_graph_normalizer.normalize_video_sg`
    to produce a canonical representation, and a
    :class:`src.negatives.hard_negatives.NegativePools` object is built per
    video for later use by the proposition generators.

Proposition generation
    Eight specialised generators are applied in a **round-robin** order over
    every question in the AGQA balanced-QA text files:

    1. ``orig_qa``       — :class:`OriginalQAGenerator`
    2. ``obj_exist``     — :class:`ObjectExistenceGenerator`
    3. ``rel_ver``       — :class:`RelationVerificationGenerator`
    4. ``rel_set``       — :class:`RelationSetGenerator`
    5. ``grounding``     — :class:`GroundingGenerator`
    6. ``temporal``      — :class:`ActionTemporalGenerator`
    7. ``compositional`` — :class:`CompositionalGenerator`
    8. ``attributes``    — :class:`AttributeGenerator`

Deduplication & validation
    Each candidate :class:`~src.schema.PropositionGroup` is fingerprinted by
    hashing the sorted proposition texts.  Duplicates (including any examples
    that were already persisted to disk from a previous run) are discarded.
    Survivors are passed through
    :func:`src.validation.validators.validate_example`; failed candidates are
    tallied and their error codes recorded.

Streaming I/O
    Accepted groups are accumulated in per-split in-memory buffers.  Whenever
    a buffer reaches ``SHARD_SIZE`` rows it is flushed to a zero-padded
    Parquet shard (``part-NNNNNN.parquet``) under
    ``data/generated/<split>/``.  This allows the pipeline to be safely
    interrupted and resumed without re-generating data that is already on disk.

Report generation
    After all data are written, four Markdown audit reports are produced in
    the working directory:

    * ``SAMPLE_REPORT.md``     — up to five example groups per generation rule.
    * ``VALIDATION_REPORT.md`` — candidate / accept / reject / duplicate counts.
    * ``SPLIT_REPORT.md``      — per-split video universe sizes and example counts.
    * ``DIVERSITY_REPORT.md``  — diversity metrics, truth-state distributions,
      temporal / grounding subtype breakdowns, and proposition-signature counts.

Constants:
    SEED (int): Global RNG seed (42).  Passed to Python's :mod:`random` module
        **and** exported as the ``PYTHONHASHSEED`` environment variable so that
        hash randomisation is deterministic across interpreter restarts.
    SHARD_SIZE (int): Number of :class:`~src.schema.PropositionGroup` rows
        accumulated in each per-split buffer before being flushed to disk as a
        Parquet shard (50 000).
    DATA_ROOT (str): Absolute path to the directory that contains this script.
        All relative asset paths (scene-graph pickles, balanced-QA text files,
        output shards) are resolved relative to this root.
"""

import os
import time
import pickle
import random
from tqdm import tqdm
from collections import Counter
from typing import List

from src.io.streaming_json import stream_questions
from src.normalization.scene_graph_normalizer import normalize_video_sg
from src.negatives.hard_negatives import NegativePools
from src.generators.original_qa import OriginalQAGenerator
from src.generators.scene_graph import ObjectExistenceGenerator, RelationVerificationGenerator, RelationSetGenerator
from src.generators.grounding import GroundingGenerator
from src.generators.temporal import ActionTemporalGenerator
from src.generators.compositional import CompositionalGenerator
from src.generators.attributes import AttributeGenerator
from src.validation.validators import validate_example
from src.storage.parquet_writer import write_shard
from src.schema import PropositionGroup

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# Reproducibility seed used for every random operation in this module.
SEED = 42

# Ensure Python's built-in hash() is also seeded the same way so that any
# set / dict iteration order that depends on hash values stays deterministic
# across interpreter invocations.
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)

# Root directory (the folder that contains this script).  All file paths are
# constructed relative to DATA_ROOT so the pipeline is portable.
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

# Maximum number of PropositionGroup rows buffered in memory per split before
# a Parquet shard is flushed to disk.
SHARD_SIZE = 50000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_signature(group: PropositionGroup) -> str:
    """Compute a compact deduplication signature for a PropositionGroup.

    The signature encodes the union of evidence identifiers that ground the
    group in the source video, enabling downstream analysis of how many
    distinct proposition groups share the same underlying evidence.  It is
    **not** the primary deduplication key (which is a hash of proposition
    texts); rather, it serves as a secondary grouping key stored in
    ``stats["examples_per_evidence"]``.

    The three evidence dimensions are each independently sorted and then
    concatenated with ``_`` separators so that ordering differences in the
    source data do not produce spurious distinct signatures.

    Args:
        group (PropositionGroup): A validated proposition group whose
            ``provenance`` attribute carries the evidence metadata.  The
            provenance is expected to expose three optional list fields:

            * ``evidence_object_ids`` — IDs of scene-graph nodes (objects /
              people) that support the propositions.
            * ``evidence_relation_ids`` — IDs of scene-graph edges (spatial or
              interaction relations) that support the propositions.
            * ``frame_ids`` — Video frame indices from which the evidence was
              drawn.

            Any of these may be ``None`` or empty; in that case the
            corresponding component of the signature is an empty tuple
            ``()``.

    Returns:
        str: A string of the form ``"(obj_ids,)_(rel_ids,)_(frame_ids,)"``
            where each component is the ``repr`` of a *sorted* tuple of the
            respective IDs.  The value is deterministic for the same set of
            IDs regardless of their original order.

    Example:
        >>> sig = get_signature(group)
        >>> stats["examples_per_evidence"][sig] += 1
    """
    # Sort each evidence list independently so that order differences in the
    # raw provenance do not create spurious distinct signatures.
    ev_obj = tuple(sorted(group.provenance.evidence_object_ids or []))
    ev_rel = tuple(sorted(group.provenance.evidence_relation_ids or []))
    ev_frm = tuple(sorted(group.provenance.frame_ids or []))
    return f"{ev_obj}_{ev_rel}_{ev_frm}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full UniProp dataset generation pipeline.

    This is the single top-level function that drives every phase of the
    pipeline from data loading through report generation.  It is called when
    the module is run as a script (see the ``if __name__ == "__main__":``
    guard at the bottom of this file).

    The function is organised into the following logical phases:

    **Phase 1 – Generator registration**
        Eight generator objects are instantiated and stored as
        ``(name, generator)`` pairs in a list.  The round-robin scheduler
        inside :func:`process_split` iterates over this list using a modular
        counter so that every question gets exactly one attempt from each
        generator before any generator is reused.

    **Phase 2 – Scene-graph loading**
        ``AGQA_train_stsgs.pkl`` and ``AGQA_test_stsgs.pkl`` are loaded with
        :mod:`pickle`.  Each file maps ``video_id -> raw_scene_graph``.  A
        sanity assertion confirms that no test video appears in the training
        set.

    **Phase 3 – Train / val split**
        The list of training video IDs is shuffled with the seeded ``rng``
        and the first 10 % are held out as the validation set.  The remaining
        90 % form the training set.  Both sets are stored as ``set`` objects
        for O(1) membership tests inside :func:`get_split`.

    **Phase 4 – Stats dictionary initialisation**
        A single ``stats`` dictionary is created to accumulate all metrics
        throughout the run.  Its fields are:

        * ``candidates``          — :class:`Counter` of raw candidate counts
          per generator name.
        * ``accepted``            — :class:`Counter` of accepted counts per
          generator name.
        * ``rejected``            — :class:`Counter` of rejected counts per
          generator name.
        * ``rejection_reasons``   — :class:`Counter` of validation error codes.
        * ``task_types``          — :class:`Counter` of
          ``group.task_type`` values.
        * ``reasoning_families``  — :class:`Counter` of
          ``group.reasoning.family`` values (e.g. ``"temporal"``,
          ``"grounding"``).
        * ``generator_families``  — :class:`Counter` of
          ``group.provenance.generator_family`` values.
        * ``complexity``          — :class:`Counter` of
          ``group.reasoning.complexity`` values.
        * ``k_counts``            — :class:`Counter` of
          ``group.num_propositions`` values (the *k* in k-way choice).
        * ``truth_states``        — :class:`Counter` of
          ``proposition.truth_state`` across all propositions in all groups.
        * ``none_included``       — ``int`` total number of propositions
          whose ``canonical_type`` is ``"logical_none"``.
        * ``none_true``           — ``int`` subset of ``none_included`` where
          ``proposition.label == 1``.
        * ``none_false``          — ``int`` subset of ``none_included`` where
          ``proposition.label != 1``.
        * ``query_templates``     — ``set`` of distinct
          ``group.query_template_id`` values seen.
        * ``temporal_subtypes``   — :class:`Counter` of temporal-reasoning
          sub-categories (e.g. specific ``temporal_relation`` strings,
          ``"multihop"``, or ``"unknown"``).
        * ``temporal_hops``       — :class:`Counter` of
          ``group.reasoning.hops`` for temporal groups.
        * ``grounding_subtypes``  — :class:`Counter` with keys
          ``"identification"`` and ``"verification"``.
        * ``source_videos``       — :class:`Counter` of examples generated
          per source video ID.
        * ``split_counts``        — :class:`Counter` of examples generated
          per split name (``"train"``, ``"val"``, ``"test"``).
        * ``examples_per_video``  — :class:`Counter` aliased view of
          per-video example counts (same semantic as ``source_videos``).
        * ``examples_per_evidence`` — :class:`Counter` keyed by the string
          returned by :func:`get_signature`; used to measure evidence reuse.
        * ``unique_prop_signatures`` — ``set`` of hashed
          ``(canonical_type, subject, predicate, object, temporal_relation,
          event_a, event_b, polarity)`` tuples across all propositions.
        * ``unique_option_sets``  — ``set`` of hashes of the *sorted* option
          text tuples for each group (group-level uniqueness metric).
        * ``duplicate_count``     — ``int`` total groups discarded by the
          hash-based deduplication gate.
        * ``total_candidates``    — ``int`` total raw candidate groups
          produced by all generators before any filtering.

    **Phase 5 – Checkpoint loading**
        Existing Parquet shards under ``data/generated/<split>/`` are scanned
        at startup.  For each shard:

        * The shard index is used to advance ``shard_indices[split]`` so that
          new shards receive non-colliding filenames.
        * The ``propositions`` column is read and the sorted-text hash of
          every existing group is added to ``seen_hashes`` so that the current
          run does not regenerate examples that are already on disk.

    **Nested helper – flush_buffer(split, force=False)**
        Writes the in-memory buffer for *split* to the next Parquet shard.

        Args:
            split (str): One of ``"train"``, ``"val"``, or ``"test"``.
            force (bool): When ``False`` (the default), the flush is skipped
                unless the buffer has reached ``SHARD_SIZE`` rows.  Pass
                ``True`` at pipeline teardown to flush any partial buffer
                that has not yet reached the threshold, ensuring no examples
                are lost.

        The shard filename follows the pattern ``part-NNNNNN.parquet`` where
        ``NNNNNN`` is a zero-padded six-digit index that increments
        monotonically per split.  On success the buffer is cleared and
        ``shard_indices[split]`` is incremented.  Exceptions are caught and
        logged without aborting the pipeline.

    **Nested helper – process_split(qs_path, sgs_raw, is_test_set, target_count)**
        The main per-question generation loop.

        Args:
            qs_path (str): Path to the AGQA balanced-QA text file for this
                pass (either ``train_balanced.txt`` or ``test_balanced.txt``).
            sgs_raw (dict): Raw scene-graph dictionary as loaded from the
                corresponding pickle file.
            is_test_set (bool): ``True`` when processing the held-out test
                questions; controls which branch of :func:`get_split` is
                consulted and which label appears in progress messages.
            target_count (int): Total number of accepted examples (across all
                splits touched by this call) at which the inner loop stops
                early.

        The function proceeds as follows:

        *SG normalisation*
            Every video in ``sgs_raw`` is normalised via
            :func:`normalize_video_sg` and a corresponding
            :class:`NegativePools` object is constructed.  Both results are
            cached in ``sgs_norm`` and ``pools_cache`` respectively so that
            the normalisation work is not repeated per question.

        *Round-robin generator selection*
            For each question ``(qid, qdata)`` yielded by
            :func:`stream_questions`, up to ``len(generators)`` generators
            are tried in sequence starting from the current ``gen_idx``.
            After each attempt ``gen_idx`` is advanced modulo the number of
            generators.  This ensures that no single generator dominates
            when the target count is reached early.

        *Hash-based deduplication*
            Before the validity gate, the sorted proposition texts are
            joined and hashed.  If the resulting hash is already in
            ``seen_hashes`` the group is counted as a duplicate and
            discarded without calling :func:`validate_example`.

        *Validation gate*
            :func:`validate_example` is called on each candidate that
            passes deduplication.  If it returns ``is_valid=False``, all
            returned error codes are tallied in
            ``stats["rejection_reasons"]`` and the group is skipped.

        *Buffer accumulation*
            Accepted groups are appended to ``buffers[split]`` and
            :func:`flush_buffer` is called immediately (with
            ``force=False``) so that shards are written as soon as the
            threshold is reached.

        *Stats tracking*
            After acceptance, the following per-group metrics are updated:

            * Task type, reasoning family, generator family, complexity,
              k-count, split, source video, evidence signature.
            * Unique proposition-text hash set and query-template ID set.
            * Per-proposition truth-state counter and logical-NONE counters.
            * **Temporal subtypes**: if ``group.reasoning.family`` is
              ``"temporal"``, the sub-category is resolved as:

              - ``"multihop"`` when ``"multihop"`` appears in
                ``group.reasoning.type``.
              - The ``temporal_relation`` of the first TRUE temporal
                proposition otherwise.
              - ``"unknown"`` if no such proposition exists.

              ``temporal_hops`` is also updated from ``group.reasoning.hops``.
            * **Grounding subtypes**: if ``group.reasoning.family`` is
              ``"grounding"``, either ``"identification"`` or
              ``"verification"`` is incremented based on whether the
              corresponding substring appears in ``group.reasoning.type``.

    **Phase 6 – Generation passes**
        :func:`process_split` is called twice:

        1. Train / val pass — ``train_balanced.txt`` × ``train_sgs_raw``,
           target 800 000 (currently set to 8 000 for development).
        2. Test pass — ``test_balanced.txt`` × ``test_sgs_raw``,
           target 200 000 (currently set to 2 000 for development).

    **Phase 7 – Teardown and report generation**
        After both passes complete:

        * Any remaining in-memory buffer rows are force-flushed to disk.
        * Five consistency assertions verify that accepted-count totals agree
          across different ``stats`` dimensions:

          - ``sum(task_types)     == total_accepted``
          - ``sum(reasoning_families) == total_accepted``
          - ``sum(split_counts)   == total_accepted``
          - ``sum(k_counts)       == total_accepted``
          - ``sum(temporal_subtypes) == reasoning_families["temporal"]``

        * Four Markdown reports are written to the working directory (see
          module docstring for content descriptions).
    """
    # Seeded RNG instance used for all shuffle operations in this function so
    # that results are deterministic regardless of global random state.
    rng = random.Random(SEED)

    # -----------------------------------------------------------------------
    # Phase 1 – Generator registration
    # Eight generators are registered in a fixed order.  The round-robin
    # scheduler in process_split() will cycle through this list so each
    # generator gets equal opportunity across the question stream.
    # -----------------------------------------------------------------------
    generators = [
        ("orig_qa", OriginalQAGenerator()),           # Wraps original AGQA QA pairs as propositions
        ("obj_exist", ObjectExistenceGenerator()),    # Generates object-existence verification groups
        ("rel_ver", RelationVerificationGenerator()), # Generates spatial-relation verification groups
        ("rel_set", RelationSetGenerator()),          # Generates relation-set membership groups
        ("grounding", GroundingGenerator()),          # Generates visual-grounding identification / verification groups
        ("temporal", ActionTemporalGenerator()),      # Generates temporal-ordering / action-duration groups
        ("compositional", CompositionalGenerator()),  # Generates multi-hop compositional reasoning groups
        ("attributes", AttributeGenerator())          # Generates attribute (colour, size, state) verification groups
    ]

    # -----------------------------------------------------------------------
    # Phase 2 – Scene-graph loading
    # -----------------------------------------------------------------------
    print("Loading AGQA Scene Graphs...", flush=True)
    train_sg_path = os.path.join(DATA_ROOT, "AGQA_scene_graphs", "AGQA_train_stsgs.pkl")
    test_sg_path  = os.path.join(DATA_ROOT, "AGQA_scene_graphs", "AGQA_test_stsgs.pkl")

    with open(train_sg_path, 'rb') as f:
        train_sgs_raw = pickle.load(f)  # dict: video_id -> raw STSG
    with open(test_sg_path, 'rb') as f:
        test_sgs_raw = pickle.load(f)   # dict: video_id -> raw STSG

    # -----------------------------------------------------------------------
    # Phase 3 – Train / val split (90 / 10, seeded)
    # -----------------------------------------------------------------------
    train_vids = list(train_sgs_raw.keys())
    test_vids  = set(test_sgs_raw.keys())

    # Sanity check: no video should appear in both train and test sources.
    assert not test_vids.intersection(set(train_vids)), "Test videos found in Train!"

    # Reproducibly shuffle training video IDs and carve out the first 10 % as
    # the validation holdout.
    rng.shuffle(train_vids)
    num_val       = int(len(train_vids) * 0.1)
    val_vids      = set(train_vids[:num_val])           # 10 % holdout
    train_vids_set = set(train_vids[num_val:])          # remaining 90 %

    def get_split(vid, is_test_set=False):
        """Map a video ID to its dataset split name.

        Args:
            vid (str): The video identifier to look up.
            is_test_set (bool): When ``True`` the function checks the test
                video universe instead of the train / val universes.

        Returns:
            str or None: One of ``"train"``, ``"val"``, ``"test"``, or
                ``None`` if the video does not belong to the current pass
                (e.g. a train video encountered during the test pass).
        """
        if is_test_set:
            # During the test pass only test-set videos are valid.
            return "test" if vid in test_vids else None
        # During the train/val pass, route to the correct sub-split.
        if vid in val_vids:
            return "val"
        if vid in train_vids_set:
            return "train"
        return None  # Video does not belong to any known split for this pass.

    # -----------------------------------------------------------------------
    # Phase 4 – Stats dictionary initialisation
    # Every metric collected during generation is stored here so that a single
    # object can be passed around without additional arguments.
    # -----------------------------------------------------------------------
    stats = {
        # Per-generator candidate / accept / reject tallies
        "candidates":          Counter(),  # Raw groups produced per generator
        "accepted":            Counter(),  # Groups that passed validation per generator
        "rejected":            Counter(),  # Groups that failed validation per generator
        "rejection_reasons":   Counter(),  # Validation error codes and their frequencies
        # Taxonomy counters (one increment per accepted group)
        "task_types":          Counter(),  # Values of group.task_type
        "reasoning_families":  Counter(),  # Values of group.reasoning.family
        "generator_families":  Counter(),  # Values of group.provenance.generator_family
        "complexity":          Counter(),  # Values of group.reasoning.complexity
        "k_counts":            Counter(),  # Values of group.num_propositions (k-way choice size)
        # Proposition-level truth state distribution
        "truth_states":        Counter(),  # "TRUE" / "FALSE" / other across all propositions
        # Logical-NONE proposition tracking (a special distractor class)
        "none_included":       0,          # Total propositions with canonical_type == "logical_none"
        "none_true":           0,          # Subset of none_included where label == 1
        "none_false":          0,          # Subset of none_included where label != 1
        # Template diversity
        "query_templates":     set(),      # Distinct query_template_id values seen
        # Temporal-reasoning breakdowns
        "temporal_subtypes":   Counter(),  # e.g. "before", "after", "during", "multihop", "unknown"
        "temporal_hops":       Counter(),  # Number of reasoning hops per temporal group
        # Grounding-reasoning breakdowns
        "grounding_subtypes":  Counter(),  # "identification" vs "verification"
        # Coverage / diversity metrics
        "source_videos":       Counter(),  # Examples generated per source video
        "split_counts":        Counter(),  # Examples generated per split name
        "examples_per_video":  Counter(),  # Per-video example counts (semantic alias of source_videos)
        "examples_per_evidence": Counter(), # Keyed by get_signature(); measures evidence reuse
        # Uniqueness metrics
        "unique_prop_signatures": set(),   # Hashed (type,subj,pred,obj,trel,ea,eb,pol) tuples
        "unique_option_sets":     set(),   # Hashed sorted-option-text tuples (group-level)
        # Global deduplication counters
        "duplicate_count":     0,          # Groups discarded by the hash-dedup gate
        "total_candidates":    0           # All raw groups produced before any filtering
    }

    # Global set of seen group hashes for deduplication across the entire run.
    # Populated from existing Parquet shards during checkpoint loading (Phase 5)
    # and extended with newly accepted groups during generation.
    seen_hashes = set()

    # Collects up to 5 example groups per generation_rule for SAMPLE_REPORT.md.
    by_rule = {}

    # Per-split in-memory accumulation buffers and their corresponding Parquet
    # shard index counters.
    buffers      = {"train": [], "val": [], "test": []}
    shard_indices = {"train": 0,  "val": 0,  "test": 0}

    # Running totals for progress reporting and final consistency checks.
    total_accepted = 0

    # generated_count: total accepted examples (including those loaded from
    #   existing checkpoints).  Used to check against target_count.
    # flushed_count:   total examples successfully written to Parquet shards
    #   during the current run.
    # written_count:   total examples recovered from existing checkpoint shards.
    generated_count = 0
    flushed_count   = 0
    written_count   = 0

    # -----------------------------------------------------------------------
    # Phase 5 – Checkpoint loading
    # Scan existing Parquet shards to resume a previous run.  For each shard
    # we (a) advance the shard index counter and (b) hash all existing
    # proposition groups into seen_hashes to prevent re-generation.
    # -----------------------------------------------------------------------
    print("Loading checkpoints...", flush=True)
    import pyarrow.parquet as pq
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATA_ROOT, "data", "generated", split)
        if os.path.exists(split_dir):
            for file in sorted(os.listdir(split_dir)):
                if file.startswith("part-") and file.endswith(".parquet"):
                    # Derive the numeric index from the filename and advance
                    # the counter so the next shard gets a unique index.
                    idx = int(file.replace("part-", "").replace(".parquet", ""))
                    shard_indices[split] = max(shard_indices[split], idx + 1)

                    # Read only the propositions column (avoid loading all
                    # columns for large shards) and add each group's hash to
                    # the deduplication cache.
                    file_path = os.path.join(split_dir, file)
                    table = pq.read_table(file_path, columns=["propositions"])
                    for row in table["propositions"].to_pylist():
                        texts = [p["text"] for p in row]
                        # Hash of sorted proposition texts — same fingerprint
                        # used during online generation.
                        seen_hashes.add(hash("".join(sorted(texts))))
                        written_count += 1

    print(f"Loaded {written_count} previous examples into deduplication cache.", flush=True)
    # Treat already-written examples as already generated so target_count
    # arithmetic remains correct.
    generated_count = written_count
    flushed_count   = written_count

    # -----------------------------------------------------------------------
    # Nested helper: flush_buffer
    # -----------------------------------------------------------------------
    def flush_buffer(split: str, force=False):
        """Flush the in-memory buffer for *split* to a Parquet shard.

        The flush is skipped unless the buffer has reached ``SHARD_SIZE``
        rows, or unless ``force=True`` is passed (used at pipeline teardown
        to write any partial buffer).  On success the buffer is cleared and
        ``shard_indices[split]`` is incremented so the next shard receives
        a unique filename.

        Args:
            split (str): Dataset split whose buffer should be flushed.
                Must be one of ``"train"``, ``"val"``, or ``"test"``.
            force (bool): When ``True``, flush even if the buffer has fewer
                than ``SHARD_SIZE`` rows.  Defaults to ``False``.

        Side Effects:
            * Writes a Parquet file to
              ``<DATA_ROOT>/data/generated/<split>/part-NNNNNN.parquet``.
            * Increments ``flushed_count`` by the number of rows flushed.
            * Clears ``buffers[split]``.
            * Increments ``shard_indices[split]``.
            * Logs a success or failure message to stdout.
        """
        nonlocal flushed_count
        n = len(buffers[split])
        # Skip if the buffer has not reached the threshold (unless forced).
        if not force and n < SHARD_SIZE:
            return
        # Nothing to write even if forced.
        if n == 0:
            return

        # Ensure the output directory exists before writing.
        os.makedirs(os.path.join(DATA_ROOT, "data", "generated", split), exist_ok=True)
        shard_idx = shard_indices[split]
        filename  = f"part-{shard_idx:06d}.parquet"
        out_path  = os.path.join(DATA_ROOT, "data", "generated", split, filename)

        try:
            write_shard(buffers[split], out_path)
            print(f"Flushed {n} examples to {out_path}", flush=True)
            flushed_count       += n
            buffers[split]       = []          # Clear the buffer after a successful write.
            shard_indices[split] += 1          # Advance so the next shard has a unique index.
        except Exception as e:
            # Log the failure but do not abort; the examples remain in the
            # buffer and will be retried on the next flush attempt.
            print(f"Failed to flush {n} examples to {out_path}: {e}", flush=True)

    # -----------------------------------------------------------------------
    # Nested helper: process_split
    # -----------------------------------------------------------------------
    def process_split(qs_path, sgs_raw, is_test_set, target_count):
        """Run the generation loop for one data split (train/val or test).

        Normalises all scene graphs, constructs negative pools, then streams
        questions from *qs_path* and applies each generator in round-robin
        order.  Accepted groups are deduplicated, validated, and accumulated
        into the appropriate per-split buffer.

        Args:
            qs_path (str): Filesystem path to the AGQA balanced-QA text file
                (``train_balanced.txt`` or ``test_balanced.txt``).
            sgs_raw (dict): Mapping of ``video_id -> raw_scene_graph`` as
                unpickled from the corresponding STSG file.
            is_test_set (bool): ``True`` when processing the test QA file.
                Affects split routing via :func:`get_split` and the label
                displayed in progress messages.
            target_count (int): The total ``generated_count`` value at which
                the inner loop terminates early.  This threshold applies
                globally (across all splits touched by this call), not per
                split.

        Side Effects:
            Mutates all outer-scope variables captured via ``nonlocal``:
            ``generated_count``, ``total_accepted``, ``written_count``,
            ``stats``, ``seen_hashes``, ``buffers``, ``shard_indices``,
            ``by_rule``.
        """
        nonlocal generated_count, total_accepted, written_count
        nonlocal total_accepted  # Explicitly re-declare to satisfy linter; same reference.

        # -------------------------------------------------------------------
        # SG normalisation phase
        # Iterate over every video in sgs_raw, normalise its scene graph, and
        # build a NegativePools object.  Both artefacts are cached by video ID
        # so that the same work is not repeated across different questions for
        # the same video.
        # -------------------------------------------------------------------
        print(f"\\nNormalizing Scene Graphs for {'TEST' if is_test_set else 'TRAIN/VAL'}...", flush=True)
        sgs_norm    = {}   # video_id -> NormalisedSceneGraph
        pools_cache = {}   # video_id -> NegativePools

        vids_needed = list(sgs_raw.keys())
        # Shuffle video order to avoid systematic bias when target_count is
        # reached before all videos are processed.
        rng.shuffle(vids_needed)
        # vids_needed = vids_needed[:1500]  # No limit for 1M generation

        for vid in tqdm(vids_needed):
            raw_sg         = sgs_raw[vid]
            norm_sg        = normalize_video_sg(vid, raw_sg)
            sgs_norm[vid]  = norm_sg
            # NegativePools pre-computes hard-negative candidate sets (e.g.
            # distractor objects and relations) that generators use to
            # construct plausible but incorrect propositions.
            pools_cache[vid] = NegativePools(norm_sg)

        print(f"Generating {'TEST' if is_test_set else 'TRAIN/VAL'} examples...", flush=True)

        # Round-robin generator index — shared across all questions in this
        # call so that the cycling is continuous, not reset per question.
        gen_idx        = 0
        added_count    = 0  # Local accepted-example counter for this call (unused externally).
        start_time     = time.time()
        last_print_time = start_time

        # -------------------------------------------------------------------
        # Main question loop
        # stream_questions yields (qid, qdata) pairs lazily from the file so
        # that the full QA file never needs to be held in memory at once.
        # -------------------------------------------------------------------
        for qid, qdata in stream_questions(qs_path):
            # Stop as soon as the global accepted count has reached the target.
            if generated_count >= target_count:
                break

            vid   = qdata.get("video_id")
            split = get_split(vid, is_test_set)
            if not split:
                # This video does not belong to any split for the current pass.
                continue

            # Retrieve the pre-computed normalised SG and negative pools.
            sg    = sgs_norm.get(vid)
            pools = pools_cache.get(vid)
            if not sg:
                # Some videos may lack scene graphs; skip silently.
                continue

            # ---------------------------------------------------------------
            # Round-robin generator selection
            # Attempt up to len(generators) generators for this question, then
            # move on to the next question regardless.  This prevents any
            # single generator from consuming all attempts for a given question.
            # ---------------------------------------------------------------
            for _ in range(len(generators)):
                if generated_count >= target_count:
                    break

                gen_name, gen_obj = generators[gen_idx]
                # Advance the index modulo the list length for the next iteration.
                gen_idx = (gen_idx + 1) % len(generators)

                # All generators except orig_qa require a valid scene graph.
                if gen_name != "orig_qa" and not sg:
                    continue

                # -----------------------------------------------------------
                # Inner generation loop
                # Each generator may yield zero or more PropositionGroup
                # candidates for the given (question, scene-graph) pair.
                # -----------------------------------------------------------
                for group in gen_obj.generate(qid, qdata, sg, split, rng, pools):
                    # Tally raw candidate before any filtering.
                    stats["candidates"][gen_name] += 1
                    stats["total_candidates"]     += 1

                    # -------------------------------------------------------
                    # Hash-based deduplication gate
                    # Compute a fingerprint from the sorted proposition texts
                    # and discard any group whose fingerprint is already known.
                    # This prevents re-generating examples present in previous
                    # checkpoint shards as well as within-run duplicates.
                    # -------------------------------------------------------
                    group_hash = hash("".join(sorted(p.text for p in group.propositions)))
                    if group_hash in seen_hashes:
                        stats["duplicate_count"] += 1
                        continue  # Already seen — skip without validation.
                    # Register the new fingerprint immediately to block
                    # duplicates that appear later in the same run.
                    seen_hashes.add(group_hash)

                    # -------------------------------------------------------
                    # Validation gate
                    # validate_example enforces schema constraints (label
                    # counts, proposition counts, etc.) and returns a boolean
                    # together with a list of error-code strings.
                    # -------------------------------------------------------
                    is_valid, errors = validate_example(group)
                    if not is_valid:
                        stats["rejected"][gen_name] += 1
                        for err in errors:
                            stats["rejection_reasons"][err] += 1
                        continue  # Discard invalid group; do not buffer.

                    # Accepted!
                    stats["accepted"][gen_name] += 1
                    generated_count += 1
                    total_accepted  += 1

                    # Periodic progress report: log every ~1 % of target or
                    # whenever more than 10 seconds have elapsed since the
                    # last log line, whichever comes first.
                    if (generated_count - written_count) % (target_count // 100 or 1) == 0 or time.time() - last_print_time > 10:
                        elapsed = time.time() - start_time
                        rate    = (generated_count - written_count) / elapsed if elapsed > 0 else 0
                        percent = ((generated_count - written_count) / target_count) * 100
                        print(f"[{percent:5.1f}%] Generated {generated_count}/{target_count} ({rate:.1f} records/sec) - {elapsed:.1f}s elapsed", flush=True)
                        last_print_time = time.time()

                    # Buffer write — appends the group and triggers a shard
                    # flush if the buffer has reached SHARD_SIZE.
                    buffers[split].append(group)
                    flush_buffer(split, force=False)

                    # Keep up to 5 example groups per generation_rule so that
                    # SAMPLE_REPORT.md has representative illustrations.
                    rule = group.provenance.generation_rule
                    if rule not in by_rule:
                        by_rule[rule] = []
                    if len(by_rule[rule]) < 5:
                        by_rule[rule].append(group)

                    # -------------------------------------------------------
                    # Stats accumulation — taxonomy and coverage metrics
                    # -------------------------------------------------------
                    stats["task_types"][group.task_type]                          += 1
                    stats["reasoning_families"][group.reasoning.family]           += 1
                    stats["generator_families"][group.provenance.generator_family] += 1
                    stats["complexity"][group.reasoning.complexity]               += 1
                    stats["k_counts"][group.num_propositions]                     += 1
                    stats["split_counts"][split]                                  += 1
                    stats["source_videos"][vid]                                   += 1

                    stats["examples_per_video"][vid] += 1
                    ev_sig = get_signature(group)
                    stats["examples_per_evidence"][ev_sig] += 1

                    # Unique option-set hash (group level).
                    stats["unique_option_sets"].add(hash(tuple(sorted(p.text for p in group.propositions))))
                    if group.query_template_id:
                        stats["query_templates"].add(group.query_template_id)

                    # Per-proposition statistics.
                    for p in group.propositions:
                        stats["truth_states"][p.truth_state] += 1
                        # Build a canonical proposition signature tuple from
                        # all semantics fields that distinguish one proposition
                        # type from another.
                        psig = (
                            p.semantics.canonical_type,
                            p.semantics.subject,
                            p.semantics.predicate,
                            p.semantics.object,
                            p.semantics.temporal_relation,
                            p.semantics.event_a,
                            p.semantics.event_b,
                            p.semantics.polarity
                        )
                        stats["unique_prop_signatures"].add(hash(psig))

                        # Track the special logical-NONE distractors that
                        # represent "none of the above" answer options.
                        if p.semantics.canonical_type == "logical_none":
                            stats["none_included"] += 1
                            if p.label == 1:
                                stats["none_true"]  += 1
                            else:
                                stats["none_false"] += 1

                    # -------------------------------------------------------
                    # Temporal subtype breakdown
                    # For temporal-family groups, classify into finer
                    # sub-categories for the diversity report.
                    # -------------------------------------------------------
                    if group.reasoning.family == "temporal":
                        if "multihop" in group.reasoning.type:
                            # Multi-hop questions span more than one temporal
                            # reasoning step; record as a single bucket.
                            stats["temporal_subtypes"]["multihop"] += 1
                        else:
                            # Single-hop: use the temporal_relation of the
                            # first TRUE temporal proposition as the subtype.
                            found = False
                            for p in group.propositions:
                                if p.semantics.canonical_type == "temporal" and p.truth_state == "TRUE":
                                    stats["temporal_subtypes"][p.semantics.temporal_relation] += 1
                                    found = True
                                    break
                            if not found:
                                # No TRUE temporal proposition found; use a
                                # catch-all bucket to keep the sum consistent.
                                stats["temporal_subtypes"]["unknown"] += 1
                        # Record hop depth independently of subtype.
                        stats["temporal_hops"][group.reasoning.hops] += 1

                    # -------------------------------------------------------
                    # Grounding subtype breakdown
                    # -------------------------------------------------------
                    if group.reasoning.family == "grounding":
                        if "identification" in group.reasoning.type:
                            # The question asks the model to identify an
                            # object / region from its description.
                            stats["grounding_subtypes"]["identification"] += 1
                        elif "verification" in group.reasoning.type:
                            # The question asks the model to verify a
                            # described grounding statement.
                            stats["grounding_subtypes"]["verification"] += 1

    # -----------------------------------------------------------------------
    # Phase 6 – Generation passes
    # -----------------------------------------------------------------------
    train_qs_path = os.path.join(DATA_ROOT, "AGQA_balanced", "AGQA_balanced", "train_balanced.txt")
    test_qs_path  = os.path.join(DATA_ROOT, "AGQA_balanced", "AGQA_balanced", "test_balanced.txt")

    # 1. Train/Val Gen (800000)
    process_split(train_qs_path, train_sgs_raw, False, 8000)
    # 2. Test Gen (200000)
    process_split(test_qs_path,  test_sgs_raw,  True,  2000)

    # -----------------------------------------------------------------------
    # Phase 7 – Teardown, validation assertions, and report generation
    # -----------------------------------------------------------------------
    print("\\nValidating Constraints and Generating Reports...", flush=True)

    # Force-flush any partial buffers that did not reach SHARD_SIZE on their own.
    for split in buffers:
        flush_buffer(split, force=True)

    # ------------------------------------------------------------------
    # Consistency assertions
    # Verify that all accepted examples were counted by every relevant
    # stats dimension.  A mismatch indicates a bookkeeping bug.
    # ------------------------------------------------------------------
    assert sum(stats["task_types"].values())        == total_accepted
    assert sum(stats["reasoning_families"].values()) == total_accepted
    assert sum(stats["split_counts"].values())       == total_accepted
    assert sum(stats["k_counts"].values())           == total_accepted
    # Every temporal group must have been classified into exactly one subtype.
    assert sum(stats["temporal_subtypes"].values()) == stats["reasoning_families"]["temporal"]

    # ------------------------------------------------------------------
    # Report 1 – SAMPLE_REPORT.md
    # Human-readable illustrations: up to 5 groups per generation rule,
    # showing proposition texts, truth labels, and source metadata.
    # ------------------------------------------------------------------
    with open("SAMPLE_REPORT.md", "w") as f:
        f.write("# V5 Logical Correctness Sample Report\\n\\n")
        for rule, groups in by_rule.items():
            f.write(f"## Generation Rule: {rule}\\n")
            for group in groups:
                f.write(f"**Example ID:** {group.example_id} | **Split:** {group.split}\\n")
                f.write(f"**Task Type:** {group.task_type} | **K:** {group.num_propositions}\\n")
                if group.query_text:
                    f.write(f"**Query:** {group.query_text}\\n")
                f.write(f"**Options:**\\n")
                for p in group.propositions:
                    l = p.label if p.label is not None else "?"
                    f.write(f"  - [{l}] {p.text} (Truth: {p.truth_state})\\n")
                f.write("\\n")

    # ------------------------------------------------------------------
    # Report 2 – VALIDATION_REPORT.md
    # Macro-level accept / reject / duplicate statistics and a breakdown
    # of the most common validation error codes.
    # ------------------------------------------------------------------
    with open("VALIDATION_REPORT.md", "w") as f:
        f.write("# V5 Validation Report\\n\\n")
        f.write(f"- Total Candidates: {stats['total_candidates']}\\n")
        f.write(f"- Total Accepted: {total_accepted}\\n")
        f.write(f"- Total Rejected: {sum(stats['rejected'].values())}\\n")
        f.write(f"- Duplicates Caught: {stats['duplicate_count']}\\n\\n")
        f.write("### Rejection Reasons\\n")
        for reason, count in stats["rejection_reasons"].items():
            f.write(f"- {count}: {reason}\\n")

    # ------------------------------------------------------------------
    # Report 3 – SPLIT_REPORT.md
    # Video-universe sizes for each split and the actual example counts
    # generated within each split.
    # ------------------------------------------------------------------
    with open("SPLIT_REPORT.md", "w") as f:
        f.write("# Dataset Split Report\\n\\n")
        f.write(f"- Train Videos (Universe): {len(train_vids_set)}\\n")
        f.write(f"- Val Videos (Universe): {len(val_vids)}\\n")
        f.write(f"- Test Videos (Universe): {len(test_vids)}\\n\\n")
        f.write("### Generated Examples by Split\\n")
        for s, c in stats["split_counts"].items():
            f.write(f"- {s.upper()}: {c}\\n")

    # ------------------------------------------------------------------
    # Report 4 – DIVERSITY_REPORT.md
    # Diversity and production-readiness metrics: coverage ratios,
    # uniqueness counts, truth-state balance, and subtype distributions.
    # ------------------------------------------------------------------
    with open("DIVERSITY_REPORT.md", "w") as f:
        f.write("# Diversity & Production Audit\\n\\n")
        # Fraction of all raw candidates that were exact duplicates.
        dup_ratio      = stats["duplicate_count"] / stats["total_candidates"] if stats["total_candidates"] else 0
        # Fraction of accepted examples that share an evidence signature with
        # at least one other accepted example (i.e. evidence reuse rate).
        ev_reuse_ratio = 1.0 - (len(stats["examples_per_evidence"]) / total_accepted)

        f.write(f"- Unique Source Videos Used: {len(stats['source_videos'])}\\n")
        f.write(f"- Average Examples / Video: {total_accepted / len(stats['source_videos']):.2f}\\n")
        f.write(f"- Average Examples / Evidence Signature: {total_accepted / len(stats['examples_per_evidence']):.2f}\\n")
        f.write(f"- Unique Proposition Signatures: {len(stats['unique_prop_signatures'])}\\n")
        f.write(f"- Unique Option Sets (Group level): {len(stats['unique_option_sets'])}\\n")
        f.write(f"- Unique Query Templates Used: {len(stats['query_templates'])}\\n")
        f.write(f"- Duplicate Catch Ratio: {dup_ratio:.2%}\\n")
        f.write(f"- Evidence Reuse Ratio: {ev_reuse_ratio:.2%}\\n\\n")

        f.write("### General Statistics\\n")
        f.write(f"- Truth States: {dict(stats['truth_states'])}\\n")
        f.write(f"- Logical NONE Included: {stats['none_included']} (True={stats['none_true']}, False={stats['none_false']})\\n")
        f.write(f"- Task Types: {dict(stats['task_types'])}\\n")
        f.write(f"- Reasoning Families: {dict(stats['reasoning_families'])}\\n")
        f.write(f"- K Counts: {dict(stats['k_counts'])}\\n")
        f.write(f"- Temporal Subtypes: {dict(stats['temporal_subtypes'])}\\n")
        f.write(f"- Temporal Hops: {dict(stats['temporal_hops'])}\\n")

    print("ALL REPORTS GENERATED!", flush=True)


if __name__ == "__main__":
    main()
