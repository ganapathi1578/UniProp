"""Config-driven dataset pipeline for the UniProp framework.

This module contains :func:`run_dataset_pipeline`, the legacy entry point
that orchestrates the full record-by-record transformation from a raw source
dataset to sharded Parquet output files.

**Pipeline stages** (in order)

1. **Configuration resolution** – merge split-specific overrides from the
   ``"splits"`` block into the top-level config dict.
2. **Adapter construction** – look up the registered
   :class:`~src.datasets.base.BaseDatasetAdapter` subclass by the
   ``"dataset"`` config key and instantiate it.
3. **Output directory preparation** – create the output directory tree
   (``<root_dir>[/<scale>]/<split>/``) if it does not already exist.
4. **Record iteration** – stream records from the adapter one at a time,
   optionally capping the total at ``max_records`` or a config-level
   ``target_records`` limit.
5. **Candidate generation** – for each record look up the registered
   :class:`~src.candidates.base.CandidateGenerator` by
   :attr:`~src.datasets.normalized.NormalizedQA.answer_type` and call
   ``generator.generate(record, context, max_k=max_k)``.
6. **Shuffle** – randomly reorder the generated ``(option, label)`` pairs
   while preserving label alignment and verifying consistency.
7. **Schema construction** – build a
   :class:`~src.schema.PropositionGroup` containing
   :class:`~src.schema.Proposition` objects for every option, annotated
   with truth states (``"TRUE"`` / ``"FALSE"`` / ``"UNKNOWN"``).
8. **Validation** – call
   :func:`~src.validation.validators.validate_example` and reject
   malformed groups.
9. **Shard writing** – buffer validated groups and flush to a Parquet shard
   (``part-NNNNNN.parquet``) each time the buffer reaches ``shard_size``.
10. **Audit report** – print a structured breakdown of processed, succeeded,
    and rejected record counts by answer domain.

**Rejection tracking**

Every record that cannot be fully processed is counted as *rejected*.
Rejection reasons are bucketed into a ``defaultdict`` and printed at the
end of the run:

* ``"unsupported_domain"`` – no generator registered for the answer type.
* ``str(ValueError)`` (lowercased) – generator raised :class:`ValueError`
  (e.g. "not enough distractors").
* ``"other_rejection"`` – generator raised any other exception.
* ``"malformed_source"`` – the constructed group failed validation.
"""

import os
import random
from typing import Iterator
from collections import defaultdict, Counter
from src.datasets.registry import dataset_registry
import src.datasets.agqa  # Initialize registry
from src.candidates.registry import candidate_registry
from src.reasoning.semantic_types import MissingEvidenceError, EvaluatorUnsupportedError
import src.candidates     # Initialize registry
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.validation.validators import validate_example
from src.storage.parquet_writer import write_shard


def run_dataset_pipeline(config: dict, split: str, max_records: int = None):
    """Execute the full UniProp optionisation pipeline for one dataset split.

    Loads a dataset adapter, iterates every source record, generates
    candidate answer propositions, validates and buffers the resulting
    :class:`~src.schema.PropositionGroup` objects, writes them to sharded
    Parquet files, and finally prints a structured audit report to stdout.

    Args:
        config: Nested configuration dictionary, typically produced by
            ``yaml.safe_load`` from a dataset YAML file.  Expected keys:

            * ``"dataset"`` *(str)* – registry name of the dataset adapter
              (e.g. ``"agqa_balanced"``).
            * ``"splits"`` *(dict, optional)* – per-split overrides; if the
              key *split* exists, its ``"source"`` sub-dict is merged into
              the top-level config before adapter construction.
            * ``"source"`` *(dict)* – paths to QA and scene-graph files
              (merged from ``splits`` when applicable).
            * ``"output"`` *(dict, optional)* – output settings:

              - ``"root_dir"`` *(str)* – base output directory
                (default ``"data/generated/<dataset>"``).
              - ``"scale"`` *(str, optional)* – inserts an extra path
                component between *root_dir* and *split* (e.g. ``"10k"``).
              - ``"shard_size"`` *(int)* – number of groups per Parquet
                shard (default ``10000``).

            * ``"selection"`` *(dict, optional)* – record-selection policy:

              - ``"mode"`` *(str)* – ``"all"`` (process everything) or
                ``"target_size"`` (stop when *target_records* successes are
                reached). Default ``"all"``.
              - ``"target_records"`` *(int, optional)* – number of
                successfully optionised records to produce before stopping.
                Only respected when ``mode == "target_size"``.

            * ``"generation"`` *(dict, optional)* – generation hyper-params:

              - ``"seed"`` *(int)* – RNG seed for candidate shuffling
                (default ``42``).
              - ``"max_k"`` *(int)* – maximum number of candidate options
                to generate per record (default ``40``).

            * ``"scenegraph_path"`` *(str, optional)* – path to scene-graph
              data, used to locate the scene-graph directory for
              :class:`~src.reasoning.evidence.TruthEvaluator`.

        split: Name of the dataset split to process (e.g. ``"train"``,
            ``"val"``, ``"test"``).  Used to select the matching entry in
            the ``"splits"`` config block and as a path component in the
            output directory.

        max_records: If not ``None``, stop after reading this many records
            from the adapter (0-indexed; the loop breaks when
            ``idx >= max_records``).  This is the CLI-level override intended
            for smoke testing and takes precedence over the adapter's natural
            end-of-data.  When ``None`` the full dataset is processed (subject
            to the ``selection.mode`` / ``target_records`` limits).

    Returns:
        tuple[int, int]: A 2-tuple ``(success_count, rejected_count)`` where

        * ``success_count`` – number of records that were successfully
          optionised, validated, and written to Parquet.
        * ``rejected_count`` – number of records that were skipped for any
          reason (unsupported domain, generator error, validation failure).

    Side Effects:
        * Creates output directories under *root_dir* if they do not exist.
        * Writes one or more ``part-NNNNNN.parquet`` shard files to the
          output directory.
        * Prints per-record rejection messages and a full audit report to
          stdout.

    Raises:
        ValueError: If ``config["dataset"]`` is not registered in
            :data:`~src.datasets.registry.dataset_registry`.
        FileNotFoundError: If a source data file referenced in *config*
            cannot be found on disk.
    """
    # ------------------------------------------------------------------ #
    # 1. Split-specific configuration resolution                           #
    # ------------------------------------------------------------------ #
    # If the config contains a "splits" block with an entry for this split,
    # merge its "source" sub-dict into the top-level config so that the
    # adapter sees the correct file paths for the requested split.
    split_cfg = config.get("splits", {}).get(split, {})
    if "source" in split_cfg:
        config["source"] = split_cfg["source"]
        
    # ------------------------------------------------------------------ #
    # 2. Adapter construction                                              #
    # ------------------------------------------------------------------ #
    dataset_name = config.get("dataset", "agqa_balanced")
    AdapterClass = dataset_registry.get_adapter(dataset_name)
    adapter = AdapterClass(config)
    
    # ------------------------------------------------------------------ #
    # 3. Output directory preparation                                      #
    # ------------------------------------------------------------------ #
    output_cfg = config.get("output", {})
    root_dir = output_cfg.get("root_dir", f"data/generated/{dataset_name}")
    scale = output_cfg.get("scale", None)
    
    # Insert optional "scale" sub-directory (e.g. "10k", "100k") when present.
    if scale:
        out_dir = os.path.join(root_dir, scale, split)
    else:
        out_dir = os.path.join(root_dir, split)
        
    os.makedirs(out_dir, exist_ok=True)
    
    # ------------------------------------------------------------------ #
    # 4. Record iteration setup                                            #
    # ------------------------------------------------------------------ #
    records = adapter.load_records()
    
    selection_cfg = config.get("selection", {})
    # "all": process every record; "target_size": stop after N successes.
    mode = selection_cfg.get("mode", "all")
    target_records = selection_cfg.get("target_records", None)
    
    # CLI-level hard cap on the number of source records read.
    cli_limit = max_records
    
    # Buffer of validated PropositionGroup objects waiting to be written.
    groups = []
    shard_idx = 0  # Zero-based shard counter used to name output files.
    shard_size = output_cfg.get("shard_size", config.get("shard_size", 10000))
    
    # ------------------------------------------------------------------ #
    # Generation hyper-parameters                                          #
    # ------------------------------------------------------------------ #
    gen_cfg = config.get("generation", {})
    # Seeded RNG ensures reproducible candidate shuffling across runs.
    rng = random.Random(gen_cfg.get("seed", config.get("seed", 42)))
    # Maximum number of candidate options to generate per record.
    max_k = gen_cfg.get("max_k", config.get("max_k", 40))
    
    # ------------------------------------------------------------------ #
    # Counters and tracking structures                                     #
    # ------------------------------------------------------------------ #
    success_count = 0   # Records fully processed and written to disk.
    rejected_count = 0  # Records skipped for any reason.
    
    # TruthEvaluator requires the directory containing scene-graph files.
    from src.reasoning.evidence import TruthEvaluator
    sg_dir = os.path.dirname(config.get("scenegraph_path", "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl"))
    evaluator = TruthEvaluator(sg_dir)

    # Buckets for the audit report printed at the end of the run.
    rejection_reasons = defaultdict(int)   # reason_string -> count
    domain_counts = defaultdict(int)        # answer_type  -> records seen
    domain_success = defaultdict(int)       # answer_type  -> records succeeded
    domain_true = defaultdict(int)          # answer_type  -> TRUE proposition count
    domain_false = defaultdict(int)         # answer_type  -> FALSE proposition count
    domain_unknown = defaultdict(int)       # answer_type  -> UNKNOWN proposition count

    # ------------------------------------------------------------------ #
    # 5. Main processing loop                                              #
    # ------------------------------------------------------------------ #
    for idx, record in enumerate(records):
        # --- CLI hard cap -----------------------------------------------
        # Stop reading source records once the CLI limit is reached.
        if cli_limit is not None and idx >= cli_limit:
            break
            
        # --- Target-size early exit ------------------------------------
        # When running in "target_size" mode, stop once we have produced
        # enough successfully validated groups.
        if mode == "target_size" and target_records is not None and success_count >= target_records:
            break
            
        # Tally the record against its answer type regardless of outcome.
        domain_counts[record.answer_type] += 1
            
        # --- Generator lookup ------------------------------------------
        # Look up the candidate generator for this record's answer type.
        # A None return means no generator is registered; reject the record.
        generator_class = candidate_registry.get_generator(record.answer_type)
        if not generator_class:
            reason = "unsupported_domain"
            print(f"REJECTED: [{record.source_question_id}] [{record.video_id}] {reason}: {record.answer_type}")
            rejection_reasons[reason] += 1
            rejected_count += 1
            continue
            
        generator = generator_class()
        
        # Scenegraph info passed via evaluator
        context = {"evaluator": evaluator}  
        
        # --- Candidate generation --------------------------------------
        try:
            # Generate candidate options, binary labels (1=correct, 0=distractor),
            # and an overall truth state for the record.
            options, labels, truth_state = generator.generate(record, context, max_k=max_k)
            assert len(options) == len(labels), "Length mismatch between options and labels"
            
            # Record original zip before shuffling to ensure label consistency
            # is verifiable after the shuffle step below.
            original_pairs = set(zip(options, labels))
        except ValueError as e:
            reason = str(e).lower()
            print(f"REJECTED: [{record.source_question_id}] [{record.video_id}] {reason}")
            rejection_reasons[reason] += 1
            rejected_count += 1
            continue
        except (MissingEvidenceError, EvaluatorUnsupportedError) as e:
            reason = str(e) or e.__class__.__name__
            if isinstance(e, MissingEvidenceError): reason = "missing_evidence"
            elif isinstance(e, EvaluatorUnsupportedError): reason = f"evaluator_unsupported:{e}"
            print(f"REJECTED: [{record.source_question_id}] [{record.video_id}] {reason}")
            rejection_reasons[reason] += 1
            rejected_count += 1
            continue
        except Exception as e:
            reason = f"other_rejection"
            print(f"REJECTED: [{record.source_question_id}] [{record.video_id}] Exception: {e}")
            rejection_reasons[reason] += 1
            rejected_count += 1
            continue
            
        # ----------------------------------------------------------------
        # 6. Shuffle options while preserving label alignment
        # ----------------------------------------------------------------
        # Zip options and labels together so that each option stays paired
        # with its correct label, then shuffle the pairs as a unit.
        combined = list(zip(options, labels))
        rng.shuffle(combined)
        options, labels = zip(*combined)
        options = list(options)
        labels = list(labels)
        
        # Verify the same option always receives the same label after shuffle.
        assert set(zip(options, labels)) == original_pairs, "Option/Label consistency corrupted during shuffle"
        
        # ----------------------------------------------------------------
        # 7. Schema construction — build Propositions and PropositionGroup
        # ----------------------------------------------------------------
        propositions = []
        for opt, lbl in zip(options, labels):
            # Map the binary label to a three-way truth state.
            # Label 1 → TRUE.  Label 0 → FALSE, unless the generator
            # signalled UNKNOWN truth (e.g. scene graph missing).
            t_state = "TRUE" if lbl == 1 else ("UNKNOWN" if truth_state == "UNKNOWN" else "FALSE")
            
            # Accumulate per-domain truth-state counts for the audit report.
            if t_state == "TRUE": domain_true[record.answer_type] += 1
            elif t_state == "FALSE": domain_false[record.answer_type] += 1
            else: domain_unknown[record.answer_type] += 1
            
            propositions.append(Proposition(
                text=opt,
                truth_state=t_state,
                label=lbl,
                semantics=PropositionSemantics(
                    canonical_type=record.semantic_type,
                    object=opt # Ensures unique semantic signature per option
                )
            ))
            
        # Assemble the top-level PropositionGroup for this record.
        group = PropositionGroup(
            example_id=f"{dataset_name}_{record.source_question_id}",
            source=dataset_name,
            split=split,
            media_id=record.video_id,
            query_text=record.query,
            query_template_id=None,
            propositions=propositions,
            # Task type is determined by the label distribution:
            # more than one correct option → multi_label;
            # exactly two options → binary; otherwise → single_choice.
            task_type="multi_label" if sum(labels) > 1 else ("binary" if len(options) == 2 else "single_choice"),
            num_propositions=len(propositions),
            reasoning=Reasoning(type=record.reasoning_type, complexity=1, family=record.semantic_type),
            provenance=Provenance(
                source_question_id=record.source_question_id,
                source_question_text=record.query,
                source_program=record.source_program,
                source_answer=record.source_answer,
                scene_graph_id=record.video_id,
                generation_rule=record.answer_type,
                generation_seed=config.get("seed", 42),
                generator_family=generator.__class__.__name__
            )
        )
        
        # ----------------------------------------------------------------
        # 8. Validation
        # ----------------------------------------------------------------
        is_valid, errs = validate_example(group)
        if is_valid:
            groups.append(group)
            success_count += 1
            domain_success[record.answer_type] += 1
        else:
            # Group is structurally malformed; log and reject.
            reason = "malformed_source"
            print(f"Validation error: {errs}")
            rejection_reasons[reason] += 1
            rejected_count += 1
            
        # ----------------------------------------------------------------
        # 9. Shard writing — flush buffer when shard_size is reached
        # ----------------------------------------------------------------
        if len(groups) >= shard_size:
            out_path = os.path.join(out_dir, f"part-{shard_idx:06d}.parquet")
            write_shard(groups, out_path)
            groups = []       # Reset buffer for the next shard.
            shard_idx += 1
            
    # Write any remaining groups that did not fill a complete shard.
    if groups:
        out_path = os.path.join(out_dir, f"part-{shard_idx:06d}.parquet")
        write_shard(groups, out_path)
        
    # ------------------------------------------------------------------ #
    # 10. Audit report                                                     #
    # ------------------------------------------------------------------ #
    # Print a structured summary of processing outcomes to stdout.
    # For each known answer domain, show input / success / rejection counts
    # and the TRUE / FALSE / UNKNOWN proposition label breakdown.
    # Domains with no registered generator are marked "NOT IMPLEMENTED".
    print(f"\n--- AUDIT REPORT ---")
    total_processed = success_count + rejected_count
    print(f"Source records read: {total_processed}")
    print(f"Successfully optionized: {success_count}")
    print(f"Rejected: {rejected_count}")
    
    print("\nRejection Reasons:")
    for r, count in rejection_reasons.items():
        print(f"  {r}: {count}")
        
    print("\nBy answer domain:")
    
    # Canonical ordered list of all answer domains known to this pipeline.
    all_possible_domains = [
        "binary", "object", "action", "count", "temporal",
        "comparison", "superlative", "three_way", "logic", "open", "other"
    ]
    
    for d in all_possible_domains:
        d_input = domain_counts.get(d, 0)
        d_success = domain_success.get(d, 0)
        d_rejected = d_input - d_success
        d_true = domain_true.get(d, 0)
        d_false = domain_false.get(d, 0)
        d_unknown = domain_unknown.get(d, 0)
        
        # Show detailed stats only for domains that have a registered generator;
        # mark unimplemented domains with a single "NOT IMPLEMENTED" line.
        is_implemented = bool(candidate_registry.get_generator(d))
        if is_implemented:
            print(f"  {d.upper()}:")
            print(f"    input: {d_input}")
            print(f"    success: {d_success}")
            print(f"    rejected: {d_rejected}")
            print(f"    TRUE labels: {d_true}")
            print(f"    FALSE labels: {d_false}")
            print(f"    UNKNOWN labels: {d_unknown}")
        else:
            print(f"  {d.upper()}: NOT IMPLEMENTED")
    print("--------------------\n")
        
    return success_count, rejected_count
