import os
import sys
import time
import json
import pickle
import random
import argparse
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Set, Optional

from src.config import UniPropConfig, load_config
from src.sampling import SamplingController, TargetSpec
from src.normalization.scene_graph_normalizer import normalize_video_sg, NormalizedSceneGraph
from src.negatives.hard_negatives import NegativePools
from src.generators.scene_graph import ObjectExistenceGenerator, RelationVerificationGenerator
from src.generators.grounding import GroundingGenerator
from src.generators.temporal import ActionTemporalGenerator
from src.generators.compositional import CompositionalGenerator
from src.validation.validators import validate_example
from src.storage.parquet_writer import write_shard
from src.schema import PropositionGroup

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_signature(group: PropositionGroup) -> str:
    ev_obj = tuple(sorted(group.provenance.evidence_object_ids or []))
    ev_rel = tuple(sorted(group.provenance.evidence_relation_ids or []))
    ev_frm = tuple(sorted(group.provenance.frame_ids or []))
    return f"{ev_obj}_{ev_rel}_{ev_frm}"


def resolve_path(base_dir: str, *candidates) -> str:
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, candidates[0])


def main():
    parser = argparse.ArgumentParser(description="UniProp Video-Jev Dataset Generation Pipeline")
    parser.add_argument("--config", type=str, default="config/generation/100k.yaml", help="Path to YAML configuration file")
    parser.add_argument("--scale", type=str, default=None, help="Override generation scale (e.g. 100k, 1m, 10m, 100m)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(DATA_ROOT, config_path)
    
    if not os.path.exists(config_path):
        fallback_path = os.path.join(DATA_ROOT, "config", "default_config.yaml")
        if os.path.exists(fallback_path):
            config_path = fallback_path

    print(f"Loading configuration from: {config_path}", flush=True)
    config = load_config(config_path)

    # CLI Overrides
    if args.scale:
        config.raw["scale"] = args.scale
    if args.output_dir:
        config.raw["paths"]["output_dir"] = args.output_dir

    seed = config.seed
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    rng = random.Random(seed)

    scale = config.scale
    shard_size = config.shard_size
    scale_output_dir = config.get_scale_output_dir(DATA_ROOT)
    print(f"Target Scale: {scale.upper()} | Output Root: {scale_output_dir}", flush=True)
    print(f"Target counts: {config.target_counts}", flush=True)

    # Prepare generator instances mapped to families
    family_generators = {
        "action": ActionTemporalGenerator(),
        "temporal": ActionTemporalGenerator(),
        "compositional": CompositionalGenerator(),
        "object": ObjectExistenceGenerator(),
        "spatial_contact": RelationVerificationGenerator(),
        "grounding": GroundingGenerator(),
    }

    # 1. Load Scene Graphs
    print("Loading AGQA Scene Graphs...", flush=True)
    train_sg_path = resolve_path(DATA_ROOT, config.paths.get("scene_graphs_train", "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl"), "AGQA_scene_graphs/AGQA_train_stsgs.pkl")
    test_sg_path = resolve_path(DATA_ROOT, config.paths.get("scene_graphs_test", "data/dataset/agqa_scene_graphs/AGQA_test_stsgs.pkl"), "AGQA_scene_graphs/AGQA_test_stsgs.pkl")

    with open(train_sg_path, "rb") as f:
        train_sgs_raw = pickle.load(f)
    with open(test_sg_path, "rb") as f:
        test_sgs_raw = pickle.load(f)

    # 2. Split videos first (Video-Level 80 / 10 / 10 Split)
    print("Partitioning videos strictly (80% Train, 10% Val, 10% Test)...", flush=True)
    raw_train_vids = sorted(list(train_sgs_raw.keys()))
    test_vids = set(test_sgs_raw.keys())

    assert not test_vids.intersection(set(raw_train_vids)), "CRITICAL: Test videos overlap with Train raw universe!"

    rng.shuffle(raw_train_vids)
    num_val = int(len(raw_train_vids) * 0.10)
    val_vids = set(raw_train_vids[:num_val])
    train_vids_set = set(raw_train_vids[num_val:])

    assert not val_vids.intersection(train_vids_set), "CRITICAL: Val and Train video sets overlap!"
    assert not val_vids.intersection(test_vids), "CRITICAL: Val and Test video sets overlap!"

    print(f"Video Partition Summary: {len(train_vids_set)} Train | {len(val_vids)} Val | {len(test_vids)} Test", flush=True)

    split_video_maps = {
        "train": (train_vids_set, train_sgs_raw),
        "val": (val_vids, train_sgs_raw),
        "test": (test_vids, test_sgs_raw),
    }

    # Tracking Statistics
    stats = {
        "candidates": Counter(), "accepted": Counter(), "rejected": Counter(),
        "rejection_reasons": Counter(), "task_types": Counter(),
        "reasoning_families": Counter(), "generator_families": Counter(),
        "complexity": Counter(), "k_counts": Counter(),
        "truth_states": Counter(), "none_included": 0, "none_true": 0, "none_false": 0,
        "query_templates": set(), "temporal_subtypes": Counter(),
        "temporal_hops": Counter(), "grounding_subtypes": Counter(),
        "source_videos": Counter(), "split_counts": Counter(),
        "examples_per_video": Counter(), "examples_per_evidence": Counter(),
        "unique_prop_signatures": set(), "unique_option_sets": set(),
        "duplicate_count": 0, "total_candidates": 0,
        "negative_difficulty": Counter(), "reasoning_depth": Counter()
    }

    seen_hashes = set()
    by_rule = {}
    total_accepted = 0

    # Shard buffering per split
    buffers = {"train": [], "val": [], "test": []}
    shard_indices = {"train": 0, "val": 0, "test": 0}

    # Load existing checkpoint parquet shards if present
    print(f"Checking for existing shards under {scale_output_dir}...", flush=True)
    import pyarrow.parquet as pq
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(scale_output_dir, split)
        if os.path.exists(split_dir):
            for file in sorted(os.listdir(split_dir)):
                if file.startswith("part-") and file.endswith(".parquet"):
                    idx = int(file.replace("part-", "").replace(".parquet", ""))
                    shard_indices[split] = max(shard_indices[split], idx + 1)
                    file_path = os.path.join(split_dir, file)
                    table = pq.read_table(file_path, columns=["video_id", "query", "options"])
                    for vid, q, row in zip(table["video_id"].to_pylist(), table["query"].to_pylist(), table["options"].to_pylist()):
                        seen_hashes.add(hash((vid, q, tuple(sorted(row)))))

    def flush_buffer(split: str, force: bool = False):
        n = len(buffers[split])
        if not force and n < shard_size:
            return
        if n == 0:
            return

        split_dir = os.path.join(scale_output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        shard_idx = shard_indices[split]
        filename = f"part-{shard_idx:06d}.parquet"
        out_path = os.path.join(split_dir, filename)

        try:
            write_shard(buffers[split], out_path)
            print(f"Flushed {n} examples to {out_path}", flush=True)
            buffers[split] = []
            shard_indices[split] += 1
        except Exception as e:
            print(f"Failed to flush {n} examples to {out_path}: {e}", flush=True)

    # Execution Loop for each split
    for split in ["train", "val", "test"]:
        target_count = config.target_counts.get(split, 0)
        if target_count <= 0:
            continue

        print(f"\n==========================================", flush=True)
        print(f"GENERATING SPLIT: {split.upper()} (Target: {target_count:,})", flush=True)
        print(f"==========================================", flush=True)

        allowed_vids, sgs_raw = split_video_maps[split]

        # Initialize split-specific sampling controller
        controller = SamplingController(config, target_count)

        vids_needed = [v for v in allowed_vids if v in sgs_raw]
        rng.shuffle(vids_needed)

        # Lazy cache for normalized scene graphs and pools
        sgs_norm_cache: Dict[str, NormalizedSceneGraph] = {}
        pools_cache: Dict[str, NegativePools] = {}

        def get_or_create_sg(v: str):
            if v not in sgs_norm_cache:
                raw_sg = sgs_raw[v]
                norm_sg = normalize_video_sg(v, raw_sg)
                sgs_norm_cache[v] = norm_sg
                pools_cache[v] = NegativePools(norm_sg)
            return sgs_norm_cache[v], pools_cache[v]

        video_idx = 0
        split_generated = 0
        start_time = time.time()
        last_print_time = start_time

        while split_generated < target_count:
            # Sample target specification using 6x4x4 controlled matrix
            spec: TargetSpec = controller.sample_target(rng)
            gen = family_generators.get(spec.family)
            if not gen:
                continue

            # Rotate through all videos in this split to maximize record coverage across the dataset
            vid = vids_needed[video_idx % len(vids_needed)]
            video_idx += 1
            sg, pools = get_or_create_sg(vid)

            gen_kwargs = {
                "target_task": spec.task_type,
                "target_k": spec.k,
                "target_hops": spec.hops,
                "target_difficulty": spec.difficulty
            }
            if spec.family in ("action", "temporal"):
                gen_kwargs["target_family"] = spec.family

            try:
                qid = f"sg_{split}_{vid}_{split_generated}_{rng.randint(0, 1000000)}"
                candidates = list(gen.generate(qid, {}, sg, split, rng, pools, **gen_kwargs))
            except Exception:
                continue

            for group in candidates:
                if split_generated >= target_count:
                    break

                stats["candidates"][spec.family] += 1
                stats["total_candidates"] += 1

                # Deduplication
                group_hash = hash((group.media_id, group.query_text, tuple(sorted(p.text for p in group.propositions))))
                if group_hash in seen_hashes:
                    stats["duplicate_count"] += 1
                    continue
                seen_hashes.add(group_hash)

                # Quality Validation Gate
                is_valid, errors = validate_example(group)
                if not is_valid:
                    stats["rejected"][spec.family] += 1
                    for err in errors:
                        stats["rejection_reasons"][err] += 1
                    continue

                # Accepted!
                stats["accepted"][spec.family] += 1
                split_generated += 1
                total_accepted += 1

                # Buffer and flush check
                buffers[split].append(group)
                flush_buffer(split, force=False)

                # Record in SamplingController to update quota deficit
                controller.record_accepted(
                    family=group.reasoning.family,
                    task_type=group.task_type,
                    k=group.num_propositions,
                    hops=group.reasoning.hops,
                    difficulty=spec.difficulty
                )

                # Maintain sample report memory
                rule = group.provenance.generation_rule
                if rule not in by_rule:
                    by_rule[rule] = []
                if len(by_rule[rule]) < 5:
                    by_rule[rule].append(group)

                # Update Global Statistics
                stats["task_types"][group.task_type] += 1
                stats["reasoning_families"][group.reasoning.family] += 1
                stats["generator_families"][group.provenance.generator_family] += 1
                stats["complexity"][group.reasoning.complexity] += 1
                stats["k_counts"][group.num_propositions] += 1
                stats["split_counts"][split] += 1
                stats["source_videos"][vid] += 1
                stats["negative_difficulty"][spec.difficulty] += 1
                stats["reasoning_depth"][group.reasoning.hops] += 1

                stats["examples_per_video"][vid] += 1
                ev_sig = get_signature(group)
                stats["examples_per_evidence"][ev_sig] += 1
                stats["unique_option_sets"].add(hash(tuple(sorted(p.text for p in group.propositions))))

                if group.query_template_id:
                    stats["query_templates"].add(group.query_template_id)

                for p in group.propositions:
                    stats["truth_states"][p.truth_state] += 1
                    psig = (p.semantics.canonical_type, p.semantics.subject, p.semantics.predicate, p.semantics.object, p.semantics.temporal_relation, p.semantics.event_a, p.semantics.event_b, p.semantics.polarity)
                    stats["unique_prop_signatures"].add(hash(psig))

                    if p.semantics.canonical_type == "logical_none":
                        stats["none_included"] += 1
                        if p.label == 1:
                            stats["none_true"] += 1
                        else:
                            stats["none_false"] += 1

                if group.reasoning.family == "temporal":
                    if "multihop" in group.reasoning.type:
                        stats["temporal_subtypes"]["multihop"] += 1
                    else:
                        found = False
                        for p in group.propositions:
                            if p.semantics.canonical_type == "temporal" and p.truth_state == "TRUE":
                                stats["temporal_subtypes"][p.semantics.temporal_relation] += 1
                                found = True
                                break
                        if not found:
                            stats["temporal_subtypes"]["unknown"] += 1
                    stats["temporal_hops"][group.reasoning.hops] += 1

                if group.reasoning.family == "grounding":
                    if "identification" in group.reasoning.type:
                        stats["grounding_subtypes"]["identification"] += 1
                    elif "verification" in group.reasoning.type:
                        stats["grounding_subtypes"]["verification"] += 1

                # Progress printing
                if split_generated % max(1, target_count // 50) == 0 or time.time() - last_print_time > 10:
                    elapsed = time.time() - start_time
                    rate = split_generated / elapsed if elapsed > 0 else 0
                    pct = (split_generated / target_count) * 100
                    print(f"[{split.upper()} {pct:5.1f}%] {split_generated}/{target_count} ({rate:.1f} rec/s) - {elapsed:.1f}s elapsed", flush=True)
                    last_print_time = time.time()

        # Flush any remaining buffer for this split
        flush_buffer(split, force=True)
        print(f"COMPLETED {split.upper()}: {split_generated:,} examples generated.", flush=True)

    print("\n==========================================", flush=True)
    print("ALL SPLITS COMPLETED! Generating Reports & Manifest...", flush=True)
    print("==========================================", flush=True)

    # Consistency assertions
    assert sum(stats["task_types"].values()) == total_accepted
    assert sum(stats["reasoning_families"].values()) == total_accepted
    assert sum(stats["split_counts"].values()) == total_accepted
    assert sum(stats["k_counts"].values()) == total_accepted

    # 1. Write Manifest
    manifest_data = {
        "dataset_name": config.dataset_name,
        "version": config.version,
        "scale": scale,
        "seed": seed,
        "total_examples": total_accepted,
        "split_counts": dict(stats["split_counts"]),
        "task_types": dict(stats["task_types"]),
        "reasoning_families": dict(stats["reasoning_families"]),
        "negative_difficulty": dict(stats["negative_difficulty"]),
        "reasoning_depth": dict(stats["reasoning_depth"]),
        "k_counts": dict(stats["k_counts"]),
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = os.path.join(scale_output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Wrote Manifest to: {manifest_path}", flush=True)

    # 2. Write Reports (to scale_output_dir and repo root)
    def write_reports(target_dir: str):
        os.makedirs(target_dir, exist_ok=True)
        # Sample Report
        with open(os.path.join(target_dir, "SAMPLE_REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# Video-Jev Controlled Distribution Sample Report\n\n")
            for rule, groups in by_rule.items():
                f.write(f"## Generation Rule: {rule}\n")
                for group in groups:
                    f.write(f"**Example ID:** {group.example_id} | **Split:** {group.split}\n")
                    f.write(f"**Task Type:** {group.task_type} | **K:** {group.num_propositions} | **Hops:** {group.reasoning.hops}\n")
                    if group.query_text:
                        f.write(f"**Query:** {group.query_text}\n")
                    f.write("**Options:**\n")
                    for p in group.propositions:
                        l = p.label if p.label is not None else "?"
                        f.write(f"  - [{l}] {p.text} (Truth: {p.truth_state})\n")
                    f.write("\n")

        # Validation Report
        with open(os.path.join(target_dir, "VALIDATION_REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# Video-Jev Validation Report\n\n")
            f.write(f"- Total Candidates: {stats['total_candidates']}\n")
            f.write(f"- Total Accepted: {total_accepted}\n")
            f.write(f"- Total Rejected: {sum(stats['rejected'].values())}\n")
            f.write(f"- Duplicates Caught: {stats['duplicate_count']}\n\n")
            f.write("### Rejection Reasons\n")
            for reason, count in stats["rejection_reasons"].items():
                f.write(f"- {count}: {reason}\n")

        # Split Report
        with open(os.path.join(target_dir, "SPLIT_REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# Dataset Split Report\n\n")
            f.write(f"- Train Videos (Universe): {len(train_vids_set)}\n")
            f.write(f"- Val Videos (Universe): {len(val_vids)}\n")
            f.write(f"- Test Videos (Universe): {len(test_vids)}\n\n")
            f.write("### Generated Examples by Split\n")
            for s, c in stats["split_counts"].items():
                f.write(f"- {s.upper()}: {c}\n")

        # Diversity & Distribution Audit Report
        with open(os.path.join(target_dir, "DIVERSITY_REPORT.md"), "w", encoding="utf-8") as f:
            dup_ratio = stats["duplicate_count"] / stats["total_candidates"] if stats["total_candidates"] else 0
            ev_reuse = 1.0 - (len(stats["examples_per_evidence"]) / total_accepted) if total_accepted else 0
            f.write("# Video-Jev Controlled Sampling Matrix & Diversity Report\n\n")
            f.write(f"- Unique Source Videos Used: {len(stats['source_videos'])}\n")
            f.write(f"- Average Examples / Video: {total_accepted / max(1, len(stats['source_videos'])):.2f}\n")
            f.write(f"- Average Examples / Evidence Signature: {total_accepted / max(1, len(stats['examples_per_evidence'])):.2f}\n")
            f.write(f"- Unique Proposition Signatures: {len(stats['unique_prop_signatures'])}\n")
            f.write(f"- Unique Option Sets (Group level): {len(stats['unique_option_sets'])}\n")
            f.write(f"- Duplicate Catch Ratio: {dup_ratio:.2%}\n")
            f.write(f"- Evidence Reuse Ratio: {ev_reuse:.2%}\n\n")

            f.write("### Controlled Sampling Matrix Statistics\n")
            f.write(f"- Reasoning Families (Target: 20% Action, 20% Temporal, 20% Comp, 15% Obj, 15% Spatial/Contact, 10% Grounding):\n")
            for fam, count in stats["reasoning_families"].items():
                f.write(f"  - {fam}: {count} ({count / total_accepted:.1%})\n")
            f.write(f"\n- Task Modalities (Target: 45% Single-Choice, 25% Binary, 20% Multi-Label, 10% Three-Way):\n")
            for task, count in stats["task_types"].items():
                f.write(f"  - {task}: {count} ({count / total_accepted:.1%})\n")
            f.write(f"\n- Reasoning Depth / Hops (Target: 20% 1-hop, 30% 2-hop, 30% 3-hop, 20% 4+ hop):\n")
            for hop, count in sorted(stats["reasoning_depth"].items()):
                f.write(f"  - {hop}-hop: {count} ({count / total_accepted:.1%})\n")
            f.write(f"\n- Negative Difficulty (Target: 20% Easy, 40% Medium, 40% Hard):\n")
            for diff, count in stats["negative_difficulty"].items():
                f.write(f"  - {diff}: {count} ({count / total_accepted:.1%})\n")
            f.write(f"\n- Truth States: {dict(stats['truth_states'])}\n")
            f.write(f"- Logical NONE Included: {stats['none_included']} (True={stats['none_true']}, False={stats['none_false']})\n")
            f.write(f"- K Counts: {dict(stats['k_counts'])}\n")

    # Write reports to scale output dir and repository root
    write_reports(scale_output_dir)
    write_reports(DATA_ROOT)

    print("ALL REPORTS AND MANIFEST SUCCESSFULLY GENERATED!", flush=True)


if __name__ == "__main__":
    main()
