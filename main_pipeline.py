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

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
SHARD_SIZE = 50000

def get_signature(group: PropositionGroup) -> str:
    ev_obj = tuple(sorted(group.provenance.evidence_object_ids or []))
    ev_rel = tuple(sorted(group.provenance.evidence_relation_ids or []))
    ev_frm = tuple(sorted(group.provenance.frame_ids or []))
    return f"{ev_obj}_{ev_rel}_{ev_frm}"

def main():
    rng = random.Random(SEED)
    
    generators = [
        ("orig_qa", OriginalQAGenerator()),
        ("obj_exist", ObjectExistenceGenerator()),
        ("rel_ver", RelationVerificationGenerator()),
        ("rel_set", RelationSetGenerator()),
        ("grounding", GroundingGenerator()),
        ("temporal", ActionTemporalGenerator()),
        ("compositional", CompositionalGenerator()),
        ("attributes", AttributeGenerator())
    ]
    
    print("Loading AGQA Scene Graphs...", flush=True)
    def resolve_path(*candidates):
        for c in candidates:
            p = os.path.join(DATA_ROOT, c)
            if os.path.exists(p):
                return p
        return os.path.join(DATA_ROOT, candidates[0])

    train_sg_path = resolve_path("data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl", "AGQA_scene_graphs/AGQA_train_stsgs.pkl")
    test_sg_path = resolve_path("data/dataset/agqa_scene_graphs/AGQA_test_stsgs.pkl", "AGQA_scene_graphs/AGQA_test_stsgs.pkl")
    
    with open(train_sg_path, 'rb') as f:
        train_sgs_raw = pickle.load(f)
    with open(test_sg_path, 'rb') as f:
        test_sgs_raw = pickle.load(f)
        
    train_vids = list(train_sgs_raw.keys())
    test_vids = set(test_sgs_raw.keys())
    
    assert not test_vids.intersection(set(train_vids)), "Test videos found in Train!"
    
    rng.shuffle(train_vids)
    num_val = int(len(train_vids) * 0.1)
    val_vids = set(train_vids[:num_val])
    train_vids_set = set(train_vids[num_val:])
    
    def get_split(vid, is_test_set=False):
        if is_test_set:
            return "test" if vid in test_vids else None
        if vid in val_vids:
            return "val"
        if vid in train_vids_set:
            return "train"
        return None

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
        "duplicate_count": 0, "total_candidates": 0
    }
    
    seen_hashes = set()
    by_rule = {}

    buffers = {"train": [], "val": [], "test": []}
    shard_indices = {"train": 0, "val": 0, "test": 0}
    total_accepted = 0
    


    generated_count = 0
    flushed_count = 0
    written_count = 0
    
    print("Loading checkpoints...", flush=True)
    import pyarrow.parquet as pq
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATA_ROOT, "data", "generated", split)
        if os.path.exists(split_dir):
            for file in sorted(os.listdir(split_dir)):
                if file.startswith("part-") and file.endswith(".parquet"):
                    # Update shard index
                    idx = int(file.replace("part-", "").replace(".parquet", ""))
                    shard_indices[split] = max(shard_indices[split], idx + 1)
                    
                    # Read hashes to prevent duplicates
                    file_path = os.path.join(split_dir, file)
                    table = pq.read_table(file_path, columns=["propositions"])
                    for row in table["propositions"].to_pylist():
                        texts = [p["text"] for p in row]
                        seen_hashes.add(hash("".join(sorted(texts))))
                        written_count += 1
                        
    print(f"Loaded {written_count} previous examples into deduplication cache.", flush=True)
    generated_count = written_count
    flushed_count = written_count

    

    def flush_buffer(split: str, force=False):
        nonlocal flushed_count
        n = len(buffers[split])
        if not force and n < SHARD_SIZE:
            return
        if n == 0:
            return
            
        os.makedirs(os.path.join(DATA_ROOT, "data", "generated", split), exist_ok=True)
        shard_idx = shard_indices[split]
        filename = f"part-{shard_idx:06d}.parquet"
        out_path = os.path.join(DATA_ROOT, "data", "generated", split, filename)
        
        try:
            write_shard(buffers[split], out_path)
            print(f"Flushed {n} examples to {out_path}", flush=True)
            flushed_count += n
            buffers[split] = []
            shard_indices[split] += 1
        except Exception as e:
            print(f"Failed to flush {n} examples to {out_path}: {e}", flush=True)
    def process_split(qs_path, sgs_raw, is_test_set, target_count):
        nonlocal generated_count, total_accepted, written_count
        nonlocal total_accepted
        print(f"\\nNormalizing Scene Graphs for {'TEST' if is_test_set else 'TRAIN/VAL'}...", flush=True)
        sgs_norm = {}
        pools_cache = {}
        
        vids_needed = list(sgs_raw.keys())
        rng.shuffle(vids_needed)
        # vids_needed = vids_needed[:1500]  # No limit for 1M generation
        
        for vid in tqdm(vids_needed):
            raw_sg = sgs_raw[vid]
            norm_sg = normalize_video_sg(vid, raw_sg)
            sgs_norm[vid] = norm_sg
            pools_cache[vid] = NegativePools(norm_sg)
            
        print(f"Generating {'TEST' if is_test_set else 'TRAIN/VAL'} examples...", flush=True)
        
        gen_idx = 0
        added_count = 0
        start_time = time.time()
        last_print_time = start_time
        
        for qid, qdata in stream_questions(qs_path):
            if generated_count >= target_count:
                break
                
            vid = qdata.get("video_id")
            split = get_split(vid, is_test_set)
            if not split: continue
                
            sg = sgs_norm.get(vid)
            pools = pools_cache.get(vid)
            if not sg: continue
            
            for _ in range(len(generators)):
                if generated_count >= target_count:
                    break
                    
                gen_name, gen_obj = generators[gen_idx]
                gen_idx = (gen_idx + 1) % len(generators)
                
                if gen_name != "orig_qa" and not sg:
                    continue
                    
                for group in gen_obj.generate(qid, qdata, sg, split, rng, pools):
                    stats["candidates"][gen_name] += 1
                    stats["total_candidates"] += 1
                    
                    group_hash = hash("".join(sorted(p.text for p in group.propositions)))
                    if group_hash in seen_hashes:
                        stats["duplicate_count"] += 1
                        continue
                    seen_hashes.add(group_hash)
                    
                    is_valid, errors = validate_example(group)
                    if not is_valid:
                        stats["rejected"][gen_name] += 1
                        for err in errors:
                            stats["rejection_reasons"][err] += 1
                        continue
                        
                    # Accepted!
                    stats["accepted"][gen_name] += 1
                    generated_count += 1
                    total_accepted += 1
                    
                    if (generated_count - written_count) % (target_count // 100 or 1) == 0 or time.time() - last_print_time > 10:
                        elapsed = time.time() - start_time
                        rate = (generated_count - written_count) / elapsed if elapsed > 0 else 0
                        percent = ((generated_count - written_count) / target_count) * 100
                        print(f"[{percent:5.1f}%] Generated {generated_count}/{target_count} ({rate:.1f} records/sec) - {elapsed:.1f}s elapsed", flush=True)
                        last_print_time = time.time()
                    
                    # Buffer write
                    buffers[split].append(group)
                    flush_buffer(split, force=False)
                    
                    # Maintain sample markdown report memory
                    rule = group.provenance.generation_rule
                    if rule not in by_rule: by_rule[rule] = []
                    if len(by_rule[rule]) < 5: by_rule[rule].append(group)
                    
                    # Update stats
                    stats["task_types"][group.task_type] += 1
                    stats["reasoning_families"][group.reasoning.family] += 1
                    stats["generator_families"][group.provenance.generator_family] += 1
                    stats["complexity"][group.reasoning.complexity] += 1
                    stats["k_counts"][group.num_propositions] += 1
                    stats["split_counts"][split] += 1
                    stats["source_videos"][vid] += 1
                    
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
                            if p.label == 1: stats["none_true"] += 1
                            else: stats["none_false"] += 1
                            
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
                        if "identification" in group.reasoning.type: stats["grounding_subtypes"]["identification"] += 1
                        elif "verification" in group.reasoning.type: stats["grounding_subtypes"]["verification"] += 1

    train_qs_path = resolve_path("data/dataset/agqa_balanced/train_balanced.txt", "data/dataset/agqa_balanced/AGQA_balanced/train_balanced.txt", "AGQA_balanced/AGQA_balanced/train_balanced.txt")
    test_qs_path = resolve_path("data/dataset/agqa_balanced/test_balanced.txt", "data/dataset/agqa_balanced/AGQA_balanced/test_balanced.txt", "AGQA_balanced/AGQA_balanced/test_balanced.txt")
    
    # 1. Train/Val Gen (80000)
    process_split(train_qs_path, train_sgs_raw, False, 80000)
    # 2. Test Gen (cumulative 100000)
    process_split(test_qs_path, test_sgs_raw, True, 100000)
    
    print("\\nValidating Constraints and Generating Reports...", flush=True)
    
    # Flush remaining buffers
    for split in buffers:
        flush_buffer(split, force=True)
    
    # Consistency assertions
    assert sum(stats["task_types"].values()) == total_accepted
    assert sum(stats["reasoning_families"].values()) == total_accepted
    assert sum(stats["split_counts"].values()) == total_accepted
    assert sum(stats["k_counts"].values()) == total_accepted
    assert sum(stats["temporal_subtypes"].values()) == stats["reasoning_families"]["temporal"]
    
    # Write Sample Report
    with open("SAMPLE_REPORT.md", "w") as f:
        f.write("# V5 Logical Correctness Sample Report\n\n")
        for rule, groups in by_rule.items():
            f.write(f"## Generation Rule: {rule}\n")
            for group in groups:
                f.write(f"**Example ID:** {group.example_id} | **Split:** {group.split}\n")
                f.write(f"**Task Type:** {group.task_type} | **K:** {group.num_propositions}\n")
                if group.query_text: f.write(f"**Query:** {group.query_text}\n")
                f.write(f"**Options:**\n")
                for p in group.propositions:
                    l = p.label if p.label is not None else "?"
                    f.write(f"  - [{l}] {p.text} (Truth: {p.truth_state})\n")
                f.write("\n")
                
    # Write Validation Report
    with open("VALIDATION_REPORT.md", "w") as f:
        f.write("# V5 Validation Report\n\n")
        f.write(f"- Total Candidates: {stats['total_candidates']}\n")
        f.write(f"- Total Accepted: {total_accepted}\n")
        f.write(f"- Total Rejected: {sum(stats['rejected'].values())}\n")
        f.write(f"- Duplicates Caught: {stats['duplicate_count']}\n\n")
        f.write("### Rejection Reasons\n")
        for reason, count in stats["rejection_reasons"].items():
            f.write(f"- {count}: {reason}\n")
            
    # Write Split Report
    with open("SPLIT_REPORT.md", "w") as f:
        f.write("# Dataset Split Report\n\n")
        f.write(f"- Train Videos (Universe): {len(train_vids_set)}\n")
        f.write(f"- Val Videos (Universe): {len(val_vids)}\n")
        f.write(f"- Test Videos (Universe): {len(test_vids)}\n\n")
        f.write("### Generated Examples by Split\n")
        for s, c in stats["split_counts"].items():
            f.write(f"- {s.upper()}: {c}\n")
            
    # Write Diversity Report
    with open("DIVERSITY_REPORT.md", "w") as f:
        f.write("# Diversity & Production Audit\n\n")
        dup_ratio = stats["duplicate_count"] / stats["total_candidates"] if stats["total_candidates"] else 0
        ev_reuse_ratio = 1.0 - (len(stats["examples_per_evidence"]) / total_accepted)
        
        f.write(f"- Unique Source Videos Used: {len(stats['source_videos'])}\n")
        f.write(f"- Average Examples / Video: {total_accepted / len(stats['source_videos']):.2f}\n")
        f.write(f"- Average Examples / Evidence Signature: {total_accepted / len(stats['examples_per_evidence']):.2f}\n")
        f.write(f"- Unique Proposition Signatures: {len(stats['unique_prop_signatures'])}\n")
        f.write(f"- Unique Option Sets (Group level): {len(stats['unique_option_sets'])}\n")
        f.write(f"- Unique Query Templates Used: {len(stats['query_templates'])}\n")
        f.write(f"- Duplicate Catch Ratio: {dup_ratio:.2%}\n")
        f.write(f"- Evidence Reuse Ratio: {ev_reuse_ratio:.2%}\n\n")
        
        f.write("### General Statistics\n")
        f.write(f"- Truth States: {dict(stats['truth_states'])}\n")
        f.write(f"- Logical NONE Included: {stats['none_included']} (True={stats['none_true']}, False={stats['none_false']})\n")
        f.write(f"- Task Types: {dict(stats['task_types'])}\n")
        f.write(f"- Reasoning Families: {dict(stats['reasoning_families'])}\n")
        f.write(f"- K Counts: {dict(stats['k_counts'])}\n")
        f.write(f"- Temporal Subtypes: {dict(stats['temporal_subtypes'])}\n")
        f.write(f"- Temporal Hops: {dict(stats['temporal_hops'])}\n")
        
    print("ALL REPORTS GENERATED!", flush=True)

if __name__ == "__main__":
    main()
