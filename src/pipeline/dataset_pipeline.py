import os
import random
from typing import Iterator
from src.datasets.registry import dataset_registry
import src.datasets.agqa  # Initialize registry
from src.candidates.registry import candidate_registry
import src.candidates     # Initialize registry
from src.schema import PropositionGroup, Proposition, PropositionSemantics, Reasoning, Provenance
from src.validation.validators import validate_example
from src.storage.parquet_writer import write_shard

def run_dataset_pipeline(config: dict, split: str, max_records: int = None):
    dataset_name = config.get("dataset", "agqa_balanced")
    AdapterClass = dataset_registry.get_adapter(dataset_name)
    adapter = AdapterClass(config)
    
    out_dir = config.get("output_dir", f"data/generated/{dataset_name}/1m")
    out_dir = os.path.join(out_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    
    records = adapter.load_records()
    
    groups = []
    shard_idx = 0
    shard_size = config.get("shard_size", 10000)
    
    rng = random.Random(config.get("seed", 42))
    
    success_count = 0
    rejected_count = 0
    
    for idx, record in enumerate(records):
        if max_records and idx >= max_records:
            break
            
        generator_class = candidate_registry.get_generator(record.answer_type)
        if not generator_class:
            # Fallback to a default generic if not registered
            # But the user said: "Do not claim support ... unless actually exists"
            # So if not found, we reject or skip
            rejected_count += 1
            continue
            
        generator = generator_class()
        
        # In a full implementation, context would contain scenegraph info
        context = {} 
        
        try:
            options, labels, truth_state = generator.generate(record, context, max_k=config.get("max_k", 40))
        except Exception as e:
            print("Generate error:", e)
            rejected_count += 1
            continue
            
        # Shuffle options while preserving label alignment
        combined = list(zip(options, labels))
        rng.shuffle(combined)
        options, labels = zip(*combined)
        options = list(options)
        labels = list(labels)
        
        # Build propositions schema
        propositions = []
        for opt, lbl in zip(options, labels):
            propositions.append(Proposition(
                text=opt,
                truth_state="TRUE" if lbl == 1 else ("UNKNOWN" if truth_state == "UNKNOWN" else "FALSE"),
                label=lbl,
                semantics=PropositionSemantics(
                    canonical_type=record.semantic_type,
                    object=opt # Ensures unique semantic signature per option
                )
            ))
            
        # Create group
        group = PropositionGroup(
            example_id=f"{dataset_name}_{record.source_question_id}",
            source=dataset_name,
            split=split,
            media_id=record.video_id,
            query_text=record.query,
            query_template_id=None,
            propositions=propositions,
            task_type="single_choice" if sum(labels) == 1 else ("binary" if len(options) == 2 else "open"),
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
        
        is_valid, errs = validate_example(group)
        if is_valid:
            groups.append(group)
            success_count += 1
        else:
            print("Validation error:", errs)
            rejected_count += 1
            
        if len(groups) >= shard_size:
            out_path = os.path.join(out_dir, f"part-{shard_idx:06d}.parquet")
            write_shard(groups, out_path)
            groups = []
            shard_idx += 1
            
    if groups:
        out_path = os.path.join(out_dir, f"part-{shard_idx:06d}.parquet")
        write_shard(groups, out_path)
        
    return success_count, rejected_count
