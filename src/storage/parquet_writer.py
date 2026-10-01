import pyarrow as pa
import pyarrow.parquet as pq

def write_shard(groups, out_path):
    data = {
        "example_id": [], "split": [], "video_id": [], "query": [], "options": [],
        "labels": [], "truth_state": [], "option_count": [], "task_type": [],
        "reasoning_family": [], "generator_family": []
    }
    for g in groups:
        data["example_id"].append(g.example_id)
        data["split"].append(g.split)
        data["video_id"].append(g.media_id if hasattr(g, 'media_id') else g.video_id)
        data["query"].append(g.query_text or "")
        data["options"].append([p.text for p in g.propositions])
        data["labels"].append([p.label if p.label is not None else -1 for p in g.propositions])
        data["truth_state"].append(g.propositions[0].truth_state if g.propositions else "UNKNOWN")
        data["option_count"].append(g.num_propositions)
        data["task_type"].append(g.task_type)
        data["reasoning_family"].append(g.reasoning.family)
        data["generator_family"].append(g.provenance.generator_family)
        
    
    assert all(v is not None and str(v).strip() != "" for v in data["video_id"]), "Missing video_id in generated examples"

    schema = pa.schema([
        ("example_id", pa.string()), ("split", pa.string()), ("video_id", pa.string()),
        ("query", pa.string()), ("options", pa.list_(pa.string())),
        ("labels", pa.list_(pa.int8())), ("truth_state", pa.string()), ("option_count", pa.int32()),
        ("task_type", pa.string()), ("reasoning_family", pa.string()), ("generator_family", pa.string())
    ])
    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, out_path)

