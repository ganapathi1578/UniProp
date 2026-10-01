import pyarrow.parquet as pq

def generate_sample_report(parquet_path: str, out_path: str = "SAMPLE_REPORT.md"):
    table = pq.read_table(parquet_path)
    records = table.to_pydict()
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# AGQA Balanced QA Optionization - Sample Report\n\n")
        
        num_records = min(10, len(records['example_id']))
        for i in range(num_records):
            f.write(f"### Example {i+1}\n")
            f.write(f"- **example_id**: `{records['example_id'][i]}`\n")
            # For provenance, we can parse example_id if we want, but video_id is here
            f.write(f"- **video_id**: `{records['video_id'][i]}`\n")
            f.write(f"- **split**: `{records['split'][i]}`\n")
            f.write(f"- **task_type**: `{records['task_type'][i]}`\n")
            f.write(f"- **reasoning_family**: `{records['reasoning_family'][i]}`\n")
            f.write(f"- **generator_family**: `{records['generator_family'][i]}`\n")
            f.write("\n")
            f.write(f"**QUERY:**\n> {records['query'][i]}\n\n")
            
            options = records['options'][i]
            labels = records['labels'][i]
            
            f.write("**OPTIONS:**\n")
            for idx, opt in enumerate(options):
                f.write(f"[{idx}] {opt}\n")
            f.write("\n")
            
            f.write("**LABELS:**\n")
            f.write(f"`{labels}`\n\n")
            
            f.write(f"**TRUTH STATE:**\n`{records['truth_state'][i]}`\n\n")
            f.write("---\n")
            
if __name__ == "__main__":
    generate_sample_report("data/generated/agqa_balanced/1m/train/part-000000.parquet")
