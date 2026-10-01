import json, sys, os

output_file = os.path.join(r'c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp', 'agqa_analysis_output.txt')
f_out = open(output_file, 'w', encoding='utf-8')

def p(s=''):
    f_out.write(str(s) + '\n')

p('='*80)
p('AGQA DATASET ANALYSIS')
p('='*80)

train_path = os.path.join(r'c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp\AGQA_balanced\AGQA_balanced', 'train_balanced.txt')
file_size = os.path.getsize(train_path)
p('\n' + '='*80)
p('FILE 1: train_balanced.txt')
p('='*80)
p('File size: {:,} bytes ({:.1f} MB)'.format(file_size, file_size/1024/1024))
p('Format: Single-line JSON (one giant dictionary)')

with open(train_path, 'r', encoding='utf-8') as f:
    chunk = f.read(300000)

p('First 100 chars: ' + repr(chunk[:100]))

def parse_entries_from_chunk(chunk):
    entries = []
    i = 1
    depth = 0
    entry_start = None
    key = None
    while i < len(chunk):
        if chunk[i] == chr(34) and entry_start is None and depth == 0:
            end_quote = chunk.index(chr(34), i+1)
            key = chunk[i+1:end_quote]
            i = end_quote + 1
            while i < len(chunk) and chunk[i] in ': ':
                i += 1
            if i < len(chunk) and chunk[i] == '{':
                entry_start = i
                depth = 1
                i += 1
                continue
        elif entry_start is not None:
            if chunk[i] == '{':
                depth += 1
            elif chunk[i] == '}':
                depth -= 1
                if depth == 0:
                    entry_json = chunk[entry_start:i+1]
                    try:
                        entry = json.loads(entry_json)
                        entries.append((key, entry))
                    except:
                        pass
                    entry_start = None
        i += 1
    return entries

entries = parse_entries_from_chunk(chunk)
p('\nEntries parsed from 300KB chunk: {}'.format(len(entries)))

q_count = chunk.count(chr(34) + 'question' + chr(34))
ratio = file_size / len(chunk)
approx_total = int(q_count * ratio)
p('Estimated total entries: ~{:,}'.format(approx_total))

all_fields = set()
ans_types = set()
semantic_types = set()
structural_types = set()
global_types = set()
answer_values = set()
video_ids = set()
steps_values = set()
novel_comp_values = set()
more_steps_values = set()

for k, entry in entries:
    all_fields.update(entry.keys())
    ans_types.add(entry.get('ans_type', ''))
    semantic_types.add(entry.get('semantic', ''))
    structural_types.add(entry.get('structural', ''))
    answer_values.add(str(entry.get('answer', '')))
    video_ids.add(entry.get('video_id', ''))
    steps_values.add(entry.get('steps', 0))
    novel_comp_values.add(entry.get('novel_comp', -1))
    more_steps_values.add(entry.get('more_steps', -1))
    for g in entry.get('global', []):
        global_types.add(g)

p('\n--- SCHEMA ---')
p('All unique field names: {}'.format(sorted(all_fields)))
p('Total fields: {}'.format(len(all_fields)))

p('\n--- FIELD TYPES (from first entry) ---')
for field in sorted(entries[0][1].keys()):
    value = entries[0][1][field]
    p('  {}: {} = {}'.format(field, type(value).__name__, repr(value)[:200]))

p('\n--- UNIQUE VALUES ---')
p('ans_type: {}'.format(sorted(ans_types)))
p('semantic: {}'.format(sorted(semantic_types)))
p('structural: {}'.format(sorted(structural_types)))
p('global (categories): {}'.format(sorted(global_types)))
p('steps range: {} to {}'.format(min(steps_values), max(steps_values)))
p('novel_comp values: {}'.format(sorted(novel_comp_values)))
p('more_steps values: {}'.format(sorted(more_steps_values)))
p('Sample video IDs: {}'.format(sorted(list(video_ids))[:10]))
p('Total unique video IDs in chunk: {}'.format(len(video_ids)))
p('Sample answer values (first 40): {}'.format(sorted(list(answer_values))[:40]))

p('\n--- SAMPLE KEY FORMAT ---')
for k, _ in entries[:5]:
    p('  ' + k)

p('\n--- COMPLETE SAMPLE ENTRIES (3) ---')
for k, entry in entries[:3]:
    p('\nKey: ' + k)
    p(json.dumps(entry, indent=2))

p('\n--- sg_grounding STRUCTURE ---')
for k, entry in entries[:3]:
    sg = entry.get('sg_grounding', {})
    p('Key: ' + k)
    p('  sg_grounding keys: {}'.format(list(sg.keys())))
    for sg_key, sg_val in sg.items():
        p('    {}: {}'.format(sg_key, sg_val))

test_path = os.path.join(r'c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp\AGQA_balanced\AGQA_balanced', 'test_balanced.txt')
file_size2 = os.path.getsize(test_path)
p('\n\n' + '='*80)
p('FILE 2: test_balanced.txt')
p('='*80)
p('File size: {:,} bytes ({:.1f} MB)'.format(file_size2, file_size2/1024/1024))

with open(test_path, 'r', encoding='utf-8') as f:
    chunk2 = f.read(300000)

entries2 = parse_entries_from_chunk(chunk2)
p('Entries parsed from chunk: {}'.format(len(entries2)))
q_count2 = chunk2.count(chr(34) + 'question' + chr(34))
ratio2 = file_size2 / len(chunk2)
approx_total2 = int(q_count2 * ratio2)
p('Estimated total entries: ~{:,}'.format(approx_total2))

test_fields = set()
for k2, entry2 in entries2:
    test_fields.update(entry2.keys())

p('All unique field names (test): {}'.format(sorted(test_fields)))
p('Test-specific fields (not in train): {}'.format(sorted(test_fields - all_fields)))
p('Train-specific fields (not in test): {}'.format(sorted(all_fields - test_fields)))

p('\n--- COMPLETE SAMPLE ENTRIES (test, 2) ---')
for k2, entry2 in entries2[:2]:
    p('\nKey: ' + k2)
    p(json.dumps(entry2, indent=2))

f_out.close()
print('Done! Output written to', output_file)
