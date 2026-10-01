import pickle, json, sys, os

output_file = os.path.join(r'c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp', 'agqa_pkl_analysis_output.txt')
f_out = open(output_file, 'w', encoding='utf-8')

def p(s=''):
    f_out.write(str(s) + '\n')

def safe_repr(obj, max_depth=5, current_depth=0):
    if current_depth >= max_depth:
        return '...'
    if isinstance(obj, dict):
        items = []
        for i, (k, v) in enumerate(obj.items()):
            if i >= 5:
                items.append('... ({} more keys)'.format(len(obj) - 5))
                break
            items.append('{}: {}'.format(repr(k), safe_repr(v, max_depth, current_depth + 1)))
        return '{' + ', '.join(items) + '}'
    elif isinstance(obj, (list, tuple)):
        items = []
        for i, v in enumerate(obj):
            if i >= 5:
                items.append('... ({} more items)'.format(len(obj) - 5))
                break
            items.append(safe_repr(v, max_depth, current_depth + 1))
        bracket = '[]' if isinstance(obj, list) else '()'
        return bracket[0] + ', '.join(items) + bracket[1]
    else:
        return repr(obj)

# ==================== TRAIN PKL ====================
p('='*80)
p('FILE 3: AGQA_train_stsgs.pkl')
p('='*80)

train_pkl_path = os.path.join(r'c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp\AGQA_scene_graphs', 'AGQA_train_stsgs.pkl')
file_size = os.path.getsize(train_pkl_path)
p('File size: {:,} bytes ({:.1f} MB)'.format(file_size, file_size/1024/1024))

p('\nLoading pickle file...')
with open(train_pkl_path, 'rb') as f:
    train_sg = pickle.load(f)

p('Top-level type: {}'.format(type(train_sg).__name__))
p('Number of top-level keys: {}'.format(len(train_sg)))

if isinstance(train_sg, dict):
    keys_list = list(train_sg.keys())
    p('Sample top-level keys (first 10): {}'.format(keys_list[:10]))
    p('Key type: {}'.format(type(keys_list[0]).__name__))
    
    # Inspect first entry
    first_key = keys_list[0]
    first_val = train_sg[first_key]
    p('\n--- First entry ---')
    p('Key: {}'.format(repr(first_key)))
    p('Value type: {}'.format(type(first_val).__name__))
    
    if isinstance(first_val, dict):
        p('Value keys: {}'.format(list(first_val.keys())))
        p('Number of sub-keys: {}'.format(len(first_val)))
        
        # Inspect sub-keys
        for sub_key in list(first_val.keys())[:5]:
            sub_val = first_val[sub_key]
            p('\n  Sub-key: {}'.format(repr(sub_key)))
            p('  Sub-value type: {}'.format(type(sub_val).__name__))
            if isinstance(sub_val, dict):
                p('  Sub-value keys: {}'.format(list(sub_val.keys())[:10]))
                p('  Number of sub-sub-keys: {}'.format(len(sub_val)))
                # Go one more level
                for sskey in list(sub_val.keys())[:3]:
                    ssval = sub_val[sskey]
                    p('    Sub-sub-key: {}'.format(repr(sskey)))
                    p('    Sub-sub-value type: {}'.format(type(ssval).__name__))
                    p('    Sub-sub-value: {}'.format(safe_repr(ssval, max_depth=4)))
            elif isinstance(sub_val, (list, tuple)):
                p('  Length: {}'.format(len(sub_val)))
                if len(sub_val) > 0:
                    p('  First element type: {}'.format(type(sub_val[0]).__name__))
                    p('  First element: {}'.format(safe_repr(sub_val[0], max_depth=4)))
                    if len(sub_val) > 1:
                        p('  Second element: {}'.format(safe_repr(sub_val[1], max_depth=4)))
            else:
                p('  Value: {}'.format(safe_repr(sub_val, max_depth=4)))
    
    # Try to print one complete entry
    p('\n--- COMPLETE FIRST ENTRY (safe repr) ---')
    p('Key: {}'.format(repr(first_key)))
    p(safe_repr(first_val, max_depth=6))
    
    # Second entry
    second_key = keys_list[1]
    second_val = train_sg[second_key]
    p('\n--- SECOND ENTRY ---')
    p('Key: {}'.format(repr(second_key)))
    if isinstance(second_val, dict):
        p('Keys: {}'.format(list(second_val.keys())[:10]))
        for sk in list(second_val.keys())[:3]:
            sv = second_val[sk]
            p('  {}: {}'.format(repr(sk), safe_repr(sv, max_depth=4)))
    
    # Collect all object types, relation types, attributes
    p('\n--- COLLECTING METADATA (from first 20 entries) ---')
    all_relations = set()
    all_objects = set()
    all_attributes = set()
    all_frame_keys = set()
    has_bbox = False
    bbox_format = None
    
    for idx, k in enumerate(keys_list[:20]):
        val = train_sg[k]
        if isinstance(val, dict):
            for frame_key, frame_val in val.items():
                all_frame_keys.add(frame_key)
                if isinstance(frame_val, dict):
                    for fk, fv in frame_val.items():
                        if fk == 'objects' or fk == 'object':
                            if isinstance(fv, dict):
                                for obj_id, obj_data in fv.items():
                                    if isinstance(obj_data, dict):
                                        for obj_field in obj_data:
                                            if obj_field in ('class', 'name', 'category'):
                                                all_objects.add(obj_data[obj_field])
                                            if obj_field in ('bbox', 'bounding_box', 'box'):
                                                has_bbox = True
                                                bbox_format = type(obj_data[obj_field]).__name__
                                            if obj_field == 'attributes':
                                                if isinstance(obj_data[obj_field], (list, tuple)):
                                                    all_attributes.update(obj_data[obj_field])
                                                elif isinstance(obj_data[obj_field], dict):
                                                    all_attributes.update(obj_data[obj_field].keys())
                                    elif isinstance(obj_data, str):
                                        all_objects.add(obj_data)
                            elif isinstance(fv, (list, tuple)):
                                for item in fv:
                                    if isinstance(item, str):
                                        all_objects.add(item)
                        if fk == 'relations' or fk == 'relation':
                            if isinstance(fv, dict):
                                for rel_id, rel_data in fv.items():
                                    if isinstance(rel_data, dict):
                                        for rel_field in rel_data:
                                            if rel_field in ('class', 'name', 'predicate', 'type'):
                                                all_relations.add(rel_data[rel_field])
                                    elif isinstance(rel_data, str):
                                        all_relations.add(rel_data)
                                    elif isinstance(rel_data, (list, tuple)):
                                        for item in rel_data:
                                            if isinstance(item, str):
                                                all_relations.add(item)
                                            elif isinstance(item, dict):
                                                for rf in item:
                                                    if rf in ('class', 'name', 'predicate'):
                                                        all_relations.add(item[rf])
                            elif isinstance(fv, (list, tuple)):
                                for item in fv:
                                    if isinstance(item, str):
                                        all_relations.add(item)
                                    elif isinstance(item, dict):
                                        for rf in item:
                                            if rf in ('class', 'name', 'predicate'):
                                                all_relations.add(item[rf])
                        if fk == 'attributes' or fk == 'attribute':
                            if isinstance(fv, (list, tuple)):
                                all_attributes.update([x for x in fv if isinstance(x, str)])
    
    p('Object categories found: {}'.format(sorted(all_objects)[:50]))
    p('Relation types found: {}'.format(sorted(all_relations)[:50]))
    p('Attribute types found: {}'.format(sorted(all_attributes)[:50]))
    p('Frame keys found: {}'.format(sorted(all_frame_keys)[:20]))
    p('Has bounding boxes: {}'.format(has_bbox))
    if has_bbox:
        p('Bounding box format: {}'.format(bbox_format))

# Free memory
del train_sg
import gc
gc.collect()

# ==================== TEST PKL ====================
p('\n\n' + '='*80)
p('FILE 4: AGQA_test_stsgs.pkl')
p('='*80)

test_pkl_path = os.path.join(r'c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp\AGQA_scene_graphs', 'AGQA_test_stsgs.pkl')
file_size2 = os.path.getsize(test_pkl_path)
p('File size: {:,} bytes ({:.1f} MB)'.format(file_size2, file_size2/1024/1024))

p('\nLoading pickle file...')
with open(test_pkl_path, 'rb') as f:
    test_sg = pickle.load(f)

p('Top-level type: {}'.format(type(test_sg).__name__))
p('Number of top-level keys: {}'.format(len(test_sg)))

if isinstance(test_sg, dict):
    keys_list2 = list(test_sg.keys())
    p('Sample top-level keys (first 10): {}'.format(keys_list2[:10]))
    
    first_key2 = keys_list2[0]
    first_val2 = test_sg[first_key2]
    p('\n--- First entry ---')
    p('Key: {}'.format(repr(first_key2)))
    p('Value type: {}'.format(type(first_val2).__name__))
    
    if isinstance(first_val2, dict):
        p('Value keys (first 10): {}'.format(list(first_val2.keys())[:10]))
        p('Number of sub-keys: {}'.format(len(first_val2)))
        
        for sub_key in list(first_val2.keys())[:3]:
            sub_val = first_val2[sub_key]
            p('\n  Sub-key: {}'.format(repr(sub_key)))
            p('  Sub-value type: {}'.format(type(sub_val).__name__))
            p('  Sub-value: {}'.format(safe_repr(sub_val, max_depth=5)))
    
    p('\n--- COMPLETE FIRST ENTRY (safe repr) ---')
    p('Key: {}'.format(repr(first_key2)))
    p(safe_repr(first_val2, max_depth=6))

del test_sg
gc.collect()

f_out.close()
print('Done! PKL analysis written to', output_file)
