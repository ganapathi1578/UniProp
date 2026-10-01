import pickle
from collections import defaultdict
with open(r"c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp\AGQA_scene_graphs\AGQA_train_stsgs.pkl", 'rb') as f:
    stsgs = pickle.load(f)

classes = defaultdict(set)

# Parse up to 200 videos to grab all vocabulary
for i, (vid, sg) in enumerate(stsgs.items()):
    if i > 200: break
    for frame_id, frame in sg.items():
        if isinstance(frame, dict) and frame.get('type') == 'frame':
            objs = frame.get('objects', {})
            if isinstance(objs, dict):
                for v in objs.get('vertices', []):
                    if isinstance(v, dict):
                        classes['object'].add((v.get('class'), v.get('metadata', {}).get('tag', '').split('/')[1] if 'tag' in v.get('metadata', {}) else ''))
                        for rtype in ['attention', 'contact', 'spatial', 'verb']:
                            for rel in v.get(rtype, []):
                                if isinstance(rel, dict):
                                    # the actual class name is sometimes missing from metadata tag, but let's try
                                    classes[rtype].add(rel.get('class'))
        if isinstance(frame, dict) and frame.get('type') == 'action':
            classes['action'].add((frame.get('charades'), frame.get('phrase')))

print("OBJECTS:", classes['object'])
print("ATTENTION:", classes['attention'])
print("CONTACT:", classes['contact'])
print("SPATIAL:", classes['spatial'])
print("VERBS:", classes['verb'])
print("ACTIONS (sample):", list(classes['action'])[:10])
