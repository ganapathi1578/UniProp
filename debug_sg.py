import pickle
import os
from src.normalization.scene_graph_normalizer import normalize_video_sg
from src.negatives.hard_negatives import NegativePools

DATA_ROOT = r"c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp"
train_sg_path = os.path.join(DATA_ROOT, "AGQA_scene_graphs", "AGQA_train_stsgs.pkl")
with open(train_sg_path, 'rb') as f:
    train_sgs_raw = pickle.load(f)

# Take first video
vid = list(train_sgs_raw.keys())[1] # N11GT
raw_sg = train_sgs_raw[vid]

norm_sg = normalize_video_sg(vid, raw_sg)
pools = NegativePools(norm_sg)

print(f"Video {vid}")
print(f"Frames: {len(norm_sg.frames)}")
rels_count = sum(len(f.relations) for f in norm_sg.frames.values())
print(f"Total relations parsed: {rels_count}")
print(f"Present relations in pool: {len(pools.present_relations)}")

if pools.present_relations:
    print(f"Sample: {list(pools.present_relations)[0]}")
else:
    # Let's see why relations weren't parsed
    frame = list(norm_sg.frames.values())[0]
    print(f"First frame relations: {frame.relations}")
    print(f"Raw frame attention: {raw_sg.get(frame.id, {}).get('attention')}")
