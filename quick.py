import pickle
import sys

with open(r"c:\Users\GANAPATHI\Desktop\NIT\Research\UniProp\AGQA_scene_graphs\AGQA_train_stsgs.pkl", 'rb') as f:
    stsgs = pickle.load(f)

vid = stsgs['N11GT']
frame = vid['000030']
print("FRAME KEYS:", list(frame.keys()))
print("ATTENTION:", frame.get('attention', []))
if frame.get('attention'):
    print("ATTENTION[0] type:", type(frame['attention'][0]))

print("CONTACT:", frame.get('contact', []))
sys.exit(0)
