from src.constants import OBJECTS, ALL_RELATIONS

# Map relation ID to a list of compatible object IDs, or True if all are compatible
COMPATIBLE_OBJECTS = {
    # Attention & Spatial can apply to almost anything, but let's be safe
    "r1": True,  # looking at
    "r2": True,  # not looking at
    "r4": True,  # above
    "r5": True,  # beneath
    "r6": True,  # in front of
    "r7": True,  # behind
    "r8": True,  # on the side of
    "r9": ["o2", "o9", "o13", "o15", "o27", "o32"],  # in (bag, closet, door, doorway, refrigerator, shelf)
    
    # Contact
    "r10": ["o2", "o4", "o6", "o7", "o8", "o10", "o12", "o17", "o19", "o21", "o23", "o24", "o25", "o26", "o30", "o35"], # carrying
    "r11": ["o4", "o10", "o26"], # covered by (towel, clothes, pillow)
    "r12": ["o12", "o17", "o21"], # drinking from (dish, food, medicine/bottle)
    "r13": ["o17", "o21"], # eating (food, medicine)
    "r14": ["o2", "o6"], # have it on the back (bag)
    "r15": ["o2", "o4", "o6", "o7", "o8", "o10", "o12", "o14", "o17", "o19", "o21", "o23", "o24", "o25", "o26", "o30", "o35"], # holding
    "r16": ["o3", "o8", "o9", "o13", "o15", "o27", "o32", "o36"], # leaning on
    "r17": ["o3", "o16"], # lying on (sofa, floor)
    "r18": True, # not contacting
    "r20": True, # other relationship
    "r21": ["o3", "o8", "o16"], # sitting on (sofa, chair, floor)
    "r22": ["o3", "o8", "o16"], # standing on (sofa, chair, floor)
    "r23": True, # touching
    "r24": ["o14"], # twisting (doorknob)
    "r25": ["o2", "o6", "o10", "o30"], # wearing (bag, clothes, shoe)
    "r26": ["o13", "o16", "o22", "o27", "o32", "o33", "o36", "o19", "o24"], # wiping (door, floor, mirror, fridge, shelf, tv, window, laptop, phone)
}

def is_compatible(rel_id: str, obj_id: str) -> bool:
    if rel_id not in COMPATIBLE_OBJECTS:
        return True
    
    val = COMPATIBLE_OBJECTS[rel_id]
    if isinstance(val, bool):
        return val
        
    return obj_id in val

