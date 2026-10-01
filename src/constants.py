"""Taxonomy and constants for AGQA and Action Genome."""

ATTENTION_RELATIONS = {
    "r1": "looking at",
    "r2": "not looking at",
}

SPATIAL_RELATIONS = {
    "r4": "above",
    "r5": "beneath",
    "r6": "in front of",
    "r7": "behind",
    "r8": "on the side of",
    "r9": "in",
}

CONTACT_RELATIONS = {
    "r10": "carrying",
    "r11": "covered by",
    "r12": "drinking from",
    "r13": "eating",
    "r14": "have it on the back",
    "r15": "holding",
    "r16": "leaning on",
    "r17": "lying on",
    "r18": "not contacting",
    "r20": "other relationship",
    "r21": "sitting on",
    "r22": "standing on",
    "r23": "touching",
    "r24": "twisting",
    "r25": "wearing",
    "r26": "wiping",
}

ALL_RELATIONS = {**ATTENTION_RELATIONS, **SPATIAL_RELATIONS, **CONTACT_RELATIONS}

OBJECTS = {
    "o2": "bag", "o3": "sofa/couch", "o4": "towel", "o6": "bag",
    "o7": "broom", "o8": "chair", "o9": "closet/cabinet",
    "o10": "clothes", "o12": "dish", "o13": "door", "o14": "doorknob",
    "o15": "doorway", "o16": "floor", "o17": "food", "o19": "laptop",
    "o20": "light", "o21": "medicine", "o22": "mirror", "o23": "book",
    "o24": "phone/camera", "o25": "picture", "o26": "pillow",
    "o27": "refrigerator", "o30": "shoe", "o32": "shelf",
    "o33": "television", "o35": "vacuum", "o36": "window",
}

# Action classes from Charades (Partial list for generation rules, actual descriptions are in the data)
# Actions are mapped directly via the 'phrase' field in the scene graph 'action' entries.

TASK_TYPES = [
    "binary", "single_choice", "multi_label", "open_answer_derived", "grounding", "temporal"
]

COMPLEXITY_LEVELS = list(range(9)) # 0 to 8
