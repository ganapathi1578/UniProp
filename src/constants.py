"""Taxonomy and constants for AGQA and Action Genome.

All vocabularies below are exhaustively extracted from the AGQA spatio-temporal
scene graph (STSG) pickle files (AGQA_train_stsgs.pkl + AGQA_test_stsgs.pkl).
No generic/invented entries — every action phrase, object class, and relation
class corresponds to an actual annotation in the dataset.
"""

# ── Relations ──────────────────────────────────────────────────────────────────

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

# ── Objects ────────────────────────────────────────────────────────────────────
# 28 unique object classes present in the AGQA scene graphs.

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

# ── Action Phrases ─────────────────────────────────────────────────────────────
# Complete set of 141 action phrases extracted from the 'phrase' field of
# every 'action' entry across all AGQA scene graphs (train + test).
# These replace the former GENERIC_ACTIONS list with ground-truth vocabulary.

AGQA_ACTION_PHRASES = [
    "awakening in bed",
    "closing a book",
    "closing a box",
    "closing a closet",
    "closing a door",
    "closing a laptop",
    "closing a refrigerator",
    "closing a window",
    "consuming some medicine",
    "dressing themselves",
    "drinking from a cup",
    "eating some food",
    "fixing a door",
    "fixing a doorknob",
    "fixing a light",
    "fixing a vacuum",
    "fixing their hair",
    "going from standing to sitting",
    "grasping onto a doorknob",
    "holding a bag",
    "holding a blanket",
    "holding a book",
    "holding a box",
    "holding a broom",
    "holding a cup of something",
    "holding a dish",
    "holding a laptop",
    "holding a mirror",
    "holding a paper",
    "holding a phone",
    "holding a picture",
    "holding a pillow",
    "holding a shoe",
    "holding a vacuum",
    "holding some clothes",
    "holding some food",
    "holding some medicine",
    "laughing at a picture",
    "laughing at something",
    "laughing at television",
    "lying on a bed",
    "lying on the floor",
    "making some food",
    "opening a bag",
    "opening a book",
    "opening a box",
    "opening a closet",
    "opening a door",
    "opening a laptop",
    "opening a refrigerator",
    "opening a window",
    "playing on a laptop",
    "playing with a phone",
    "pouring something into a cup",
    "putting a bag somewhere",
    "putting a blanket somewhere",
    "putting a book somewhere",
    "putting a box somewhere",
    "putting a broom somewhere",
    "putting a cup somewhere",
    "putting a dish somewhere",
    "putting a laptop somewhere",
    "putting a phone somewhere",
    "putting a picture somewhere",
    "putting a pillow somewhere",
    "putting clothes somewhere",
    "putting on a shoe",
    "putting shoes somewhere",
    "putting some food somewhere",
    "putting something on a table",
    "putting their paper somewhere",
    "reaching for and grabbing a picture",
    "running somewhere",
    "sitting at a table",
    "sitting in a bed",
    "sitting in a chair",
    "sitting on a table",
    "sitting on the floor",
    "smiling at a book",
    "smiling at something",
    "smiling in a mirror",
    "sneezing somewhere",
    "snuggling with a blanket",
    "snuggling with a pillow",
    "standing on a chair",
    "standing up",
    "taking a bag from somewhere",
    "taking a blanket from somewhere",
    "taking a book from somewhere",
    "taking a box from somewhere",
    "taking a broom from somewhere",
    "taking a cup from somewhere",
    "taking a dish from somewhere",
    "taking a laptop from somewhere",
    "taking a phone from somewhere",
    "taking a picture of something",
    "taking a pillow from somewhere",
    "taking a vacuum from somewhere",
    "taking food from somewhere",
    "taking off some shoes",
    "taking paper from somewhere",
    "taking shoes from somewhere",
    "taking some clothes from somewhere",
    "taking something from a box",
    "talking on a phone",
    "throwing a bag somewhere",
    "throwing a blanket somewhere",
    "throwing a book somewhere",
    "throwing a box somewhere",
    "throwing a broom somewhere",
    "throwing a pillow somewhere",
    "throwing clothes somewhere",
    "throwing food somewhere",
    "throwing shoes somewhere",
    "throwing something on the floor",
    "tidying some clothes",
    "tidying something on the floor",
    "tidying up a blanket",
    "tidying up a closet",
    "tidying up a table",
    "tidying up with a broom",
    "turning off a light",
    "turning on a light",
    "undressing themselves",
    "walking through a doorway",
    "washing a cup",
    "washing a dish",
    "washing a mirror",
    "washing a table",
    "washing a window",
    "washing some clothes",
    "washing something with a blanket",
    "washing their hands",
    "watching a book",
    "watching a laptop or something on a laptop",
    "watching a picture",
    "watching outside of a window",
    "watching something in a mirror",
    "watching television",
    "working at a table",
    "working on a book",
]

# Frozen set for O(1) lookups
AGQA_ACTION_PHRASES_SET = frozenset(AGQA_ACTION_PHRASES)

# ── Charades Action IDs ────────────────────────────────────────────────────────
# 141 Charades action class IDs present across all AGQA scene graphs.

CHARADES_ACTION_IDS = [
    "c000", "c001", "c002", "c003", "c004", "c005", "c006", "c007",
    "c008", "c009", "c010", "c011", "c012", "c013", "c014", "c015",
    "c016", "c017", "c018", "c019", "c020", "c021", "c022", "c023",
    "c024", "c025", "c026", "c027", "c028", "c029", "c030", "c031",
    "c032", "c038", "c039", "c040", "c041", "c042", "c043", "c044",
    "c045", "c046", "c047", "c048", "c049", "c050", "c051", "c052",
    "c053", "c054", "c055", "c056", "c057", "c058", "c059", "c060",
    "c061", "c062", "c063", "c064", "c065", "c066", "c070", "c071",
    "c072", "c073", "c074", "c075", "c076", "c077", "c078", "c079",
    "c080", "c083", "c084", "c085", "c086", "c087", "c088", "c089",
    "c090", "c091", "c092", "c093", "c094", "c095", "c096", "c097",
    "c098", "c099", "c100", "c101", "c102", "c103", "c104", "c105",
    "c106", "c107", "c108", "c109", "c110", "c111", "c112", "c113",
    "c114", "c115", "c116", "c117", "c118", "c119", "c120", "c121",
    "c124", "c125", "c126", "c127", "c128", "c129", "c131", "c132",
    "c133", "c134", "c135", "c136", "c137", "c138", "c139", "c140",
    "c141", "c142", "c143", "c144", "c145", "c148", "c149", "c150",
    "c151", "c152", "c153", "c154", "c155",
]

# ── Semantic groupings for action-level hard negatives ─────────────────────────
# Actions clustered by verb/activity type for generating plausible distractors.

ACTION_VERB_GROUPS = {
    "holding": [a for a in AGQA_ACTION_PHRASES if a.startswith("holding ")],
    "putting": [a for a in AGQA_ACTION_PHRASES if a.startswith("putting ")],
    "taking": [a for a in AGQA_ACTION_PHRASES if a.startswith("taking ")],
    "throwing": [a for a in AGQA_ACTION_PHRASES if a.startswith("throwing ")],
    "opening": [a for a in AGQA_ACTION_PHRASES if a.startswith("opening ")],
    "closing": [a for a in AGQA_ACTION_PHRASES if a.startswith("closing ")],
    "washing": [a for a in AGQA_ACTION_PHRASES if a.startswith("washing ")],
    "fixing": [a for a in AGQA_ACTION_PHRASES if a.startswith("fixing ")],
    "watching": [a for a in AGQA_ACTION_PHRASES if a.startswith("watching ")],
    "tidying": [a for a in AGQA_ACTION_PHRASES if a.startswith("tidying ")],
    "sitting": [a for a in AGQA_ACTION_PHRASES if a.startswith("sitting ")],
    "lying": [a for a in AGQA_ACTION_PHRASES if a.startswith("lying ")],
    "smiling_laughing": [a for a in AGQA_ACTION_PHRASES if a.startswith("smiling ") or a.startswith("laughing ")],
    "snuggling": [a for a in AGQA_ACTION_PHRASES if a.startswith("snuggling ")],
}

# ── Task and Complexity ────────────────────────────────────────────────────────

TASK_TYPES = [
    "binary", "single_choice", "multi_label", "open_answer_derived", "grounding", "temporal"
]

COMPLEXITY_LEVELS = list(range(9))  # 0 to 8
