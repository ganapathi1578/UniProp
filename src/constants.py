"""Taxonomy and constants for the AGQA / Action Genome dataset pipeline.

This module centralises every hard-coded identifier, vocabulary term, and
configuration constant used throughout the UniProp generation and evaluation
pipeline.  Importing this module provides a single source of truth for:

* **Relation taxonomies** – the three disjoint relation families
  (:data:`ATTENTION_RELATIONS`, :data:`SPATIAL_RELATIONS`,
  :data:`CONTACT_RELATIONS`) plus their union (:data:`ALL_RELATIONS`).
* **Object vocabulary** – the canonical Action Genome object catalogue
  (:data:`OBJECTS`).
* **Task types** – the set of admissible evaluation protocols
  (:data:`TASK_TYPES`).
* **Complexity levels** – the integer difficulty scale
  (:data:`COMPLEXITY_LEVELS`).

All relation and object dicts use **Action Genome relation/object IDs** as
keys (e.g. ``"r1"``, ``"o8"``) and human-readable English labels as values.
These IDs match those used in the raw Action Genome scene-graph JSON files.

Note:
    Action class labels (from the Charades action taxonomy) are intentionally
    *not* stored here as a constant dict because they are read directly from
    the ``"phrase"`` field of each scene graph's ``"action"`` entries at
    runtime.  Only a partial reference comment is kept below for clarity.

Example::

    from src.constants import ALL_RELATIONS, OBJECTS, TASK_TYPES

    # Look up a relation label by its Action Genome ID.
    label = ALL_RELATIONS["r15"]   # -> "holding"

    # Check whether a task type is valid.
    assert "binary" in TASK_TYPES
"""


# ---------------------------------------------------------------------------
# Attention Relations
# ---------------------------------------------------------------------------

# Maps Action Genome attention-relation IDs to their English labels.
# Attention relations capture where the person's gaze is directed.
# These are a disjoint subset of ALL_RELATIONS (keys: "r1" – "r2").
#
# Keys  : Action Genome relation ID strings (format "r<N>").
# Values: Human-readable English relation phrase used in proposition text.
ATTENTION_RELATIONS = {
    "r1": "looking at",      # Person's gaze is directed toward the object.
    "r2": "not looking at",  # Person's gaze is explicitly directed away from the object.
}

# ---------------------------------------------------------------------------
# Spatial Relations
# ---------------------------------------------------------------------------

# Maps Action Genome spatial-relation IDs to their English labels.
# Spatial relations describe the 3-D geometric arrangement of an object
# relative to the person (subject).  They are *not* symmetric.
# These are a disjoint subset of ALL_RELATIONS (keys: "r4" – "r9").
#
# Keys  : Action Genome relation ID strings (format "r<N>").
# Values: Human-readable English relation phrase used in proposition text.
SPATIAL_RELATIONS = {
    "r4": "above",        # Object is positioned above the person.
    "r5": "beneath",      # Object is positioned below / under the person.
    "r6": "in front of",  # Object is in front of the person (closer to camera).
    "r7": "behind",       # Object is behind the person (farther from camera).
    "r8": "on the side of",  # Object is laterally adjacent to the person.
    "r9": "in",           # Person is contained within the object (e.g. in a room).
}

# ---------------------------------------------------------------------------
# Contact Relations
# ---------------------------------------------------------------------------

# Maps Action Genome contact-relation IDs to their English labels.
# Contact relations describe physical interactions or bodily contact between
# the person and an object.  This is the largest relation family.
# These are a disjoint subset of ALL_RELATIONS (keys: "r10" – "r26").
#
# Note: IDs are **not** contiguous (e.g. "r19" and "r28"–"r29" are absent)
# because some relation IDs were removed from the Action Genome taxonomy.
#
# Keys  : Action Genome relation ID strings (format "r<N>").
# Values: Human-readable English relation phrase used in proposition text.
CONTACT_RELATIONS = {
    "r10": "carrying",           # Person is carrying the object.
    "r11": "covered by",         # Person is covered by the object (e.g. blanket).
    "r12": "drinking from",      # Person is drinking from the object.
    "r13": "eating",             # Person is eating the object.
    "r14": "have it on the back",# Person has the object strapped to their back.
    "r15": "holding",            # Person is holding the object in their hand(s).
    "r16": "leaning on",         # Person is leaning against the object.
    "r17": "lying on",           # Person is lying on top of the object.
    "r18": "not contacting",     # Explicit annotation that no contact exists.
    "r20": "other relationship",  # Catch-all for contact not covered by other labels.
    "r21": "sitting on",         # Person is seated on the object.
    "r22": "standing on",        # Person is standing on top of the object.
    "r23": "touching",           # Person is touching the object (generic contact).
    "r24": "twisting",           # Person is twisting / turning the object.
    "r25": "wearing",            # Person is wearing the object as clothing/accessory.
    "r26": "wiping",             # Person is wiping the object (or wiping with it).
}

# ---------------------------------------------------------------------------
# Combined Relation Vocabulary
# ---------------------------------------------------------------------------

# Union of all three relation families.
# Used when a lookup must cover the full relation vocabulary regardless of
# family (e.g. when resolving relation IDs read from a scene graph file).
#
# Keys  : Action Genome relation ID strings (all "r<N>" values above).
# Values: Human-readable English relation phrase.
ALL_RELATIONS = {**ATTENTION_RELATIONS, **SPATIAL_RELATIONS, **CONTACT_RELATIONS}

# ---------------------------------------------------------------------------
# Object Vocabulary
# ---------------------------------------------------------------------------

# Maps Action Genome object IDs to their canonical English class names.
# Only the subset of objects that appear in Charades / Action Genome and
# are used by the proposition-generation rules is listed here.
#
# Note: IDs are **not** contiguous — some object IDs (e.g. "o1", "o5",
# "o11", "o18", "o28", "o29", "o31", "o34") do not appear in the
# taxonomy and are therefore absent.
#
# Keys  : Action Genome object ID strings (format "o<N>").
# Values: Human-readable English object class label used in proposition text.
OBJECTS = {
    "o2":  "bag",             # Generic bag (hand bag, shopping bag, etc.).
    "o3":  "sofa/couch",      # Sofa or couch.
    "o4":  "towel",           # Towel (hand towel, bath towel, etc.).
    "o6":  "bag",             # Secondary bag entry (different annotation context).
    "o7":  "broom",           # Broom used for sweeping.
    "o8":  "chair",           # Chair (desk chair, dining chair, etc.).
    "o9":  "closet/cabinet",  # Closet or cabinet (storage furniture).
    "o10": "clothes",         # Generic clothing item.
    "o12": "dish",            # Dish, plate, or bowl.
    "o13": "door",            # Door (interior or exterior).
    "o14": "doorknob",        # Doorknob or door handle.
    "o15": "doorway",         # Doorway / threshold (architectural feature).
    "o16": "floor",           # Floor surface.
    "o17": "food",            # Generic food item.
    "o19": "laptop",          # Laptop computer.
    "o20": "light",           # Light fixture or lamp.
    "o21": "medicine",        # Medicine (pill bottle, inhaler, etc.).
    "o22": "mirror",          # Mirror.
    "o23": "book",            # Book or notebook.
    "o24": "phone/camera",    # Mobile phone or camera.
    "o25": "picture",         # Picture frame or photograph.
    "o26": "pillow",          # Pillow or cushion.
    "o27": "refrigerator",    # Refrigerator / fridge.
    "o30": "shoe",            # Shoe or footwear.
    "o32": "shelf",           # Shelf (book shelf, wall shelf, etc.).
    "o33": "television",      # Television / TV screen.
    "o35": "vacuum",          # Vacuum cleaner.
    "o36": "window",          # Window.
}

# ---------------------------------------------------------------------------
# Action Classes (informational note — NOT stored as a constant dict)
# ---------------------------------------------------------------------------

# Action classes from Charades (Partial list for generation rules, actual
# descriptions are in the data).
# Actions are mapped directly via the 'phrase' field in the scene graph
# 'action' entries.

# ---------------------------------------------------------------------------
# Task Types
# ---------------------------------------------------------------------------

# Ordered list of all admissible evaluation-task-type identifiers.
# Every ``PropositionGroup.task_type`` value must appear in this list.
#
# Values:
#   "binary"               – Single yes/no proposition (label ∈ {0, 1}).
#   "single_choice"        – One correct answer among mutually exclusive options.
#   "multi_label"          – Multiple propositions may be simultaneously true.
#   "open_answer_derived"  – Open-ended answer derived from a source QA pair.
#   "grounding"            – Model must localise an entity in the video frame.
#   "temporal"             – Model must reason about event ordering over time.
TASK_TYPES = [
    "binary",
    "single_choice",
    "multi_label",
    "open_answer_derived",
    "grounding",
    "temporal",
]

# ---------------------------------------------------------------------------
# Complexity Levels
# ---------------------------------------------------------------------------

# Integer difficulty scale used to stratify evaluation results.
# Range: 0 (simplest, single-hop object-existence) to 8 (most complex,
# multi-hop compositional / temporal chains).
# Generated via list(range(9)) so the list is always [0, 1, 2, …, 8].
#
# Each ``Reasoning.complexity`` value must fall within this range.
COMPLEXITY_LEVELS = list(range(9))  # 0 to 8
