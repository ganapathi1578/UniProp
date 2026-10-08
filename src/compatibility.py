"""Relation–object compatibility registry for the UniProp benchmark.

This module encodes domain knowledge about which (relation, object) pairs are
semantically plausible in the Action Genome / AG-derived relation taxonomy used
by UniProp.  The information is consumed by the question-generation pipeline to
filter out nonsensical distractors — e.g. a question that asks whether a person
is *eating a chair* would never be generated because ``"r13"`` (eating) is not
compatible with the chair object class.

Two compatibility modes are supported:

* **Universal** (``True``): the relation is plausible with *any* object in the
  benchmark ontology.  Spatial predicates such as *above*, *beneath*, and
  attention predicates such as *looking at* fall into this category.
* **Restricted** (``list[str]``): the relation is plausible only with the
  enumerated object IDs.  Contact/interaction predicates such as *carrying*,
  *eating*, or *wearing* fall into this category.

Typical usage::

    from src.compatibility import is_compatible

    if is_compatible("r13", "o17"):   # eating + food → True
        ...  # include as a valid distractor candidate

Attributes:
    COMPATIBLE_OBJECTS (dict[str, bool | list[str]]): Maps each relation ID
        (``"r1"`` … ``"r26"``) to either ``True`` — meaning the relation is
        universally applicable — or a ``list[str]`` of the object IDs with
        which the relation is semantically plausible.  Relation IDs not present
        in the mapping are treated as universally compatible by
        :func:`is_compatible`.
"""

from src.constants import OBJECTS, ALL_RELATIONS

# ---------------------------------------------------------------------------
# Relation–object compatibility map
# ---------------------------------------------------------------------------
# Keys are relation IDs drawn from the AG relation taxonomy used in UniProp.
# Values are either:
#   True          – the relation is plausible with every object class.
#   list[str]     – the relation is plausible only with the listed object IDs.
#
# Object ID legend (selected):
#   o2  = bag          o3  = bed/sofa    o4  = blanket/towel  o6  = clothes
#   o7  = cup/glass    o8  = chair       o9  = closet         o10 = dish
#   o12 = dish/cup     o13 = door        o14 = doorknob       o15 = doorway
#   o16 = floor        o17 = food        o19 = laptop         o21 = medicine
#   o22 = mirror       o23 = paper       o24 = phone          o25 = pillow
#   o26 = shoe/bag     o27 = refrigerator o30 = towel/clothes  o32 = shelf
#   o33 = tv           o35 = umbrella    o36 = window
# ---------------------------------------------------------------------------

COMPATIBLE_OBJECTS = {
    # ------------------------------------------------------------------
    # Attention relations — apply universally; a person can look at (or
    # deliberately *not* look at) any object in the scene.
    # ------------------------------------------------------------------
    "r1": True,   # looking at
    "r2": True,   # not looking at

    # ------------------------------------------------------------------
    # Spatial / positional relations — purely geometric predicates that
    # are independent of object category and therefore universally valid.
    # ------------------------------------------------------------------
    "r4": True,   # above
    "r5": True,   # beneath
    "r6": True,   # in front of
    "r7": True,   # behind
    "r8": True,   # on the side of

    # Containment: only concave/enclosed objects can physically contain a
    # person's limb or body — bag, closet, door, doorway, refrigerator, shelf.
    "r9": ["o2", "o9", "o13", "o15", "o27", "o32"],  # in

    # ------------------------------------------------------------------
    # Contact / interaction relations
    # ------------------------------------------------------------------

    # Carrying: portable, hand-holdable objects.
    "r10": ["o2", "o4", "o6", "o7", "o8", "o10", "o12", "o17", "o19",
            "o21", "o23", "o24", "o25", "o26", "o30", "o35"],  # carrying

    # Covered by: objects that can drape or wrap around the body.
    "r11": ["o4", "o10", "o26"],  # covered by (towel, clothes, pillow)

    # Drinking from: vessels or containers that hold liquid/medicine.
    "r12": ["o12", "o17", "o21"],  # drinking from (dish, food, medicine/bottle)

    # Eating: consumable items — food and medicine only.
    "r13": ["o17", "o21"],  # eating (food, medicine)

    # Have it on the back: wearable/carryable items worn on the back.
    "r14": ["o2", "o6"],  # have it on the back (bag)

    # Holding: broader than carrying — includes stationary held objects.
    "r15": ["o2", "o4", "o6", "o7", "o8", "o10", "o12", "o14", "o17",
            "o19", "o21", "o23", "o24", "o25", "o26", "o30", "o35"],  # holding

    # Leaning on: large, structurally stable surfaces or furniture.
    "r16": ["o3", "o8", "o9", "o13", "o15", "o27", "o32", "o36"],  # leaning on

    # Lying on: flat, horizontal surfaces large enough to support a body.
    "r17": ["o3", "o16"],  # lying on (sofa, floor)

    # Not contacting — universally valid; absence of contact is always a
    # meaningful and plausible label regardless of object class.
    "r18": True,  # not contacting

    # Other relationship — catch-all category; universally applicable.
    "r20": True,  # other relationship

    # Sitting on: surfaces designed or large enough to be sat upon.
    "r21": ["o3", "o8", "o16"],  # sitting on (sofa, chair, floor)

    # Standing on: same surface classes as sitting on.
    "r22": ["o3", "o8", "o16"],  # standing on (sofa, chair, floor)

    # Touching — any physical contact, universally applicable.
    "r23": True,  # touching

    # Twisting: only the doorknob object class is typically twisted.
    "r24": ["o14"],  # twisting (doorknob)

    # Wearing: wearable items only — bag (worn on shoulder), clothes, shoe.
    "r25": ["o2", "o6", "o10", "o30"],  # wearing (bag, clothes, shoe)

    # Wiping: flat, cleanable surfaces or objects.
    "r26": ["o13", "o16", "o22", "o27", "o32", "o33", "o36",
            "o19", "o24"],  # wiping (door, floor, mirror, fridge, shelf, tv, window, laptop, phone)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_compatible(rel_id: str, obj_id: str) -> bool:
    """Check whether a relation–object pair is semantically compatible.

    A pair is considered compatible when the given object class is a
    plausible argument of the given relation according to the domain
    knowledge encoded in :data:`COMPATIBLE_OBJECTS`.  The function is used
    by the question-generation pipeline to exclude nonsensical distractors
    before they are shown to annotators or models.

    Compatibility is resolved as follows:

    1. If ``rel_id`` is **not** a key in :data:`COMPATIBLE_OBJECTS`, the pair
       is assumed compatible (open-world assumption for unknown relations).
    2. If the mapped value is ``True``, the relation is universally applicable
       and the pair is compatible regardless of ``obj_id``.
    3. Otherwise, the mapped value is a ``list[str]`` of allowed object IDs;
       the pair is compatible if and only if ``obj_id`` appears in that list.

    Args:
        rel_id: The relation identifier to check (e.g. ``"r13"`` for *eating*).
            Must be a string matching the keys used in :data:`COMPATIBLE_OBJECTS`.
        obj_id: The object identifier to test against the relation (e.g.
            ``"o17"`` for *food*).  Must be a string matching the object ID
            scheme used throughout the UniProp ontology.

    Returns:
        ``True`` if the ``(rel_id, obj_id)`` pair is semantically plausible;
        ``False`` otherwise.

    Examples::

        >>> is_compatible("r13", "o17")   # eating + food → compatible
        True
        >>> is_compatible("r13", "o8")    # eating + chair → not compatible
        False
        >>> is_compatible("r1", "o8")     # looking at + chair → universally compatible
        True
        >>> is_compatible("r99", "o8")    # unknown relation → assumed compatible
        True
    """
    if rel_id not in COMPATIBLE_OBJECTS:
        # Unknown relation ID — apply open-world assumption and treat the
        # pair as compatible to avoid silently dropping valid annotations.
        return True

    val = COMPATIBLE_OBJECTS[rel_id]
    if isinstance(val, bool):
        # Universal compatibility flag (currently always True in the registry,
        # but the isinstance check future-proofs against a False sentinel).
        return val

    # Restricted compatibility — check membership in the allowed object list.
    return obj_id in val
