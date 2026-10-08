"""Hard negative and UNKNOWN generation logic for scene graphs.

This module constructs pools of plausible-but-false propositions (hard
negatives) from a :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`.
It is used during dataset construction to produce high-quality negative
samples that require genuine visual understanding to distinguish from true
propositions, rather than superficial lexical differences.

Two sources of hard negatives are supported:

* **Relation negatives** – relation labels that are demonstrably false for a
  given (subject, object) pair, obtained either via explicit mutual exclusion
  (e.g. *looking at* ↔ *not looking at*) or via the closed-world assumption
  applied to Action Genome's exhaustive per-object annotation scheme.

* **Object negatives** – object class labels that are verifiably absent from
  every frame of the video.

Module-level constants
----------------------
RELATION_FAMILIES : dict[str, str]
    Maps every relation class ID (string) to its semantic family:
    ``"attention"``, ``"spatial"``, or ``"contact"``.  Verb-class relations
    (e.g. ``v000``) are intentionally excluded because Action Genome does not
    provide human-readable names for them, making them unusable in natural-
    language propositions.

MUTUALLY_EXCLUSIVE_RELATIONS : dict[str, list[str]]
    Sparse lookup table of relation class IDs whose truth values are
    logically incompatible.  For example, ``r1`` (*looking at*) and ``r2``
    (*not looking at*) cannot both hold for the same subject–object pair
    simultaneously.
"""

import random
from typing import List, Set, Tuple, Optional
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.constants import ATTENTION_RELATIONS, SPATIAL_RELATIONS, CONTACT_RELATIONS, OBJECTS

# ---------------------------------------------------------------------------
# Build a flat family-lookup table from the three relation-class dicts.
# Verb relations are not added; they have no printable name (e.g. "v000").
# ---------------------------------------------------------------------------

RELATION_FAMILIES = {}
for r_id in ATTENTION_RELATIONS:
    RELATION_FAMILIES[r_id] = "attention"
for r_id in SPATIAL_RELATIONS:
    RELATION_FAMILIES[r_id] = "spatial"
for r_id in CONTACT_RELATIONS:
    RELATION_FAMILIES[r_id] = "contact"

# ---------------------------------------------------------------------------
# Hand-curated mutual exclusion table.
# Keys and values are relation class IDs defined in src.constants.
# ---------------------------------------------------------------------------

MUTUALLY_EXCLUSIVE_RELATIONS = {
    "r1": ["r2"], # looking at -> not looking at
    "r2": ["r1"],
    "r4": ["r5"], # above -> beneath
    "r5": ["r4"],
}


class NegativePools:
    """Pre-computed pools of hard-negative candidates for a single video.

    Given a :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`,
    this class scans every frame to collect:

    * The set of **object class IDs that are present** in at least one frame.
    * The set of **relation triples that are annotated** as true, expressed as
      ``(family, class_id, target_object_id)`` tuples.

    These pools are then queried at generation time to find relation or object
    labels that are either logically contradicted or simply unobserved
    (closed-world false).

    Attributes:
        sg (NormalizedSceneGraph): The source scene graph from which the pools
            are built.
        present_objects (set[str]): Object class IDs observed in at least one
            frame of the video.
        present_relations (set[tuple[str, str, str]]): Relation triples of the
            form ``(family, class_id, target_object_id)`` that are annotated
            as true anywhere in the video.  Only relations belonging to the
            ``"attention"``, ``"spatial"``, and ``"contact"`` families are
            included; verb-class relations are excluded.
    """

    def __init__(self, sg: NormalizedSceneGraph):
        """Initialise the pools by scanning the provided scene graph.

        Args:
            sg (NormalizedSceneGraph): The normalised scene graph for a single
                video clip.  Its ``frames`` dictionary is iterated to populate
                :attr:`present_objects` and :attr:`present_relations`.
        """
        self.sg = sg
        self._build_pools()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pools(self):
        """Populate :attr:`present_objects` and :attr:`present_relations`.

        Iterates over every frame in the scene graph.  For each frame:

        * Every annotated object's ``class_id`` is added to
          :attr:`present_objects`.
        * Every relation of type ``"attention"``, ``"spatial"``, or
          ``"contact"`` expands into one entry per ``object_id`` in
          ``rel.object_ids``, stored as the triple
          ``(rel.type, rel.class_id, obj_target)`` in
          :attr:`present_relations`.

        Verb-class relations are silently skipped because Action Genome does
        not expose human-readable labels for them (e.g. ``v000``), making
        them unsuitable for proposition text.
        """
        self.present_objects = set()
        self.present_relations = set()  # (type, class_id, obj_target)

        for frame in self.sg.frames.values():
            for obj in frame.objects.values():
                self.present_objects.add(obj.class_id)
            for rel in frame.relations:
                # ONLY use attention, spatial, contact. IGNORE verbs since we don't have names for v000.
                if rel.type in ['attention', 'spatial', 'contact']:
                    for obj_target in rel.object_ids:
                        self.present_relations.add((rel.type, rel.class_id, obj_target))

    def _is_compatible(self, rel_class: str, target_obj: str) -> bool:
        """Check whether a relation class is semantically plausible for an object.

        Applies a rule-based type-compatibility filter to block obviously
        nonsensical propositions such as *"eating a window"* or
        *"drinking a floor"*.  The filter covers six verb stems; all other
        relations default to ``True`` (permitted) because Action Genome
        annotations already serve as the ground-truth upper bound for
        false-negative detection.

        Args:
            rel_class (str): The relation class ID (e.g. ``"r12"``).
            target_obj (str): The object class ID of the relation's target.

        Returns:
            bool: ``True`` if the relation is plausible for the given object
                class, ``False`` if it is a known semantically invalid
                combination.
        """
        obj_name = OBJECTS.get(target_obj, "")
        if not obj_name:
            return False

        # We don't have human names mapped inside constants easily for all,
        # but we know some bad ones from the prompt (eating window, drinking window, standing on window)
        # We will block the most obvious bad combos based on keywords
        from src.constants import ALL_RELATIONS
        rel_name = ALL_RELATIONS.get(rel_class, "")

        # Eating is only plausible for edible objects.
        if "eat" in rel_name:
            return obj_name in {"food", "sandwich"}
        # Drinking is only plausible for liquid/vessel objects.
        if "drink" in rel_name:
            return obj_name in {"cup_glass_bottle", "medicine", "water"}
        # Standing requires a horizontal surface.
        if "stand" in rel_name:
            return obj_name in {"floor", "bed", "chair", "table"}
        # Sitting requires a seatable surface.
        if "sit" in rel_name:
            return obj_name in {"chair", "sofa_couch", "bed", "floor", "table"}
        # Wearing requires clothing or accessories.
        if "wear" in rel_name:
            return obj_name in {"clothes", "shoe", "bag"}
        # Wiping requires a cleanable surface.
        if "wipe" in rel_name:
            return obj_name in {"table", "mirror", "window", "floor", "television", "refrigerator"}

        # Default allow (since explicit annotation overrides it, but this is for generating fake negatives)
        return True

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_mutually_exclusive_relation(
        self,
        true_rel_class: str,
        target_obj: str,
        rng: random.Random,
    ) -> Optional[str]:
        """Return a relation class that directly contradicts ``true_rel_class``.

        Looks up ``true_rel_class`` in :data:`MUTUALLY_EXCLUSIVE_RELATIONS`.
        If an entry exists, one of the contradictory relation class IDs is
        returned at random using the caller-supplied RNG.

        This is the strongest form of hard negative: it produces a
        semantically *logically false* proposition (e.g. *"not looking at"*
        when the ground truth is *"looking at"*).

        Args:
            true_rel_class (str): The relation class ID that is annotated as
                true for the given subject–object pair.
            target_obj (str): The object class ID of the relation's target.
                Currently unused by this method but accepted for API
                consistency with the other query methods.
            rng (random.Random): A seeded :class:`random.Random` instance used
                for deterministic sampling.

        Returns:
            Optional[str]: A relation class ID whose truth is mutually
                exclusive with ``true_rel_class``, or ``None`` if no explicit
                contradiction is defined for ``true_rel_class``.
        """
        # 1. Direct explicit contradiction
        if true_rel_class in MUTUALLY_EXCLUSIVE_RELATIONS:
            return rng.choice(MUTUALLY_EXCLUSIVE_RELATIONS[true_rel_class])
        return None

    def get_closed_world_false_relation(
        self,
        true_rel_class: str,
        target_obj: str,
        rng: random.Random,
    ) -> Optional[str]:
        """Return a relation class that is false under the closed-world assumption.

        Action Genome annotates *all* relations between a person and each
        interacted object within each frame.  Therefore, if a relation from
        the same semantic family (attention / spatial / contact) is *not*
        annotated for a given ``target_obj`` anywhere in the video, we can
        safely treat it as false.

        Algorithm:

        1. Determine the family of ``true_rel_class`` via
           :data:`RELATION_FAMILIES`.
        2. Enumerate all relation class IDs in that family.
        3. Keep only those that:

           * Are **different** from ``true_rel_class`` (avoiding trivial
             same-label negatives).
           * Are **not** present in :attr:`present_relations` for
             ``target_obj`` (closed-world filter).
           * Pass the semantic compatibility check via
             :meth:`_is_compatible` (syntactic plausibility filter).

        4. Return one uniformly sampled candidate.

        Args:
            true_rel_class (str): The relation class ID that is annotated as
                true for the given subject–object pair.
            target_obj (str): The object class ID of the relation's target.
                Used to restrict the closed-world lookup to relations
                involving this specific object.
            rng (random.Random): A seeded :class:`random.Random` instance used
                for deterministic sampling.

        Returns:
            Optional[str]: A relation class ID from the same family as
                ``true_rel_class`` that is not annotated for ``target_obj``
                and is semantically compatible with it, or ``None`` if no
                qualifying candidate is found.
        """
        family = RELATION_FAMILIES.get(true_rel_class)
        if not family:
            return None

        possible_classes = []
        if family == "attention":
            possible_classes = list(ATTENTION_RELATIONS.keys())
        elif family == "spatial":
            possible_classes = list(SPATIAL_RELATIONS.keys())
        elif family == "contact":
            possible_classes = list(CONTACT_RELATIONS.keys())

        # Filter to those definitely not present for this object in the video
        # AND are typed correctly
        false_classes = [
            c for c in possible_classes
            if c != true_rel_class and (family, c, target_obj) not in self.present_relations and self._is_compatible(c, target_obj)
        ]

        if false_classes:
            return rng.choice(false_classes)
        return None

    def get_false_object(self, rng: random.Random) -> Optional[str]:
        """Return an object class that is verifiably absent from the video.

        Computes the set difference between *all* known object class IDs
        (from :data:`~src.constants.OBJECTS`) and the set of object classes
        actually observed in any frame of the video (:attr:`present_objects`).
        One absent class is sampled uniformly at random.

        Args:
            rng (random.Random): A seeded :class:`random.Random` instance used
                for deterministic sampling.

        Returns:
            Optional[str]: An object class ID from :data:`~src.constants.OBJECTS`
                that does not appear in any frame of the video, or ``None``
                if every known object class is present (extremely unlikely in
                practice).
        """
        all_objects = set(OBJECTS.keys())
        absent_objects = list(all_objects - self.present_objects)
        if absent_objects:
            return rng.choice(absent_objects)
        return None
