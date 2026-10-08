"""Query template registry for the UniProp benchmark.

This module centralises all natural-language query templates used to probe a
video-language model.  Templates are organised into *semantic families* — each
family groups paraphrases that test the same capability.  At evaluation time a
single template is sampled at random from the chosen family so that benchmark
results are not tied to any particular phrasing.

Typical usage::

    import random
    from src.templates import get_query

    rng = random.Random(42)
    template_id, question_text = get_query("ACTION", rng)

Attributes:
    QUERY_TEMPLATES (dict[str, list[tuple[str, str]]]): Registry mapping each
        semantic family name to a non-empty list of ``(template_id,
        query_text)`` tuples.  ``template_id`` is a short stable identifier
        used for logging and reproducibility; ``query_text`` is the
        human-readable question string fed to the model.
"""

import random
from typing import Tuple

# ---------------------------------------------------------------------------
# Query template registry
# ---------------------------------------------------------------------------

QUERY_TEMPLATES = {
    # ------------------------------------------------------------------
    # OBJECT_SINGLE
    # Single-object identification tasks.  The model must pick exactly one
    # object from a candidate set that is present in the scene.
    # ------------------------------------------------------------------
    "OBJECT_SINGLE": [
        ("obj_s_01", "Which of the following objects is present?"),
        ("obj_s_02", "Identify the single object present in the scene."),
        ("obj_s_03", "Which one of these items appears here?"),
        ("obj_s_04", "What specific object is shown?"),
        ("obj_s_05", "Select the one object that is visible.")
    ],

    # ------------------------------------------------------------------
    # OBJECT_SET
    # Multi-object identification tasks.  The model must enumerate all
    # objects from a candidate set that are visible in the scene.  This
    # family is also used as the fallback when an unrecognised family name
    # is requested via :func:`get_query`.
    # ------------------------------------------------------------------
    "OBJECT_SET": [
        ("obj_m_01", "Which objects are present in the video?"),
        ("obj_m_02", "Select all objects that can be seen."),
        ("obj_m_03", "What items does the scene contain?"),
        ("obj_m_04", "Which of these are visible?"),
        ("obj_m_05", "Identify the objects occurring here.")
    ],

    # ------------------------------------------------------------------
    # OBJECT_VERIFY
    # Binary (yes/no) verification of whether a specific object is present.
    # ------------------------------------------------------------------
    "OBJECT_VERIFY": [
        ("obj_v_01", "Is this object present in the scene?"),
        ("obj_v_02", "Can this object be seen?"),
        ("obj_v_03", "Verify if the object is visible."),
        ("obj_v_04", "Does the video show this item?"),
        ("obj_v_05", "Is the item observed here?")
    ],

    # ------------------------------------------------------------------
    # ACTION
    # Action-recognition tasks.  The model must identify one or more
    # actions or activities performed by the person in the video.
    # ------------------------------------------------------------------
    "ACTION": [
        ("act_01", "What is the person doing?"),
        ("act_02", "Which actions occur?"),
        ("act_03", "What activity is the person performing?"),
        ("act_04", "What is happening in the video?"),
        ("act_05", "Select the actions being performed.")
    ],

    # ------------------------------------------------------------------
    # RELATION
    # Open-ended person–object relation tasks.  The model selects all
    # spatial or contact relationships that hold between the person and a
    # given object.
    # ------------------------------------------------------------------
    "RELATION": [
        ("rel_01", "Which relationships hold between the person and the object?"),
        ("rel_02", "How is the person related to the object?"),
        ("rel_03", "What relation holds between them?"),
        ("rel_04", "Which relations are true?"),
        ("rel_05", "Describe the interaction between the person and object.")
    ],

    # ------------------------------------------------------------------
    # RELATION_VERIFY
    # Binary (yes/no) verification of a specific person–object relation.
    # ------------------------------------------------------------------
    "RELATION_VERIFY": [
        ("rel_v_01", "Does this relationship hold?"),
        ("rel_v_02", "Is the person interacting with the object in this way?"),
        ("rel_v_03", "Verify this spatial relationship."),
        ("rel_v_04", "Is this interaction occurring?"),
        ("rel_v_05", "Can you confirm this relationship?")
    ],

    # ------------------------------------------------------------------
    # TEMPORAL
    # Temporal-reasoning tasks.  The model must determine the chronological
    # ordering of events or actions within a video clip.
    # ------------------------------------------------------------------
    "TEMPORAL": [
        ("temp_01", "Which event happens first?"),
        ("temp_02", "What occurs before this event?"),
        ("temp_03", "What occurs after this event?"),
        ("temp_04", "Which action happens earlier?"),
        ("temp_05", "What is the chronological sequence?")
    ],

    # ------------------------------------------------------------------
    # GROUNDING_SINGLE
    # Spatial-grounding tasks with a single target region.  Given a
    # bounding-box or localised region, the model must identify the object
    # it contains.
    # ------------------------------------------------------------------
    "GROUNDING_SINGLE": [
        ("grnd_s_01", "Which object corresponds to this localized region?"),
        ("grnd_s_02", "What item is found precisely here?"),
        ("grnd_s_03", "Identify the object in this specific box."),
        ("grnd_s_04", "Which of these is located in the region?"),
        ("grnd_s_05", "Name the localized item.")
    ],

    # ------------------------------------------------------------------
    # GROUNDING_VERIFY
    # Binary (yes/no) verification of whether a specific object is located
    # within a given bounding box or region.
    # ------------------------------------------------------------------
    "GROUNDING_VERIFY": [
        ("grnd_v_01", "Is the specified object located in this region?"),
        ("grnd_v_02", "Does this box contain the item?"),
        ("grnd_v_03", "Verify the object's presence at this location."),
        ("grnd_v_04", "Can you confirm the item is in this bounding box?"),
        ("grnd_v_05", "Is the localization correct?")
    ],

    # ------------------------------------------------------------------
    # UNCERTAINTY
    # Meta-cognitive tasks that probe whether the model can recognise when
    # a question cannot be answered from the available visual evidence.
    # This family intentionally has fewer variants to keep the set concise.
    # ------------------------------------------------------------------
    "UNCERTAINTY": [
        ("unc_01", "Can this be determined from the video?"),
        ("unc_02", "Is there enough evidence to tell?")
    ]
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_query(family: str, rng: random.Random) -> Tuple[str, str]:
    """Return a randomly sampled query template from the requested family.

    If ``family`` is not a key in :data:`QUERY_TEMPLATES`, the function
    silently falls back to the ``"OBJECT_SET"`` family so that callers always
    receive a valid template without raising an exception.

    Args:
        family: The name of the semantic query family to sample from (e.g.
            ``"ACTION"``, ``"RELATION"``, ``"TEMPORAL"``).  Must match one of
            the keys in :data:`QUERY_TEMPLATES`; unrecognised values trigger
            the ``"OBJECT_SET"`` fallback.
        rng: A seeded :class:`random.Random` instance used for the selection.
            Passing an explicit ``rng`` (rather than using the module-level
            random state) makes sampling fully reproducible across calls.

    Returns:
        A ``(template_id, query_text)`` tuple where ``template_id`` is the
        stable short identifier of the chosen template (e.g. ``"act_03"``)
        and ``query_text`` is the corresponding human-readable question string
        (e.g. ``"What activity is the person performing?"``).

    Example::

        >>> import random
        >>> from src.templates import get_query
        >>> rng = random.Random(0)
        >>> tid, text = get_query("ACTION", rng)
        >>> tid.startswith("act_")
        True
    """
    if family not in QUERY_TEMPLATES:
        # Unknown family — fall back to a safe multi-object default so the
        # pipeline never stalls due to a misconfigured family name.
        family = "OBJECT_SET"
    return rng.choice(QUERY_TEMPLATES[family])
