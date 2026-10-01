import random
from typing import Tuple

QUERY_TEMPLATES = {
    "OBJECT_SINGLE": [
        ("obj_s_01", "Which of the following objects is present?"),
        ("obj_s_02", "Identify the single object present in the scene."),
        ("obj_s_03", "Which one of these items appears here?"),
        ("obj_s_04", "What specific object is shown?"),
        ("obj_s_05", "Select the one object that is visible.")
    ],
    "OBJECT_SET": [
        ("obj_m_01", "Which objects are present in the video?"),
        ("obj_m_02", "Select all objects that can be seen."),
        ("obj_m_03", "What items does the scene contain?"),
        ("obj_m_04", "Which of these are visible?"),
        ("obj_m_05", "Identify the objects occurring here.")
    ],
    "OBJECT_VERIFY": [
        ("obj_v_01", "Is this object present in the scene?"),
        ("obj_v_02", "Can this object be seen?"),
        ("obj_v_03", "Verify if the object is visible."),
        ("obj_v_04", "Does the video show this item?"),
        ("obj_v_05", "Is the item observed here?")
    ],
    "ACTION": [
        ("act_01", "What is the person doing?"),
        ("act_02", "Which actions occur?"),
        ("act_03", "What activity is the person performing?"),
        ("act_04", "What is happening in the video?"),
        ("act_05", "Select the actions being performed.")
    ],
    "RELATION": [
        ("rel_01", "Which relationships hold between the person and the object?"),
        ("rel_02", "How is the person related to the object?"),
        ("rel_03", "What relation holds between them?"),
        ("rel_04", "Which relations are true?"),
        ("rel_05", "Describe the interaction between the person and object.")
    ],
    "RELATION_VERIFY": [
        ("rel_v_01", "Does this relationship hold?"),
        ("rel_v_02", "Is the person interacting with the object in this way?"),
        ("rel_v_03", "Verify this spatial relationship."),
        ("rel_v_04", "Is this interaction occurring?"),
        ("rel_v_05", "Can you confirm this relationship?")
    ],
    "TEMPORAL": [
        ("temp_01", "Which event happens first?"),
        ("temp_02", "What occurs before this event?"),
        ("temp_03", "What occurs after this event?"),
        ("temp_04", "Which action happens earlier?"),
        ("temp_05", "What is the chronological sequence?")
    ],
    "GROUNDING_SINGLE": [
        ("grnd_s_01", "Which object corresponds to this localized region?"),
        ("grnd_s_02", "What item is found precisely here?"),
        ("grnd_s_03", "Identify the object in this specific box."),
        ("grnd_s_04", "Which of these is located in the region?"),
        ("grnd_s_05", "Name the localized item.")
    ],
    "GROUNDING_VERIFY": [
        ("grnd_v_01", "Is the specified object located in this region?"),
        ("grnd_v_02", "Does this box contain the item?"),
        ("grnd_v_03", "Verify the object's presence at this location."),
        ("grnd_v_04", "Can you confirm the item is in this bounding box?"),
        ("grnd_v_05", "Is the localization correct?")
    ],
    "UNCERTAINTY": [
        ("unc_01", "Can this be determined from the video?"),
        ("unc_02", "Is there enough evidence to tell?")
    ]
}

def get_query(family: str, rng: random.Random) -> Tuple[str, str]:
    if family not in QUERY_TEMPLATES:
        family = "OBJECT_SET"
    return rng.choice(QUERY_TEMPLATES[family])
