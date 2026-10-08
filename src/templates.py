"""Query templates for question generation.

Each template family contains 15-30 diverse phrasings to avoid repetitive
question formulation.  Every template is a ``(template_id, query_text)`` tuple.
Template IDs follow the ``{family_prefix}_{nn}`` convention.
"""
import random
from typing import Tuple

QUERY_TEMPLATES = {
    # ── Object Existence: Single Choice ─────────────────────────────────────
    "OBJECT_SINGLE": [
        ("obj_s_01", "Which of the following objects is present?"),
        ("obj_s_02", "Identify the single object present in the scene."),
        ("obj_s_03", "Which one of these items appears here?"),
        ("obj_s_04", "What specific object is shown?"),
        ("obj_s_05", "Select the one object that is visible."),
        ("obj_s_06", "Out of the listed objects, which one is in the video?"),
        ("obj_s_07", "Pick the object that actually appears in this scene."),
        ("obj_s_08", "Which item from the choices below can be spotted?"),
        ("obj_s_09", "Determine the object that is present in the footage."),
        ("obj_s_10", "From the options, which object is shown in this clip?"),
        ("obj_s_11", "Which object is visible in this particular video?"),
        ("obj_s_12", "Name the object that is clearly present here."),
        ("obj_s_13", "Indicate the single item that exists in the scene."),
        ("obj_s_14", "Which listed item can be observed in the video?"),
        ("obj_s_15", "Among the following, which object is actually there?"),
        ("obj_s_16", "Point out the one object that appears in this footage."),
        ("obj_s_17", "What object from the choices is depicted here?"),
        ("obj_s_18", "Tell me which object is present among these options."),
        ("obj_s_19", "Which of these items exists within the scene?"),
        ("obj_s_20", "Select the correct object that shows up in the video."),
    ],

    # ── Object Existence: Multi-label Set ───────────────────────────────────
    "OBJECT_SET": [
        ("obj_m_01", "Which objects are present in the video?"),
        ("obj_m_02", "Select all objects that can be seen."),
        ("obj_m_03", "What items does the scene contain?"),
        ("obj_m_04", "Which of these are visible?"),
        ("obj_m_05", "Identify the objects occurring here."),
        ("obj_m_06", "Mark every object that appears in this clip."),
        ("obj_m_07", "Which items from the list can be found in the video?"),
        ("obj_m_08", "Indicate all objects visible in the footage."),
        ("obj_m_09", "What objects show up in this scene?"),
        ("obj_m_10", "From the choices, select those present in the video."),
        ("obj_m_11", "Which of the listed items exist in the scene?"),
        ("obj_m_12", "Choose every object that is actually depicted."),
        ("obj_m_13", "Point out all visible items in this video."),
        ("obj_m_14", "List the objects that are present in this footage."),
        ("obj_m_15", "Identify every item that can be spotted here."),
        ("obj_m_16", "Which objects from the following appear in the video?"),
        ("obj_m_17", "Check off each object that is shown in the scene."),
        ("obj_m_18", "Tell me all items that are present in the clip."),
        ("obj_m_19", "Select the objects that exist within this video."),
        ("obj_m_20", "Among these options, which objects are visible?"),
    ],

    # ── Object Existence: Binary Verification ───────────────────────────────
    "OBJECT_VERIFY": [
        ("obj_v_01", "Is this object present in the scene?"),
        ("obj_v_02", "Can this object be seen?"),
        ("obj_v_03", "Verify if the object is visible."),
        ("obj_v_04", "Does the video show this item?"),
        ("obj_v_05", "Is the item observed here?"),
        ("obj_v_06", "Can you spot this object in the video?"),
        ("obj_v_07", "Does this particular item appear in the scene?"),
        ("obj_v_08", "Is this item actually in the footage?"),
        ("obj_v_09", "Confirm whether this object is present."),
        ("obj_v_10", "Does the scene contain this item?"),
        ("obj_v_11", "Check if this object can be found in the video."),
        ("obj_v_12", "Is the following object visible in the clip?"),
        ("obj_v_13", "Is this specific item part of the scene?"),
        ("obj_v_14", "Determine if the object exists in the video."),
        ("obj_v_15", "Can this item be identified in the scene?"),
        ("obj_v_16", "Is the depicted object actually there?"),
        ("obj_v_17", "Verify the presence of this item in the footage."),
        ("obj_v_18", "Do you see this object anywhere in the video?"),
        ("obj_v_19", "Does this object show up in the scene?"),
        ("obj_v_20", "Tell me if this object is in the video."),
    ],

    # ── Action Recognition ──────────────────────────────────────────────────
    "ACTION": [
        ("act_01", "What is the person doing?"),
        ("act_02", "Which actions occur?"),
        ("act_03", "What activity is the person performing?"),
        ("act_04", "What is happening in the video?"),
        ("act_05", "Select the actions being performed."),
        ("act_06", "Describe the activity shown in this clip."),
        ("act_07", "What action is taking place here?"),
        ("act_08", "Which activity does the person engage in?"),
        ("act_09", "Identify the person's action in this scene."),
        ("act_10", "What is the person currently engaged in doing?"),
        ("act_11", "Determine the activity occurring in the video."),
        ("act_12", "Which of the following actions is the person performing?"),
        ("act_13", "What behavior is displayed in the footage?"),
        ("act_14", "Tell me what the person is doing in this scene."),
        ("act_15", "Name the action the person carries out."),
        ("act_16", "Pick the activity that matches what is shown."),
        ("act_17", "Which action best describes what is happening?"),
        ("act_18", "What task is the person working on?"),
        ("act_19", "Identify the action captured in this video."),
        ("act_20", "Characterize what the person is doing here."),
        ("act_21", "Choose the activity depicted in this clip."),
        ("act_22", "Which of the listed actions takes place in the video?"),
        ("act_23", "What is the person occupied with?"),
        ("act_24", "Select the correct action from the options."),
        ("act_25", "Recognize the activity being performed."),
    ],

    # ── Relation: Multi-choice / Set ────────────────────────────────────────
    "RELATION": [
        ("rel_01", "Which relationships hold between the person and the object?"),
        ("rel_02", "How is the person related to the object?"),
        ("rel_03", "What relation holds between them?"),
        ("rel_04", "Which relations are true?"),
        ("rel_05", "Describe the interaction between the person and object."),
        ("rel_06", "What is the person's spatial or contact relationship with the object?"),
        ("rel_07", "Select the relationships that apply here."),
        ("rel_08", "How does the person interact with the object?"),
        ("rel_09", "Which interaction best describes the scene?"),
        ("rel_10", "Pick the correct person-object relationship."),
        ("rel_11", "What kind of contact exists between the person and the object?"),
        ("rel_12", "Identify the relationship shown in the video."),
        ("rel_13", "Among the choices, which relation is occurring?"),
        ("rel_14", "In what way is the person engaging with the object?"),
        ("rel_15", "Which of these interactions is happening?"),
        ("rel_16", "Determine the relationship depicted between person and object."),
        ("rel_17", "What connection can be observed between the person and the object?"),
        ("rel_18", "From the options, which describes the person-object relation?"),
        ("rel_19", "Select all valid relationships between the person and the item."),
        ("rel_20", "Which interaction is captured in this scene?"),
        ("rel_21", "Tell me how the person relates to the object."),
        ("rel_22", "What is the nature of the person-object interaction?"),
        ("rel_23", "Identify which relational statement is correct."),
        ("rel_24", "Choose the relation that accurately describes the scene."),
        ("rel_25", "Which of the following person-object relations holds true?"),
    ],

    # ── Relation: Binary Verification ───────────────────────────────────────
    "RELATION_VERIFY": [
        ("rel_v_01", "Does this relationship hold?"),
        ("rel_v_02", "Is the person interacting with the object in this way?"),
        ("rel_v_03", "Verify this spatial relationship."),
        ("rel_v_04", "Is this interaction occurring?"),
        ("rel_v_05", "Can you confirm this relationship?"),
        ("rel_v_06", "Is the described person-object relation true?"),
        ("rel_v_07", "Check if this interaction is happening in the video."),
        ("rel_v_08", "Does the scene show this kind of relationship?"),
        ("rel_v_09", "Confirm whether this relation exists in the video."),
        ("rel_v_10", "Is this particular interaction taking place?"),
        ("rel_v_11", "Determine if the person-object relation holds."),
        ("rel_v_12", "Verify whether this contact relationship applies."),
        ("rel_v_13", "Is the stated relationship accurate?"),
        ("rel_v_14", "Does this interaction match what is shown?"),
        ("rel_v_15", "Can this person-object relation be observed?"),
        ("rel_v_16", "Is the person really interacting with the object this way?"),
        ("rel_v_17", "Tell me if this relationship is depicted."),
        ("rel_v_18", "Check whether this relational statement is correct."),
        ("rel_v_19", "Does the footage support this relationship?"),
        ("rel_v_20", "Is the following interaction present in the scene?"),
    ],

    # ── Temporal Ordering ───────────────────────────────────────────────────
    "TEMPORAL": [
        ("temp_01", "Which event happens first?"),
        ("temp_02", "What occurs before this event?"),
        ("temp_03", "What occurs after this event?"),
        ("temp_04", "Which action happens earlier?"),
        ("temp_05", "What is the chronological sequence?"),
        ("temp_06", "Determine the temporal order of these events."),
        ("temp_07", "Which of the following activities occurs first?"),
        ("temp_08", "What event precedes the other?"),
        ("temp_09", "Identify which action takes place before the other."),
        ("temp_10", "In what order do these events unfold?"),
        ("temp_11", "Which activity comes first chronologically?"),
        ("temp_12", "Establish the sequence of the given events."),
        ("temp_13", "Tell me which action happens prior to the other."),
        ("temp_14", "What is the correct ordering of these activities?"),
        ("temp_15", "Which event takes place earlier in the video?"),
        ("temp_16", "Order these actions from earliest to latest."),
        ("temp_17", "Before which event does the first activity occur?"),
        ("temp_18", "Rank these events by when they happen."),
        ("temp_19", "Pick the event that occurs first in time."),
        ("temp_20", "Which action is performed before the other?"),
        ("temp_21", "What is the temporal relationship between these events?"),
        ("temp_22", "Identify the event that precedes all others."),
        ("temp_23", "Determine which activity starts earlier."),
        ("temp_24", "Sort these events in the order they appear."),
        ("temp_25", "From the events listed, which one happens first?"),
        ("temp_26", "Which of these activities was done before the rest?"),
        ("temp_27", "Establish which action comes first in the footage."),
        ("temp_28", "Decide the temporal precedence of these events."),
        ("temp_29", "What event does the person do before the other?"),
        ("temp_30", "In the video, which action starts sooner?"),
    ],

    # ── Temporal: Before/After specific ─────────────────────────────────────
    "TEMPORAL_BEFORE": [
        ("temp_b_01", "Which event happened before the person {event}?"),
        ("temp_b_02", "What did the person do before {event}?"),
        ("temp_b_03", "Identify the activity that preceded {event}."),
        ("temp_b_04", "Which action occurred prior to {event}?"),
        ("temp_b_05", "What took place before the person started {event}?"),
        ("temp_b_06", "Select the event that happened before {event}."),
        ("temp_b_07", "Before {event}, what was the person doing?"),
        ("temp_b_08", "Which activity came first, before {event}?"),
        ("temp_b_09", "Name the action that preceded {event} in the video."),
        ("temp_b_10", "What event was performed prior to {event}?"),
        ("temp_b_11", "Tell me what happened before the person began {event}."),
        ("temp_b_12", "Determine which action took place before {event}."),
        ("temp_b_13", "From the options, which event occurred before {event}?"),
        ("temp_b_14", "Pick the activity the person did before {event}."),
        ("temp_b_15", "Which of these was done earlier than {event}?"),
    ],

    "TEMPORAL_AFTER": [
        ("temp_a_01", "What did the person do after {event}?"),
        ("temp_a_02", "Which event happened after the person {event}?"),
        ("temp_a_03", "Identify the activity that followed {event}."),
        ("temp_a_04", "Which action occurred after {event}?"),
        ("temp_a_05", "What took place after {event} ended?"),
        ("temp_a_06", "Select the event that happened after {event}."),
        ("temp_a_07", "After {event}, what did the person do next?"),
        ("temp_a_08", "Which activity came after {event}?"),
        ("temp_a_09", "Name the action that followed {event} in the video."),
        ("temp_a_10", "What did the person proceed to do after {event}?"),
        ("temp_a_11", "Tell me what happened after the person finished {event}."),
        ("temp_a_12", "Determine which action took place after {event}."),
        ("temp_a_13", "From the options, which event followed {event}?"),
        ("temp_a_14", "Pick the next activity after {event}."),
        ("temp_a_15", "Which of these was done later than {event}?"),
    ],

    # ── Temporal: During/While ──────────────────────────────────────────────
    "TEMPORAL_DURING": [
        ("temp_d_01", "What was the person doing while {event}?"),
        ("temp_d_02", "Which action occurred at the same time as {event}?"),
        ("temp_d_03", "Identify the activity happening simultaneously with {event}."),
        ("temp_d_04", "During {event}, what else was the person doing?"),
        ("temp_d_05", "Which of these events overlapped with {event}?"),
        ("temp_d_06", "What activity took place concurrently with {event}?"),
        ("temp_d_07", "While {event}, what other action occurred?"),
        ("temp_d_08", "Select the action that happened during {event}."),
        ("temp_d_09", "Name the activity that co-occurred with {event}."),
        ("temp_d_10", "Tell me what the person was also doing during {event}."),
    ],

    # ── Grounding: Single Choice ────────────────────────────────────────────
    "GROUNDING_SINGLE": [
        ("grnd_s_01", "Which object corresponds to this localized region?"),
        ("grnd_s_02", "What item is found precisely here?"),
        ("grnd_s_03", "Identify the object in this specific box."),
        ("grnd_s_04", "Which of these is located in the region?"),
        ("grnd_s_05", "Name the localized item."),
        ("grnd_s_06", "What object occupies this bounding box?"),
        ("grnd_s_07", "Select the object that matches the highlighted area."),
        ("grnd_s_08", "Which item from the list is inside this region?"),
        ("grnd_s_09", "Tell me what object is in the marked area."),
        ("grnd_s_10", "Identify the item within the indicated bounding box."),
        ("grnd_s_11", "What is located at the specified spatial position?"),
        ("grnd_s_12", "Pick the object that corresponds to this location."),
        ("grnd_s_13", "Determine the object occupying the boxed region."),
        ("grnd_s_14", "Which object sits inside this specific area?"),
        ("grnd_s_15", "From the options, identify what is in this box."),
        ("grnd_s_16", "Name the item found at the indicated position."),
        ("grnd_s_17", "Which listed object is at this location?"),
        ("grnd_s_18", "Point to the object that belongs in this region."),
        ("grnd_s_19", "Select the item that the bounding box highlights."),
        ("grnd_s_20", "What object is captured within this region?"),
    ],

    # ── Grounding: Binary Verification ──────────────────────────────────────
    "GROUNDING_VERIFY": [
        ("grnd_v_01", "Is the specified object located in this region?"),
        ("grnd_v_02", "Does this box contain the item?"),
        ("grnd_v_03", "Verify the object's presence at this location."),
        ("grnd_v_04", "Can you confirm the item is in this bounding box?"),
        ("grnd_v_05", "Is the localization correct?"),
        ("grnd_v_06", "Does the object appear inside this bounding box?"),
        ("grnd_v_07", "Is this the right object for the marked region?"),
        ("grnd_v_08", "Confirm if the item is within the highlighted area."),
        ("grnd_v_09", "Is the object located at this position?"),
        ("grnd_v_10", "Does the bounding box correctly capture this item?"),
        ("grnd_v_11", "Check whether this object occupies the region."),
        ("grnd_v_12", "Verify if the highlighted area contains the object."),
        ("grnd_v_13", "Is the described item at this spatial location?"),
        ("grnd_v_14", "Does this region correspond to the given object?"),
        ("grnd_v_15", "Tell me if the object belongs to this bounding box."),
        ("grnd_v_16", "Is this item correctly localized here?"),
        ("grnd_v_17", "Determine if the object matches the boxed region."),
        ("grnd_v_18", "Can the item be found in the indicated area?"),
        ("grnd_v_19", "Is the object at the specified location?"),
        ("grnd_v_20", "Verify whether this is the correct grounding."),
    ],

    # ── Grounding: Multi-label ──────────────────────────────────────────────
    "GROUNDING_SET": [
        ("grnd_m_01", "Which of these objects are localized in the frame?"),
        ("grnd_m_02", "Select all objects present at this location."),
        ("grnd_m_03", "Identify the items that appear in this frame region."),
        ("grnd_m_04", "Which listed objects can be found in this frame?"),
        ("grnd_m_05", "Mark all items visible in the specified region."),
        ("grnd_m_06", "Pick every object that exists in this frame area."),
        ("grnd_m_07", "Tell me which objects occupy this frame."),
        ("grnd_m_08", "From the options, which items are in this frame?"),
        ("grnd_m_09", "Indicate all objects located in the frame."),
        ("grnd_m_10", "Which of the following items appear in this region?"),
    ],

    # ── Compositional ───────────────────────────────────────────────────────
    "COMPOSITIONAL": [
        ("comp_01", "Which joint interaction is occurring?"),
        ("comp_02", "Select the combined activity taking place."),
        ("comp_03", "What pair of interactions is happening simultaneously?"),
        ("comp_04", "Which compound relation describes the scene?"),
        ("comp_05", "Identify the joint person-object interaction."),
        ("comp_06", "Pick the composite activity shown in the video."),
        ("comp_07", "Which combined relationship holds true?"),
        ("comp_08", "Determine the pair of interactions occurring."),
        ("comp_09", "What multi-part activity is depicted?"),
        ("comp_10", "From the options, which joint interaction applies?"),
        ("comp_11", "Select the correct combination of relations."),
        ("comp_12", "Tell me which pair of actions is taking place."),
        ("comp_13", "Which composite interaction describes the footage?"),
        ("comp_14", "Identify the compound relationship in the scene."),
        ("comp_15", "Choose the correct joint interaction."),
        ("comp_16", "Which set of simultaneous interactions is true?"),
        ("comp_17", "What combined person-object relation is shown?"),
        ("comp_18", "Determine which pair of relationships holds."),
        ("comp_19", "Pick the composite relation that applies here."),
        ("comp_20", "Name the joint interaction depicted in this video."),
    ],

    # ── Compositional: Verification ─────────────────────────────────────────
    "COMPOSITIONAL_VERIFY": [
        ("comp_v_01", "Is this combined interaction occurring?"),
        ("comp_v_02", "Does this joint relationship hold?"),
        ("comp_v_03", "Verify the composite person-object interaction."),
        ("comp_v_04", "Is this pair of interactions happening?"),
        ("comp_v_05", "Confirm whether both relations are true."),
        ("comp_v_06", "Is this compound activity taking place?"),
        ("comp_v_07", "Does the video show this joint interaction?"),
        ("comp_v_08", "Check if both interactions occur simultaneously."),
        ("comp_v_09", "Is the described composite relation accurate?"),
        ("comp_v_10", "Tell me if this combined interaction is correct."),
        ("comp_v_11", "Verify whether the compound relationship holds."),
        ("comp_v_12", "Does the scene depict this joint relation?"),
        ("comp_v_13", "Can you confirm this pair of interactions?"),
        ("comp_v_14", "Is the stated combined activity true?"),
        ("comp_v_15", "Determine if this composite interaction is occurring."),
    ],

    # ── Uncertainty ─────────────────────────────────────────────────────────
    "UNCERTAINTY": [
        ("unc_01", "Can this be determined from the video?"),
        ("unc_02", "Is there enough evidence to tell?"),
        ("unc_03", "Based on the video, can we be sure?"),
        ("unc_04", "Is the answer determinable from the footage?"),
        ("unc_05", "Can this question be answered from what is shown?"),
        ("unc_06", "Is there sufficient visual evidence to decide?"),
        ("unc_07", "Do we have enough information from the video?"),
        ("unc_08", "Can a definitive answer be given based on the clip?"),
        ("unc_09", "Is it possible to determine this from the scene?"),
        ("unc_10", "Does the video provide enough context to answer?"),
    ],

    # ── Action: Binary Verification ─────────────────────────────────────────
    "ACTION_VERIFY": [
        ("act_v_01", "Is the person performing this action?"),
        ("act_v_02", "Does the video show the person doing this?"),
        ("act_v_03", "Verify if this activity is taking place."),
        ("act_v_04", "Is this action occurring in the scene?"),
        ("act_v_05", "Can you confirm the person is doing this?"),
        ("act_v_06", "Check whether this activity is happening."),
        ("act_v_07", "Is the described action being performed?"),
        ("act_v_08", "Does the person carry out this activity?"),
        ("act_v_09", "Tell me if this action is present in the video."),
        ("act_v_10", "Verify the occurrence of this action."),
        ("act_v_11", "Is the person currently engaged in this activity?"),
        ("act_v_12", "Determine if this action takes place in the footage."),
        ("act_v_13", "Does the scene depict this particular action?"),
        ("act_v_14", "Can this activity be observed in the video?"),
        ("act_v_15", "Confirm whether the person performs this action."),
        ("act_v_16", "Is the stated activity shown in the clip?"),
        ("act_v_17", "Check if the person is doing the described action."),
        ("act_v_18", "Does this action match what is happening?"),
        ("act_v_19", "Is the following activity being carried out?"),
        ("act_v_20", "Verify whether this action is depicted in the scene."),
    ],
}


def get_query(family: str, rng: random.Random) -> Tuple[str, str]:
    """Return a random (template_id, query_text) pair for the given family."""
    if family not in QUERY_TEMPLATES:
        family = "OBJECT_SET"
    return rng.choice(QUERY_TEMPLATES[family])


def get_query_formatted(family: str, rng: random.Random, **kwargs) -> Tuple[str, str]:
    """Return a random (template_id, query_text) pair with placeholder substitution.

    Use for template families that contain ``{event}`` or other placeholders.
    Pass the placeholder values as keyword arguments, e.g.
    ``get_query_formatted("TEMPORAL_BEFORE", rng, event="opening a door")``.
    """
    if family not in QUERY_TEMPLATES:
        family = "OBJECT_SET"
    tid, raw_text = rng.choice(QUERY_TEMPLATES[family])
    try:
        formatted = raw_text.format(**kwargs)
    except KeyError:
        formatted = raw_text
    return tid, formatted
