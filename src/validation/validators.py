"""Validators for generated proposition-group examples.

This module provides a single public entry point,
:func:`validate_example`, which runs a battery of structural, semantic, and
surface-level quality checks on a :class:`~src.schema.PropositionGroup`.

The function is invoked as a post-generation gate: any group that fails one or
more checks is discarded (or flagged for review) before being written to the
final dataset.  All checks are **non-destructive** — they never mutate the
input; they only accumulate human-readable error strings.

Checks performed (in order)
----------------------------
1. **Proposition count range** – the number of propositions must be in
   ``[2, 40]`` for non-``three_way`` tasks.
2. **Three-way count** – ``three_way`` tasks must have exactly 3 propositions.
3. **Label cardinality** – ``binary``, ``single_choice``, and ``three_way``
   tasks must each have exactly one correct (label == 1) proposition.
4. **Query-to-task compatibility** – the ``query_template_id`` prefix must
   match the reasoning type and task type.
5. **Semantic contradiction detection** – the same canonical semantic
   signature cannot be simultaneously labelled positive and negative.
6. **Mutual exclusivity detection** – relation pairs that cannot both be true
   (e.g. *above* / *beneath*) are checked across all positive-truth
   propositions.
7. **Temporal consistency checks** – detects cyclic or contradictory
   *before* / *after* temporal orderings.
8. **NONE proposition consistency** – if a ``logical_none`` proposition is
   present, its correctness must be consistent with the other labels.
9. **Duplicate text detection** – proposition texts within a group must be
   unique.
10. **Bad grammar / surface-form checks** – detects unresolved IDs, known
    bad-grammar patterns, generic filler phrases, and double negations.

Module-level constants
----------------------
MUTUALLY_EXCLUSIVE_RELATIONS : dict[str, list[str]]
    Maps a relation name (string, human-readable) to the list of relation
    names that cannot simultaneously be true for the same subject–object pair.
    Used by the mutual-exclusivity check (step 6).
"""

import re
from typing import List, Tuple
from src.schema import PropositionGroup

# ---------------------------------------------------------------------------
# Mutual-exclusion table (human-readable relation names, not class IDs).
# Each key maps to the list of relations whose simultaneous truth would be
# logically contradictory.
# ---------------------------------------------------------------------------

MUTUALLY_EXCLUSIVE_RELATIONS = {
    "looking at": ["not looking at"],
    "not looking at": ["looking at"],
    "above": ["beneath"],
    "beneath": ["above"],
}


def validate_example(group: PropositionGroup) -> Tuple[bool, List[str]]:
    """Run all validators on a :class:`~src.schema.PropositionGroup`.

    Executes ten validation passes on ``group`` in sequence.  Each pass
    appends human-readable error strings to an internal ``errors`` list when
    a violation is detected.  The function does **not** short-circuit on the
    first error; it always runs all checks so that the caller receives a
    complete diagnostic report.

    Args:
        group (PropositionGroup): The proposition group to validate.  This
            object is never mutated.

    Returns:
        Tuple[bool, List[str]]: A 2-tuple where the first element is ``True``
            if **all** checks pass (i.e. ``errors`` is empty) and ``False``
            otherwise, and the second element is the (possibly empty) list of
            error description strings.

    Example::

        ok, errors = validate_example(my_group)
        if not ok:
            for err in errors:
                logging.warning("Validation error: %s", err)
    """
    errors = []

    # ------------------------------------------------------------------
    # Check 1: Proposition count — must be in [2, 40] for non-three_way tasks.
    # three_way tasks are exempt from this range; they are handled separately.
    # ------------------------------------------------------------------
    if not (2 <= group.num_propositions <= 40) and group.task_type != "three_way":
        errors.append(f"Proposition count {group.num_propositions} out of range [2, 40]")

    # ------------------------------------------------------------------
    # Check 2: three_way tasks must have exactly 3 propositions.
    # This enforces the fixed-arity contract for three-option questions.
    # ------------------------------------------------------------------
    if group.task_type == "three_way" and group.num_propositions != 3:
        errors.append(f"Task type three_way must have exactly 3 propositions, found {group.num_propositions}")

    # ------------------------------------------------------------------
    # Check 3: Label cardinality for discriminative task types.
    # binary, single_choice, and three_way must each have exactly 1 correct
    # proposition (label == 1).  multi_label is intentionally excluded here.
    # ------------------------------------------------------------------
    labels = [p.label for p in group.propositions if p.label is not None]
    truth_state = getattr(group, "truth_state", None) or (group.propositions[0].truth_state if group.propositions else "TRUE")
    if truth_state != "UNKNOWN" and group.task_type in ["binary", "single_choice", "three_way"]:
        if sum(labels) != 1:
            errors.append(f"Task type {group.task_type} must have exactly 1 correct proposition, found {sum(labels)}")

    # ------------------------------------------------------------------
    # Check 4: Query-to-task compatibility.
    # The query_template_id prefix encodes the reasoning modality (object /
    # relation / temporal / grounding) and the task type (verification /
    # single-choice / multi-label).  Mismatches indicate that the wrong
    # template was used during generation.
    # ------------------------------------------------------------------
    if group.query_template_id:
        tid = group.query_template_id

        if group.reasoning.type == "object_existence":
            # Object existence questions: binary/three_way → "obj_v" prefix;
            # single_choice → "obj_s"; multi_label → "obj_m".
            if group.task_type in ["binary", "three_way"] and not tid.startswith("obj_v"):
                errors.append("Binary object proposition must use an existence/verification query")
            elif group.task_type == "single_choice" and not tid.startswith("obj_s"):
                errors.append("Single-choice object task must use a single-object selection query")
            elif group.task_type == "multi_label" and not tid.startswith("obj_m"):
                errors.append("Multi-label object task must use a selection/set query")

        elif group.reasoning.family in ["spatial", "contact", "attention"]:
            # Relation questions: binary/three_way must use "rel_v" (verification) queries.
            if group.task_type in ["binary", "three_way"] and not tid.startswith("rel_v"):
                errors.append("Binary relation proposition must use a relation verification query")

        elif group.reasoning.family == "temporal" and not tid.startswith("temp"):
            # All temporal questions must use the "temp" query family.
            errors.append("Temporal proposition must use a temporal query")

        elif group.reasoning.family == "grounding":
            # Grounding questions: single_choice → "grnd_s"; binary/three_way → "grnd_v".
            if group.task_type == "single_choice" and not tid.startswith("grnd_s"):
                errors.append("Single-choice grounding task must use a region/object identification query")
            elif group.task_type in ["binary", "three_way"] and not tid.startswith("grnd_v"):
                errors.append("Binary grounding task must use verification query")

    # ------------------------------------------------------------------
    # Check 5: Semantic contradiction detection.
    #
    # Every proposition is reduced to a canonical *semantic signature* tuple
    # and placed into one of two sets:
    #   positive_semantics — signatures that are asserted to be TRUE.
    #   negative_semantics — signatures that are asserted to be FALSE.
    #
    # A signature appears in positive_semantics when:
    #   (truth_state == "TRUE"  AND polarity == "positive"), or
    #   (truth_state == "FALSE" AND polarity == "negative")  [double negation].
    # A signature appears in negative_semantics when:
    #   (truth_state == "FALSE" AND polarity == "positive"), or
    #   (truth_state == "TRUE"  AND polarity == "negative").
    #
    # If the same signature appears in both sets, two propositions assign
    # contradictory truth values to the same fact → validation error.
    #
    # Special cases:
    #   - logical_none propositions are skipped (they carry no atomic fact).
    #   - UNKNOWN propositions are skipped (truth is undetermined).
    #   - compositional_and propositions: if TRUE, each constituent relation
    #     is individually asserted positive; if FALSE, we cannot determine
    #     which conjunct fails, so nothing is added to negative_semantics.
    # ------------------------------------------------------------------
    positive_semantics = set()
    negative_semantics = set()

    def add_semantics(p_truth, p_pol, p_sig):
        """Classify a semantic signature into positive or negative truth sets.

        Args:
            p_truth (str): The proposition's ``truth_state`` field, either
                ``"TRUE"`` or ``"FALSE"``.
            p_pol (str): The proposition's ``polarity`` field, either
                ``"positive"`` or ``"negative"``.
            p_sig (tuple): The canonical semantic signature tuple for this
                proposition, as constructed by the caller.
        """
        if (p_truth == "TRUE" and p_pol == "positive") or (p_truth == "FALSE" and p_pol == "negative"):
            positive_semantics.add(p_sig)
        elif (p_truth == "FALSE" and p_pol == "positive") or (p_truth == "TRUE" and p_pol == "negative"):
            negative_semantics.add(p_sig)

    for p in group.propositions:
        # logical_none propositions don't encode an atomic semantic fact.
        if p.semantics.canonical_type == "logical_none":
            continue
        # UNKNOWN truth state means we cannot classify the proposition.
        if p.truth_state == "UNKNOWN":
            continue

        if p.semantics.canonical_type == "compositional_and":
            # predicate is "rel1 and rel2"
            rels = p.semantics.predicate.split(" and ")
            for r in rels:
                sig = ("relation", p.semantics.subject, r, p.semantics.object)
                if p.truth_state == "TRUE":
                    add_semantics("TRUE", "positive", sig)
                # If FALSE, we don't know which is false, so we don't add to negative_semantics
            continue

        if p.semantics.canonical_type == "temporal":
            # Temporal signatures encode a directed event-ordering relationship.
            sig = (p.semantics.canonical_type, p.semantics.event_a, p.semantics.temporal_relation, p.semantics.event_b)
        else:
            # Standard propositions: (type, subject, predicate, object).
            sig = (p.semantics.canonical_type, p.semantics.subject, p.semantics.predicate, p.semantics.object)

        add_semantics(p.truth_state, p.semantics.polarity, sig)

    # Any overlap between the two sets is a direct label contradiction.
    if positive_semantics.intersection(negative_semantics):
        errors.append("Contradictory labels for same semantic signature")

    # ------------------------------------------------------------------
    # Check 6 & 7: Recursive satisfiability checks.
    #
    # Iterates over all positively-asserted signatures to detect:
    #   6. Mutual exclusivity: two relations whose simultaneous truth is
    #      logically impossible (e.g. "above" and "beneath" for the same pair).
    #   7. Temporal consistency: cyclic or directly contradictory orderings
    #      among "before" / "after" event pairs.
    # ------------------------------------------------------------------
    for sig in positive_semantics:

        # -- Check 6: Mutual exclusivity --------------------------------
        if sig[0] == "relation":
            pred = sig[2]
            if pred in MUTUALLY_EXCLUSIVE_RELATIONS:
                for excl in MUTUALLY_EXCLUSIVE_RELATIONS[pred]:
                    if ("relation", sig[1], excl, sig[3]) in positive_semantics:
                        errors.append(f"Mutually exclusive relations {pred} and {excl} simultaneously true")

        # -- Check 7: Temporal consistency ------------------------------
        if sig[0] == "temporal":
            event_a, rel, event_b = sig[1], sig[2], sig[3]
            if rel == "before":
                # "A before B" and "A after B" are contradictory.
                # Check if A after B is also true
                if ("temporal", event_a, "after", event_b) in positive_semantics:
                    errors.append("A before B and A after B both true")
                # "A before B" and "B before A" form a cycle.
                # Check if B before A is also true
                if ("temporal", event_b, "before", event_a) in positive_semantics:
                    errors.append("A before B and B before A both true")
            elif rel == "after":
                # "A after B" and "B after A" form a cycle.
                if ("temporal", event_b, "after", event_a) in positive_semantics:
                    errors.append("A after B and B after A both true")

    # ------------------------------------------------------------------
    # Check 8: NONE proposition consistency.
    #
    # In multi-label tasks a special "logical_none" proposition means
    # "none of the above".  The following invariants must hold:
    #   - If NONE is correct (label == 1), no other proposition may be
    #     correct (other_labels_sum == 0).
    #   - If NONE is incorrect (label == 0), at least one other proposition
    #     must be correct (other_labels_sum > 0).
    # ------------------------------------------------------------------
    none_props = [p for p in group.propositions if p.semantics.canonical_type == "logical_none"]
    if none_props:
        none_p = none_props[0]
        other_labels_sum = sum([p.label for p in group.propositions if p != none_p and p.label is not None])
        if none_p.label == 1 and other_labels_sum > 0:
            errors.append("NONE proposition is correct, but other propositions are also correct")
        elif none_p.label == 0 and other_labels_sum == 0:
            errors.append("NONE proposition is incorrect, but no other propositions are correct")

    # ------------------------------------------------------------------
    # Check 9: Deduplication.
    # All proposition texts within a group must be distinct; duplicate texts
    # indicate a generation fault (same template instantiated twice, etc.).
    # ------------------------------------------------------------------
    texts = [p.text for p in group.propositions]
    if len(set(texts)) != len(texts):
        errors.append("Duplicate proposition texts found")

    # ------------------------------------------------------------------
    # Check 10: Surface-form and grammar checks.
    #
    # Iterates every proposition text and applies the following sub-checks:
    #   a. Empty text — the proposition has no text at all.
    #   b. Unresolved ID — raw class IDs (e.g. "o012", "r03", "v07") were
    #      not substituted with human-readable names before writing the text.
    #   c. "have it on the back" — a known bad grammatical pattern produced
    #      by a specific template bug.
    #   d. Uncountable article — "A food", "A clothes", etc. are grammatically
    #      incorrect; these nouns require "some" or no article.
    #   e. Generic filler — "doing something else entirely" is a placeholder
    #      phrase that should never appear in the final dataset.
    #   f. Double negation — "not not" in any casing suggests a template
    #      composed two negations incorrectly.
    # ------------------------------------------------------------------
    # Matches raw ID tokens: 'o', 'r', 'v', or 'c' followed by 2+ digits.
    unresolved_pattern = re.compile(r'\b[orvc]\d{2,}\b', re.IGNORECASE)

    for p in group.propositions:
        # Sub-check a: Empty text.
        if not p.text:
            errors.append("Empty proposition text")
        # Sub-check b: Unresolved ID in surface text.
        if unresolved_pattern.search(p.text):
            errors.append("Unresolved ID in text")
        # Sub-check c: Known bad-grammar pattern from a specific template.
        if "have it on the back" in p.text:
            errors.append("Bad grammar: have it on the back")
        # Sub-check d: Uncountable noun with indefinite article "A".
        if "A food" in p.text or "A clothes" in p.text or "A water" in p.text or "A medicine" in p.text or "A paper" in p.text:
            errors.append("Bad grammar: uncountable article")
        # Sub-check e: Generic filler phrase used as a lazy negative.
        if "doing something else entirely" in p.text:
            errors.append("Generic filler negative used")
        # Sub-check f: Double negation from mis-composed negative templates.
        if "not not" in p.text.lower():
            errors.append("Double negation")

    return len(errors) == 0, errors
