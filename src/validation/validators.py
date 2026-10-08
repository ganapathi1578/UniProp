"""Validators for generated examples."""
import re
from typing import List, Tuple
from src.schema import PropositionGroup

MUTUALLY_EXCLUSIVE_RELATIONS = {
    "looking at": ["not looking at"],
    "not looking at": ["looking at"],
    "above": ["beneath"],
    "beneath": ["above"],
}

def validate_example(group: PropositionGroup) -> Tuple[bool, List[str]]:
    """Run all validators on a proposition group."""
    errors = []
    
    if not (2 <= group.num_propositions <= 40) and group.task_type != "three_way":
        errors.append(f"Proposition count {group.num_propositions} out of range [2, 40]")
        
    if group.task_type == "three_way" and group.num_propositions != 3:
        errors.append(f"Task type three_way must have exactly 3 propositions, found {group.num_propositions}")

    labels = [p.label for p in group.propositions if p.label is not None]
    if group.task_type in ["binary", "single_choice", "three_way"]:
        if sum(labels) != 1:
            errors.append(f"Task type {group.task_type} must have exactly 1 correct proposition, found {sum(labels)}")
            
    # Query-to-task compatibility
    if group.query_template_id:
        tid = group.query_template_id
        if group.reasoning.type == "object_existence":
            if group.task_type in ["binary", "three_way"] and not tid.startswith("obj_v"):
                errors.append("Binary object proposition must use an existence/verification query")
            elif group.task_type == "single_choice" and not tid.startswith("obj_s"):
                errors.append("Single-choice object task must use a single-object selection query")
            elif group.task_type == "multi_label" and not tid.startswith("obj_m"):
                errors.append("Multi-label object task must use a selection/set query")
        elif group.reasoning.family in ["spatial", "contact", "attention", "spatial_contact"]:
            if group.task_type in ["binary", "three_way"] and not tid.startswith("rel_v"):
                errors.append("Binary relation proposition must use a relation verification query")
        elif group.reasoning.family == "temporal" and not tid.startswith("temp"):
            errors.append("Temporal proposition must use a temporal query")
        elif group.reasoning.family == "grounding":
            if group.task_type == "single_choice" and not tid.startswith("grnd_s"):
                errors.append("Single-choice grounding task must use a region/object identification query")
            elif group.task_type in ["binary", "three_way"] and not tid.startswith("grnd_v"):
                errors.append("Binary grounding task must use verification query")


    # Contradiction Detection
    positive_semantics = set()
    negative_semantics = set()
    
    def add_semantics(p_truth, p_pol, p_sig):
        if (p_truth == "TRUE" and p_pol == "positive") or (p_truth == "FALSE" and p_pol == "negative"):
            positive_semantics.add(p_sig)
        elif (p_truth == "FALSE" and p_pol == "positive") or (p_truth == "TRUE" and p_pol == "negative"):
            negative_semantics.add(p_sig)

    for p in group.propositions:
        if p.semantics.canonical_type == "logical_none": continue
        if p.truth_state == "UNKNOWN": continue
        
        if p.semantics.canonical_type == "compositional_and":
            # predicate is "rel1 and rel2"
            rels = p.semantics.predicate.split(" and ")
            for r in rels:
                sig = ("relation", p.semantics.subject, r, p.semantics.object)
                if p.truth_state == "TRUE" and p.semantics.polarity == "positive":
                    add_semantics("TRUE", "positive", sig)
                # If FALSE or negative polarity, we don't know which is false, so we don't add to negative_semantics
            continue
            
        if p.semantics.canonical_type == "temporal":
            sig = (p.semantics.canonical_type, p.semantics.event_a, p.semantics.temporal_relation, p.semantics.event_b)
        else:
            sig = (p.semantics.canonical_type, p.semantics.subject, p.semantics.predicate, p.semantics.object)
            
        add_semantics(p.truth_state, p.semantics.polarity, sig)
            
    if positive_semantics.intersection(negative_semantics):
        errors.append("Contradictory labels for same semantic signature")
        
    # Recursive Satisfiability Checks
    for sig in positive_semantics:
        if sig[0] == "relation":
            pred = sig[2]
            if pred in MUTUALLY_EXCLUSIVE_RELATIONS:
                for excl in MUTUALLY_EXCLUSIVE_RELATIONS[pred]:
                    if ("relation", sig[1], excl, sig[3]) in positive_semantics:
                        errors.append(f"Mutually exclusive relations {pred} and {excl} simultaneously true")
        if sig[0] == "temporal":
            event_a, rel, event_b = sig[1], sig[2], sig[3]
            if rel == "before":
                # Check if A after B is also true
                if ("temporal", event_a, "after", event_b) in positive_semantics:
                    errors.append("A before B and A after B both true")
                # Check if B before A is also true
                if ("temporal", event_b, "before", event_a) in positive_semantics:
                    errors.append("A before B and B before A both true")
            elif rel == "after":
                if ("temporal", event_b, "after", event_a) in positive_semantics:
                    errors.append("A after B and B after A both true")
                    
    # NONE consistency
    none_props = [p for p in group.propositions if p.semantics.canonical_type == "logical_none"]
    if none_props:
        none_p = none_props[0]
        other_labels_sum = sum([p.label for p in group.propositions if p != none_p and p.label is not None])
        if none_p.label == 1 and other_labels_sum > 0:
            errors.append("NONE proposition is correct, but other propositions are also correct")
        elif none_p.label == 0 and other_labels_sum == 0:
            errors.append("NONE proposition is incorrect, but no other propositions are correct")
            
    # Deduplication
    texts = [p.text for p in group.propositions]
    if len(set(texts)) != len(texts):
        errors.append("Duplicate proposition texts found")
        
    unresolved_pattern = re.compile(r'\b[orvc]\d{2,}\b', re.IGNORECASE)
    for p in group.propositions:
        if not p.text:
            errors.append("Empty proposition text")
        if unresolved_pattern.search(p.text):
            errors.append("Unresolved ID in text")
        if "have it on the back" in p.text:
            errors.append("Bad grammar: have it on the back")
        if "A food" in p.text or "A clothes" in p.text or "A water" in p.text or "A medicine" in p.text or "A paper" in p.text:
            errors.append("Bad grammar: uncountable article")
        if "doing something else entirely" in p.text:
            errors.append("Generic filler negative used")
        if "not not" in p.text.lower():
            errors.append("Double negation")
            
    return len(errors) == 0, errors
