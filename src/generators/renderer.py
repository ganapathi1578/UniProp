"""Natural Language Renderer for Propositions."""

# Mapping to handle grammar
UNCOUNTABLE_OBJECTS = {"food", "clothes", "medicine", "water", "paper"}
VOWEL_SOUNDS = {"a", "e", "i", "o", "u"}

def get_article(noun: str) -> str:
    """Return appropriate article (a/an/some) for a noun."""
    if not noun:
        return ""
    if noun.lower() in UNCOUNTABLE_OBJECTS:
        return ""
    if noun[0].lower() in VOWEL_SOUNDS:
        return "an"
    return "a"

def render_noun_phrase(noun: str, definite: bool = False) -> str:
    if not noun:
        return "something"
    
    if definite:
        if noun.lower() in UNCOUNTABLE_OBJECTS:
            return f"the {noun}"
        return f"the {noun}"
        
    art = get_article(noun)
    if art:
        return f"{art} {noun}"
    return noun

def render_existence(obj_name: str, polarity: str) -> str:
    np = render_noun_phrase(obj_name, definite=False).capitalize()
    verb = "are" if obj_name.lower() == "clothes" else "is"
    neg = " not" if polarity == "negative" else ""
    return f"{np} {verb}{neg} present."

def render_relation(rel_name: str, obj_name: str, polarity: str) -> str:
    rel_name = rel_name.replace("have it on the back", "carrying on their back")
    rel_name = rel_name.replace("other relationship", "interacting with")
    np = render_noun_phrase(obj_name, definite=True)
    if polarity == "negative":
        if rel_name.startswith("not "):
            return f"The person is {rel_name[4:]} {np}."
        return f"The person is not {rel_name} {np}."
    return f"The person is {rel_name} {np}."

def render_action(phrase: str, polarity: str = "positive") -> str:
    if not phrase:
        return "The person is doing something."
    neg = "not " if polarity == "negative" else ""
    return f"The person is {neg}{phrase}."

def render_temporal(event_a: str, event_b: str, relation: str, polarity: str = "positive") -> str:
    def clean_event(e):
        if e.startswith("The person is "): e = e.replace("The person is ", "")
        if e.startswith("The person "): e = e.replace("The person ", "")
        if e.endswith("."): e = e[:-1]
        return e
    ea = clean_event(event_a)
    eb = clean_event(event_b)
    
    neg = "not " if polarity == "negative" else ""
    
    if relation == "before":
        return f"The person is {neg}{ea} before {eb}."
    elif relation == "after":
        return f"The person is {neg}{ea} after {eb}."
    elif relation == "while":
        return f"The person is {neg}{ea} while {eb}."
    return f"The person is {neg}{ea} {relation} {eb}."

def render_grounding(obj_name: str, polarity: str) -> str:
    np = render_noun_phrase(obj_name, definite=True)
    if polarity == "positive":
        return f"This region contains {np}."
    else:
        return f"This region does not contain {np}."

