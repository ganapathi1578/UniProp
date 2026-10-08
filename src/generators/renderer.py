"""Natural-language rendering helpers for UniProp propositions.

This module converts structured proposition components (objects, relations,
actions, temporal events, and spatial groundings) into fluent English sentences
that can be presented to a model or human annotator as multiple-choice options.

Each public function accepts a small set of semantic arguments and returns a
single, well-formed English sentence.  Polarity control (``"positive"`` /
``"negative"``) lets callers generate both affirmative and negated variants of
the same proposition from a unified code path.

Module-level constants:
    UNCOUNTABLE_OBJECTS (set[str]): Nouns that are grammatically mass/uncountable
        and therefore take no indefinite article (e.g. *food*, *water*).
        These nouns also use *"is"* rather than *"are"* in most contexts
        (exception: *clothes*, handled explicitly in :func:`render_existence`).
    VOWEL_SOUNDS (set[str]): The set of lowercase English vowel letters used to
        decide between the articles *"a"* and *"an"*.

Typical usage::

    from src.generators.renderer import render_existence, render_action

    pos = render_existence("book", polarity="positive")  # "A book is present."
    neg = render_action("walking", polarity="negative")  # "The person is not walking."
"""

# ---------------------------------------------------------------------------
# Module-level grammar helpers
# ---------------------------------------------------------------------------

# Mapping to handle grammar
# Nouns in this set are treated as uncountable: they receive no indefinite
# article and are returned as bare nouns by render_noun_phrase.
UNCOUNTABLE_OBJECTS = {"food", "clothes", "medicine", "water", "paper"}

# Single-character lowercase vowels used for a/an article selection.
VOWEL_SOUNDS = {"a", "e", "i", "o", "u"}


# ---------------------------------------------------------------------------
# Article & noun-phrase helpers
# ---------------------------------------------------------------------------

def get_article(noun: str) -> str:
    """Return the appropriate English indefinite article for a noun.

    Handles three cases:

    1. **Empty string** – returns ``""`` so callers can safely concatenate.
    2. **Uncountable noun** – nouns in :data:`UNCOUNTABLE_OBJECTS` are mass
       nouns that take no article in English; returns ``""``.
    3. **Vowel-initial noun** – returns ``"an"``; otherwise returns ``"a"``.

    The check is case-insensitive throughout.

    Args:
        noun (str): The singular noun (or noun phrase head) to article-ise.
            Leading/trailing whitespace is not stripped; pass a clean token.

    Returns:
        str: One of ``"a"``, ``"an"``, or ``""`` (empty string for uncountable
        or empty input).

    Examples::

        >>> get_article("apple")
        'an'
        >>> get_article("book")
        'a'
        >>> get_article("water")
        ''
        >>> get_article("")
        ''
    """
    if not noun:
        return ""
    # Mass/uncountable nouns do not take an indefinite article.
    if noun.lower() in UNCOUNTABLE_OBJECTS:
        return ""
    # Select "an" before vowel sounds, "a" before consonants.
    if noun[0].lower() in VOWEL_SOUNDS:
        return "an"
    return "a"


def render_noun_phrase(noun: str, definite: bool = False) -> str:
    """Render a noun as a complete noun phrase with an appropriate article.

    Produces either a *definite* phrase (``"the <noun>"``) or an *indefinite*
    phrase (``"<article> <noun>"``).  Falls back to ``"something"`` when
    *noun* is empty or falsy.

    For indefinite uncountable nouns (e.g. *food*) the bare noun is returned
    without any article, since :func:`get_article` returns ``""`` for them.

    Args:
        noun (str): The head noun to wrap in a phrase.  Should be a single
            common noun without a determiner already attached.
        definite (bool): If ``True``, the phrase uses the definite article
            ``"the"``.  If ``False`` (the default), an indefinite article is
            selected via :func:`get_article`.

    Returns:
        str: The rendered noun phrase, e.g. ``"a book"``, ``"the apple"``,
        ``"food"`` (uncountable, indefinite), or ``"something"`` (empty noun).

    Examples::

        >>> render_noun_phrase("umbrella")
        'an umbrella'
        >>> render_noun_phrase("book", definite=True)
        'the book'
        >>> render_noun_phrase("food")
        'food'
        >>> render_noun_phrase("")
        'something'
    """
    if not noun:
        return "something"

    if definite:
        # Both countable and uncountable nouns take "the" in the definite case.
        if noun.lower() in UNCOUNTABLE_OBJECTS:
            return f"the {noun}"
        return f"the {noun}"

    # Indefinite: prepend article when non-empty; bare noun for uncountables.
    art = get_article(noun)
    if art:
        return f"{art} {noun}"
    return noun


# ---------------------------------------------------------------------------
# Proposition renderers
# ---------------------------------------------------------------------------

def render_existence(obj_name: str, polarity: str) -> str:
    """Render an existence proposition as a complete English sentence.

    Produces a sentence of the form::

        "<NounPhrase> is [not] present."

    The copula is *"are"* for the plural noun *"clothes"* and *"is"* for all
    other nouns.  The noun phrase is always indefinite (no ``"the"``).

    Args:
        obj_name (str): The object whose existence is asserted or denied
            (e.g. ``"book"``, ``"umbrella"``, ``"clothes"``).
        polarity (str): ``"positive"`` to assert existence; any other value
            (typically ``"negative"``) to negate it.

    Returns:
        str: A capitalised sentence ending with a full stop, e.g.:
            - ``"A book is present."``
            - ``"Clothes are not present."``
            - ``"An umbrella is not present."``

    Examples::

        >>> render_existence("book", "positive")
        'A book is present.'
        >>> render_existence("clothes", "negative")
        'Clothes are not present.'
    """
    # Build an indefinite noun phrase and capitalise the first letter.
    np = render_noun_phrase(obj_name, definite=False).capitalize()
    # "clothes" is a plural-only noun; all others use singular "is".
    verb = "are" if obj_name.lower() == "clothes" else "is"
    # Insert " not" immediately after the copula for negative polarity.
    neg = " not" if polarity == "negative" else ""
    return f"{np} {verb}{neg} present."


def render_relation(rel_name: str, obj_name: str, polarity: str) -> str:
    """Render a person–object relation proposition as a complete English sentence.

    Normalises two informal relation strings before composing the sentence:

    * ``"have it on the back"`` → ``"carrying on their back"``
    * ``"other relationship"``  → ``"interacting with"``

    Produces a sentence of the form::

        "The person is [not] <relation> the <object>."

    Args:
        rel_name (str): The relation label string (e.g. ``"holding"``,
            ``"wearing"``, ``"have it on the back"``).  Normalisation is
            applied in-place before rendering.
        obj_name (str): The object noun involved in the relation
            (e.g. ``"bag"``, ``"food"``).
        polarity (str): ``"positive"`` for an affirmative relation; any other
            value (typically ``"negative"``) to negate it.

    Returns:
        str: A sentence describing the relation, e.g.:
            - ``"The person is holding the bag."``
            - ``"The person is not wearing the hat."``
            - ``"The person is carrying on their back the food."``

    Examples::

        >>> render_relation("holding", "bag", "positive")
        'The person is holding the bag.'
        >>> render_relation("wearing", "hat", "negative")
        'The person is not wearing the hat.'
    """
    # Normalise informal or legacy relation strings to natural English phrases.
    rel_name = rel_name.replace("have it on the back", "carrying on their back")
    rel_name = rel_name.replace("other relationship", "interacting with")
    # Use the definite article for the object since the relation is specific.
    np = render_noun_phrase(obj_name, definite=True)
    neg = "not " if polarity == "negative" else ""
    return f"The person is {neg}{rel_name} {np}."


def render_action(phrase: str, polarity: str = "positive") -> str:
    """Render an action proposition as a complete English sentence.

    Wraps *phrase* in the progressive construction ``"The person is [not] …"``
    and appends a full stop.  Falls back to ``"The person is doing something."``
    when *phrase* is empty or falsy.

    Args:
        phrase (str): The action verb phrase in present-participle form
            (e.g. ``"walking"``, ``"eating a sandwich"``).
        polarity (str): ``"positive"`` (the default) for an affirmative
            statement; any other value (typically ``"negative"``) to negate.

    Returns:
        str: A sentence describing the action, e.g.:
            - ``"The person is walking."``
            - ``"The person is not eating a sandwich."``
            - ``"The person is doing something."`` (when *phrase* is empty).

    Examples::

        >>> render_action("walking")
        'The person is walking.'
        >>> render_action("eating", polarity="negative")
        'The person is not eating.'
        >>> render_action("")
        'The person is doing something.'
    """
    if not phrase:
        # Provide a generic fallback rather than producing a malformed sentence.
        return "The person is doing something."
    neg = "not " if polarity == "negative" else ""
    return f"The person is {neg}{phrase}."


def render_temporal(event_a: str, event_b: str, relation: str, polarity: str = "positive") -> str:
    """Render a temporal relation between two events as a complete English sentence.

    Strips common prefixes (``"The person is "`` / ``"The person "``) and a
    trailing period from both event strings so they can be embedded cleanly
    into the output template.

    Supported *relation* values with dedicated templates:

    * ``"before"`` → ``"The person is [not] <ea> before <eb>."``
    * ``"after"``  → ``"The person is [not] <ea> after <eb>."``
    * ``"while"``  → ``"The person is [not] <ea> while <eb>."``

    Any other relation string is interpolated directly as a fallback.

    Args:
        event_a (str): The primary event description, either raw or already
            rendered by :func:`render_action` (prefix stripping handles both).
        event_b (str): The secondary event description used as the temporal
            reference point.
        relation (str): The temporal connective: ``"before"``, ``"after"``,
            or ``"while"``.  Unknown values are interpolated as-is.
        polarity (str): ``"positive"`` (the default) for an affirmative
            statement; any other value (typically ``"negative"``) to negate
            only *event_a*.

    Returns:
        str: A temporally-qualified sentence, e.g.:
            - ``"The person is walking before running."``
            - ``"The person is not sitting while standing."``
            - ``"The person is eating after cooking."``

    Examples::

        >>> render_temporal("The person is walking.", "running", "before")
        'The person is walking before running.'
        >>> render_temporal("eating", "cooking", "after", polarity="negative")
        'The person is not eating after cooking.'
    """
    def clean_event(e):
        """Strip leading person-subject prefixes and trailing period from *e*.

        Args:
            e (str): Raw event string, potentially prefixed by ``"The person
                is "`` or ``"The person "`` and/or suffixed with ``"."``.

        Returns:
            str: The cleaned verb-phrase fragment suitable for embedding in a
            larger sentence.
        """
        # Remove the most verbose progressive prefix first.
        if e.startswith("The person is "): e = e.replace("The person is ", "")
        # Then remove the shorter subject-only prefix.
        if e.startswith("The person "): e = e.replace("The person ", "")
        # Drop a trailing period so the embedding sentence can add its own.
        if e.endswith("."): e = e[:-1]
        return e

    # Clean both event strings before composing the sentence.
    ea = clean_event(event_a)
    eb = clean_event(event_b)

    # "not " is prepended to event_a only; event_b is always affirmative.
    neg = "not " if polarity == "negative" else ""

    # Select the appropriate temporal template based on the relation label.
    if relation == "before":
        return f"The person is {neg}{ea} before {eb}."
    elif relation == "after":
        return f"The person is {neg}{ea} after {eb}."
    elif relation == "while":
        return f"The person is {neg}{ea} while {eb}."
    # Fallback: interpolate unknown relation strings directly.
    return f"The person is {neg}{ea} {relation} {eb}."


def render_grounding(obj_name: str, polarity: str) -> str:
    """Render a spatial grounding proposition as a complete English sentence.

    Produces a sentence asserting or denying that a bounding-box region
    contains a specific object, using a definite noun phrase.

    Positive form::

        "This region contains the <object>."

    Negative form::

        "This region does not contain the <object>."

    Args:
        obj_name (str): The object noun to look for in the region
            (e.g. ``"bag"``, ``"food"``).
        polarity (str): ``"positive"`` to assert containment; any other value
            (typically ``"negative"``) to deny it.

    Returns:
        str: A grounding sentence, e.g.:
            - ``"This region contains the bag."``
            - ``"This region does not contain the food."``

    Examples::

        >>> render_grounding("bag", "positive")
        'This region contains the bag.'
        >>> render_grounding("food", "negative")
        'This region does not contain the food.'
    """
    # Use a definite noun phrase because the region provides a specific context.
    np = render_noun_phrase(obj_name, definite=True)
    if polarity == "positive":
        return f"This region contains {np}."
    else:
        return f"This region does not contain {np}."
