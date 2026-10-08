"""Canonicalization utilities for object and action labels in AGQA scene graphs.

AGQA annotations (and the natural-language answers generated from them) often
contain minor surface-form variations for the same underlying entity:
articles (``"a"``, ``"an"``, ``"the"``), common plural forms (``"chairs"``
vs ``"chair"``), or trivial case differences.  Naïve string equality would
therefore miss many true matches.

This module provides lightweight, *deterministic* canonicalization functions
that normalize object and action labels before comparison.  The approach is
intentionally conservative: rather than applying automated stemming or substring
matching (which can produce false positives), it uses an explicit alias table
for a controlled set of known plural/singular pairs.

Typical usage::

    from src.reasoning.canonicalization import (
        canonicalize_object,
        are_objects_equivalent,
        deduplicate_candidates,
    )

    canon = canonicalize_object("The Chairs")   # -> "chair"
    same  = are_objects_equivalent("shoes", "shoe")  # -> True
    unique = deduplicate_candidates(["Chair", "chairs", "table"], "object")
    # -> ["Chair", "table"]  (first occurrence wins per canonical form)
"""

from typing import Set


def canonicalize_object(obj_name: str) -> str:
    """Normalize an object label to a canonical lowercase form.

    The normalization pipeline applies the following steps in order:

    1. **Lower-case and strip** surrounding whitespace.
    2. **Strip leading articles** — ``"a "``, ``"an "``, ``"the "``, and
       ``"some "`` are removed if the label starts with them.  Only the
       *longest matching prefix* is stripped (iteration order in the ``for``
       loop handles this implicitly because only the first matching prefix is
       consumed).
    3. **Alias resolution** — a curated mapping converts a fixed set of plural
       forms to their singular canonical equivalents.  This list is kept
       intentionally small to avoid unsafe substring-based false positives.

    Args:
        obj_name: The raw object label string as it appears in an AGQA
            annotation or a model prediction.  May contain leading/trailing
            whitespace, mixed case, leading articles, or a known plural form.

    Returns:
        A lowercase, article-free, alias-resolved string suitable for direct
        equality comparison.  If ``obj_name`` does not match any alias, the
        de-articled, lowercase form is returned unchanged.

    Examples::

        >>> canonicalize_object("a Chair")
        'chair'
        >>> canonicalize_object("The Doors")
        'door'
        >>> canonicalize_object("some bottles")
        'bottle'
        >>> canonicalize_object("laptop")
        'laptop'
    """
    obj_name = obj_name.lower().strip()
    # Remove articles
    for prefix in ["a ", "an ", "the ", "some "]:
        if obj_name.startswith(prefix):
            obj_name = obj_name[len(prefix):]

    # Handle common plurals/aliases manually for strict control
    aliases = {
        "clothes": "clothing",
        "shoes": "shoe",
        "doors": "door",
        "windows": "window",
        "books": "book",
        "chairs": "chair",
        "dishes": "dish",
        "bottles": "bottle"
    }
    return aliases.get(obj_name, obj_name)


def canonicalize_action(act_name: str) -> str:
    """Normalize an action phrase to a canonical lowercase form.

    Currently applies only lower-casing and whitespace stripping.  No alias
    table is used for actions because AGQA action phrases are already
    relatively consistent across annotations.

    Args:
        act_name: The raw action phrase string (e.g. ``"Walking"``,
            ``" running "``).

    Returns:
        A lowercase, whitespace-stripped string suitable for equality
        comparison.

    Examples::

        >>> canonicalize_action("Walking")
        'walking'
        >>> canonicalize_action("  Running ")
        'running'
    """
    act_name = act_name.lower().strip()
    return act_name


def are_objects_equivalent(a: str, b: str) -> bool:
    """Determine whether two object label strings refer to the same entity.

    Both labels are passed through :func:`canonicalize_object` before
    comparison, so surface variations such as articles, case, and known plural
    forms are handled transparently.

    Args:
        a: First object label string.
        b: Second object label string.

    Returns:
        ``True`` if the canonical forms of ``a`` and ``b`` are identical;
        ``False`` otherwise.

    Examples::

        >>> are_objects_equivalent("a Chair", "chairs")
        True
        >>> are_objects_equivalent("bottle", "book")
        False
    """
    return canonicalize_object(a) == canonicalize_object(b)


def are_actions_equivalent(a: str, b: str) -> bool:
    """Determine whether two action phrase strings refer to the same action.

    Both phrases are passed through :func:`canonicalize_action` before
    comparison, so case and whitespace differences are handled transparently.

    Args:
        a: First action phrase string.
        b: Second action phrase string.

    Returns:
        ``True`` if the canonical forms of ``a`` and ``b`` are identical;
        ``False`` otherwise.

    Examples::

        >>> are_actions_equivalent("Walking", "walking")
        True
        >>> are_actions_equivalent("running", "sitting")
        False
    """
    return canonicalize_action(a) == canonicalize_action(b)


def deduplicate_candidates(candidates: list, domain: str) -> list:
    """Remove duplicate candidates while preserving insertion order.

    Two candidates are considered duplicates if their canonical forms (as
    produced by :func:`canonicalize_object` or :func:`canonicalize_action`,
    depending on ``domain``) are identical.  When duplicates are found, only
    the **first occurrence** (by position in ``candidates``) is kept in the
    output list; subsequent duplicates are silently dropped.

    This is useful when building answer-candidate lists from heterogeneous
    sources (e.g. scene-graph objects combined with model predictions) where
    the same entity may appear under slightly different surface forms.

    Args:
        candidates: An ordered list of raw label strings to deduplicate.
        domain: Selects the canonicalization function to apply.  Supported
            values are:

            - ``"object"`` — uses :func:`canonicalize_object`.
            - ``"action"`` — uses :func:`canonicalize_action`.
            - Any other string — applies a generic ``str.lower().strip()``
              normalization.

    Returns:
        A new list containing, in original order, the first occurrence of each
        canonically distinct candidate from ``candidates``.

    Examples::

        >>> deduplicate_candidates(["Chair", "chairs", "table"], "object")
        ['Chair', 'table']
        >>> deduplicate_candidates(["Walking", "walking", "Running"], "action")
        ['Walking', 'Running']
        >>> deduplicate_candidates(["Yes", "yes", "No"], "other")
        ['Yes', 'No']
    """
    seen = set()
    result = []

    for cand in candidates:
        if domain == "object":
            canon = canonicalize_object(cand)
        elif domain == "action":
            canon = canonicalize_action(cand)
        else:
            canon = str(cand).lower().strip()

        if canon not in seen:
            seen.add(canon)
            result.append(cand)

    return result
