"""Parser for AGQA functional program strings.

AGQA (Action Genome Question Answering) represents compositional questions as
*functional programs* — nested function-call expressions written in a
Lisp-like prefix notation.  For example::

    Exists(object, Filter(video, [objects]))
    And(Exists(walking, Filter(video, [actions])),
        Exists(chair, Filter(video, [objects])))

This module provides :func:`parse_agqa_program`, a single-pass recursive
descent parser that converts such strings into a nested Python AST
(Abstract Syntax Tree) consisting of tuples, lists, and plain strings.

The resulting AST is consumed by
:class:`~src.reasoning.semantic_engine.SemanticEngine`, which dispatches each
node to the appropriate ``eval_*`` method.

AST node types
--------------
+--------------------------+---------------------------------------------------+
| Input token(s)           | AST representation                                |
+==========================+===================================================+
| ``FuncName(arg, ...)``   | ``(str, list)`` — ``("FuncName", [arg_ast, ...])``|
+--------------------------+---------------------------------------------------+
| ``[item, ...]``          | ``list`` — ``[item_ast, ...]``                    |
+--------------------------+---------------------------------------------------+
| ``bare_token``           | ``str`` — the raw token string                    |
+--------------------------+---------------------------------------------------+
| ``(empty)``              | ``None``                                          |
+--------------------------+---------------------------------------------------+

Example::

    >>> parse_agqa_program("Exists(object, Filter(video, [objects]))")
    ('Exists', ['object', ('Filter', ['video', ['objects']])])

    >>> parse_agqa_program("walking")
    'walking'

    >>> parse_agqa_program("[actions]")
    ['actions']
"""

import re


def parse_agqa_program(prog_str: str):
    """Parse an AGQA functional program string into a nested AST.

    The function implements a two-phase approach:

    **Phase 1 — Tokenization** (inner function :func:`tokenize`):
    The input string is split into a flat list of atomic tokens using a
    single regular expression.  The regex ``r'\\[|\\]|\\(|\\)|,|[^\\[\\]\\(\\),]+'``
    matches one of the four structural delimiters (``[``, ``]``, ``(``,
    ``)``), the argument separator (``,``), or any maximal run of non-delimiter
    characters (identifiers, whitespace-padded names, etc.).  Tokens are then
    stripped of surrounding whitespace and empty strings are removed so that
    the subsequent parsing step never has to handle blank tokens.

    **Phase 2 — Recursive descent parsing** (inner function :func:`parse`):
    Tokens are consumed left-to-right from a mutable list (used as a stack).
    The parser distinguishes three cases at each recursive call:

    1. **List literal** — the current token is ``[``.  The parser consumes
       tokens recursively until it sees the matching ``]``, building a Python
       ``list`` of sub-ASTs (skipping bare ``","`` tokens).
    2. **Function call** — the current token is an identifier *immediately
       followed* by ``(``.  The parser records the identifier as the function
       name, consumes the ``(``, then recursively parses arguments until it
       sees the matching ``)``, producing a ``(name, args_list)`` tuple.
    3. **Atom** — any other token is returned as a bare ``str``.

    Commas (``,``) are consumed during list/argument parsing but are **not**
    added to the result list; they serve purely as separators.

    Args:
        prog_str: A well-formed AGQA functional program string such as::

                "Exists(object, Filter(video, [objects]))"
                "And(Exists(walking, Filter(video, [actions])), ...)"

    Returns:
        The root AST node, which is one of:

        - ``(str, list)`` — a function-call node where the first element is
          the function name and the second is a list of argument ASTs.
        - ``list`` — a list-literal node containing zero or more AST nodes.
        - ``str`` — a bare atom (identifier, keyword, or literal).
        - ``None`` — if ``prog_str`` is empty or contains only whitespace /
          separators.

    Examples::

        >>> parse_agqa_program("Exists(object, Filter(video, [objects]))")
        ('Exists', ['object', ('Filter', ['video', ['objects']])])

        >>> parse_agqa_program("[actions]")
        ['actions']

        >>> parse_agqa_program("walking")
        'walking'

        >>> parse_agqa_program(
        ...     "And(Exists(walking, Filter(video, [actions])), "
        ...     "Exists(chair, Filter(video, [objects])))"
        ... )
        ('And', [('Exists', ['walking', ('Filter', ['video', ['actions']])]),
                 ('Exists', ['chair', ('Filter', ['video', ['objects']])])])
    """
    def tokenize(s):
        """Split the program string into a flat list of atomic tokens.

        Uses a single regular expression to match structural delimiters
        (``[``, ``]``, ``(``, ``)``, ``,``) individually and non-delimiter
        runs as single tokens.

        Args:
            s: The raw program string to tokenize.

        Returns:
            A list of raw token strings (including whitespace-padded ones
            that are later stripped by the caller).
        """
        return re.findall(r'\[|\]|\(|\)|,|[^\[\]\(\),]+', s)

    # Tokenize and clean: strip surrounding whitespace, drop blank tokens.
    tokens = tokenize(prog_str)
    tokens = [t.strip() for t in tokens if t.strip()]

    def parse(tokens):
        """Recursively consume tokens and build one AST node.

        This is the core recursive descent routine.  It pops tokens from the
        front of the shared mutable list ``tokens`` and dispatches based on
        the value of the popped token:

        - ``'['`` → enter list-literal mode; collect sub-ASTs until ``']'``.
        - An identifier followed by ``'('`` → function-call mode; collect
          arguments until ``')'``.
        - Anything else → return the token as a bare ``str`` atom.

        Comma tokens are consumed during collection loops but discarded (not
        appended to result lists), because they are purely syntactic
        separators.

        Args:
            tokens: Mutable ``list`` of remaining token strings.  Tokens are
                consumed (popped from the front) as the AST is built.

        Returns:
            One AST node: a ``(str, list)`` tuple, a ``list``, a bare
            ``str``, or ``None`` if ``tokens`` is already empty on entry.
        """
        if not tokens: return None
        tok = tokens.pop(0)
        if tok == '[':
            # Build a list literal: collect items until the closing ']'.
            L = []
            while tokens and tokens[0] != ']':
                res = parse(tokens)
                if res != ',': L.append(res)
            if tokens: tokens.pop(0)  # consume ']'
            return L
        elif tokens and tokens[0] == '(':
            # Function call: current token is the function name.
            func = tok
            tokens.pop(0)  # pop '('
            args = []
            while tokens and tokens[0] != ')':
                res = parse(tokens)
                if res != ',': args.append(res)
            if tokens: tokens.pop(0)  # pop ')'
            return (func, args)
        else:
            # Bare atom: return the token string as-is.
            return tok

    return parse(tokens)
