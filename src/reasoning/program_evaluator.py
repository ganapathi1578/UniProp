"""Legacy stub evaluator for AGQA functional programs.

.. warning::
    This module is a **stub / prototype** implementation that is superseded by
    :mod:`src.reasoning.semantic_engine`.  It does **not** perform actual
    semantic evaluation — every ``eval_*`` method returns a *lazy tuple*
    ``(FuncName, args)`` rather than a resolved :class:`~src.reasoning.semantic_types.SemanticValue`.

    It is retained in the codebase for historical reference and for situations
    where an un-evaluated (unevaluated, symbolic) representation of the AST is
    useful (e.g. for debugging the parser output or as a placeholder during
    incremental development).

    For production truth-value evaluation, use
    :class:`~src.reasoning.semantic_engine.SemanticEngine` via
    :class:`~src.reasoning.evidence.TruthEvaluator`.

Design rationale for lazy evaluation
-------------------------------------
The original design intent was to support *lazy* / *deferred* evaluation where
sub-expressions would only be evaluated on demand.  In this stub each
``eval_*`` method simply re-wraps its arguments in a tagged tuple, effectively
building a *thunk* rather than executing the operation.  This allowed the
codebase to evolve iteratively: the parser and dispatcher infrastructure could
be tested end-to-end before each ``eval_*`` method was given a real
implementation (which eventually happened in
:class:`~src.reasoning.semantic_engine.SemanticEngine`).
"""

from src.normalization.scene_graph_normalizer import NormalizedSceneGraph


class EvaluatorException(Exception):
    """Raised when :class:`ProgramEvaluator` encounters an unsupported AST node.

    This is the legacy equivalent of
    :class:`~src.reasoning.semantic_types.EvaluatorUnsupportedError`.  It is
    raised by :meth:`ProgramEvaluator.evaluate` when the AST contains a
    function name for which no ``eval_*`` handler method exists on the class.

    Args:
        message: A human-readable description of the unsupported function.
            Typically formatted as ``"UNSUPPORTED: <FuncName>"``.
    """
    pass


class ProgramEvaluator:
    """Stub / legacy evaluator for AGQA AST programs.

    .. deprecated::
        Use :class:`~src.reasoning.semantic_engine.SemanticEngine` for
        production evaluation.  This class returns lazy tuples rather than
        resolved semantic values and does not interact with the scene graph
        in any meaningful way.

    This evaluator accepts a parsed AGQA AST (as produced by
    :func:`~src.reasoning.program_parser.parse_agqa_program`) and dispatches
    each ``(FuncName, args)`` tuple to a corresponding ``eval_<funcname>``
    method.  However, each such method simply returns
    ``(FuncName, args)`` unchanged — i.e. the evaluator performs *no actual
    computation* and merely echoes the AST back in tagged-tuple form.

    The class holds a reference to a
    :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
    (``self.sg``) for potential future use, but none of the stub methods
    currently query it.

    Contrast with :class:`~src.reasoning.semantic_engine.SemanticEngine`:

    +---------------------------+-------------------------------+------------------------------------------+
    | Feature                   | ``ProgramEvaluator`` (stub)   | ``SemanticEngine`` (full)                |
    +===========================+===============================+==========================================+
    | Returns                   | lazy ``(FuncName, args)`` tuple | resolved :class:`~src.reasoning.semantic_types.SemanticValue` |
    +---------------------------+-------------------------------+------------------------------------------+
    | Queries scene graph       | No                            | Yes                                      |
    +---------------------------+-------------------------------+------------------------------------------+
    | Produces truth values     | No                            | Yes (``BOOLEAN`` / ``UNKNOWN``)          |
    +---------------------------+-------------------------------+------------------------------------------+
    | Used in production        | No                            | Yes                                      |
    +---------------------------+-------------------------------+------------------------------------------+

    Attributes:
        sg: The :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
            instance provided at construction time.  Unused by the current stub
            methods, but kept for API compatibility with
            :class:`~src.reasoning.semantic_engine.SemanticEngine`.

    Example::

        from src.reasoning.program_parser import parse_agqa_program
        from src.reasoning.program_evaluator import ProgramEvaluator

        ast = parse_agqa_program("Exists(object, Filter(video, [objects]))")
        evaluator = ProgramEvaluator(sg=some_normalized_sg)
        result = evaluator.evaluate(ast)
        # result == ('Exists', ['object', ('Filter', ['video', ['objects']])])
        # NOTE: No actual scene graph lookup has occurred.
    """

    def __init__(self, sg: NormalizedSceneGraph):
        """Initialise the stub evaluator with a scene graph.

        Args:
            sg: A :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
                instance.  Stored as ``self.sg`` for potential future use; not
                accessed by any of the current stub ``eval_*`` methods.
        """
        self.sg = sg

    def evaluate(self, ast):
        """Dispatch an AST node to the appropriate ``eval_*`` stub method.

        The dispatcher mirrors the interface of
        :meth:`~src.reasoning.semantic_engine.SemanticEngine.evaluate` so that
        the two classes can be swapped during testing.  However, unlike the
        full engine, this method does **not** recursively resolve sub-expressions
        to :class:`~src.reasoning.semantic_types.SemanticValue` objects; it
        instead echoes the AST nodes back as lazy tuples.

        Dispatch logic:

        1. If ``ast`` is a ``str``, return it unchanged (base case for atoms).
        2. If ``ast`` is a ``list``, recursively apply :meth:`evaluate` to
           each element and return the resulting list.
        3. If ``ast`` is a ``tuple`` of ``(func_name, args)``, look up
           ``eval_<func_name.lower()>`` on ``self``.  If found, delegate to it;
           otherwise raise :class:`EvaluatorException`.
        4. Any other type is returned unchanged (fallback for unexpected nodes).

        Args:
            ast: An AST node as returned by
                :func:`~src.reasoning.program_parser.parse_agqa_program`.
                May be a ``str``, ``list``, ``tuple``, or ``None``.

        Returns:
            For ``str`` and non-tuple scalars: the node itself.
            For ``list``: a new list with each element recursively evaluated.
            For ``tuple`` ``(func, args)``: the lazy tuple returned by the
            matching ``eval_*`` stub method.

        Raises:
            EvaluatorException: If the AST contains a function name for which
                no ``eval_*`` method is defined on this class.
        """
        if isinstance(ast, str):
            return ast
        if isinstance(ast, list):
            return [self.evaluate(a) for a in ast]

        if not isinstance(ast, tuple):
            return ast

        func, args = ast

        # Dispatch to specific functions
        method_name = f"eval_{func.lower()}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(args)

        raise EvaluatorException(f"UNSUPPORTED: {func}")

    def eval_filter(self, args):
        """Stub for the AGQA ``Filter`` operator.

        In the full engine, ``Filter(source, [criteria])`` returns all objects
        or actions from ``source`` that satisfy the given filter criteria.  In
        this stub the arguments are returned unchanged as a lazy tuple.

        ``Filter`` signature in AGQA::

            Filter(frame|video, [relations|actions|objects, ...])

        Args:
            args: The evaluated argument list as returned by the parser, e.g.
                ``['video', ['objects']]``.

        Returns:
            The lazy tuple ``("Filter", args)`` with no evaluation performed.
        """
        # Filter(frame, [relations, touching, objects])
        # Returns a set of things depending on context?
        # Let's just return the args for now, as we might need lazy evaluation
        return ("Filter", args)

    def eval_iterate(self, args):
        """Stub for the AGQA ``Iterate`` operator.

        In the full engine, ``Iterate(localized_frame_set, collection)`` filters
        ``collection`` to those items that temporally overlap the given frame set.
        In this stub the arguments are returned unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list, e.g.
                ``[('Localize', [...]), ('Filter', [...])]``.

        Returns:
            The lazy tuple ``("Iterate", args)`` with no evaluation performed.
        """
        # Iterate(video, Filter(frame, ...))
        return ("Iterate", args)

    def eval_onlyitem(self, args):
        """Stub for the AGQA ``OnlyItem`` operator.

        In the full engine, ``OnlyItem(collection)`` asserts that ``collection``
        contains exactly one element and returns it.  In this stub the arguments
        are returned unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list, e.g. ``[('Filter', [...])]``.

        Returns:
            The lazy tuple ``("OnlyItem", args)`` with no evaluation performed.
        """
        return ("OnlyItem", args)

    def eval_query(self, args):
        """Stub for the AGQA ``Query`` operator.

        In the full engine, ``Query(attribute, item)`` retrieves the specified
        attribute (e.g. ``"action"``, ``"start"``, ``"end"``) of the given
        item from the scene graph.  In this stub the arguments are returned
        unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list, e.g. ``['action', ('Iterate', [...])]``.

        Returns:
            The lazy tuple ``("Query", args)`` with no evaluation performed.
        """
        return ("Query", args)

    def eval_exists(self, args):
        """Stub for the AGQA ``Exists`` operator.

        In the full engine, ``Exists(target, collection)`` checks whether
        ``target`` is a member of ``collection`` and returns a boolean
        :class:`~src.reasoning.semantic_types.SemanticValue`.  In this stub
        the arguments are returned unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list, e.g.
                ``['object', ('Filter', ['video', ['objects']])]``.

        Returns:
            The lazy tuple ``("Exists", args)`` with no evaluation performed.
        """
        return ("Exists", args)

    def eval_localize(self, args):
        """Stub for the AGQA ``Localize`` operator.

        In the full engine, ``Localize(relation, ref_action)`` looks up the
        temporal interval of ``ref_action`` in the scene graph and returns a
        :class:`~src.reasoning.semantic_types.SemanticType.FRAME_SET` value
        describing the temporal window implied by ``relation`` (``"before"``,
        ``"after"``, ``"while"``).  In this stub the arguments are returned
        unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list, e.g. ``['after', 'walking']``.

        Returns:
            The lazy tuple ``("Localize", args)`` with no evaluation performed.
        """
        return ("Localize", args)

    def eval_compare(self, args):
        """Stub for the AGQA ``Compare`` operator.

        In the full engine, ``Compare`` selects between two options by applying
        a mapping function (e.g. duration) and returning the option corresponding
        to the higher or lower score.  In this stub the arguments are returned
        unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list, e.g.
                ``[['longer', 'shorter'], [item1_ast, item2_ast], mapping_fn_ast]``.

        Returns:
            The lazy tuple ``("Compare", args)`` with no evaluation performed.
        """
        return ("Compare", args)

    def eval_toaction(self, args):
        """Stub for the AGQA ``ToAction`` operator.

        ``ToAction`` is an AGQA primitive that converts an object reference into
        the associated action context.  In this stub the arguments are returned
        unchanged as a lazy tuple.

        Args:
            args: The evaluated argument list.

        Returns:
            The lazy tuple ``("ToAction", args)`` with no evaluation performed.
        """
        return ("ToAction", args)
