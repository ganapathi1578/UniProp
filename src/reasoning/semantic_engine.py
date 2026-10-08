from src.reasoning.sg_access import iter_actions, iter_objects, iter_frames, iter_relations
"""Full semantic evaluator for AGQA functional programs.

This module contains :class:`SemanticEngine`, the production-grade evaluator
that executes parsed AGQA AST programs against a
:class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph` and
produces typed :class:`~src.reasoning.semantic_types.SemanticValue` results.

AGQA (Action Genome Question Answering) expresses compositional questions as
nested functional programs.  Each node in the AST is a ``(FuncName, args)``
tuple.  The engine maps every function name to a dedicated ``eval_*`` method
via a name-based dispatch (``eval_<funcname.lower()>``).  The methods are
grouped into three broad categories:

**Boolean composition** — ``And``, ``Or``, ``Xor``, ``Not``, ``Subtract``
    Combine or negate boolean or numeric
    :class:`~src.reasoning.semantic_types.SemanticValue` instances produced by
    sub-expressions.

**Core domain operators** — ``Exists``, ``Query``, ``OnlyItem``
    Check membership in sets or retrieve attributes from the scene graph.

**Set / temporal operators** — ``Filter``, ``Iterate``, ``Localize``
    Construct and filter sets of objects or actions, optionally constrained to
    a temporal window derived from the video's action graph.

**Comparison / selection operators** — ``Choose``, ``Compare``, ``Equals``,
``Superlative``
    Select between or rank candidates based on numeric scores or equality.

Typical usage::

    from src.reasoning.program_parser import parse_agqa_program
    from src.reasoning.semantic_engine import SemanticEngine

    ast = parse_agqa_program("Exists(object, Filter(video, [objects]))")
    engine = SemanticEngine(scene_graph=normalized_sg)
    result = engine.evaluate(ast)
    # result.type == SemanticType.BOOLEAN
    # result.value in {True, False}
"""

from typing import Any, List, Dict, Set, Optional, Tuple
from src.reasoning.sg_access import iter_actions, iter_objects, iter_frames, iter_relations
from src.reasoning.semantic_types import SemanticValue, SemanticType, EvaluatorUnsupportedError, MissingEvidenceError
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.reasoning.canonicalization import are_objects_equivalent, are_actions_equivalent, canonicalize_object, canonicalize_action


class SemanticEngine:
    """Full semantic evaluator that executes AGQA AST programs against a scene graph.

    Each AGQA question is associated with a *functional program* — a
    compositional expression that, when evaluated against the spatio-temporal
    scene graph (STSG) of the corresponding video, yields a ground-truth answer.
    :class:`SemanticEngine` implements this evaluation by:

    1. Accepting an AST produced by
       :func:`~src.reasoning.program_parser.parse_agqa_program`.
    2. Recursively evaluating each ``(FuncName, args)`` node by dispatching to
       the matching ``eval_<funcname>`` method.
    3. Querying the underlying
       :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
       (``self.sg``) for frame-level object annotations and video-level action
       intervals.
    4. Returning a :class:`~src.reasoning.semantic_types.SemanticValue` whose
       ``type`` field indicates the result category (e.g. ``BOOLEAN``,
       ``OBJECT_SET``, ``UNKNOWN``).

    The engine raises :class:`~src.reasoning.semantic_types.EvaluatorUnsupportedError`
    for function names it does not recognise, and
    :class:`~src.reasoning.semantic_types.MissingEvidenceError` when the scene
    graph lacks the data required to answer a sub-expression.

    Attributes:
        sg: The :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
            instance that provides frame-level object annotations
            (``sg.frames``) and video-level action intervals (``sg.actions``).

    Example::

        engine = SemanticEngine(scene_graph=normalized_sg)
        result = engine.evaluate(
            parse_agqa_program("Exists(walking, Filter(video, [actions]))")
        )
        print(result.type, result.value)
        # SemanticType.BOOLEAN  True
    """

    def __init__(self, scene_graph: NormalizedSceneGraph):
        """Initialise the engine with a normalised scene graph.

        Args:
            scene_graph: A :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
                instance for the video being evaluated.  All ``eval_*`` methods
                query this object for evidence.
        """
        self.sg = scene_graph

    def evaluate(self, ast: Any, env: Optional[Dict[str, Any]] = None) -> SemanticValue:
        """Recursively evaluate an AST node and return a typed semantic value.

        This is the central dispatch method.  It handles three cases:

        1. **String atom** — if the string is present in ``env`` (the local
           variable environment), return the bound value; otherwise wrap it as
           an ``UNKNOWN`` semantic value so that downstream operators can still
           inspect the raw string.
        2. **Non-tuple / malformed node** — return
           :meth:`~src.reasoning.semantic_types.SemanticValue.unknown` as a
           safe fallback.
        3. **``(operation, args)`` tuple** — look up
           ``eval_<operation.lower()>`` on ``self`` and delegate; raise
           :class:`~src.reasoning.semantic_types.EvaluatorUnsupportedError` if
           no such method exists.

        Args:
            ast: An AST node as produced by
                :func:`~src.reasoning.program_parser.parse_agqa_program`.
                May be a ``str``, ``tuple``, ``list``, or ``None``.
            env: Optional mapping from variable names (``str``) to
                :class:`~src.reasoning.semantic_types.SemanticValue` objects.
                Used by :meth:`eval_compare` and :meth:`eval_superlative` to
                inject per-candidate bindings (e.g. ``{"action": SemanticValue.action("walking")}``).
                Defaults to an empty dict when ``None``.

        Returns:
            A :class:`~src.reasoning.semantic_types.SemanticValue` representing
            the evaluated result of ``ast``.

        Raises:
            EvaluatorUnsupportedError: If ``ast`` is a tuple whose operation
                name has no corresponding ``eval_*`` method on this class.
        """
        env = env or {}

        if isinstance(ast, str):
            if ast in env:
                return env[ast]
            return SemanticValue(SemanticType.UNKNOWN, ast)

        if not isinstance(ast, tuple) or len(ast) != 2:
            return SemanticValue.unknown()

        operation, args = ast
        handler_name = f"eval_{operation.lower()}"
        if hasattr(self, handler_name):
            return getattr(self, handler_name)(args, env)

        raise EvaluatorUnsupportedError(operation)

    # --- BOOLEAN COMPOSITION ---

    def eval_and(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``And`` operator (logical conjunction).

        Both sub-expressions are evaluated eagerly.  If both resolve to
        ``BOOLEAN`` values, their Python ``and`` is returned as a new
        ``BOOLEAN`` result.  If either sub-expression is not boolean, the
        result degrades to ``UNKNOWN``.

        AGQA semantic::

            And(expr1, expr2)  →  expr1 ∧ expr2

        Args:
            args: Exactly two AST nodes ``[expr1, expr2]``.
            env: The local variable environment (passed through to
                recursive :meth:`evaluate` calls).

        Returns:
            A ``BOOLEAN`` :class:`~src.reasoning.semantic_types.SemanticValue`
            with the conjunction result, or an ``UNKNOWN`` value if either
            operand is not boolean.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("And(wrong_args)")
        v1 = self.evaluate(args[0], env)
        v2 = self.evaluate(args[1], env)
        if v1.type == SemanticType.BOOLEAN and v2.type == SemanticType.BOOLEAN:
            return SemanticValue.boolean(v1.value and v2.value)
        return SemanticValue.unknown()

    def eval_or(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Or`` operator (logical disjunction).

        Both sub-expressions are evaluated eagerly.  If both resolve to
        ``BOOLEAN`` values, their Python ``or`` is returned.  If either is not
        boolean, the result is ``UNKNOWN``.

        AGQA semantic::

            Or(expr1, expr2)  →  expr1 ∨ expr2

        Args:
            args: Exactly two AST nodes ``[expr1, expr2]``.
            env: The local variable environment.

        Returns:
            A ``BOOLEAN`` :class:`~src.reasoning.semantic_types.SemanticValue`
            with the disjunction result, or ``UNKNOWN`` if either operand is
            not boolean.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Or(wrong_args)")
        v1 = self.evaluate(args[0], env)
        v2 = self.evaluate(args[1], env)
        if v1.type == SemanticType.BOOLEAN and v2.type == SemanticType.BOOLEAN:
            return SemanticValue.boolean(v1.value or v2.value)
        return SemanticValue.unknown()

    def eval_xor(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Xor`` operator (exclusive disjunction).

        Both sub-expressions are evaluated eagerly.  If both resolve to
        ``BOOLEAN`` values, their exclusive-or (``!=`` on booleans) is
        returned.  If either is not boolean, the result is ``UNKNOWN``.

        AGQA semantic::

            Xor(expr1, expr2)  →  expr1 ⊕ expr2  (true iff exactly one is true)

        Args:
            args: Exactly two AST nodes ``[expr1, expr2]``.
            env: The local variable environment.

        Returns:
            A ``BOOLEAN`` :class:`~src.reasoning.semantic_types.SemanticValue`
            with the XOR result, or ``UNKNOWN`` if either operand is not boolean.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Xor(wrong_args)")
        v1 = self.evaluate(args[0], env)
        v2 = self.evaluate(args[1], env)
        if v1.type == SemanticType.BOOLEAN and v2.type == SemanticType.BOOLEAN:
            return SemanticValue.boolean(v1.value != v2.value)
        return SemanticValue.unknown()

    def eval_not(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Not`` operator (logical negation).

        The single sub-expression is evaluated.  If it resolves to a
        ``BOOLEAN`` value, its Python ``not`` is returned.  Otherwise the
        result is ``UNKNOWN``.

        AGQA semantic::

            Not(expr)  →  ¬ expr

        Args:
            args: Exactly one AST node ``[expr]``.
            env: The local variable environment.

        Returns:
            A ``BOOLEAN`` :class:`~src.reasoning.semantic_types.SemanticValue`
            with the negated result, or ``UNKNOWN`` if the operand is not boolean.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 1``.
        """
        if len(args) != 1: raise EvaluatorUnsupportedError("Not(wrong_args)")
        v = self.evaluate(args[0], env)
        if v.type == SemanticType.BOOLEAN:
            return SemanticValue.boolean(not v.value)
        return SemanticValue.unknown()

    def eval_subtract(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Subtract`` operator (numeric difference).

        Both sub-expressions are evaluated.  If both resolve to ``UNKNOWN``
        type values that can be coerced to ``float``, their arithmetic
        difference is computed and returned as an ``UNKNOWN``-typed value
        (because the result is a raw numeric quantity, not a categorised
        semantic value).  If coercion fails, the operation is unsupported.

        Note: ``UNKNOWN`` is used as the result type rather than ``NUMBER``
        because AGQA temporal durations are represented as floating-point
        seconds and do not map cleanly to the integer-only ``NUMBER`` type.

        AGQA semantic::

            Subtract(expr1, expr2)  →  float(expr1) − float(expr2)

        Args:
            args: Exactly two AST nodes ``[expr1, expr2]``.
            env: The local variable environment.

        Returns:
            An ``UNKNOWN``-typed :class:`~src.reasoning.semantic_types.SemanticValue`
            whose ``value`` is the floating-point difference.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2`` or if the
                sub-expression values cannot be coerced to ``float``.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Subtract(wrong_args)")
        v1 = self.evaluate(args[0], env)
        v2 = self.evaluate(args[1], env)
        if v1.type == SemanticType.UNKNOWN and v2.type == SemanticType.UNKNOWN:
            try:
                return SemanticValue(SemanticType.UNKNOWN, float(v1.value) - float(v2.value))
            except (ValueError, TypeError):
                pass
        raise EvaluatorUnsupportedError("Subtract(non_numeric)")

    # --- CORE DOMAINS ---

    def eval_exists(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Exists`` operator (set membership test).

        ``Exists`` checks whether a target label appears in a collection
        (object set or action set) returned by a sub-expression such as
        ``Filter``.  Comparison is performed via the canonicalization helpers
        :func:`~src.reasoning.canonicalization.are_objects_equivalent` and
        :func:`~src.reasoning.canonicalization.are_actions_equivalent`, so
        minor surface variations (articles, case, known plural forms) do not
        cause false negatives.

        AGQA semantic::

            Exists(target, collection)  →  target ∈ collection

        Args:
            args: Exactly two AST nodes ``[target_expr, collection_expr]``
                where ``target_expr`` resolves to a string label and
                ``collection_expr`` resolves to an ``OBJECT_SET`` or
                ``ACTION_SET``.
            env: The local variable environment.

        Returns:
            A ``BOOLEAN`` :class:`~src.reasoning.semantic_types.SemanticValue`
            — ``True`` if ``target`` is found in ``collection``, ``False`` if
            the collection is non-empty but does not contain the target.
            Returns ``UNKNOWN`` if ``collection`` is neither an ``OBJECT_SET``
            nor an ``ACTION_SET``.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Exists(wrong_args)")
        target_val = self.evaluate(args[0], env).value
        collection = self.evaluate(args[1], env)

        if collection.type == SemanticType.OBJECT_SET:
            for obj in collection.value:
                if are_objects_equivalent(target_val, obj):
                    return SemanticValue.boolean(True)
            return SemanticValue.boolean(False)

        elif collection.type == SemanticType.ACTION_SET:
            for act in collection.value:
                if are_actions_equivalent(target_val, act):
                    return SemanticValue.boolean(True)
            return SemanticValue.boolean(False)

        return SemanticValue.unknown()

    def eval_query(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Query`` operator (attribute retrieval).

        ``Query`` retrieves a specific attribute of a scene-graph entity.
        The supported attribute keys and their behaviours are:

        - ``"action"`` — If the item is an ``ACTION_SET`` or ``ACTION``,
          return it as-is (pass-through for action-typed collections).
        - ``"class"`` / ``"object"`` — If the item is an ``OBJECT_SET`` or
          ``OBJECT``, return it as-is.
        - ``"start"`` / ``"end"`` — Search ``self.sg.actions`` for an action
          whose phrase matches the item's value and return the corresponding
          ``start_secs`` or ``end_secs`` timestamp as an ``UNKNOWN``-typed
          float value.

        AGQA semantic::

            Query(attribute, item)  →  item.attribute

        Args:
            args: Exactly two AST nodes ``[attribute_expr, item_expr]`` where
                ``attribute_expr`` resolves to a string attribute name.
            env: The local variable environment.

        Returns:
            A :class:`~src.reasoning.semantic_types.SemanticValue` whose type
            and value depend on the requested attribute (see above).

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2`` or if the
                attribute name is not one of the supported values.
            MissingEvidenceError: If ``attribute`` is ``"start"`` or ``"end"``
                but no matching action is found in the scene graph.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Query(wrong_args)")
        attr = self.evaluate(args[0], env).value
        item = self.evaluate(args[1], env)

        # e.g., Query(action, Iterate(Localize(...)))
        if attr == "action":
            # Just return the action set or a representative action
            if item.type == SemanticType.ACTION_SET:
                return item
            if item.type == SemanticType.ACTION:
                return item

        if attr in ("class", "object"):
            if item.type == SemanticType.OBJECT_SET:
                return item
            if item.type == SemanticType.OBJECT:
                return item
            if item.type == SemanticType.UNKNOWN:
                return SemanticValue(SemanticType.OBJECT, item.value)

        if attr == "start" or attr == "end":
            act_names = item.value if isinstance(item.value, (set, list, tuple)) else [item.value]
            for act in iter_actions(self.sg):
                if any(are_actions_equivalent(act.phrase, an) for an in act_names):
                    return SemanticValue(SemanticType.UNKNOWN, act.start_secs if attr == "start" else act.end_secs)
            raise MissingEvidenceError()

        raise EvaluatorUnsupportedError(f"Query({attr})")

    def eval_onlyitem(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``OnlyItem`` operator (singleton assertion / pass-through).

        In AGQA, ``OnlyItem`` is often used when the program author asserts
        that a collection contains exactly one element and wants to extract it.
        The current implementation passes the evaluated collection through
        unchanged, because the downstream operators (e.g. ``Exists``, ``Query``)
        can already handle both single-element and multi-element sets.

        AGQA semantic::

            OnlyItem(collection)  →  the unique element of collection
                                     (semantically; pragmatically a pass-through)

        Args:
            args: Exactly one AST node ``[collection_expr]``.
            env: The local variable environment.

        Returns:
            The :class:`~src.reasoning.semantic_types.SemanticValue` produced
            by evaluating the single argument — i.e. the collection is returned
            as-is without attempting to extract a single element.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 1``.
        """
        if len(args) != 1: raise EvaluatorUnsupportedError("OnlyItem(wrong_args)")
        collection = self.evaluate(args[0], env)
        # In AGQA, OnlyItem often just asserts there's a unique item or just extracts it.
        # We will just pass through the collection for now, as we treat single vs multiple downstream.
        return collection

    # --- SET OPERATIONS ---

    def eval_filter(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Filter`` operator (set construction from the scene graph).

        ``Filter`` interrogates the scene graph to build a set of entities that
        match the given criteria.  The ``source`` argument is either ``"video"``
        or ``"frame"``, and the ``filters`` argument is a list of keyword strings
        that determine which entity type to retrieve:

        - ``"actions"`` in filter strings → return all action phrases from
          ``self.sg.actions`` as an ``ACTION_SET``.
        - ``"objects"`` in filter strings (without ``"relations"``) → return
          all canonicalized object names from every frame as an ``OBJECT_SET``.
        - ``"relations"`` in filter strings → extract a specific spatial
          relation name from the filter strings (any token that is not one of
          the reserved keywords ``"relations"``, ``"objects"``, ``"video"``,
          ``"frame"``), then return all target-object names of that relation
          across all frames as an ``OBJECT_SET``.

        AGQA semantic::

            Filter(source, [criteria, ...])  →  {e | e satisfies criteria in source}

        Args:
            args: Exactly two elements: ``[source_ast, filters_ast]`` where
                ``source_ast`` is typically the string ``"video"`` or
                ``"frame"``, and ``filters_ast`` is either a ``("List", [...])``,
                a ``list``, or a single node that resolves to a filter keyword.
            env: The local variable environment.

        Returns:
            An ``ACTION_SET`` or ``OBJECT_SET``
            :class:`~src.reasoning.semantic_types.SemanticValue` populated from
            the scene graph.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``, or if the filter
                combination is not one of the three recognised patterns above.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Filter(wrong_args)")
        source_ast = args[0]
        filters = args[1]

        if isinstance(filters, tuple) and filters[0] == "List":
            filter_list = filters[1]
        elif isinstance(filters, list):
            filter_list = filters
        else:
            filter_list = [filters]

        # Get actual strings from filters
        f_strs = [self.evaluate(f, env).value for f in filter_list]

        if source_ast == "video" or source_ast == "frame":
            # Top level collection
            if "actions" in f_strs:
                valid = set()
                # If specific actions are requested, filter them
                specific_actions = [f for f in f_strs if f not in ("actions", "video", "frame")]
                for a in self.sg.actions.values():
                    if not specific_actions or any(are_actions_equivalent(a.phrase, sa) for sa in specific_actions):
                        valid.add(a.phrase)
                return SemanticValue.action_set(valid)
            elif "objects" in f_strs:
                objs = set()
                for f in self.sg.frames.values():
                    for obj in f.objects.values():
                        objs.add(canonicalize_object(obj.name))
                return SemanticValue.object_set(objs)
            elif "relations" in f_strs:
                # If they ask for relation-based filtering, we need to extract objects meeting those relations
                # This requires parsing the exact relations from f_strs (e.g. ['relations', 'touching', 'objects'])
                rel_name = None
                for f in f_strs:
                    if f not in ("relations", "objects", "video", "frame"):
                        rel_name = f

                if rel_name:
                    objs = set()
                    for f in iter_frames(self.sg):
                        for rel in f.relations:
                            if rel.name == rel_name:
                                for obj_id in rel.object_ids:
                                    obj_inst = f.objects.get(obj_id)
                                    if obj_inst:
                                        objs.add(canonicalize_object(obj_inst.name))
                    return SemanticValue.object_set(objs)
                else:
                    rels = set()
                    for f in iter_frames(self.sg):
                        for rel in f.relations:
                            rels.add(rel.name.lower().strip())
                    return SemanticValue.action_set(rels)

        raise EvaluatorUnsupportedError(f"Filter({f_strs})")

    def eval_iterate(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Iterate`` operator (temporal filtering of a collection).

        ``Iterate`` applies a temporal window (a ``FRAME_SET`` value produced by
        :meth:`eval_localize`) to an entity collection (an ``ACTION_SET`` or
        ``OBJECT_SET``) and returns only those entities that fall within the
        window.

        Temporal relation semantics:

        - ``"after"`` — include entities whose interval/timestamp is strictly
          *after* ``t_min`` (the end time of the reference action).
        - ``"before"`` — include entities whose interval/timestamp is strictly
          *before* ``t_max`` (the start time of the reference action).
        - ``"while"`` — include entities whose interval/timestamp overlaps the
          window ``[t_min, t_max]`` (inclusive).

        For **action sets**: each action in ``self.sg.actions`` is tested
        against the temporal relation using its ``start_secs`` / ``end_secs``
        fields.  Slight overlaps are permitted ("AGQA temporal bounds can
        overlap slightly").

        For **object sets**: each frame in ``self.sg.frames`` is tested against
        the temporal relation using its ``secs`` timestamp; objects from
        qualifying frames are collected.

        If the localizer does not return a ``FRAME_SET``, the collection is
        passed through unchanged (fallback for unrecognised localization types).

        AGQA semantic::

            Iterate(localize_expr, collection_expr)
                →  {e ∈ collection | e falls within temporal window}

        Args:
            args: Exactly two AST nodes ``[localize_expr, collection_expr]``.
                ``localize_expr`` should evaluate to a ``FRAME_SET`` value;
                ``collection_expr`` should evaluate to an ``ACTION_SET`` or
                ``OBJECT_SET``.
            env: The local variable environment.

        Returns:
            An ``ACTION_SET`` or ``OBJECT_SET``
            :class:`~src.reasoning.semantic_types.SemanticValue` containing
            only the entities that satisfy the temporal constraint.  If the
            localization type is not ``FRAME_SET``, the original ``collection``
            is returned unchanged.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``.
        """
        if len(args) == 1:
            return self.evaluate(args[0], env)
        if len(args) != 2: raise EvaluatorUnsupportedError("Iterate(wrong_args)")
        loc = self.evaluate(args[0], env)
        collection = self.evaluate(args[1], env)

        if loc.type == SemanticType.FRAME_SET and collection.type == SemanticType.ACTION_SET:
            rel, t_min, t_max = loc.value
            valid = set()
            for a in self.sg.actions.values():
                # AGQA temporal bounds can overlap slightly
                if rel == "after" and a.end_secs > t_min:
                    valid.add(a.phrase)
                elif rel == "before" and a.start_secs < t_max:
                    valid.add(a.phrase)
                elif rel == "while" and not (a.end_secs < t_min or a.start_secs > t_max):
                    valid.add(a.phrase)
            return SemanticValue.action_set(valid)

        if loc.type == SemanticType.FRAME_SET and collection.type == SemanticType.OBJECT_SET:
            rel, t_min, t_max = loc.value
            valid = set()
            for f_id, f in self.sg.frames.items():
                t = f.secs
                # For objects, strict bounds may miss objects that are in the scene.
                # Usually we just check the timestamps.
                is_valid = False
                if rel == "after" and t > t_min:
                    is_valid = True
                elif rel == "before" and t < t_max:
                    is_valid = True
                elif rel == "while" and t_min <= t <= t_max:
                    is_valid = True

                if is_valid:
                    for obj in f.objects.values():
                        valid.add(canonicalize_object(obj.name))
            return SemanticValue.object_set(valid)

        return collection

    def eval_iterateuntil(self, args: List[Any], env: dict) -> SemanticValue:
        return self.eval_iterate(args, env)

    def eval_localize(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Localize`` operator (temporal window construction).

        ``Localize`` looks up the temporal span of a *reference action* in the
        scene graph and derives a temporal window based on the specified
        relation:

        - ``"after"`` → window ``("after", r_max, 9999.0)`` where ``r_max`` is
          the latest end-time of any matching action.
        - ``"before"`` → window ``("before", 0.0, r_min)`` where ``r_min`` is
          the earliest start-time of any matching action.
        - ``"while"`` / ``"during"`` → window ``("while", r_min, r_max)``
          spanning the full extent of all matching action intervals.

        The returned ``FRAME_SET`` value encodes the temporal window as a
        ``(relation, t_min, t_max)`` triple that is consumed by
        :meth:`eval_iterate`.

        AGQA semantic::

            Localize(relation, ref_action)
                →  temporal window defined by relation w.r.t. ref_action's span

        Args:
            args: Exactly two AST nodes ``[relation_expr, ref_action_expr]``
                where ``relation_expr`` resolves to one of ``"after"``,
                ``"before"``, ``"while"``, ``"during"``, and
                ``ref_action_expr`` resolves to an action phrase string.
            env: The local variable environment.

        Returns:
            A ``FRAME_SET`` :class:`~src.reasoning.semantic_types.SemanticValue`
            whose ``value`` is a ``(relation, t_min, t_max)`` tuple of
            ``(str, float, float)``.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2`` or if the
                relation string is not one of the four supported values.
            MissingEvidenceError: If no action in ``self.sg.actions`` matches
                the reference action phrase.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Localize(wrong_args)")
        rel = self.evaluate(args[0], env).value
        ref_val = self.evaluate(args[1], env)
        ref_actions = ref_val.value if isinstance(ref_val.value, (set, list, tuple)) else [ref_val.value]

        ref_starts, ref_ends = [], []
        for a in iter_actions(self.sg):
            if any(are_actions_equivalent(a.phrase, ra) for ra in ref_actions):
                ref_starts.append(a.start_secs)
                ref_ends.append(a.end_secs)

        if not ref_starts:
            raise MissingEvidenceError()

        r_min, r_max = min(ref_starts), max(ref_ends)
        t_min, t_max = 0.0, 9999.0

        if rel == "after":
            return SemanticValue(SemanticType.FRAME_SET, ("after", r_max, 9999.0))
        elif rel == "before":
            return SemanticValue(SemanticType.FRAME_SET, ("before", 0.0, r_min))
        elif rel in ("while", "during"):
            return SemanticValue(SemanticType.FRAME_SET, ("while", r_min, r_max))
        else:
            raise EvaluatorUnsupportedError(f"Localize({rel})")

    # --- OTHERS ---

    def eval_equals(self, args: List[Any], env: dict) -> SemanticValue:
        if len(args) != 2: raise EvaluatorUnsupportedError("Equals(wrong_args)")
        left = self.evaluate(args[0], env)
        right = self.evaluate(args[1], env)
        if left.type == SemanticType.UNKNOWN or right.type == SemanticType.UNKNOWN:
            return SemanticValue.unknown()
        return SemanticValue.boolean(left.value == right.value)

    def eval_toaction(self, args: List[Any], env: dict) -> SemanticValue:
        if len(args) != 2: raise EvaluatorUnsupportedError("ToAction(wrong_args)")
        verb = self.evaluate(args[0], env).value
        obj = self.evaluate(args[1], env).value
        action_phrase = f"{verb} {obj}" if obj else verb
        return SemanticValue(SemanticType.ACTION, action_phrase)

    def eval_and(self, args: List[Any], env: dict) -> SemanticValue:
        if len(args) != 2: raise EvaluatorUnsupportedError("AND(wrong_args)")
        left = self.evaluate(args[0], env)
        right = self.evaluate(args[1], env)
        if left.type == SemanticType.UNKNOWN or right.type == SemanticType.UNKNOWN: return SemanticValue.unknown()
        return SemanticValue.boolean(bool(left.value and right.value))

    def eval_xor(self, args: List[Any], env: dict) -> SemanticValue:
        if len(args) != 2: raise EvaluatorUnsupportedError("XOR(wrong_args)")
        left = self.evaluate(args[0], env)
        right = self.evaluate(args[1], env)
        if left.type == SemanticType.UNKNOWN or right.type == SemanticType.UNKNOWN: return SemanticValue.unknown()
        return SemanticValue.boolean(bool(left.value) ^ bool(right.value))

    def eval_choose(self, args: List[Any], env: dict) -> SemanticValue:
        if len(args) != 3:
            raise EvaluatorUnsupportedError(f"Choose(wrong_args: {len(args)})")
            
        opt1 = self.evaluate(args[0], env).value
        opt2 = self.evaluate(args[1], env).value
        result = self.evaluate(args[2], env)
        
        if result.type == SemanticType.BOOLEAN:
            return SemanticValue(SemanticType.OBJECT, opt1 if result.value else opt2)
            
        if result.value == opt1: return result
        elif result.value == opt2: return result
        return SemanticValue.unknown()

    def eval_compare(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Compare`` operator as a BOOLEAN predicate.

        Returns True if opt0 satisfies the condition (or item1 >= item2),
        Returns False if opt1 satisfies the condition (or item1 < item2).
        Returns Unknown if neither or error.
        """
        opts = args[0]
        if not isinstance(opts, list) or len(opts) < 2:
            raise EvaluatorUnsupportedError("Compare(malformed_options)")

        opt0, opt1 = str(opts[0]).strip(), str(opts[1]).strip()

        if len(args) == 2:
            condition_ast = args[1]
            
            def substitute_temporal_tag(ast_node: Any, replacement: str) -> Any:
                if isinstance(ast_node, str):
                    return replacement if ast_node == "temporal tag" else ast_node
                if isinstance(ast_node, (list, tuple)):
                    return type(ast_node)(substitute_temporal_tag(child, replacement) for child in ast_node)
                return ast_node
                
            ast0 = substitute_temporal_tag(condition_ast, opt0)
            val0 = self.evaluate(ast0, env)
            if val0.type == SemanticType.BOOLEAN and val0.value is True:
                return SemanticValue(SemanticType.BOOLEAN, True)
                
            ast1 = substitute_temporal_tag(condition_ast, opt1)
            val1 = self.evaluate(ast1, env)
            if val1.type == SemanticType.BOOLEAN and val1.value is True:
                return SemanticValue(SemanticType.BOOLEAN, False)
                
            return SemanticValue.unknown()
            
        elif len(args) == 3:
            items = args[1]
            if not isinstance(items, list) or len(items) != 2:
                raise EvaluatorUnsupportedError("Compare(malformed_items)")
                
            mapping_fn = args[2]
            
            v1_set = self.evaluate(items[0], env)
            v2_set = self.evaluate(items[1], env)
            
            if v1_set.type == SemanticType.UNKNOWN or v2_set.type == SemanticType.UNKNOWN:
                return SemanticValue.unknown()
                
            l1 = list(v1_set.value) if isinstance(v1_set.value, (set, list)) else [v1_set.value]
            l2 = list(v2_set.value) if isinstance(v2_set.value, (set, list)) else [v2_set.value]
            
            if not l1 or not l2:
                return SemanticValue.unknown()
                
            c1, c2 = l1[0], l2[0]
            
            env1 = env.copy()
            env1["action"] = SemanticValue(SemanticType.ACTION, c1)
            env1["object"] = SemanticValue(SemanticType.OBJECT, c1)
            score1_val = self.evaluate(mapping_fn, env1)
            
            env2 = env.copy()
            env2["action"] = SemanticValue(SemanticType.ACTION, c2)
            env2["object"] = SemanticValue(SemanticType.OBJECT, c2)
            score2_val = self.evaluate(mapping_fn, env2)
            
            if score1_val.type == SemanticType.UNKNOWN or score2_val.type == SemanticType.UNKNOWN:
                return SemanticValue.unknown()
                
            try:
                s1 = float(score1_val.value)
                s2 = float(score2_val.value)
            except (ValueError, TypeError):
                return SemanticValue.unknown()
                
            return SemanticValue(SemanticType.BOOLEAN, s1 >= s2)
            
        else:
            raise EvaluatorUnsupportedError(f"Compare(wrong_args: {len(args)})")

    def eval_count(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA Count operator."""
        if not args:
            return SemanticValue.unknown()
        val = self.evaluate(args[0], env)
        if isinstance(val.value, (set, list, tuple)):
            return SemanticValue(SemanticType.UNKNOWN, str(len(val.value)))
        return SemanticValue.unknown()

    def eval_equals(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Equals`` operator (equality test).

        Both sub-expressions are evaluated and their values are compared for
        equality.  If both values are strings, comparison is performed after
        applying *both* :func:`~src.reasoning.canonicalization.canonicalize_object`
        and :func:`~src.reasoning.canonicalization.canonicalize_action`, so the
        comparison succeeds if the strings are equivalent under either domain's
        normalization.  Non-string values are compared with Python's ``==``
        operator directly.

        AGQA semantic::

            Equals(expr1, expr2)  →  canonical(expr1) == canonical(expr2)

        Args:
            args: Exactly two AST nodes ``[expr1, expr2]``.
            env: The local variable environment.

        Returns:
            A ``BOOLEAN`` :class:`~src.reasoning.semantic_types.SemanticValue`.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 2``.
        """
        if len(args) != 2: raise EvaluatorUnsupportedError("Equals(wrong_args)")
        v1 = self.evaluate(args[0], env).value
        v2 = self.evaluate(args[1], env).value

        # We can canonicalize if they are strings
        if isinstance(v1, str) and isinstance(v2, str):
            res = (canonicalize_object(v1) == canonicalize_object(v2) or canonicalize_action(v1) == canonicalize_action(v2))
        else:
            res = (v1 == v2)

        return SemanticValue.boolean(res)

    def eval_superlative(self, args: List[Any], env: dict) -> SemanticValue:
        """Evaluate the AGQA ``Superlative`` operator (argmin / argmax selection).

        ``Superlative`` finds the candidate from a collection that minimises or
        maximises a numeric mapping function.  The operation is parameterised by:

        - ``mode`` — either ``"min"`` (return the candidate with the smallest
          score) or ``"max"`` (return the candidate with the largest score).
        - ``collections`` — one or more AST nodes that evaluate to
          ``ACTION_SET`` or ``OBJECT_SET`` values; all candidates are pooled
          into a flat list.
        - ``mapping_fn`` — an AST node that, when evaluated with a candidate
          bound to both ``"action"`` and ``"object"`` in the environment,
          produces a numeric :class:`~src.reasoning.semantic_types.SemanticValue`.
          Candidates for which the mapping function fails (raises
          ``ValueError``, ``TypeError``, or
          :class:`~src.reasoning.semantic_types.MissingEvidenceError`) are
          silently skipped.

        AGQA semantic::

            Superlative(mode, collections, fn)
                →  argmin_{c ∈ pool} fn(c)   if mode == "min"
                   argmax_{c ∈ pool} fn(c)   if mode == "max"

        Args:
            args: Exactly three AST nodes ``[mode_expr, collections_ast, mapping_fn_ast]``.
            env: The local variable environment.

        Returns:
            An ``UNKNOWN``-typed :class:`~src.reasoning.semantic_types.SemanticValue`
            whose ``value`` is the winning candidate string.

        Raises:
            EvaluatorUnsupportedError: If ``len(args) != 3``.
            MissingEvidenceError: If the candidate pool is empty or if every
                candidate fails the mapping function.
        """
        if len(args) != 3: raise EvaluatorUnsupportedError("Superlative(wrong_args)")
        mode = self.evaluate(args[0], env).value
        collections = args[1]  # list of ASTs
        mapping_fn = args[2]

        candidates = []
        if isinstance(collections, list):
            for c_ast in collections:
                c_val = self.evaluate(c_ast, env)
                if c_val.type in (SemanticType.ACTION_SET, SemanticType.OBJECT_SET):
                    candidates.extend(list(c_val.value))
                else:
                    candidates.append(c_val.value)
        else:
            c_val = self.evaluate(collections, env)
            if c_val.type in (SemanticType.ACTION_SET, SemanticType.OBJECT_SET):
                candidates.extend(list(c_val.value))
            else:
                candidates.append(c_val.value)

        if not candidates:
            raise MissingEvidenceError()

        best_val = None
        best_cand = None

        for cand in candidates:
            # We don't strictly know if it's 'action' or 'object' variable in the AST.
            # Usually AGQA uses "action" or "object" as the variable name. We can just inject both.
            local_env = dict(env)
            local_env["action"] = SemanticValue(SemanticType.ACTION, cand)
            local_env["object"] = SemanticValue(SemanticType.OBJECT, cand)

            try:
                score_val = self.evaluate(mapping_fn, local_env)
                score = float(score_val.value)
            except (ValueError, TypeError, MissingEvidenceError):
                continue

            if best_val is None:
                best_val = score
                best_cand = cand
            else:
                if mode == "min" and score < best_val:
                    best_val = score
                    best_cand = cand
                elif mode == "max" and score > best_val:
                    best_val = score
                    best_cand = cand

        if best_cand is None:
            raise MissingEvidenceError()

        # Is it an action or object? The engine doesn't strictly know here, return UNKNOWN string for now
        # Or we can guess based on domain, but string is fine.
        return SemanticValue(SemanticType.UNKNOWN, best_cand)
