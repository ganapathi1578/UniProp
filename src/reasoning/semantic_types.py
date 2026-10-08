"""Semantic type system for the UniProp AGQA program evaluator.

This module defines the core vocabulary of typed values that flow through the
semantic evaluation pipeline.  Every sub-expression in an AGQA functional
program is evaluated to a :class:`SemanticValue`, which pairs a runtime Python
object with a :class:`SemanticType` tag so that downstream operators can
dispatch correctly without inspecting the raw value.

Typical usage::

    from src.reasoning.semantic_types import SemanticValue, SemanticType

    # Create a boolean result
    result = SemanticValue.boolean(True)

    # Create a set of object labels
    objs = SemanticValue.object_set({"person", "chair"})

    # Signal missing evidence during evaluation
    raise MissingEvidenceError("No actions found for video 0123")
"""

from dataclasses import dataclass
from typing import Any, List, Set, Optional
from enum import Enum


class SemanticType(Enum):
    BOOLEAN = "BOOLEAN"
    STRING = "STRING"
    OBJECT = "OBJECT"
    OBJECT_SET = "OBJECT_SET"
    ACTION = "ACTION"
    ACTION_SET = "ACTION_SET"
    EVENT = "EVENT"
    EVENT_SET = "EVENT_SET"
    FRAME_SET = "FRAME_SET"
    NUMBER = "NUMBER"
    DURATION = "DURATION"
    ORDER = "ORDER"
    UNKNOWN = "UNKNOWN"


@dataclass
class SemanticValue:
    """A typed value produced during AGQA program evaluation.

    Every sub-expression in an AGQA AST is reduced to a ``SemanticValue`` by
    :class:`~src.reasoning.semantic_engine.SemanticEngine`.  The ``type``
    field carries a :class:`SemanticType` tag; the ``value`` field carries the
    actual Python runtime object whose concrete type depends on ``type``:

    +-----------------------+----------------------------------------------+
    | ``type``              | Expected ``value`` Python type               |
    +=======================+==============================================+
    | ``BOOLEAN``           | ``bool``                                     |
    +-----------------------+----------------------------------------------+
    | ``OBJECT``            | ``str``                                      |
    +-----------------------+----------------------------------------------+
    | ``OBJECT_SET``        | ``set[str]``                                 |
    +-----------------------+----------------------------------------------+
    | ``ACTION``            | ``str``                                      |
    +-----------------------+----------------------------------------------+
    | ``ACTION_SET``        | ``set[str]``                                 |
    +-----------------------+----------------------------------------------+
    | ``FRAME_SET``         | ``tuple[str, float, float]``                 |
    +-----------------------+----------------------------------------------+
    | ``NUMBER``            | ``int``                                      |
    +-----------------------+----------------------------------------------+
    | ``UNKNOWN``           | ``Any`` (often ``str``, ``float``, or None)  |
    +-----------------------+----------------------------------------------+

    Attributes:
        type: The :class:`SemanticType` tag identifying the semantic category.
        value: The underlying Python value; its concrete type is determined by
            ``type`` (see table above).

    Example::

        sv = SemanticValue(SemanticType.BOOLEAN, True)
        assert sv.type == SemanticType.BOOLEAN
        assert sv.value is True
    """

    type: SemanticType
    value: Any

    @staticmethod
    def boolean(val: bool) -> 'SemanticValue':
        """Create a ``BOOLEAN``-typed semantic value.

        Args:
            val: The Python boolean to wrap.

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.BOOLEAN`` and
            ``value=val``.

        Example::

            sv = SemanticValue.boolean(False)
            assert sv.type == SemanticType.BOOLEAN
            assert sv.value is False
        """
        return SemanticValue(SemanticType.BOOLEAN, val)

    @staticmethod
    def object(obj_id: str) -> 'SemanticValue':
        """Create an ``OBJECT``-typed semantic value for a single entity label.

        Args:
            obj_id: A canonicalized object label string (e.g. ``"chair"``).

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.OBJECT`` and
            ``value=obj_id``.

        Example::

            sv = SemanticValue.object("person")
            assert sv.type == SemanticType.OBJECT
        """
        return SemanticValue(SemanticType.OBJECT, obj_id)

    @staticmethod
    def object_set(obj_ids: set) -> 'SemanticValue':
        """Create an ``OBJECT_SET``-typed semantic value for a collection of labels.

        Args:
            obj_ids: A ``set`` of canonicalized object label strings.

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.OBJECT_SET`` and
            ``value=obj_ids``.

        Example::

            sv = SemanticValue.object_set({"person", "chair", "bottle"})
            assert sv.type == SemanticType.OBJECT_SET
        """
        return SemanticValue(SemanticType.OBJECT_SET, obj_ids)

    @staticmethod
    def action(act_id: str) -> 'SemanticValue':
        """Create an ``ACTION``-typed semantic value for a single action phrase.

        Args:
            act_id: An action phrase string (e.g. ``"walking"``).

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.ACTION`` and
            ``value=act_id``.

        Example::

            sv = SemanticValue.action("running")
            assert sv.type == SemanticType.ACTION
        """
        return SemanticValue(SemanticType.ACTION, act_id)

    @staticmethod
    def action_set(act_ids: set) -> 'SemanticValue':
        """Create an ``ACTION_SET``-typed semantic value for a collection of action phrases.

        Args:
            act_ids: A ``set`` of action phrase strings.

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.ACTION_SET`` and
            ``value=act_ids``.

        Example::

            sv = SemanticValue.action_set({"walking", "sitting"})
            assert sv.type == SemanticType.ACTION_SET
        """
        return SemanticValue(SemanticType.ACTION_SET, act_ids)

    @staticmethod
    def number(val: int) -> 'SemanticValue':
        """Create a ``NUMBER``-typed semantic value for an integer count.

        Args:
            val: The integer count to wrap.

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.NUMBER`` and
            ``value=val``.

        Example::

            sv = SemanticValue.number(3)
            assert sv.type == SemanticType.NUMBER
        """
        return SemanticValue(SemanticType.NUMBER, val)

    @staticmethod
    def unknown() -> 'SemanticValue':
        """Create an ``UNKNOWN``-typed sentinel value representing a missing or unresolvable result.

        This factory is used when a sub-expression cannot be evaluated (e.g.
        a string token that is not in the environment dictionary, or a query
        attribute type that is not currently handled).

        Returns:
            A :class:`SemanticValue` with ``type=SemanticType.UNKNOWN`` and
            ``value=None``.

        Example::

            sv = SemanticValue.unknown()
            assert sv.type == SemanticType.UNKNOWN
            assert sv.value is None
        """
        return SemanticValue(SemanticType.UNKNOWN, None)


class MissingEvidenceError(ValueError):
    """Raised when required evidence is absent from the scene graph.

    This exception signals that the evaluation of an AGQA program sub-expression
    requires data (e.g. an action interval, an object annotation) that is not
    present in the :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
    for the current video.  Callers such as
    :meth:`~src.reasoning.evidence.TruthEvaluator.evaluate_program` catch this
    exception and return ``"UNKNOWN"`` to indicate that the truth value cannot
    be determined from the available evidence.

    Args:
        message: Human-readable description of what evidence is missing.
            Defaults to ``"missing_evidence"``.

    Example::

        raise MissingEvidenceError("No action matching 'running' found")
    """

    def __init__(self, message="missing_evidence"):
        super().__init__(message)


class EvaluatorUnsupportedError(ValueError):
    """Raised when an AGQA operation is not implemented by the current evaluator.

    This exception is raised by :class:`~src.reasoning.semantic_engine.SemanticEngine`
    when it encounters a functional node whose handler method either does not
    exist or has received arguments in a shape that is not supported.  The
    error message encodes the unsupported operation name so that callers can
    log or surface the exact failure point.

    Args:
        operation: A string identifying the unsupported AGQA function or
            argument combination (e.g. ``"Choose"`` or ``"Filter([unknown])"``).
            The resulting exception message is formatted as
            ``"evaluator_unsupported:<operation>"``.

    Example::

        raise EvaluatorUnsupportedError("Choose")
        # str(exc) -> "evaluator_unsupported:Choose"
    """

    def __init__(self, operation: str):
        super().__init__(f"evaluator_unsupported:{operation}")
