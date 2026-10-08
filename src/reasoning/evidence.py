"""Ground-truth truth-value evaluator for AGQA programs.

This module provides :class:`TruthEvaluator`, the top-level entry point for
evaluating whether a given AGQA functional program is ``TRUE``, ``FALSE``, or
``UNKNOWN`` for a specific video.

Pipeline overview
-----------------
::

    program string  ──► parse_agqa_program()  ──► AST
                                                    │
    AGQA STSG pickle ──► NormalizedSceneGraph ◄─────┘
                                │
                         SemanticEngine.evaluate()
                                │
                         SemanticValue  ──► "TRUE" / "FALSE" / "UNKNOWN"

The class handles two expensive concerns transparently:

1. **Lazy pickle loading** — the large AGQA Spatio-Temporal Scene Graph (STSG)
   pickle files (one per dataset split) are loaded from disk only on the first
   request for any video in that split.  Subsequent requests reuse the cached
   raw dict.

2. **Per-video scene graph caching** — normalizing a raw AGQA STSG entry into a
   :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph` is
   done at most once per video per process lifetime.  The normalized object is
   stored in ``self.sg_cache`` and reused for all subsequent program
   evaluations against the same video.

Typical usage::

    evaluator = TruthEvaluator(scenegraph_dir="/data/agqa/stsg")
    verdict = evaluator.evaluate_program(
        program="Exists(walking, Filter(video, [actions]))",
        video_id="0001",
        split="test",
    )
    # verdict in {"TRUE", "FALSE", "UNKNOWN"}
"""

from typing import Dict, Any, Optional, List
import pickle
import os
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.reasoning.program_parser import parse_agqa_program
from src.reasoning.semantic_engine import SemanticEngine
from src.reasoning.semantic_types import EvaluatorUnsupportedError, MissingEvidenceError, SemanticType


class TruthEvaluator:
    """Loads AGQA STSG evidence and evaluates functional programs to truth values.

    :class:`TruthEvaluator` sits at the top of the reasoning stack.  It
    orchestrates pickle loading, scene-graph normalization, AST parsing, and
    semantic evaluation, and reduces the result to a simple three-valued
    string: ``"TRUE"``, ``"FALSE"``, or ``"UNKNOWN"``.

    **Caching strategy**

    Two levels of caching are maintained:

    - ``raw_data_cache`` (``Dict[str, Any]``) — maps ``split`` name to the raw
      dict loaded from the corresponding AGQA STSG pickle file.  A split is
      loaded at most once per :class:`TruthEvaluator` instance.
    - ``sg_cache`` (``Dict[str, NormalizedSceneGraph]``) — maps ``video_id`` to
      the normalized scene graph.  A video's scene graph is normalized at most
      once; subsequent evaluations for the same video skip the normalization step.

    **Pickle file convention**

    The evaluator expects pickle files named ``AGQA_<split>_stsgs.pkl`` inside
    ``scenegraph_dir``.  If the split-specific file does not exist, it falls
    back to ``AGQA_train_stsgs.pkl`` as a generic substitute (e.g. for
    evaluation on a custom split that reuses training scene graphs).

    **Return value semantics**

    +----------------------------------+-------------------+
    | Condition                        | Return value      |
    +==================================+===================+
    | Program evaluates to ``True``    | ``"TRUE"``        |
    +----------------------------------+-------------------+
    | Program evaluates to ``False``   | ``"FALSE"``       |
    +----------------------------------+-------------------+
    | Video not found in the STSG data | ``"UNKNOWN"``     |
    +----------------------------------+-------------------+
    | :class:`MissingEvidenceError`    | ``"UNKNOWN"``     |
    | raised during evaluation         |                   |
    +----------------------------------+-------------------+
    | Non-boolean result type          | ``"UNKNOWN"``     |
    +----------------------------------+-------------------+
    | :class:`EvaluatorUnsupportedError`| re-raised as    |
    | raised during evaluation         | ``ValueError``    |
    +----------------------------------+-------------------+
    | Any other exception              | re-raised as      |
    |                                  | ``ValueError``    |
    +----------------------------------+-------------------+

    Attributes:
        scenegraph_dir: Absolute path to the directory containing the AGQA STSG
            pickle files.
        raw_data_cache: In-memory cache of loaded STSG dicts, keyed by split name.
        sg_cache: In-memory cache of
            :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
            objects, keyed by video ID string.

    Example::

        evaluator = TruthEvaluator("/data/agqa/stsg")
        result = evaluator.evaluate_program(
            "Exists(walking, Filter(video, [actions]))",
            video_id="0001",
            split="train",
        )
        assert result in {"TRUE", "FALSE", "UNKNOWN"}
    """

    def __init__(self, scenegraph_dir: str):
        """Initialise the evaluator with the path to the STSG pickle directory.

        No files are loaded at construction time; loading is deferred until
        the first call to :meth:`get_scenegraph` or :meth:`evaluate_program`.

        Args:
            scenegraph_dir: Path to the directory that contains AGQA STSG
                pickle files (``AGQA_<split>_stsgs.pkl``).
        """
        self.scenegraph_dir = scenegraph_dir
        self.raw_data_cache = {}
        self.sg_cache: Dict[str, NormalizedSceneGraph] = {}

    def _load_split(self, split: str):
        """Load and cache the raw STSG dict for the given dataset split.

        If ``split`` has already been loaded (i.e. its key is present in
        ``self.raw_data_cache``), this method returns immediately without doing
        any I/O.  Otherwise it resolves the pickle file path, prints a loading
        message, deserializes the pickle with ``pickle.load``, and stores the
        result in ``self.raw_data_cache[split]``.

        **Fallback behaviour**: If the split-specific file
        ``AGQA_<split>_stsgs.pkl`` does not exist on disk, the method falls
        back to ``AGQA_train_stsgs.pkl`` in the same directory.  This allows
        evaluation on custom or unseen splits that do not have their own STSG
        file but can share the training scene graphs.

        Args:
            split: Dataset split identifier string, e.g. ``"train"``,
                ``"test"``, ``"balanced_test"``.
        """
        if split not in self.raw_data_cache:
            path = os.path.join(self.scenegraph_dir, f"AGQA_{split}_stsgs.pkl")
            if not os.path.exists(path):
                # Fallback or generic path
                path = os.path.join(self.scenegraph_dir, f"AGQA_train_stsgs.pkl")
            print(f"Loading scenegraph evidence from {path}...")
            with open(path, 'rb') as f:
                self.raw_data_cache[split] = pickle.load(f)

    def get_scenegraph(self, video_id: str, split: str = "train") -> Optional[NormalizedSceneGraph]:
        """Retrieve (or construct) the normalized scene graph for a given video.

        This method first ensures the raw STSG data for ``split`` is loaded via
        :meth:`_load_split`.  It then checks whether ``video_id`` is present in
        the raw data dict and whether its normalized form is already cached in
        ``self.sg_cache``.

        If the normalized scene graph is not yet cached, it is constructed by
        calling ``normalize_video_sg(video_id, raw_data[video_id])`` (imported
        lazily from :mod:`src.normalization.scene_graph_normalizer` to avoid
        circular imports at module load time) and stored in ``self.sg_cache``.

        Args:
            video_id: The AGQA video identifier string (e.g. ``"0001"``).
            split: Dataset split identifier string.  Defaults to ``"train"``.

        Returns:
            The :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
            for ``video_id``, or ``None`` if ``video_id`` is not present in the
            STSG data for the requested split.
        """
        self._load_split(split)

        raw_data = self.raw_data_cache[split]
        if video_id not in raw_data:
            return None

        if video_id not in self.sg_cache:
            from src.normalization.scene_graph_normalizer import normalize_video_sg
            sg = normalize_video_sg(video_id, raw_data[video_id])
            self.sg_cache[video_id] = sg

        return self.sg_cache[video_id]

    def evaluate_program(self, program: str, video_id: str, split: str = "train") -> str:
        """Evaluate an AGQA functional program against a video's scene graph.

        This is the primary public API of :class:`TruthEvaluator`.  It performs
        the full evaluation pipeline:

        1. Load (or reuse cached) :class:`~src.normalization.scene_graph_normalizer.NormalizedSceneGraph`
           for ``video_id`` via :meth:`get_scenegraph`.
        2. If no scene graph is found, return ``"UNKNOWN"`` immediately.
        3. Parse ``program`` into an AST using
           :func:`~src.reasoning.program_parser.parse_agqa_program`.
        4. Construct a :class:`~src.reasoning.semantic_engine.SemanticEngine`
           bound to the scene graph and call its :meth:`~src.reasoning.semantic_engine.SemanticEngine.evaluate`
           method on the AST.
        5. Map the resulting :class:`~src.reasoning.semantic_types.SemanticValue`
           to a string verdict:
           - ``BOOLEAN True`` → ``"TRUE"``
           - ``BOOLEAN False`` → ``"FALSE"``
           - ``UNKNOWN`` → ``"UNKNOWN"``
           - Any other type → ``"UNKNOWN"``

        Exception handling:

        - :class:`~src.reasoning.semantic_types.MissingEvidenceError` → returns
          ``"UNKNOWN"`` (evidence absent from the scene graph is treated as
          unanswerable, not as an error).
        - :class:`~src.reasoning.semantic_types.EvaluatorUnsupportedError` →
          re-raised as :class:`ValueError` with the original error string, so
          that callers can detect and log unsupported program patterns.
        - Any other :class:`Exception` → re-raised as :class:`ValueError` with
          a ``"evaluator_unsupported: <message>"`` prefix.

        Args:
            program: An AGQA functional program string, e.g.::

                    "Exists(walking, Filter(video, [actions]))"
                    "And(Exists(chair, Filter(video, [objects])), ...)"

            video_id: The AGQA video identifier string used to look up the
                scene graph.
            split: Dataset split identifier string.  Defaults to ``"train"``.

        Returns:
            One of the three string verdicts: ``"TRUE"``, ``"FALSE"``, or
            ``"UNKNOWN"``.

        Raises:
            ValueError: If the program contains an unsupported AGQA operator
                (wraps :class:`~src.reasoning.semantic_types.EvaluatorUnsupportedError`),
                or if any other unexpected exception occurs during evaluation.

        Example::

            evaluator = TruthEvaluator("/data/agqa/stsg")
            verdict = evaluator.evaluate_program(
                "Exists(walking, Filter(video, [actions]))",
                video_id="0001",
                split="train",
            )
            # verdict == "TRUE" or "FALSE" depending on the video content
        """
        sg = self.get_scenegraph(video_id, split)
        if not sg:
            return "UNKNOWN"

        try:
            ast = parse_agqa_program(program)
            engine = SemanticEngine(sg)
            result = engine.evaluate(ast)

            if result.type == SemanticType.BOOLEAN:
                return "TRUE" if result.value else "FALSE"
            elif result.type == SemanticType.UNKNOWN:
                return "UNKNOWN"
            else:
                return "UNKNOWN"

        except MissingEvidenceError:
            return "UNKNOWN"
        except EvaluatorUnsupportedError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"evaluator_unsupported: {str(e)}")
