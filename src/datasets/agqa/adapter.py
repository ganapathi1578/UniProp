"""AGQA Balanced dataset adapter.

This module provides :class:`AGQABalancedAdapter`, which reads the AGQA
Balanced QA JSON file, infers an abstract answer type for every record, and
yields :class:`~src.datasets.normalized.NormalizedQA` instances consumed by
the pipeline.

**AGQA Overview**
AGQA (Action Genome Question Answering) is a compositional video QA benchmark
built on top of Action Genome scene graphs.  The "balanced" variant equalises
the yes/no ratio to prevent models from exploiting label frequency biases.
Each question is annotated with:

* ``semantic`` – coarse topic (``"action"``, ``"object"``, …).
* ``structural`` – syntactic template category (``"query"``, ``"count"``,
  ``"compare"``, …).
* ``global`` – list of compositional reasoning tags (e.g.
  ``["superlative", "action"]``).
* ``program`` – compositional functional program string.

**Answer-type inference**
The adapter maps raw answer strings and structural metadata to one of eight
abstract answer-type labels that the candidate registry recognises:

=============  ===============================================================
Label          Detection rule
=============  ===============================================================
``binary``     Answer is ``"yes"`` or ``"no"`` (case-insensitive).
``temporal``   Answer is ``"before"`` or ``"after"`` (case-insensitive).
``count``      ``structural`` field equals ``"count"`` *or* the answer string
               is a pure integer digit sequence.
``comparison`` ``structural`` field contains the substring ``"compare"``.
``superlative`` The string ``"superlative"`` appears in the ``global`` tag list.
``action``     ``semantic`` field equals ``"action"`` (after the checks above).
``object``     ``semantic`` field equals ``"object"`` (after the checks above).
``open``       Fallback for all other answers.
=============  ===============================================================

The rules are evaluated in priority order from top to bottom; the first
matching rule wins.

Usage::

    from src.datasets.agqa.adapter import AGQABalancedAdapter

    adapter = AGQABalancedAdapter({
        "source": {
            "qa_path": "data/dataset/agqa_balanced/train_balanced.txt",
            "scenegraph_path": "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl",
        }
    })
    for record in adapter.load_records():
        print(record.answer_type, record.query)
"""

import json
from typing import Iterator, Dict
from src.datasets.base import BaseDatasetAdapter
from src.datasets.normalized import NormalizedQA


class AGQABalancedAdapter(BaseDatasetAdapter):
    """Adapter that loads the AGQA Balanced QA JSON and yields NormalizedQA records.

    The adapter lazily loads the QA JSON on the first call to
    :meth:`load_records` (via the internal :meth:`_load_data` helper).
    Scene-graph data is intentionally *not* loaded here; it is resolved on
    demand by :class:`~src.reasoning.evidence.TruthEvaluator` during
    candidate generation, so the adapter remains lightweight.

    Attributes:
        config: Full configuration dictionary as passed by the pipeline.
        balanced_path: Filesystem path to the balanced QA JSON file.
            Resolved in priority order:
            ``config["source"]["qa_path"]`` →
            ``config["balanced_path"]`` →
            default ``"data/dataset/agqa_balanced/train_balanced.txt"``.
        scenegraph_path: Filesystem path to the scene-graph pickle file.
            Resolved in priority order:
            ``config["source"]["scenegraph_path"]`` →
            ``config["scenegraph_path"]`` →
            default ``"data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl"``.
            Stored for downstream consumers; not opened by this adapter.
        _qa_data: In-memory cache of the parsed QA JSON (``dict[qid, qdata]``).
            ``None`` until :meth:`_load_data` is called for the first time.
        _sg_data: Reserved for a future in-memory scene-graph cache.
            Currently always ``None``.
    """

    def __init__(self, config: dict):
        """Initialise the adapter with a pipeline configuration block.

        Extracts ``qa_path`` and ``scenegraph_path`` from *config*, falling
        back to legacy top-level keys and then hard-coded defaults.  No file
        I/O is performed at construction time.

        Args:
            config: Nested configuration dictionary.  Recognised keys:

                * ``config["source"]["qa_path"]`` – preferred path to the QA
                  JSON.
                * ``config["balanced_path"]`` – legacy top-level fallback.
                * ``config["source"]["scenegraph_path"]`` – preferred path to
                  the scene-graph pickle.
                * ``config["scenegraph_path"]`` – legacy top-level fallback.
        """
        self.config = config
        source_cfg = config.get("source", {})
        # Resolve QA path: prefer nested source block, then top-level legacy key, then default.
        self.balanced_path = source_cfg.get("qa_path", config.get("balanced_path", "data/dataset/agqa_balanced/train_balanced.txt"))
        # Resolve scene-graph path the same way; path is stored but file is not opened here.
        self.scenegraph_path = source_cfg.get("scenegraph_path", config.get("scenegraph_path", "data/dataset/agqa_scene_graphs/AGQA_train_stsgs.pkl"))
        # Lazy-load caches; populated on first call to _load_data / load_records.
        self._qa_data = None
        self._sg_data = None

    def _load_data(self):
        """Load and cache the QA JSON file on first access.

        Reads :attr:`balanced_path` as a UTF-8 JSON file and stores the
        parsed result in :attr:`_qa_data`.  Subsequent calls are no-ops
        because the guard ``if self._qa_data is None`` short-circuits.

        The JSON is expected to be a top-level object whose keys are question
        IDs (QIDs) and whose values are dicts containing at least the fields
        ``"video_id"``, ``"question"``, ``"answer"``, ``"semantic"``,
        ``"structural"``, ``"global"``, and ``"program"``.

        Raises:
            FileNotFoundError: If :attr:`balanced_path` does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
        """
        if self._qa_data is None:
            # We assume it's a JSON file mapping QID to data
            with open(self.balanced_path, 'r', encoding='utf-8') as f:
                self._qa_data = json.load(f)

    def load_records(self) -> Iterator[NormalizedQA]:
        """Iterate over QA records and yield normalised representations.

        Calls :meth:`_load_data` to ensure the JSON is in memory, then
        iterates over every ``(qid, qdata)`` pair.  For each pair the method:

        1. Extracts raw fields from *qdata*.
        2. Infers the abstract ``answer_type`` label via an ordered set of
           heuristic rules (see module docstring for the full rule table).
        3. Yields a fully populated :class:`~src.datasets.normalized.NormalizedQA`.

        **Answer-type inference detail** (rules applied in order):

        * ``binary`` – answer (lowercased, stripped) is ``"yes"`` or ``"no"``.
        * ``temporal`` – answer is ``"before"`` or ``"after"``.
        * ``count`` – ``structural`` equals ``"count"`` *or* the raw answer
          string consists entirely of digit characters.
        * ``comparison`` – ``structural`` (lowercased) contains ``"compare"``.
        * ``superlative`` – the string ``"superlative"`` is present in
          ``qdata["global"]`` (a list of compositional reasoning tags).
        * ``action`` – ``semantic`` equals ``"action"``.
        * ``object`` – ``semantic`` equals ``"object"``.
        * ``open`` – fallback when no earlier rule matched.

        The ``reasoning_type`` field is built by joining the ``global`` tag
        list with hyphens (e.g. ``["superlative", "action"]`` →
        ``"superlative-action"``).

        Note:
            ``scenegraph_reference`` is explicitly set to ``None`` here; it is
            populated later by
            :class:`~src.reasoning.evidence.TruthEvaluator` during
            candidate generation ("Loaded from SG dataset during optionization").

        Yields:
            NormalizedQA: One normalised record per question-answer pair in
                the source JSON, in iteration order of the JSON object.

        Raises:
            FileNotFoundError: Propagated from :meth:`_load_data` if the
                source file is missing.
        """
        self._load_data()
        
        for qid, qdata in self._qa_data.items():
            # Extract raw fields from the source JSON record.
            video_id = qdata.get("video_id")
            query = qdata.get("question")
            answer = qdata.get("answer")
            ans_type = qdata.get("ans_type", "open")
            semantic = qdata.get("semantic", "")
            structural = qdata.get("structural", "")
            program = qdata.get("program", "")
            
            # Identify abstract answer type (e.g., binary, count, object, action, temporal, etc.)
            # Normalise the answer string once for all string-equality checks below.
            ans_lower = str(answer).lower().strip()
            
            if ans_lower in ["yes", "no"]:
                # Closed yes/no questions are treated as binary classification.
                a_type = "binary"
            elif ans_lower in ["before", "after"]:
                # Temporal-ordering questions whose answer is a relation word.
                a_type = "temporal"
            elif structural == "count" or str(answer).isdigit():
                # Count questions: either flagged by structural label or the
                # answer is a pure integer (e.g. "3").
                a_type = "count"
            elif "compare" in structural.lower():
                # Comparison questions explicitly labelled in the structural field.
                a_type = "comparison"
            elif "superlative" in qdata.get("global", []):
                # Superlative questions flagged in the compositional tag list.
                a_type = "superlative"
            elif semantic == "action":
                # Action-domain questions not caught by earlier rules.
                a_type = "action"
            elif semantic == "object":
                # Object-domain questions not caught by earlier rules.
                a_type = "object"
            else:
                # Fallback for all other answer forms (e.g. free-text responses).
                a_type = "open"

            yield NormalizedQA(
                source_dataset="agqa_balanced",
                source_question_id=qid,
                video_id=video_id,
                query=query,
                source_answer=answer,
                answer_type=a_type,
                semantic_type=semantic,
                structural_type=structural,
                # Join the compositional reasoning tag list into a hyphen-separated string.
                reasoning_type="-".join(qdata.get("global", [])),
                source_program=program,
                scenegraph_reference=None, # Loaded from SG dataset during optionization
                source_metadata=qdata
            )
