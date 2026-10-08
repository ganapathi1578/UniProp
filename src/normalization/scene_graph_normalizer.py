"""Scene graph normalization for the AGQA / Action Genome dataset.

This module converts the raw, heterogeneous scene graph dictionaries stored in
the AGQA pickle files into a set of clean, strongly-typed dataclasses that the
rest of the UniProp pipeline can depend on without worrying about the
idiosyncrasies of the original data format.

The raw scene graph for each video is a flat dictionary whose values can be
*frame* entries, *action* entries, or other metadata blobs.  This module
provides:

- **Data classes** — ``NormalizedObject``, ``NormalizedRelation``,
  ``NormalizedAction``, ``NormalizedFrame``, and ``NormalizedSceneGraph`` —
  that give each concept a stable, documented schema.
- **Helper look-up functions** — ``get_human_readable_name`` and
  ``get_relation_name`` — that translate integer-like class IDs into the
  human-readable labels defined in ``src.constants``.
- **``normalize_video_sg``** — the main entry point that performs the
  two-pass conversion from raw dict to ``NormalizedSceneGraph``.

Typical usage::

    import pickle
    from src.normalization.scene_graph_normalizer import normalize_video_sg

    with open("agqa_scene_graphs.pkl", "rb") as fh:
        raw_data = pickle.load(fh)   # Dict[video_id, raw_sg_dict]

    for video_id, raw_sg in raw_data.items():
        nsg = normalize_video_sg(video_id, raw_sg)
        # nsg is a NormalizedSceneGraph ready for downstream use.

Dependencies:
    dataclasses: Standard library — used for defining the data model.
    src.constants: Provides ``OBJECTS`` (class_id → object label) and
        ``ALL_RELATIONS`` (class_id → relation label) mapping dictionaries.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data-model dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NormalizedObject:
    """A single object instance detected within one frame of a video.

    Attributes:
        id: Unique identifier for this object instance within the frame.
            Typically a string-cast integer assigned by the Action Genome
            annotation pipeline (e.g. ``"0"``, ``"1"``).
        class_id: The numeric class identifier (stored as a string) that
            maps to a human-readable label via ``src.constants.OBJECTS``.
            For example ``"1"`` might map to ``"person"``.
        name: Human-readable class label resolved from *class_id* using
            ``src.constants.OBJECTS``.  Falls back to *class_id* itself if no
            mapping is found.
        visible: Whether the object is visible in this frame.  Objects that
            are present in the scene graph but out of frame or occluded may
            have ``visible=False``.
        bbox: Axis-aligned bounding box expressed as a 4-tuple of floats
            ``(x_min, y_min, x_max, y_max)`` in pixel coordinates, or
            ``None`` if no bounding-box annotation is available for this
            object in this frame.
    """

    id: str
    class_id: str
    name: str
    visible: bool
    bbox: Optional[Tuple[float, float, float, float]]


@dataclasses.dataclass
class NormalizedRelation:
    """A directed relationship between objects in a single video frame.

    Action Genome defines four categories of pairwise relations between the
    *person* and other objects:

    - **attention** — where the person is looking.
    - **contact** — physical contact between the person and an object.
    - **spatial** — coarse spatial relationships (e.g. "next to", "behind").
    - **verb** — action-level relationships (e.g. "holding", "sitting on").

    Attributes:
        id: Unique identifier for this relation instance (e.g. ``"r0"``).
            May be an empty string if the source data omits it.
        type: Semantic category of the relation.  One of ``"attention"``,
            ``"contact"``, ``"spatial"``, or ``"verb"``.
        class_id: The numeric class identifier (stored as a string) for this
            specific relationship, mapping to a label via
            ``src.constants.ALL_RELATIONS``.
        name: Human-readable relationship label resolved from *class_id*.
            Falls back to *class_id* if no mapping exists.
        object_ids: List of target object identifiers (source object class
            IDs as extracted from the raw ``"objects"`` sub-list inside the
            relation dict).  In Action Genome annotations, this typically
            contains the class identifier of the object involved in the
            relationship.
    """

    id: str
    type: str  # attention, contact, spatial, verb
    class_id: str
    name: str
    object_ids: List[str]


@dataclasses.dataclass
class NormalizedAction:
    """A temporal action segment associated with a video.

    Actions are derived from the Charades dataset and describe activities
    performed by the person over a contiguous time interval.  Each action is
    linked to the subset of frames it spans and to the primary object and
    verb that define it.

    Attributes:
        id: Unique key for this action entry as it appears in the raw scene
            graph dictionary (e.g. ``"a0"``, ``"a1"``).
        charades_id: The original Charades dataset action class identifier
            (e.g. ``"c001"``), enabling cross-dataset look-ups.
        phrase: Free-text description of the activity, as provided by the
            Charades annotation (e.g. ``"holding a cup"``).
        start_secs: Temporal start of the action segment in seconds from the
            beginning of the video clip.
        end_secs: Temporal end of the action segment in seconds from the
            beginning of the video clip.  Always >= *start_secs*.
        frame_ids: List of frame identifiers (matching ``NormalizedFrame.id``)
            that fall within the ``[start_secs, end_secs]`` interval for this
            action.  Sourced from the raw ``"all_f"`` field.
        object_id: Identifier of the primary object involved in the action
            (e.g. the object being held).  May be an empty string if the
            annotation is absent.
        verb_id: Identifier of the verb describing the activity type.  May be
            an empty string if the annotation is absent.
    """

    id: str
    charades_id: str
    phrase: str
    start_secs: float
    end_secs: float
    frame_ids: List[str]
    object_id: str
    verb_id: str


@dataclasses.dataclass
class NormalizedFrame:
    """A single annotated frame extracted from a video.

    Each frame captures the spatial state of the scene at one point in time,
    including all detected objects and their pairwise relationships.

    Attributes:
        id: Unique frame identifier within the video (e.g. ``"000001"``),
            typically zero-padded to six digits following Action Genome
            conventions.
        secs: Timestamp of this frame in seconds from the start of the video
            clip.  Derived from the raw ``"secs"`` field.
        objects: Mapping from object instance ID to its ``NormalizedObject``
            descriptor.  Keys match the ``NormalizedObject.id`` values of the
            contained objects.
        relations: Ordered list of ``NormalizedRelation`` instances describing
            all pairwise relationships present in this frame across all four
            relation categories (attention, contact, spatial, verb).
    """

    id: str
    secs: float
    objects: Dict[str, "NormalizedObject"]
    relations: List[NormalizedRelation]


@dataclasses.dataclass
class NormalizedSceneGraph:
    """The complete, normalized scene graph for a single video.

    This is the top-level container produced by ``normalize_video_sg``.  It
    bundles together all frames and actions for one video clip into a
    coherent, typed structure.

    Attributes:
        video_id: The Charades video identifier string (e.g. ``"ABCDE"``).
            This is the same ID used as the key in the raw pickle dictionary.
        split: Dataset partition this video belongs to, as inferred from the
            frame metadata in the raw scene graph.  Typically ``"train"``,
            ``"test"``, or ``"val"``.  Defaults to ``"unknown"`` if the
            metadata field is absent or unrecognised.
        frames: Mapping from frame ID to its ``NormalizedFrame`` descriptor.
            Keys are the same strings stored in ``NormalizedFrame.id``.
        actions: Mapping from action key (as it appears in the raw dict) to
            its ``NormalizedAction`` descriptor.
    """

    video_id: str
    split: str
    frames: Dict[str, NormalizedFrame]
    actions: Dict[str, NormalizedAction]


# ---------------------------------------------------------------------------
# Helper look-up functions
# ---------------------------------------------------------------------------

def get_human_readable_name(class_id: str, objects_dict: dict) -> str:
    """Resolve a numeric object class ID to its human-readable label.

    Performs a look-up in the ``OBJECTS`` mapping imported from
    ``src.constants``.  If *class_id* is not present in that mapping the
    raw *class_id* string is returned unchanged, ensuring the function never
    raises a ``KeyError``.

    Args:
        class_id: The numeric class identifier stored as a string (e.g.
            ``"15"``), as found in the ``"class"`` field of raw object
            vertex dicts.
        objects_dict: The parent objects sub-dictionary from the raw frame
            entry.  Currently unused in the implementation but accepted for
            forward-compatibility with callers that may pass additional
            context.

    Returns:
        A human-readable label string such as ``"cup"`` or ``"chair"``.
        Returns *class_id* verbatim when no mapping is found.
    """
    from src.constants import OBJECTS
    return OBJECTS.get(class_id, class_id)


def get_relation_name(class_id: str) -> str:
    """Resolve a numeric relation class ID to its human-readable label.

    Performs a look-up in the ``ALL_RELATIONS`` mapping imported from
    ``src.constants``.  Returns the raw *class_id* if no entry is found.

    Args:
        class_id: The numeric relation class identifier stored as a string
            (e.g. ``"0"``), as found in the ``"class"`` field of raw
            relation dicts inside object vertex entries.

    Returns:
        A human-readable label string such as ``"looking at"`` or
        ``"holding"``.  Returns *class_id* verbatim when no mapping is found.
    """
    from src.constants import ALL_RELATIONS
    return ALL_RELATIONS.get(class_id, class_id)


# ---------------------------------------------------------------------------
# Main normalization entry point
# ---------------------------------------------------------------------------

def normalize_video_sg(video_id: str, raw_sg: dict) -> NormalizedSceneGraph:
    """Normalize a raw scene graph dictionary for a single video.

    Converts the flat, heterogeneous dictionary that the AGQA/Action Genome
    pickle files store for each video into a ``NormalizedSceneGraph`` instance
    whose fields have stable types and well-defined semantics.

    The function performs **two passes** over *raw_sg*:

    1. **Pre-pass** — scans all entries to:
       - Detect the dataset split (``"train"``, ``"test"``, etc.) from frame
         metadata.
       - Collect and build ``NormalizedAction`` objects.

    2. **Main pass** — iterates over entries again to build
       ``NormalizedFrame`` objects, including:
       - Parsing each object vertex into a ``NormalizedObject``.
       - Iterating over the four relation categories embedded in each vertex
         to produce ``NormalizedRelation`` objects.

    Args:
        video_id: The Charades video identifier (e.g. ``"ABCDE"``).  Used
            verbatim to populate ``NormalizedSceneGraph.video_id``.
        raw_sg: A flat dictionary loaded from the AGQA scene graph pickle
            file for one video.  Each value is expected to be a dict
            containing a ``"type"`` field set to either ``"frame"`` or
            ``"action"``.  Non-dict values and entries missing the
            ``"type"`` field are silently skipped.

    Returns:
        A fully populated ``NormalizedSceneGraph`` dataclass instance
        containing:

        - ``video_id`` — passed through from the *video_id* argument.
        - ``split`` — inferred from the first frame entry's ``"metadata"``
          field; ``"unknown"`` if no frame metadata is found.
        - ``frames`` — dict mapping frame IDs to ``NormalizedFrame`` objects.
        - ``actions`` — dict mapping action keys to ``NormalizedAction``
          objects.

    Notes:
        **Split inference**: The split label is taken from the ``"metadata"``
        field of any ``"frame"`` type entry.  If that field is a plain
        string it is used directly; if it is a dict the ``"set"`` sub-key is
        used.  The first non-empty match wins (subsequent frames may
        overwrite it if the dict ordering differs, but in practice all frames
        in the same raw_sg share the same split).

        **Bounding boxes**: Raw bbox values are coerced to a 4-tuple of
        ``float`` only when exactly four elements are present.  Any other
        shape (including ``None`` or empty lists) results in ``bbox=None``.

        **Relation target IDs**: Action Genome encodes target objects of a
        relation as dicts with a ``"class"`` field.  This class value is used
        directly as the target object identifier in ``NormalizedRelation.
        object_ids``, rather than looking up the vertex ID.

        **Missing fields**: All raw field accesses use ``dict.get`` with safe
        defaults (empty strings, ``0.0``, empty lists, ``True`` for
        visibility) so that partially annotated entries do not raise
        ``KeyError``.

    Example::

        import pickle
        from src.normalization.scene_graph_normalizer import normalize_video_sg

        with open("ag_scene_graphs_train.pkl", "rb") as fh:
            raw_data = pickle.load(fh)

        video_id = "ABCDE"
        nsg = normalize_video_sg(video_id, raw_data[video_id])

        print(nsg.split)                          # e.g. "train"
        print(len(nsg.frames))                    # number of annotated frames
        for fid, frame in nsg.frames.items():
            for obj in frame.objects.values():
                print(obj.name, obj.bbox)
    """
    frames = {}
    actions = {}
    split = "unknown"

    # ------------------------------------------------------------------
    # Pre-pass: determine dataset split and collect action entries.
    # ------------------------------------------------------------------
    for key, entry in raw_sg.items():
        if not isinstance(entry, dict):
            # Skip non-dict values (e.g. scalar metadata blobs).
            continue
        entry_type = entry.get('type')
        if not entry_type:
            # Entries without a "type" key are not part of the scene graph.
            continue

        if entry_type == 'frame':
            # Infer the dataset split from the frame's metadata field.
            meta = entry.get('metadata', '')
            if isinstance(meta, str):
                # Metadata is a plain string (e.g. "train").
                split = meta
            elif isinstance(meta, dict) and 'set' in meta:
                # Metadata is a dict with a "set" key (e.g. {"set": "train"}).
                split = meta['set']
        elif entry_type == 'action':
            # Build a NormalizedAction from the raw action entry.
            actions[key] = NormalizedAction(
                id=key,
                charades_id=entry.get('charades', ''),
                phrase=entry.get('phrase', ''),
                start_secs=entry.get('start', 0.0),
                end_secs=entry.get('end', 0.0),
                frame_ids=entry.get('all_f', []),
                object_id=entry.get('object_id', ''),
                verb_id=entry.get('verb_id', '')
            )

    # ------------------------------------------------------------------
    # Main pass: build NormalizedFrame objects from "frame" type entries.
    # ------------------------------------------------------------------
    for key, entry in raw_sg.items():
        if not isinstance(entry, dict):
            continue
        if entry.get('type') == 'frame':
            frame_id = entry.get('id', key)
            secs = entry.get('secs', 0.0)

            # --------------------------------------------------------------
            # Parse object vertices into NormalizedObject instances.
            # --------------------------------------------------------------
            norm_objects = {}
            raw_objects = entry.get('objects', {})
            if isinstance(raw_objects, dict):
                for vertex in raw_objects.get('vertices', []):
                    if isinstance(vertex, dict):
                        obj_id = vertex.get('id', '')
                        class_id = vertex.get('class', '')
                        name = get_human_readable_name(class_id, raw_objects)
                        bbox = vertex.get('bbox')
                        # Ensure bbox is a tuple of 4 floats if present
                        if bbox and len(bbox) == 4:
                            bbox = tuple(float(x) for x in bbox)
                        else:
                            bbox = None

                        norm_objects[obj_id] = NormalizedObject(
                            id=obj_id,
                            class_id=class_id,
                            name=name,
                            visible=vertex.get('visible', True),
                            bbox=bbox
                        )

            # --------------------------------------------------------------
            # Parse relations embedded inside each object vertex.
            # Relations are embedded inside the object vertices
            # --------------------------------------------------------------
            norm_relations = []
            if isinstance(raw_objects, dict):
                for vertex in raw_objects.get('vertices', []):
                    if isinstance(vertex, dict):
                        # Iterate over all four Action Genome relation categories.
                        for rel_type in ['attention', 'contact', 'spatial', 'verb']:
                            for rel_data in vertex.get(rel_type, []):
                                if isinstance(rel_data, dict):
                                    rel_id = rel_data.get('id', '')
                                    class_id = rel_data.get('class', '')

                                    # Get target object IDs
                                    # The 'objects' list inside the relation usually contains the target object dicts
                                    target_obj_ids = []
                                    for target_obj in rel_data.get('objects', []):
                                        if isinstance(target_obj, dict):
                                            # Action Genome uses the object class as the 'class' field
                                            target_class = target_obj.get('class', '')
                                            if target_class:
                                                target_obj_ids.append(target_class)

                                    norm_relations.append(NormalizedRelation(
                                        id=rel_id,
                                        type=rel_type,
                                        class_id=class_id,
                                        name=get_relation_name(class_id),
                                        object_ids=target_obj_ids
                                    ))

            frames[frame_id] = NormalizedFrame(
                id=frame_id,
                secs=secs,
                objects=norm_objects,
                relations=norm_relations
            )

    return NormalizedSceneGraph(
        video_id=video_id,
        split=split,
        frames=frames,
        actions=actions
    )
