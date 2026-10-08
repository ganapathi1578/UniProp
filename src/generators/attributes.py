"""Object attribute proposition generator (placeholder stub).

This module defines :class:`AttributeGenerator`, a concrete but currently
inert subclass of :class:`~src.generators.base.GeneratorBase`.

Why attribute generation is not yet active
------------------------------------------
The Action Genome (AG) dataset — the video scene-graph backbone used by
UniProp — ships its annotations as Spatio-Temporal Scene-Graph (STSG)
pickle files.  These pickles encode:

* **Objects** — bounding boxes and class labels per frame.
* **Relations** — person-to-object predicates (e.g. ``holding``,
  ``looking_at``).

They do **not** include rich visual attributes such as colour, material,
texture, size, or state (e.g. "the red cup", "the wooden chair",
"the open laptop").  Attribute labels of that kind are absent from the AG
STSG pickles entirely.

Consequence
-----------
Without a source of ground-truth attribute annotations, any attribute
proposition (e.g. "The cup is red.") would be unverifiable and could not be
assigned a reliable ``truth_state``.  Generating unverifiable propositions
would corrupt benchmark integrity.

Future extension path
---------------------
Should attribute annotations become available — either by extracting them
from the AGQA text questions via an attribute parser, or by using an
external attribute-annotation source aligned with Action Genome — this
generator can be activated by implementing the ``yield`` logic inside
:meth:`AttributeGenerator.generate`.  No other pipeline changes are required,
because the generator is already registered in the generator registry.
"""

import random
from typing import Iterator

from src.schema import PropositionGroup
from src.normalization.scene_graph_normalizer import NormalizedSceneGraph
from src.generators.base import GeneratorBase
from src.negatives.hard_negatives import NegativePools


class AttributeGenerator(GeneratorBase):
    """Placeholder generator for object attribute propositions.

    Generates object attribute propositions (e.g. "The cup is red.",
    "The chair is wooden.").

    Note:
        Action Genome baseline does not include rich attributes (like color,
        material).  This acts as a placeholder that yields 0 examples for now
        unless AGQA raw attributes are provided.

    This generator is currently inert because the Action Genome STSG pickle
    files do not carry visual attribute annotations.  See the module-level
    docstring for a full explanation and the future extension path.
    """

    def generate(
        self,
        qid: str,
        qdata: dict,
        sg: NormalizedSceneGraph,
        split: str,
        rng: random.Random,
        pools: NegativePools,
    ) -> Iterator[PropositionGroup]:
        """Yield nothing — attribute data is absent from Action Genome pickles.

        Action Genome STSG pickles contain only object bounding boxes and
        person-object relation labels; they carry no visual attribute
        annotations (colour, material, size, state, etc.).  Without verified
        ground-truth attribute labels it is impossible to assign a reliable
        ``truth_state`` to attribute propositions, so this method is a
        deliberate no-op.

        Future extension: if attributes are extracted from the AGQA text
        questions or supplied from an external annotation source, the
        generation logic should be added here.

        Args:
            qid (str): Unused.  Present only to satisfy the base interface.
            qdata (dict): Unused.  Present only to satisfy the base interface.
            sg (NormalizedSceneGraph): Unused.  Present only to satisfy the
                base interface.
            split (str): Unused.  Present only to satisfy the base interface.
            rng (random.Random): Unused.  Present only to satisfy the base
                interface.
            pools (NegativePools): Unused.  Present only to satisfy the base
                interface.

        Yields:
            Nothing.  No attribute annotations are available in the current
            Action Genome STSG pickles.
        """
        # No attributes in Action Genome STSG pickles.
        # Future extension: if we extract attributes from the text questions, we could generate them here.
        return
        yield
