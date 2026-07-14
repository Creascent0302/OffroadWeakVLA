"""Sample schema.

`slip_curve` is the ground-truth mean slip at ALL three gears for this sample's
terrain.  It exists only in simulation.  It is what the ORACLE gear label is
derived from -- see data/labels.py.  Note the asymmetry it exposes:

    modular  needs only `slip` (the gear you actually drove)  -> self-supervised
    e2e/hybrid need `best_gear`, which needs `slip_curve`     -> counterfactual

That asymmetry is a real, reportable advantage of the modular route and is one
of the things the paper measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    image: np.ndarray  # (H, W, 3) uint8, forward-looking RGB
    roi_mask: np.ndarray  # (H, W) bool, "terrain about to be traversed"
    gear: int  # 0/1/2, the gear actually used when this sample was recorded
    slip: float  # [0, 1], measured slip at that gear (self-supervised label)
    slip_curve: np.ndarray  # (3,) float32, TRUE mean slip at each gear (sim only)
    meta: dict = field(default_factory=dict)  # {terrain, run_id, seed, ...}
