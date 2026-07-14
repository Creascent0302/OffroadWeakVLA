"""Baselines.  These are what the three modes have to beat.

FixedS / FixedM / FixedL
    No perception at all. They bound the Pareto plot: FixedS is maximally
    efficient and slips a lot; FixedL never slips and always costs the most. Any
    method worth having must sit strictly inside the region they define.

ReactiveOnly
    Proprioception, no vision. It only learns a terrain is slippery BY SLIPPING
    ON IT, then corrects. This is the baseline that isolates the value of *seeing
    ahead*: an anticipatory policy pays no such entry cost.

VisionOnlyNoConformal
    Lives in policies.PointSlipPolicy: the full vision model with the conformal
    layer removed and nothing else changed. It isolates exactly what the
    guarantee is worth.
"""

from __future__ import annotations

import numpy as np

from constants import GEARS, NUM_GEARS
from conformal.select import Selection
from eval.metrics import EvalData
from eval.policies import Policy
from language import RiskBudget


class FixedGear(Policy):
    def __init__(self, gear: int):
        self.gear = int(gear)
        self.name = f"fixed_{'SML'[self.gear]}"

    def select(self, out, data: EvalData, budget: RiskBudget) -> Selection:
        n = len(data)
        return Selection(
            gear=np.full(n, self.gear, np.int64),
            low_conf=np.zeros(n, bool),
            info={},
        )


class ReactiveOnly(Policy):
    """React to the slip you just measured; never look ahead.

    The vehicle is currently in `gear_driven` and has just measured
    `slip_driven`.  If that exceeds the budget, shift up (more traction).  If it
    is comfortably below, shift down (save effort).  Otherwise hold -- the
    dead-band stops it oscillating.

    Structurally it can never do better than "slip once, then fix it", which is
    precisely the cost that a vision model avoids.
    """

    name = "reactive_only"

    def __init__(self, downshift_frac: float = 0.5):
        self.downshift_frac = float(downshift_frac)

    def select(self, out, data: EvalData, budget: RiskBudget) -> Selection:
        gear = data.gear_driven.astype(np.int64).copy()
        slip = data.slip_driven

        up = slip > budget.tau
        down = slip < self.downshift_frac * budget.tau

        gear[up] += 1
        gear[down] -= 1
        gear = np.clip(gear, min(GEARS), max(GEARS))

        return Selection(
            gear=gear,
            low_conf=np.zeros(len(gear), bool),  # it never knows it is uncertain
            info={"shifted_up": up, "shifted_down": down},
        )


class RandomGear(Policy):
    """Chance level, so 'gear_acc' has a floor to be read against."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def select(self, out, data: EvalData, budget: RiskBudget) -> Selection:
        rng = np.random.default_rng(self.seed)
        n = len(data)
        return Selection(
            gear=rng.integers(0, NUM_GEARS, n).astype(np.int64),
            low_conf=np.zeros(n, bool),
            info={},
        )


class OracleGear(Policy):
    """Knows the true slip curve.  The ceiling: no policy can beat it on gear_acc."""

    name = "oracle"

    def select(self, out, data: EvalData, budget: RiskBudget) -> Selection:
        gear = data.oracle(budget)
        return Selection(gear=gear, low_conf=np.zeros(len(gear), bool), info={})


def model_free_baselines(seed: int = 0) -> list[Policy]:
    """The baselines that need no trained model at all."""
    return [
        FixedGear(0),
        FixedGear(1),
        FixedGear(2),
        ReactiveOnly(),
        RandomGear(seed),
        OracleGear(),
    ]
