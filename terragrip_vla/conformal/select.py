"""Gear selection.  This is where LANGUAGE meets the model.

The training data has no language, and the slip head never sees any.  The
instruction enters here, and only here, as a risk budget (alpha, tau, lambda):

    alpha  widens or tightens the conformal slip bound
    tau    is the slip the operator is willing to tolerate
    lambda trades slip against effort among the gears that are acceptable

The two routes DO NOT get the same guarantee, and pretending otherwise was a real
error in an earlier draft of this file.  They guarantee different EVENTS:

    modular  P( realised_slip(chosen) <= tau ) >= 1 - alpha
             The slip bound is calibrated on y = the MEASURED (noisy) slip, so the
             event it covers is the outcome you actually get.

    e2e      P( mean_slip(chosen) <= tau ) >= 1 - alpha
             APS covers `best_gear`, and best_gear is defined on the true MEAN slip
             curve.  Covering it gives chosen >= best, hence (by monotonicity)
             mean_slip(chosen) <= mean_slip(best) <= tau.  It says nothing about
             the noise around that mean.

So `violation` (realised) and `violation_mean` (mean) are BOTH reported in
eval/metrics.py, and each route is read against the event it actually certifies.

The monotonicity assumption -- more contact area never slips more -- is a claim
about tracked vehicles.  It holds by construction in the mock; on a real robot it
must be stated.

FAILURE IS EXPLICIT, AND THE TWO KINDS OF FAILURE ARE NOT THE SAME:

    modular, empty acceptable set
        EVIDENCE: no gear can be certified to keep slip under tau.  The terrain is
        genuinely beyond budget.  -> fall back to maximum traction, low_conf.

    e2e, empty APS set
        NOT evidence.  Under randomised APS the set is empty exactly when the
        draw u exceeds q -- a coin flip independent of the image, at rate ~alpha
        for a confident classifier.  It says nothing about danger.

    Sending that to the max-effort gear (as an earlier version did) makes alpha
    act on the two routes with OPPOSITE SIGN: raising alpha makes modular's bound
    tighter and its choice CHEAPER, while it makes e2e abstain more often and its
    choice MORE EXPENSIVE.  Since alpha is exactly the knob language turns, that
    silently corrupts every effort / language-direction comparison.  So an empty
    APS set resolves to the model's own argmax -- "no certificate, use the point
    estimate" -- and is reported through `low_conf` instead.
"""

# ============================================================================
# 【中文导读】选档 —— 【语言与模型相遇的地方】。
#
#   训练数据里没有语言，slip 头也从不看语言。指令只在这里进入，形式是风险预算。
#
#   ⚠ 两条路线的保证【不是同一个事件】（早期版本声称“相同”，那是错的）：
#       modular : P( 实测(带噪) slip <= tau ) >= 1-alpha   ← 保形是在实测 slip 上标定的
#       e2e     : P( 均值   slip <= tau ) >= 1-alpha       ← APS 覆盖的 best_gear 定义在均值曲线上
#     所以 metrics 里同时报 violation(实测) 和 violation_mean(均值)，各按各的承诺打分。
#
#   ⚠ 两种“失败”也不是同一回事：
#       modular 可接受集合为空 = 【证据】：没有任何档位能被认证到 tau 以下 → 退最大档
#       e2e   APS 集合为空     = 【不是证据】：只是随机化里 u > q 的抛硬币，与图像无关
#     把后者也送去最大档，会让 alpha 对两条路线【符号相反】地起作用（alpha 变大时
#     modular 变便宜、e2e 反而变贵），而 alpha 恰恰是语言在拧的那个旋钮 —— 这会
#     悄悄污染所有能耗与语义方向的对比。所以空集合退回模型自己的 argmax。
# ============================================================================


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import CONTACT_AREA, FALLBACK_GEAR, GEARS
from language import RiskBudget

AREA = np.array([CONTACT_AREA[g] for g in GEARS], np.float64)

# How to resolve an APS set with more than one gear in it.
#   safe      -> largest contact area  (gives e2e the traction guarantee above)
#   efficient -> smallest contact area (cheaper, but then B has NO guarantee)
AMBIGUITY = ("safe", "efficient")


@dataclass
class Selection:
    gear: np.ndarray  # (N,) int
    low_conf: np.ndarray  # (N,) bool -- nothing could be certified
    info: dict


def select_modular(
    slip_upper: np.ndarray,  # (N, 3) conformal upper bound per gear
    budget: RiskBudget,
    slip_point: np.ndarray | None = None,  # (N, 3) point estimate, for the lambda term
) -> Selection:
    """Accept the gears whose CERTIFIED slip fits the budget; take the cheapest."""
    upper = np.asarray(slip_upper, np.float64)
    point = np.asarray(slip_point if slip_point is not None else upper, np.float64)

    acceptable = upper <= budget.tau  # (N, 3)
    # Same cost as the oracle label (data/labels.gear_cost): slip is normalised by
    # tau so lambda is commensurate with the 0.5 contact-area gap.
    cost = AREA[None, :] + budget.lam * (point / max(budget.tau, 1e-9))  # (N, 3)

    masked = np.where(acceptable, cost, np.inf)
    gear = masked.argmin(axis=1)

    empty = ~acceptable.any(axis=1)
    gear[empty] = FALLBACK_GEAR  # cannot certify anything -> maximum traction

    return Selection(
        gear=gear.astype(np.int64),
        low_conf=empty,
        info={"acceptable": acceptable, "upper": upper, "tau": budget.tau},
    )


def select_e2e(
    sets: list[list[int]],
    probs: np.ndarray | None = None,
    ambiguity: str = "safe",
) -> Selection:
    """Resolve an APS prediction set to a single gear.

    The set is produced by conformal/aps.py (which owns the randomisation and its
    RNG).  This function only decides what to DO with it:

        one gear   -> take it, confident
        many gears -> `safe` takes the largest (the mean-slip guarantee above);
                      `efficient` takes the smallest -- cheaper, certifies nothing
        empty      -> NO certificate. Fall back to the model's own argmax, NOT to
                      the max-effort gear: see the module docstring. `probs` is
                      required for this; without it we can only take the fallback
                      gear, and the alpha-sign asymmetry comes back.
    """
    if ambiguity not in AMBIGUITY:
        raise ValueError(f"ambiguity must be one of {AMBIGUITY}")

    gear = np.empty(len(sets), np.int64)
    low_conf = np.empty(len(sets), bool)
    empty = np.zeros(len(sets), bool)

    for i, s in enumerate(sets):
        if len(s) == 0:
            empty[i] = True
            low_conf[i] = True
            gear[i] = (
                int(np.asarray(probs)[i].argmax()) if probs is not None else FALLBACK_GEAR
            )
        elif len(s) == 1:
            gear[i], low_conf[i] = s[0], False
        else:
            key = (lambda g: CONTACT_AREA[g]) if ambiguity == "efficient" else (
                lambda g: -CONTACT_AREA[g]
            )
            gear[i], low_conf[i] = min(s, key=key), True

    return Selection(
        gear=gear,
        low_conf=low_conf,
        info={
            "sets": sets,
            "set_size": np.array([len(s) for s in sets]),
            "abstained": empty,  # reported, not silently folded into the gear choice
        },
    )


def select_point(slip_point: np.ndarray, budget: RiskBudget) -> Selection:
    """No conformal, no fallback: just trust the point estimate.

    This is the `VisionOnlyNoConformal` baseline.  It isolates exactly what the
    conformal layer buys, by removing it and changing nothing else.
    """
    point = np.asarray(slip_point, np.float64)
    acceptable = point <= budget.tau
    cost = AREA[None, :] + budget.lam * (point / max(budget.tau, 1e-9))
    masked = np.where(acceptable, cost, np.inf)
    gear = masked.argmin(axis=1)

    # "No fallback" means: when nothing fits the budget, just take the gear the
    # point estimate thinks slips least. No certificate, no low_conf flag ever.
    empty = ~acceptable.any(axis=1)
    if empty.any():
        gear[empty] = point[empty].argmin(axis=1)

    return Selection(
        gear=gear.astype(np.int64),
        low_conf=np.zeros(len(gear), bool),
        info={"acceptable": acceptable},
    )
