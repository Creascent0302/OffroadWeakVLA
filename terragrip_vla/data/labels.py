"""Gear labels derived from slip.

`best_gear` is the single decision rule the whole project is built around.  It is
used in two different roles, and keeping them the same function is what makes the
three modes comparable:

    * as the ORACLE TRAINING LABEL for e2e / hybrid, applied to the true slip
      curve  ->  best_gear(slip_curve, tau, lam)
    * as the SELECTION OBJECTIVE for modular, applied to conformal upper bounds
      ->  see conformal/select.py

The rule:  among gears whose slip is within the budget tau, take the cheapest
(smallest contact area, optionally traded against slip via lam).  If no gear is
within budget, fall back to the largest gear.
"""

# ============================================================================
# 【中文导读】由 slip 派生档位标签。best_gear 是整个项目的唯一决策规则，
# 它同时扮演两个角色（保持是同一个函数，三种模式才可比）：
#   · e2e/hybrid 的【oracle 训练标签】：作用在真值 slip 曲线上
#   · modular 的【选档目标】：作用在保形上界上（见 conformal/select.py）
#
# 规则：在“打滑不超过预算 tau”的档位里，挑最省力的（可用 lambda 拿打滑换省力）；
#      若无档位达标，退到最大档。
#
# ⚠ 注意 best_gear 是 ORACLE：它需要【所有档位】的 slip，即反事实信息。
#   这正是 e2e/hybrid 的监督成本远高于 modular 的原因 —— modular 只需要
#   “你实际走的那一档的 slip”，那是真·自监督。
# ============================================================================


from __future__ import annotations

from typing import Sequence

import numpy as np

from constants import CONTACT_AREA, FALLBACK_GEAR, GEARS


def gear_cost(gear: int, slip: float, tau: float, lam: float) -> float:
    """Effort + lam * (how much of the slip budget this gear spends).

    The slip term is divided by tau on purpose. With a raw `lam * slip`, lambda is
    a DEAD KNOB: the contact-area gap between adjacent gears is 0.5, while inside
    the acceptable set slip <= tau <= 0.35, so `lam * slip` can never span 0.5 at
    any lambda the risk table uses. Measured: no value in RISK_TABLE ever changed a
    single decision. Normalised by tau, slip/tau is in [0, 1] for every acceptable
    gear -- commensurate with the 0.5 area gap -- so lambda actually trades
    traction against effort, which is what the paper says it does.
    `test_selection.py::test_lambda_actually_bites_on_the_real_risk_table` pins it.
    """
    return CONTACT_AREA[gear] + lam * (slip / max(tau, 1e-9))


def best_gear(slip_per_gear: Sequence[float], tau: float, lam: float = 0.0) -> int:
    """Cheapest gear whose slip fits the budget; else the fallback gear.

    ORACLE: needs slip at *every* gear, i.e. counterfactual knowledge.  Only the
    simulator (or a very expensive real data collection) can provide this.
    """
    acceptable = [g for g in GEARS if float(slip_per_gear[g]) <= tau]
    if not acceptable:
        return FALLBACK_GEAR
    return min(acceptable, key=lambda g: gear_cost(g, float(slip_per_gear[g]), tau, lam))


def best_gear_batch(slip_curves: np.ndarray, taus: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Vectorised `best_gear`.

    slip_curves : (N, 3)
    taus, lams  : (N,)   per-sample risk budget, decoded from that sample's
                         instruction -- this is what makes the label
                         language-conditioned.
    returns     : (N,) int64
    """
    slip_curves = np.asarray(slip_curves, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).reshape(-1, 1)
    lams = np.asarray(lams, dtype=np.float64).reshape(-1, 1)

    area = np.array([CONTACT_AREA[g] for g in GEARS], dtype=np.float64)  # (3,)
    acceptable = slip_curves <= taus  # (N, 3)
    cost = area[None, :] + lams * (slip_curves / np.maximum(taus, 1e-9))  # (N, 3)

    masked = np.where(acceptable, cost, np.inf)
    choice = masked.argmin(axis=1)
    choice[~acceptable.any(axis=1)] = FALLBACK_GEAR
    return choice.astype(np.int64)
