"""APS: conformal prediction SETS over gears.  Used by e2e (B) and hybrid (C).

Guarantee: P( true best_gear in S(x) ) >= 1 - alpha.

This is what lets B be compared to A on equal terms.  Without it, B is a bare
classifier with no statistical guarantee at all, and the Pareto comparison is
rigged in A's favour.

RANDOMISED by default, and that is not a detail.  The deterministic variant grows
the set until the cumulative mass reaches q.  When the classifier is confident --
and here it is, ~1.0 accuracy with softmax mass ~0.99 on the top class -- almost
every calibration score is ~0.99, so q lands at ~0.999 and EVERY test set needs
two or three classes to reach it.  Measured: sets were ambiguous 81% of the time
and e2e ended up picking the largest gear constantly, costing more effort than a
fixed-medium baseline.  That is an artefact of the score, not a property of the
model, and it would have silently handed the comparison to modular.

The randomised score (Romano, Sesia & Candes, 2020) removes it:

    E(x, y, u) = sum_{y' : p(y') > p(y)} p(y')  +  u * p(y),     u ~ U(0,1)
    S(x, u)    = { y : E(x, y, u) <= q }

It gives EXACT coverage rather than conservative coverage, and singleton sets on
confident inputs.  The set may legitimately be empty, which the selector reads as
an abstention and answers with the fallback gear.
"""

# ============================================================================
# 【中文导读】APS —— 档位上的保形【预测集合】（e2e / hybrid 用）。
#   保证：P( 真实最优档 ∈ S(x) ) >= 1 - alpha
#   这是让 B 能和 A 平起平坐的关键：没有它，e2e 就是个没有任何统计保证的
#   裸分类器，帕累托对比从一开始就是做局。
#
#   默认【随机化】版本（Romano et al. 2020），这不是细节：
#   本项目的档位分类器准确率≈100%、softmax 质量≈0.99，确定性 APS 会让每个
#   校准分数都≈0.99 → 阈值 q≈0.999 → 每个测试集合都要装 2~3 个类才够。
#   实测 81% 的集合是模糊的，e2e 被逼得一直选最大档，比“固定中档”还费力。
#   那是打分函数的人为产物，不是模型的性质，会把对比悄悄送给 modular。
# ============================================================================


from __future__ import annotations

import math

import numpy as np


def _sorted_view(probs: np.ndarray):
    p = np.asarray(probs, np.float64)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    # mass strictly above each class, in the sorted order
    above = np.cumsum(sorted_p, axis=1) - sorted_p
    return p, order, sorted_p, above


def aps_scores(
    probs: np.ndarray,
    true_gear: np.ndarray,
    randomized: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Conformity score of the TRUE class, for every calibration point."""
    p, order, _, above = _sorted_view(probs)
    n = p.shape[0]
    true = np.asarray(true_gear).astype(int).ravel()

    rank = np.argmax(order == true[:, None], axis=1)  # where the truth sits
    mass_above = above[np.arange(n), rank]
    p_true = p[np.arange(n), true]

    if not randomized:
        return mass_above + p_true  # cumulative up to and including the truth

    rng = rng or np.random.default_rng(0)
    return mass_above + rng.random(n) * p_true


def aps_score(probs: np.ndarray, true_gear: int) -> float:
    """Deterministic single-sample score.  Kept for tests and for reference."""
    return float(aps_scores(np.asarray(probs)[None, :], np.array([true_gear]), False)[0])


def aps_calibrate(
    cal_probs: np.ndarray,
    cal_true: np.ndarray,
    alpha: float,
    randomized: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    """The (1-alpha) quantile of the calibration scores, with the (n+1) correction."""
    s = np.sort(aps_scores(cal_probs, cal_true, randomized, rng))
    n = s.size
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return 1.0  # too few calibration points to certify: the vacuous answer
    return float(s[max(k, 1) - 1])


def aps_sets(
    probs: np.ndarray,
    q: float,
    randomized: bool = True,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """{ y : E(x, y, u) <= q }.  E is increasing along the sorted order, so we stop early."""
    p, order, sorted_p, above = _sorted_view(probs)
    n, k = p.shape

    u = (rng or np.random.default_rng(0)).random(n) if randomized else np.ones(n)

    # Tolerance, not sloppiness: `above` is a cumulative sum, so E can land a few
    # ULPs above a q it should exactly equal (0.8 + 0.15 = 0.9500000000000001).
    # Without this a class that belongs in the set is dropped. Measure-zero for
    # coverage; decisive for the boundary cases.
    eps = 1e-9

    out: list[list[int]] = []
    for i in range(n):
        s: list[int] = []
        for j in range(k):
            e = above[i, j] + u[i] * sorted_p[i, j]
            if e > q + eps:
                break  # E is non-decreasing in j, so nothing further can qualify
            s.append(int(order[i, j]))
        out.append(s)
    return out


def aps_set(probs: np.ndarray, q: float, randomized: bool = False) -> list[int]:
    return aps_sets(np.asarray(probs)[None, :], q, randomized)[0]


def aps_coverage(
    probs: np.ndarray,
    true_gear: np.ndarray,
    q: float,
    randomized: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    sets = aps_sets(probs, q, randomized, rng)
    return float(np.mean([int(t) in s for s, t in zip(sets, np.asarray(true_gear).ravel())]))


class GearConformal:
    """APS thresholds, one per alpha.

    The RNG is seeded and stored, so calibration and inference are reproducible
    and a re-run gives the identical sets.
    """

    def __init__(self, randomized: bool = True, seed: int = 0):
        self.randomized = randomized
        self.seed = seed
        self.Q: dict[float, float] = {}

    def _rng(self, tag: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + tag)

    def fit(self, cal_probs, cal_true, alphas: list[float]) -> "GearConformal":
        for i, a in enumerate(alphas):
            self.Q[float(a)] = aps_calibrate(
                cal_probs, cal_true, float(a), self.randomized, self._rng(i + 1)
            )
        return self

    def q(self, alpha: float) -> float:
        a = float(alpha)
        if a not in self.Q:
            raise KeyError(f"alpha={a} was never calibrated (have {sorted(self.Q)})")
        return self.Q[a]

    def sets(self, probs, alpha: float) -> list[list[int]]:
        return aps_sets(probs, self.q(alpha), self.randomized, self._rng(1000))

    def coverage(self, probs, true_gear, alpha: float) -> float:
        return aps_coverage(
            probs, true_gear, self.q(alpha), self.randomized, self._rng(1000)
        )

    def to_dict(self) -> dict:
        return {"randomized": self.randomized, "Q": {str(a): v for a, v in self.Q.items()}}
