"""One-sided split conformal regression on slip.  Used by modular (A) and hybrid (C).

Guarantee (finite sample, distribution free): if the calibration and test points
are exchangeable, then

    P( true_slip <= slip_upper(x, g) ) >= 1 - alpha

Two score functions:

    absolute   s = y - mu
    normalized s = (y - mu) / sigma_hat        <-- default

Both are valid: conformal holds for ANY score.  `normalized` exists because the
slip noise here is strongly heteroscedastic (sigma 0.02 on concrete, 0.15 on
mud).  With `absolute`, the single global quantile has to cover mud, so it comes
out around 0.16; the upper bound on concrete then becomes 0.05 + 0.16 = 0.21,
which exceeds tau_careful = 0.15.  The acceptable set is empty everywhere, and
modular collapses to "always gear L" -- even with a *perfect* slip predictor.
`normalized` divides that away and the bound tracks the terrain.  Keep
`absolute` only to demonstrate the collapse as an ablation.

Assumption worth stating in the paper: calibration residuals are collected at
the gear that was actually driven, but at test time the bound is applied to all
three gears counterfactually.  With `per_gear=False` this assumes residuals are
exchangeable across gears; `per_gear=True` drops that assumption at the cost of
splitting the calibration set three ways.
"""

# ============================================================================
# 【中文导读】单边 split conformal 回归（modular / hybrid 用）。
#
#   保证（有限样本、分布无关）：若 cal 与 test 可交换，则
#       P( 真实 slip <= slip_upper(x, g) ) >= 1 - alpha
#
#   两种打分函数，默认 normalized：
#       absolute   s = y - mu               （教科书写法）
#       normalized s = (y - mu) / sigma_hat （局部自适应）← 默认
#
#   为什么必须用 normalized：本项目的打滑噪声是强异方差的。用 absolute 时，
#   唯一的全局分位数必须覆盖泥地，于是 Q(0.05)=0.158；水泥地的上界就变成
#   0.05+0.158=0.208 > tau_careful=0.15 —— 可接受集合处处为空，modular 退化成
#   “永远 L 档”，【即使 slip 预测器是完美的】。保形对任意打分函数都成立，
#   所以有限样本保证丝毫未损。这【不是】CQR（没有分位数回归/pinball 损失）。
# ============================================================================


from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from constants import GEARS

SCORES = ("absolute", "normalized")


def conformity_scores(
    mu: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None, score: str = "normalized"
) -> np.ndarray:
    """One-sided residuals.  Positive means the truth was above the prediction."""
    mu = np.asarray(mu, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    if score == "absolute":
        return y - mu
    if score == "normalized":
        if sigma is None:
            raise ValueError("normalized score needs sigma")
        s = np.asarray(sigma, np.float64).ravel()
        return (y - mu) / np.maximum(s, 1e-6)
    raise ValueError(f"score must be one of {SCORES}, got {score!r}")


def quantile(scores: np.ndarray, alpha: float) -> float:
    """The (1-alpha) conformal quantile, with the finite-sample (n+1) correction.

    k = ceil((n+1)(1-alpha)).  If k > n the calibration set is too small to
    certify this alpha and the honest answer is +inf (a vacuous, but still valid,
    bound) rather than a silently over-confident one.
    """
    s = np.sort(np.asarray(scores, np.float64).ravel())
    n = s.size
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(s[max(k, 1) - 1])


class SlipConformal:
    """Calibrated slip upper bounds, one quantile per alpha (optionally per gear)."""

    def __init__(self, score: str = "normalized", per_gear: bool = False):
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")
        self.score = score
        self.per_gear = per_gear
        self.Q: dict[float, float | dict[int, float]] = {}

    # ------------------------------------------------------------------
    def fit(
        self,
        mu: np.ndarray,       # (N,) prediction at the gear that was driven
        y: np.ndarray,        # (N,) measured slip
        alphas: list[float],
        sigma: np.ndarray | None = None,  # (N,)
        gear: np.ndarray | None = None,   # (N,) needed only when per_gear
    ) -> "SlipConformal":
        scores = conformity_scores(mu, y, sigma, self.score)

        for a in alphas:
            a = float(a)
            if not self.per_gear:
                self.Q[a] = quantile(scores, a)
            else:
                if gear is None:
                    raise ValueError("per_gear=True needs the gear array")
                g = np.asarray(gear).ravel()
                self.Q[a] = {int(k): quantile(scores[g == k], a) for k in GEARS}
        return self

    # ------------------------------------------------------------------
    def q(self, alpha: float) -> np.ndarray:
        """(3,) quantile per gear (broadcast when a single global quantile is used)."""
        a = float(alpha)
        if a not in self.Q:
            raise KeyError(f"alpha={a} was never calibrated (have {sorted(self.Q)})")
        v = self.Q[a]
        if isinstance(v, dict):
            return np.array([v[g] for g in GEARS], np.float64)
        return np.full(len(GEARS), float(v), np.float64)

    def upper(
        self, mu: np.ndarray, alpha: float, sigma: np.ndarray | None = None
    ) -> np.ndarray:
        """mu (B,3) [, sigma (B,3)] -> slip upper bound (B,3)."""
        mu = np.asarray(mu, np.float64)
        q = self.q(alpha)[None, :]
        if self.score == "absolute":
            return mu + q
        if sigma is None:
            raise ValueError("normalized score needs sigma")
        return mu + q * np.asarray(sigma, np.float64)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "per_gear": self.per_gear,
            "Q": {str(a): v for a, v in self.Q.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SlipConformal":
        d = json.loads(Path(path).read_text())
        obj = cls(d["score"], d["per_gear"])
        for a, v in d["Q"].items():
            obj.Q[float(a)] = (
                {int(k): float(x) for k, x in v.items()} if isinstance(v, dict) else float(v)
            )
        return obj


def empirical_coverage(y: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of test points whose true slip really is below the bound."""
    return float((np.asarray(y).ravel() <= np.asarray(upper).ravel() + 1e-12).mean())
