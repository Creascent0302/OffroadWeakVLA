"""Scoring.  Pure functions over a Selection -- no model, no torch.

The counterfactual world is drawn ONCE per split and shared by every policy
(common random numbers).  Policy P choosing gear 2 and policy Q choosing gear 0
are scored against the same realisation of the same terrain, so any difference
between them is the policy, not the dice.
"""

# ============================================================================
# 【中文导读】评分。纯函数，不碰模型也不碰 torch。
#   【共同随机数】：每个 split 的反事实世界只抽一次噪声，所有策略共用。
#   策略 P 选 2 档、策略 Q 选 0 档，是在【同一个地形的同一次实现】上被打分的，
#   于是两者的差异只来自策略本身，而不是骰子。
# ============================================================================


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import CONTACT_AREA, GEARS, NUM_GEARS
from conformal.select import Selection
from data.labels import best_gear_batch
from data.mock_generator import TERRAIN_SIGMA
from language import RiskBudget

AREA = np.array([CONTACT_AREA[g] for g in GEARS], np.float64)


@dataclass
class EvalData:
    """Everything needed to score a decision on one split."""

    terrain: np.ndarray  # (N,) str
    slip_curve: np.ndarray  # (N, 3) true mean slip per gear
    sigma: np.ndarray  # (N,) terrain noise std
    gear_driven: np.ndarray  # (N,) the gear actually recorded
    slip_driven: np.ndarray  # (N,) the slip actually measured
    realized: np.ndarray  # (N, 3) what slip WOULD have been at each gear

    @classmethod
    def from_source(cls, source, seed: int = 1234) -> "EvalData":
        lab = source.all_labels()
        terrain = lab["terrain"]
        curve = lab["slip_curve"].astype(np.float64)
        sigma = np.array([TERRAIN_SIGMA[t] for t in terrain], np.float64)

        # Common random numbers: one noise draw per (sample, gear), fixed seed.
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, size=(len(terrain), NUM_GEARS))
        realized = np.clip(curve + noise * sigma[:, None], 0.0, 1.0)

        return cls(terrain, curve, sigma, lab["gear"], lab["slip"].astype(np.float64), realized)

    def __len__(self) -> int:
        return len(self.terrain)

    def oracle(self, budget: RiskBudget) -> np.ndarray:
        """The gear a policy that knew the true slip curve would pick."""
        n = len(self)
        return best_gear_batch(
            self.slip_curve, np.full(n, budget.tau), np.full(n, budget.lam)
        )

    def take(self, gear: np.ndarray) -> np.ndarray:
        """Realised slip at the chosen gear."""
        return self.realized[np.arange(len(self)), np.asarray(gear).astype(int)]


def score_selection(sel: Selection, data: EvalData, budget: RiskBudget) -> dict:
    """Every headline metric for one (policy, instruction) pair."""
    gear = np.asarray(sel.gear).astype(int)
    slip = data.take(gear)  # realised slip if we drive this gear
    expected = data.slip_curve[np.arange(len(data)), gear]  # noise-free slip
    energy = AREA[gear]  # effort proxy
    oracle = data.oracle(budget)
    ok = slip <= budget.tau

    # The two routes certify DIFFERENT events (see conformal/select.py):
    #   modular's bound is on the REALISED (noisy) slip     -> `violation`
    #   e2e's APS covers best_gear, defined on the MEAN slip -> `violation_mean`
    # Reporting only one of them would score one route against the other's promise.
    ok_mean = expected <= budget.tau

    m = {
        "slip_mean": float(expected.mean()),  # low-variance version of "how much we slip"
        "slip_realized_mean": float(slip.mean()),
        "energy": float(energy.mean()),  # contact area = effort per unit distance
        "success": float(ok.mean()),  # task proxy: chosen gear kept us inside tau
        "violation": float((~ok).mean()),          # realised slip > tau (modular's event)
        "violation_mean": float((~ok_mean).mean()),  # mean slip > tau  (e2e's event)
        "gear_acc": float((gear == oracle).mean()),
        "fallback_rate": float(np.asarray(sel.low_conf).mean()),
        "gear_hist": {int(g): float((gear == g).mean()) for g in GEARS},
    }

    # The guarantee only ever claims something where the system was confident.
    # Reporting the violation rate there is the honest test of it.
    confident = ~np.asarray(sel.low_conf)
    m["violation_when_confident"] = (
        float((~ok)[confident].mean()) if confident.any() else float("nan")
    )
    m["confident_rate"] = float(confident.mean())
    return m


def per_terrain(sel: Selection, data: EvalData, budget: RiskBudget) -> dict:
    out = {}
    for t in sorted(set(data.terrain)):
        mask = data.terrain == t
        sub = Selection(
            gear=np.asarray(sel.gear)[mask],
            low_conf=np.asarray(sel.low_conf)[mask],
            info={},
        )
        sub_data = EvalData(
            data.terrain[mask],
            data.slip_curve[mask],
            data.sigma[mask],
            data.gear_driven[mask],
            data.slip_driven[mask],
            data.realized[mask],
        )
        out[t] = score_selection(sub, sub_data, budget)
    return out


# --------------------------------------------------------------------------
# Concept fidelity
# --------------------------------------------------------------------------
def bayes_slip_mae(data: EvalData) -> float:
    """The MAE a PERFECT slip predictor would still incur -- the noise floor.

    Reporting raw MAE without this is meaningless: with these sigmas the floor is
    ~0.061, so the spec's "MAE < 0.06" target is below what any model can reach.
    What matters is the ratio to this floor.
    """
    driven = np.arange(len(data)), data.gear_driven.astype(int)
    return float(np.abs(data.slip_driven - data.slip_curve[driven]).mean())


def slip_fidelity(slip_mu: np.ndarray, data: EvalData) -> dict:
    """How well the physical concept is actually predicted (modular / hybrid only)."""
    idx = np.arange(len(data)), data.gear_driven.astype(int)
    pred = np.asarray(slip_mu, np.float64)[idx]
    mae = float(np.abs(pred - data.slip_driven).mean())
    floor = bayes_slip_mae(data)

    out = {"slip_mae": mae, "slip_mae_bayes": floor, "slip_mae_ratio": mae / max(floor, 1e-9)}
    for t in sorted(set(data.terrain)):
        mask = data.terrain == t
        out[f"slip_mae_{t}"] = float(np.abs(pred[mask] - data.slip_driven[mask]).mean())
    return out


# --------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------
def language_compliance(by_intent: dict[str, dict]) -> dict:
    """Does the instruction move the policy in the direction it promises?

    careful -> more contact area (safer);  fast -> less (more efficient).
    """
    e = {k: v["energy"] for k, v in by_intent.items()}
    monotone = e["careful"] >= e["normal"] >= e["fast"]
    return {
        "lang_monotone": float(monotone),
        "lang_spread": float(e["careful"] - e["fast"]),  # 0 = ignores the instruction
        "energy_careful": e["careful"],
        "energy_normal": e["normal"],
        "energy_fast": e["fast"],
    }


def aggregate(by_intent: dict[str, dict]) -> dict:
    """Average the headline metrics over instructions, then add the language ones."""
    keys = [
        "slip_mean", "slip_realized_mean", "energy", "success",
        "violation", "violation_mean",
        "gear_acc", "fallback_rate", "violation_when_confident", "confident_rate",
    ]
    out = {k: float(np.nanmean([v[k] for v in by_intent.values()])) for k in keys}
    out.update(language_compliance(by_intent))
    return out


def mean_std(rows: list[dict]) -> dict:
    """mean +- std across seeds, for the main table."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], float)
        out[k] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v))
    return out
