"""Policies: everything that turns (image, instruction) into a gear.

A `Policy` is a pure function of pre-computed model outputs plus the risk budget.
Pulling the torch forward pass out (into `run_model`) and making policies numpy
means the three modes, the ablations and the baselines all go through the SAME
scoring code -- which is exactly what "controlled comparison" requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from conformal.aps import GearConformal
from conformal.select import Selection, select_e2e, select_modular, select_point
from conformal.split_conformal import SlipConformal
from eval.metrics import EvalData
from language import RiskBudget
from models.model import TerraGripModel, language_features


@dataclass
class ModelOutputs:
    slip_mu: np.ndarray | None = None  # (N, 3)
    slip_sigma: np.ndarray | None = None  # (N, 3)
    probs: np.ndarray | None = None  # (N, 3)
    gear_feat: np.ndarray | None = None  # (N, 128) penultimate, for the probe


@torch.no_grad()
def run_model(
    model: TerraGripModel,
    phi: torch.Tensor,
    lang: dict,
    device,
    batch_size: int = 1024,
    concept_override: np.ndarray | None = None,
    want_features: bool = False,
) -> ModelOutputs:
    """One forward pass over a whole split, returned as numpy."""
    model.eval()
    lang_t = language_features(lang, model.lang_condition).to(device)
    override = (
        torch.as_tensor(concept_override, dtype=torch.float32, device=device)
        if concept_override is not None
        else None
    )

    mu, sg, pr, ft = [], [], [], []
    for i in range(0, len(phi), batch_size):
        sl = slice(i, i + batch_size)
        out = model(
            phi=phi[sl].to(device),
            lang=lang_t[sl],
            concept_override=override[sl] if override is not None else None,
            return_features=want_features,
        )
        if "slip" in out:
            mu.append(out["slip"].float().cpu().numpy())
            sg.append(out["slip_sigma"].float().cpu().numpy())
        if "gear_logits" in out:
            pr.append(torch.softmax(out["gear_logits"].float(), -1).cpu().numpy())
        if want_features and "gear_feat" in out:
            ft.append(out["gear_feat"].float().cpu().numpy())

    cat = lambda xs: np.concatenate(xs) if xs else None
    return ModelOutputs(cat(mu), cat(sg), cat(pr), cat(ft))


# --------------------------------------------------------------------------
class Policy:
    """Subclasses implement `select`.  `name` is what shows up in the tables."""

    name: str = "policy"

    def select(self, out: ModelOutputs, data: EvalData, budget: RiskBudget) -> Selection:
        raise NotImplementedError


class ConformalSlipPolicy(Policy):
    """A: certified slip bound per gear -> cheapest gear inside the budget.

    This is the only policy where the instruction acts through an ANALYTIC rule,
    so a brand-new (alpha, tau, lambda) works with no retraining at all.
    """

    def __init__(self, conformal: SlipConformal, name: str = "modular"):
        self.conformal = conformal
        self.name = name

    def select(self, out, data, budget):
        upper = self.conformal.upper(out.slip_mu, budget.alpha, sigma=out.slip_sigma)
        return select_modular(upper, budget, slip_point=out.slip_mu)


class APSPolicy(Policy):
    """B / C: conformal prediction set over gears -> resolve to one gear."""

    def __init__(self, conformal: GearConformal, ambiguity: str = "safe", name: str = "e2e"):
        self.conformal = conformal
        self.ambiguity = ambiguity
        self.name = name

    def select(self, out, data, budget):
        sets = self.conformal.sets(out.probs, budget.alpha)
        return select_e2e(sets, probs=out.probs, ambiguity=self.ambiguity)


class ArgmaxPolicy(Policy):
    """B without conformal.  Isolates what the guarantee costs and buys."""

    name = "e2e_argmax"

    def select(self, out, data, budget):
        gear = out.probs.argmax(1).astype(np.int64)
        return Selection(gear=gear, low_conf=np.zeros(len(gear), bool), info={})


class PointSlipPolicy(Policy):
    """VisionOnlyNoConformal: trust the point estimate, no bound, no fallback."""

    name = "vision_only_no_conformal"

    def select(self, out, data, budget):
        return select_point(out.slip_mu, budget)
