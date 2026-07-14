"""Slip head: (phi_vis, gear) -> predicted slip.  The physical concept bottleneck.

It deliberately does NOT take language.  Slip is a fact about terrain and gear;
telling the robot "be careful" does not change how much mud slips.  Language
belongs at the decision, not at the physics.  `test_modes.py` asserts this.

Alongside the mean it predicts a sigma, used ONLY to normalise the conformal
score -- that is what lets one global quantile adapt to terrain-dependent noise
(0.02 on concrete vs 0.15 on mud).

TWO THINGS THE SIGMA PATH GETS RIGHT, because the obvious implementation gets
both wrong:

1. `mu.detach()` IN THE LOSS IS NOT ENOUGH.  If mu and log_sigma share a trunk,
   detaching mu in the NLL severs only the DIRECT path: the NLL still reaches
   log_sigma's weights and flows back through the SHARED hidden state into the
   trunk -- which mu depends on.  So the NLL silently retrains mu after all, and
   the documented "mu is trained by MSE alone" invariant is false.  Here the sigma
   branch reads a DETACHED hidden state, so no NLL gradient can reach any
   parameter mu uses.  `test_modes.py::test_sigma_nll_cannot_move_mu` asserts it.

2. SIGMA MUST MODEL TERRAIN NOISE, NOT DROPOUT NOISE.  During training mu is
   computed with dropout active, so the residual (y - mu) carries
   terrain_noise + dropout_noise and sigma is fitted to the sum -- inflating it on
   exactly the low-noise terrain where the tight bound matters.  The sigma path
   therefore runs the trunk WITHOUT dropout, and the NLL is fitted against that
   same dropout-free mean (`slip_clean`).

The extra sigma parameters do not affect head-capacity matching: they cannot
influence mu (point 1), so the decision capacity is still the shared 256/128
trunk that GearHead also has.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import GEARS, NUM_GEARS

# sigma in [~2.5e-3, ~2.7]. The floor stops the normalised score from exploding
# on very easy terrain; the ceiling keeps the NLL well conditioned.
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -6.0, 1.0


def mlp(in_dim: int, hidden: tuple[int, ...], dropout: float = 0.0) -> nn.Sequential:
    """Shared trunk builder. Both heads use it, so capacity stays matched."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        d = h
    return nn.Sequential(*layers)


def _no_dropout(module: nn.Module):
    """Context manager: run `module` with its Dropout layers switched off."""

    class _Ctx:
        def __enter__(self):
            self.was = [(m, m.training) for m in module.modules() if isinstance(m, nn.Dropout)]
            for m, _ in self.was:
                m.eval()

        def __exit__(self, *exc):
            for m, was in self.was:
                m.train(was)
            return False

    return _Ctx()


class SlipHead(nn.Module):
    def __init__(self, vis_dim: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.4):
        super().__init__()
        self.vis_dim = vis_dim
        self.trunk = mlp(vis_dim + NUM_GEARS, hidden, dropout)
        self.mu = nn.Linear(hidden[-1], 1)
        self.log_sigma = nn.Linear(hidden[-1], 1)
        nn.init.constant_(self.log_sigma.bias, -2.5)  # sigma ~= 0.08 at init

    # ------------------------------------------------------------------
    def _input(self, phi: torch.Tensor, gear: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(gear.long(), NUM_GEARS).to(phi.dtype)
        return torch.cat([phi, onehot], dim=-1)

    def forward(self, phi: torch.Tensor, gear: torch.Tensor):
        """-> (mu [B], log_sigma [B], mu_clean [B]).

        mu         dropout-active mean; trained by MSE; this is the concept.
        log_sigma  fitted on a DETACHED, dropout-free hidden state (see docstring).
        mu_clean   dropout-free mean, no grad: the target the sigma NLL is fitted
                   against, so sigma models terrain noise rather than dropout noise.
        """
        x = self._input(phi, gear)

        h = self.trunk(x)
        mu = torch.sigmoid(self.mu(h)).squeeze(-1)

        with _no_dropout(self.trunk):
            h_clean = self.trunk(x)
            with torch.no_grad():
                mu_clean = torch.sigmoid(self.mu(h_clean)).squeeze(-1)

        # detach: no NLL gradient may reach any parameter mu depends on.
        log_sigma = self.log_sigma(h_clean.detach()).squeeze(-1)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)

        return mu, log_sigma, mu_clean

    # ------------------------------------------------------------------
    def predict(self, phi: torch.Tensor, gear: int | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(gear):
            gear = torch.full((phi.shape[0],), int(gear), dtype=torch.long, device=phi.device)
        return self.forward(phi, gear)[0]

    def predict_all(self, phi: torch.Tensor):
        """Counterfactual slip at EVERY gear -> (mu [B,3], sigma [B,3], mu_clean [B,3]).

        This is the query the selector needs: "what would slip be if I used gear g?".
        In modular it is answered without ever having driven gear g here.

        The three gears are evaluated in ONE trunk call on a [B*3] batch, and the
        dropout mask is drawn once per SAMPLE and shared across its three gears.
        Evaluating them in three separate calls would give each column of the
        concept an independent mask, so the [B,3] vector the hybrid gear head reads
        would have its cross-gear structure scrambled by noise that has nothing to
        do with the terrain.
        """
        b = phi.shape[0]
        g = torch.arange(NUM_GEARS, device=phi.device).repeat(b)  # [B*3]
        phi_rep = phi.repeat_interleave(NUM_GEARS, dim=0)  # [B*3, D]
        x = self._input(phi_rep, g)

        h = self._trunk_shared_mask(x, b)
        mu = torch.sigmoid(self.mu(h)).squeeze(-1).view(b, NUM_GEARS)

        with _no_dropout(self.trunk):
            h_clean = self.trunk(x)
            with torch.no_grad():
                mu_clean = torch.sigmoid(self.mu(h_clean)).squeeze(-1).view(b, NUM_GEARS)

        log_sigma = self.log_sigma(h_clean.detach()).squeeze(-1)
        sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp().view(b, NUM_GEARS)

        return mu, sigma, mu_clean

    def _trunk_shared_mask(self, x: torch.Tensor, b: int) -> torch.Tensor:
        """Run the trunk with one dropout mask per sample, shared across its gears."""
        if not self.training:
            return self.trunk(x)

        h = x
        for layer in self.trunk:
            if isinstance(layer, nn.Dropout):
                keep = 1.0 - layer.p
                # one Bernoulli mask per sample, broadcast over that sample's 3 gears
                mask = torch.rand(b, 1, h.shape[-1], device=h.device) < keep
                h = h.view(b, NUM_GEARS, -1) * mask / keep
                h = h.view(b * NUM_GEARS, -1)
            else:
                h = layer(h)
        return h
