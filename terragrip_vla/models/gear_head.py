"""Gear head: features -> 3 gear logits.  The 'A' of the VLA.

Hidden widths are identical to SlipHead's (256, 128) so the three modes have
matched head capacity.  The input dimension necessarily differs, because the
modes consume different things -- that difference IS the experiment:

    e2e     : [phi_vis, phi_lang]                      (no physical concept)
    hybrid  : [slip_concept, scw * phi_vis, phi_lang]  (concept + leak knob)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from constants import NUM_GEARS
from models.slip_head import mlp


class GearHead(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.4):
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = hidden[-1]
        self.trunk = mlp(in_dim, hidden, dropout)
        self.out = nn.Linear(hidden[-1], NUM_GEARS)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """x [B, in_dim] -> logits [B, 3]  (+ penultimate features [B, 128])."""
        h = self.trunk(x)
        logits = self.out(h)
        return (logits, h) if return_features else logits
