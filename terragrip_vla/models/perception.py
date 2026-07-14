"""Frozen visual front-end, shared verbatim by all three modes.

Freezing it is what makes the A/B/C comparison controlled: every mode sees the
exact same phi_vis, so any difference between modes comes from the head and the
decision route, never from a different representation.

It also means phi_vis is a pure function of the image, so it can be cached --
see data/dataset.FeatureDataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import PATCH_SIZE

DINOV2_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}


class _DummyBackbone(nn.Module):
    """Fixed random patch embedding.  Offline/CI stand-in for DINOv2.

    It is a real (if weak) frozen feature extractor -- a random linear projection
    of each 14x14 patch -- so shape/plumbing tests are meaningful without a
    300 MB download.  Never use it for reported numbers.
    """

    def __init__(self, dim: int = 768, patch: int = PATCH_SIZE):
        super().__init__()
        self.embed_dim = dim
        self.patch_size = patch
        self.proj = nn.Conv2d(3, dim, patch, patch, bias=False)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            w = torch.randn(self.proj.weight.shape, generator=g) / (3 * patch * patch) ** 0.5
            self.proj.weight.copy_(w)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward_features(self, x: torch.Tensor) -> dict:
        tokens = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]
        return {"x_norm_patchtokens": self.norm(torch.tanh(tokens))}


class Perception(nn.Module):
    """image + roi_mask -> phi_vis [B, D] by masked mean pooling of patch tokens."""

    def __init__(self, backbone: str = "dinov2_vitb14"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "dummy":
            self.net = _DummyBackbone()
        elif backbone in DINOV2_DIMS:
            self.net = torch.hub.load("facebookresearch/dinov2", backbone, trust_repo=True)
        else:
            raise ValueError(f"unknown backbone: {backbone}")

        self.dim = int(getattr(self.net, "embed_dim", DINOV2_DIMS.get(backbone, 768)))
        self.patch = PATCH_SIZE

        self.net.eval()
        self.net.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D102 - the backbone stays frozen
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def encode(self, image: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        """image [B,3,H,W] (ImageNet-normalised), roi [B,H,W] bool -> phi [B,D].

        H and W must be multiples of the patch size (14).
        """
        b, _, h, w = image.shape
        if h % self.patch or w % self.patch:
            raise ValueError(f"image {h}x{w} is not a multiple of patch {self.patch}")

        gh, gw = h // self.patch, w // self.patch
        tokens = self.net.forward_features(image)["x_norm_patchtokens"]  # [B, gh*gw, D]
        if tokens.shape[1] != gh * gw:
            raise RuntimeError(f"expected {gh * gw} patch tokens, got {tokens.shape[1]}")

        # Fraction of each patch that lies inside the ROI -> soft pooling weights.
        weight = F.adaptive_avg_pool2d(roi.to(tokens.dtype).unsqueeze(1), (gh, gw))
        weight = weight.flatten(1)  # [B, gh*gw]

        total = weight.sum(1, keepdim=True)
        # Degenerate ROI (nothing selected) -> fall back to a global mean rather
        # than dividing by zero.
        empty = total.squeeze(1) <= 1e-6
        if empty.any():
            weight = weight.clone()
            weight[empty] = 1.0
            total = weight.sum(1, keepdim=True)

        weight = weight / total
        return (tokens * weight.unsqueeze(-1)).sum(1)  # [B, D]

    def forward(self, image: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        return self.encode(image, roi)
