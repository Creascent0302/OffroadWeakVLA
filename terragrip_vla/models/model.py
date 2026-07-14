"""TerraGripModel: one class, three decision routes.

All three modes take exactly the same input -- (image, instruction) -- and emit a
gear.  All three are genuine VLAs.  The only thing that changes is HOW the gear
is reached:

    modular  vision -> slip(g) --[conformal + risk budget from language]--> gear
             The arrow in brackets is analytic, not learned.  No selection
             gradient ever touches the slip head.

    e2e      (vision, language) -> gear
             One black box.  The physical concept never exists.

    hybrid   (vision, language) -> slip(g) bottleneck (+ scw * vision) -> gear
             Jointly trained: the gear loss backprops THROUGH the concept.
             `side_channel_weight` is the leakage knob: at 0 the decision can
             only see vision through slip; above 0 it can go around it.

Language never reaches the slip head, in any mode: slip is physics, and physics
does not listen to instructions.  Language only ever shapes the *decision*.
"""

# ============================================================================
# 【中文导读】统一模型：一个类，三条决策路径，靠 config 的 mode 一键切换。
#
#   modular  视觉 → slip(g) --[保形上界 + 语言给的风险预算]--> 档位
#            中括号里那一步是【解析的，不是学的】。选档梯度永远碰不到 slip 头。
#   e2e      (视觉, 语言) → 档位。纯黑箱，物理概念根本不存在。
#   hybrid   (视觉, 语言) → slip 瓶颈(+ scw·视觉旁路) → 档位，联合训练。
#            选档的 CE 梯度会【穿过概念】回流到 slip 头。
#            scw = 泄漏旋钮：=0 时决策只能透过 slip 看世界；>0 时可以绕过去。
#
#   任何模式下，语言都到不了 slip 头 —— 物理不听指令。语言只塑造【决策】。
# ============================================================================


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import NUM_GEARS
from models.gear_head import GearHead
from models.perception import Perception
from models.slip_head import SlipHead

MODES = ("modular", "e2e", "hybrid")
BUDGET_DIM = 3  # (alpha, tau, lambda)


def language_features(lang_batch: dict, condition: str, device=None) -> torch.Tensor:
    """Turn a LanguageAugmenter batch into the tensor the gear head consumes.

    text   : frozen MiniLM embedding      -> the honest VLA setting
    budget : the (alpha, tau, lam) vector -> lets e2e accept a risk budget it was
             never trained on, which is how we test risk-budget generalisation
             on equal footing with modular
    none   : nothing                      -> ablation; instruction following dies
    """
    if condition == "text":
        out = lang_batch["lang"]
    elif condition == "budget":
        out = torch.stack(
            [lang_batch["alpha"], lang_batch["tau"], lang_batch["lam"]], dim=-1
        )
    elif condition == "none":
        n = len(lang_batch["intent"])
        out = torch.zeros(n, 0)
    else:
        raise ValueError(f"unknown language.condition: {condition}")
    return out.to(device) if device is not None else out


class TerraGripModel(nn.Module):
    def __init__(
        self,
        mode: str,
        backbone: str = "dinov2_vitb14",
        lang_dim: int = 384,
        lang_condition: str = "text",
        side_channel_weight: float = 0.0,
        dropout: float = 0.4,
        perception: Perception | None = None,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        self.mode = mode
        self.scw = float(side_channel_weight)
        self.lang_condition = lang_condition

        # Shared and frozen. Passing an existing Perception lets several models
        # reuse one backbone (and one feature cache).
        self.perception = perception if perception is not None else Perception(backbone)
        d_vis = self.perception.dim

        if lang_condition == "none":
            self.lang_dim = 0
        elif lang_condition == "budget":
            self.lang_dim = BUDGET_DIM
        else:
            self.lang_dim = int(lang_dim)

        self.dropout = float(dropout)
        self.slip_head = SlipHead(d_vis, dropout=self.dropout) if mode in ("modular", "hybrid") else None

        if mode == "e2e":
            self.gear_head = GearHead(d_vis + self.lang_dim, dropout=self.dropout)
        elif mode == "hybrid":
            self.gear_head = GearHead(NUM_GEARS + d_vis + self.lang_dim, dropout=self.dropout)
        else:
            self.gear_head = None  # modular selects analytically, at inference

    # ------------------------------------------------------------------
    @property
    def has_concept(self) -> bool:
        return self.slip_head is not None

    def encode(self, image: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        return self.perception.encode(image, roi)

    def gear_logits_from(
        self,
        concept: torch.Tensor | None,
        phi: torch.Tensor,
        lang: torch.Tensor,
        return_features: bool = False,
    ):
        """The gear head, exposed so the analyses can feed it a doctored concept.

        leakage.py shuffles `concept`; intervention.py replaces it with the true
        slip.  Both need this entry point.
        """
        if self.mode == "e2e":
            x = torch.cat([phi, lang], dim=-1)
        elif self.mode == "hybrid":
            # L2-normalise before scaling, so ||side|| == scw exactly. Without
            # this the knob is useless: raw phi has norm ~sqrt(768) ~= 28, versus
            # ~0.5 for the concept and 1.0 for the language embedding, so even
            # scw=0.1 would swamp the bottleneck and the whole sweep would
            # saturate at its first non-zero point. Normalised, scw is literally
            # "how much vision energy is allowed around the concept", on the same
            # scale as the other two inputs, and 0 is still exactly zero.
            side = self.scw * F.normalize(phi, dim=-1)
            x = torch.cat([concept, side, lang], dim=-1)
        else:
            raise RuntimeError("modular has no gear head; use conformal/select.py")
        return self.gear_head(x, return_features=return_features)

    def forward(
        self,
        phi: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        roi: torch.Tensor | None = None,
        lang: torch.Tensor | None = None,
        concept_override: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
        """Either `phi` (cached) or (`image`, `roi`) must be given.

        Returns, depending on mode:
            slip       [B, 3] predicted slip at every gear (modular, hybrid)
            slip_sigma [B, 3] predictive std, used only to normalise the
                              conformal score (modular, hybrid)
            slip_clean [B, 3] dropout-free mean; what the sigma NLL is fitted
                              against, so sigma models terrain noise not dropout
            gear_logits[B, 3] (e2e, hybrid)
            phi        [B, D]
        """
        if phi is None:
            if image is None or roi is None:
                raise ValueError("give either phi, or both image and roi")
            phi = self.encode(image, roi)

        out: dict[str, torch.Tensor] = {"phi": phi}

        if self.has_concept:
            slip_mu, slip_sigma, slip_clean = self.slip_head.predict_all(phi)  # [B,3] each
            out["slip"] = slip_mu            # the concept (dropout-active in train)
            out["slip_sigma"] = slip_sigma   # conformal normaliser; no grad into mu
            out["slip_clean"] = slip_clean   # dropout-free mean; the sigma NLL's target

        if self.mode == "modular":
            # No gear head on purpose: the gear is produced at inference time by
            # the conformal selector, which is where the risk budget (language)
            # enters. Nothing here can backprop into the slip head from a gear.
            return out

        if lang is None:
            lang = phi.new_zeros(phi.shape[0], self.lang_dim)
        if lang.shape[-1] != self.lang_dim:
            raise ValueError(f"lang dim {lang.shape[-1]} != expected {self.lang_dim}")
        lang = lang.to(phi.dtype)

        concept = None
        if self.mode == "hybrid":
            concept = concept_override if concept_override is not None else out["slip"]
            concept = concept.to(phi.dtype)

        res = self.gear_logits_from(concept, phi, lang, return_features=return_features)
        if return_features:
            out["gear_logits"], out["gear_feat"] = res
        else:
            out["gear_logits"] = res
        return out

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict_heads(self) -> dict:
        """Only the heads: the backbone is frozen and never needs saving."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("perception.")}

    def load_state_dict_heads(self, sd: dict) -> None:
        self.load_state_dict(sd, strict=False)
