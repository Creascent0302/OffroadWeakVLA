"""Shared setup used by train / eval / analysis, so nothing is duplicated.

One config, one data build, one frozen backbone, one feature cache -- every mode
and every seed goes through here.  That is what makes the comparison controlled
rather than three scripts that drifted apart.
"""

# ============================================================================
# 【中文导读】共享运行时。train / eval / analysis 全都从这里取上下文，
#   保证：同一份配置、同一份数据、同一个冻结骨干、同一份特征缓存。
#   这正是“受控对比”的前提 —— 否则三个脚本会各自漂移，对比就没意义了。
#
#   Context 里最关键的一步：phi 先按 train 统计量标准化，再做 L2 归一化。
#   这不是化妆，是承重的：原始 ||phi||≈28，而档位 one-hot / 句向量 / slip 概念
#   的范数都≈1，不归一化的话视觉会以 28:1 的比例淹没其余输入。
# ============================================================================


from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

import features as feat
from constants import SPLITS
from data.dataset import FeatureDataset, LanguageAugmenter, MockSource
from data.mock_generator import generate_all
from language import LanguageEncoder
from models.model import TerraGripModel
from models.perception import Perception

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"


def load_cfg(config_name: str = "default", overrides: list[str] | None = None) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: DictConfig | None = None) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_data(cfg: DictConfig) -> Path:
    """Generate the mock data, and REGENERATE it if the config no longer matches.

    Checking only that the .npz files exist is not enough: change data.seed or
    data.sizes and the stale files are silently reused, so you train on data that
    is not the data you asked for and nothing anywhere says so. The generating
    (seed, sizes) is therefore recorded next to the data and compared.
    """
    root = ROOT / cfg.data.root if not Path(cfg.data.root).is_absolute() else Path(cfg.data.root)
    want = {"seed": int(cfg.data.seed), "sizes": {k: int(v) for k, v in cfg.data.sizes.items()}}
    stamp = root / "generated_with.json"

    have = None
    if stamp.exists():
        try:
            have = json.loads(stamp.read_text())
        except Exception:
            have = None

    fresh = all((root / f"{s}.npz").exists() for s in SPLITS) and have == want
    if not fresh:
        generate_all(root, sizes=want["sizes"], seed=want["seed"])
        stamp.write_text(json.dumps(want, indent=2))
    return root


def checkpoint_path(cfg: DictConfig, mode: str, scw: float, seed: int) -> Path:
    tag = f"{mode}_scw{scw:g}_seed{seed}" if mode == "hybrid" else f"{mode}_seed{seed}"
    return ROOT / cfg.artifacts_dir / f"{tag}.pt"


@dataclass
class Context:
    """Everything a run needs, built exactly once."""

    cfg: DictConfig
    device: torch.device
    root: Path
    perception: Perception
    encoder: LanguageEncoder
    sources: dict[str, MockSource]
    phi: dict[str, torch.Tensor]

    def dataset(self, split: str) -> FeatureDataset:
        return FeatureDataset(self.sources[split], self.phi[split])

    def augmenter(self, paraphrase_split: str = "train", seed: int = 0) -> LanguageAugmenter:
        return LanguageAugmenter(self.encoder, paraphrase_split, seed=seed)

    @property
    def lang_dim(self) -> int:
        return self.encoder.dim


def build_context(cfg: DictConfig, splits: tuple[str, ...] = SPLITS) -> Context:
    device = get_device(cfg)
    root = ensure_data(cfg)

    perception = Perception(cfg.backbone).to(device)
    encoder = LanguageEncoder(cfg.language.encoder)

    sources = {s: MockSource(root, s) for s in splits}
    phi = {
        s: feat.get_features(
            sources[s], perception, s, ROOT / cfg.artifacts_dir, int(cfg.data.seed), device
        )
        for s in splits
    }

    # Standardise with TRAIN statistics, then L2-normalise to unit length.
    #
    # The normalisation is not cosmetic, it is load-bearing. Raw phi has L2 norm
    # ~sqrt(768) ~= 28, while every other input to every head has norm ~1: the
    # gear one-hot in SlipHead, the language embedding in GearHead, the slip
    # concept in hybrid. Concatenated raw, vision outweighs them ~28:1 and the
    # heads barely use them -- measured: the slip head's systematic error per
    # (terrain, gear) cell was 0.047 raw vs 0.017 normalised, i.e. it was largely
    # ignoring which gear it was asked about. One fixed, mode-agnostic affine map
    # fixes all three imbalances at once, so it cannot favour any mode.
    if "train" in phi:
        mu = phi["train"].mean(0, keepdim=True)
        sd = phi["train"].std(0, keepdim=True).clamp_min(1e-6)
        phi = {s: torch.nn.functional.normalize((v - mu) / sd, dim=-1) for s, v in phi.items()}

    return Context(cfg, device, root, perception, encoder, sources, phi)


def build_model(cfg: DictConfig, ctx: Context, mode: str | None = None, scw: float | None = None):
    return TerraGripModel(
        mode=mode or cfg.mode,
        lang_dim=ctx.lang_dim,
        lang_condition=cfg.language.condition,
        side_channel_weight=cfg.side_channel_weight if scw is None else scw,
        dropout=float(cfg.model.dropout),
        perception=ctx.perception,
    ).to(ctx.device)


def save_model(model: TerraGripModel, path: Path, cfg: DictConfig, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "heads": model.state_dict_heads(),
            "meta": {
                "mode": model.mode,
                "scw": model.scw,
                "lang_condition": model.lang_condition,
                "lang_dim": model.lang_dim,
                "backbone": cfg.backbone,
                "dropout": float(cfg.model.dropout),
                **(extra or {}),
            },
        },
        path,
    )


def load_model(path: Path, ctx: Context) -> TerraGripModel:
    ckpt = torch.load(path, map_location=ctx.device, weights_only=False)
    meta = ckpt["meta"]
    model = TerraGripModel(
        mode=meta["mode"],
        lang_dim=ctx.lang_dim,
        lang_condition=meta["lang_condition"],
        side_channel_weight=meta["scw"],
        dropout=float(meta.get("dropout", ctx.cfg.model.dropout)),
        perception=ctx.perception,
    ).to(ctx.device)
    model.load_state_dict_heads(ckpt["heads"])
    model.eval()
    return model


def to_device(lang: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in lang.items()
    }


def cfg_summary(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)
