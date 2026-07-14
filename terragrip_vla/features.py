"""Frozen-backbone feature cache.

The backbone never trains, and the ROI is fixed, so phi_vis is a pure function of
the sample.  Computing it once and reusing it is mathematically identical to
recomputing it every epoch, and it turns the 3 modes x 5 seeds x 4 scw sweep from
hours into minutes.  It is also what lets the whole pipeline be validated on a
laptop CPU.
"""

# ============================================================================
# 【中文导读】冻结骨干的特征缓存。
#   骨干不训练、ROI 固定 ⇒ phi_vis 是样本的纯函数 ⇒ 算一次存起来即可。
#   这在数学上和每个 epoch 重算完全等价，但让 3 模式 × 5 种子 × 7 个 scw 的
#   全套实验从几小时变成几分钟，也让整条流水线能在笔记本 CPU 上跑通。
#   缓存键含【图像内容哈希】：改了数据规模/地形，缓存不会被静默复用。
# ============================================================================


from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import TerraGripDataset, make_loader
from models.perception import Perception


def source_fingerprint(source) -> str:
    """Hash of what the images actually ARE, not of how they were configured.

    Keying the cache on (backbone, seed, split) alone is not enough: change
    data.sizes, or the terrain palette, and the manifest changes while the key does
    not. get_features would then load a cache of the right LENGTH but the wrong
    CONTENT and train on features belonging to different images -- silently, with
    no error anywhere. Hashing the identity of every image closes that.
    """
    h = hashlib.sha256()
    h.update(np.asarray(source.seed_img, np.int64).tobytes())
    h.update(np.asarray(source.terrain_idx, np.int64).tobytes())
    h.update(",".join(source.terrains).encode())
    return h.hexdigest()[:12]


def cache_path(
    artifacts_dir: str | Path, backbone: str, split: str, data_seed: int, fingerprint: str = ""
) -> Path:
    tag = f"{backbone}_seed{data_seed}_{split}" + (f"_{fingerprint}" if fingerprint else "")
    return Path(artifacts_dir) / "features" / f"{tag}.pt"


@torch.no_grad()
def build_features(
    source,
    perception: Perception,
    device: torch.device,
    batch_size: int = 64,
    progress: bool = True,
) -> torch.Tensor:
    perception.to(device).eval()
    loader = make_loader(TerraGripDataset(source), batch_size, shuffle=False)
    chunks = []
    it = tqdm(loader, desc="phi_vis", leave=False) if progress else loader
    for batch in it:
        phi = perception.encode(batch["image"].to(device), batch["roi"].to(device))
        chunks.append(phi.float().cpu())
    return torch.cat(chunks)


def get_features(
    source,
    perception: Perception,
    split: str,
    artifacts_dir: str | Path,
    data_seed: int,
    device: torch.device,
    batch_size: int = 64,
    rebuild: bool = False,
) -> torch.Tensor:
    """Load the cache, or build and save it.  Keyed by image content, so it cannot go stale."""
    path = cache_path(
        artifacts_dir, perception.backbone_name, split, data_seed, source_fingerprint(source)
    )
    if path.exists() and not rebuild:
        phi = torch.load(path, map_location="cpu")
        if len(phi) == len(source):
            return phi

    phi = build_features(source, perception, device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(phi, path)
    return phi
