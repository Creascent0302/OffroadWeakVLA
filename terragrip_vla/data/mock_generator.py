"""Synthetic terrain data with HETEROSCEDASTIC slip noise.

Why heteroscedastic: the noise level per terrain is the whole reason the
conformal layer has anything to do.  On concrete slip is almost deterministic, on
mud it is very noisy.  A selector that ignores this either over-trusts mud or
needlessly refuses concrete.  It is also what gives the leakage / probe /
intervention analyses something to separate.

Images are NOT stored.  A sample is (terrain, gear, slip, seed) and the image is
re-rendered deterministically from (terrain, seed).  The manifest stays tiny, the
data are exactly reproducible, and the frozen-backbone feature cache makes the
render cost a one-off.
"""

# ============================================================================
# 【中文导读】合成数据生成器。三个设计要点：
#
#  1. 异方差噪声（关键）。每种地形的打滑噪声 sigma 不同（水泥 0.02，泥地 0.15）。
#     这不是装饰 —— 它是保形层存在的全部理由：忽略它的选择器要么过度信任泥地，
#     要么无谓地拒绝水泥地。leakage / 探针 / 干预三个分析也全靠它才有区分度。
#
#  2. OOD 地形的真值【刻意避开 tau 边界】。第一版用了 [0.35,0.20,0.15]，
#     恰好精确落在三个 tau (0.15/0.25/0.35) 上，于是 "slip <= tau" 变成了浮点
#     抛硬币，一个 epsilon 量级的保形宽度就能把 oracle 干预精度从 1.00 掀到 0.67。
#     真值落在刀刃上，下游每个数字都不可信。
#
#  3. 图像不存盘。样本只是 (地形, 档位, slip, seed)，图像由 (地形, seed) 确定性
#     重渲染。manifest 极小、完全可复现，而冻结骨干的特征缓存让渲染只发生一次。
# ============================================================================


from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from constants import GEARS, IMAGE_SIZE, NUM_GEARS, SLIP_MAX, SLIP_MIN

# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
# Mean slip at gear [S, M, L].  Larger contact area -> less slip, always.
TERRAIN_SLIP: dict[str, list[float]] = {
    "concrete": [0.05, 0.04, 0.04],
    "grass": [0.20, 0.10, 0.06],
    "mud": [0.55, 0.30, 0.12],
    "sand": [0.40, 0.22, 0.10],
    # OOD: never seen in train/cal/test.
    #
    # These values are deliberately kept OFF the tau boundaries (0.15/0.25/0.35).
    # The first draft used [0.35, 0.20, 0.15], which sits EXACTLY on all three, so
    # `slip <= tau` became a floating-point coin flip and adding an epsilon-sized
    # conformal width (1.65 * 1e-3) flipped the decision. The oracle-intervention
    # accuracy on OOD swung from 1.00 to 0.67 on nothing but round-off. Ground truth
    # on a knife edge makes every downstream number untrustworthy.
    # With [0.30, 0.18, 0.10] the oracle gears are still (L, M, S) across the three
    # instructions -- the full spread we want -- but with >= 0.03 of margin.
    "wet_tile": [0.30, 0.18, 0.10],
    # A SECOND OOD terrain, and it is not optional. With one OOD terrain the
    # oracle gear is a constant per instruction, so the language-only floor in
    # eval/analysis/leakage.py is exactly 1.0, the headroom is <= 0, and the whole
    # OOD leakage curve is void by construction. Two terrains with DIFFERENT oracle
    # gears (wet_tile -> L,M,S ; loose_gravel -> L,L,M) make the floor 0.67.
    "loose_gravel": [0.45, 0.28, 0.12],
}

# Per-terrain slip noise std.  This is the heteroscedasticity.
TERRAIN_SIGMA: dict[str, float] = {
    "concrete": 0.02,
    "grass": 0.05,
    "mud": 0.15,
    "sand": 0.10,
    "wet_tile": 0.12,
    "loose_gravel": 0.08,
}

TRAIN_TERRAINS: list[str] = ["concrete", "grass", "mud", "sand"]
OOD_TERRAINS: list[str] = ["wet_tile", "loose_gravel"]

DEFAULT_SIZES: dict[str, int] = {"train": 2000, "cal": 1000, "test": 1000, "ood": 500}

# Deterministic per-split offset so no image seed is ever reused across splits.
_SPLIT_SEED_BASE: dict[str, int] = {
    "train": 1_000_000,
    "cal": 2_000_000,
    "test": 3_000_000,
    "ood": 4_000_000,
}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _band_noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    """Spatial noise in [0,1] whose characteristic blob size is ~`scale` px."""
    low = max(2, size // max(1, scale))
    coarse = rng.random((low, low), dtype=np.float32)
    up = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(up.min()), float(up.max())
    return (up - lo) / (hi - lo + 1e-8)


def render_terrain(terrain: str, seed: int, size: int = IMAGE_SIZE) -> np.ndarray:
    """Deterministic (terrain, seed) -> HxWx3 uint8 RGB.

    Each terrain gets a distinct base colour AND a distinct spatial statistic, so
    a frozen backbone can separate them without relying on colour alone.
    """
    rng = np.random.default_rng(seed)

    if terrain == "concrete":
        base = np.array([142, 142, 145], np.float32)
        tex = 0.5 * _band_noise(rng, size, 2) + 0.5 * _band_noise(rng, size, 24)
        img = base[None, None] * (0.88 + 0.24 * tex[..., None])
        # a few hairline cracks
        for _ in range(rng.integers(1, 4)):
            p = rng.integers(0, size, size=4)
            cv2.line(img, (p[0], p[1]), (p[2], p[3]), (105, 105, 108), 1)

    elif terrain == "grass":
        base = np.array([72, 118, 54], np.float32)
        tex = 0.7 * _band_noise(rng, size, 4) + 0.3 * _band_noise(rng, size, 32)
        img = base[None, None] * (0.70 + 0.60 * tex[..., None])
        img[..., 1] *= 1.10  # push green

    elif terrain == "mud":
        base = np.array([78, 58, 42], np.float32)
        tex = 0.75 * _band_noise(rng, size, 20) + 0.25 * _band_noise(rng, size, 5)
        img = base[None, None] * (0.70 + 0.65 * tex[..., None])
        wet = (_band_noise(rng, size, 14) > 0.80).astype(np.float32)  # specular pools
        img += 55.0 * cv2.GaussianBlur(wet, (0, 0), 2.0)[..., None]

    elif terrain == "sand":
        base = np.array([198, 176, 128], np.float32)
        tex = 0.35 * _band_noise(rng, size, 2) + 0.65 * _band_noise(rng, size, 40)
        img = base[None, None] * (0.85 + 0.28 * tex[..., None])

    elif terrain == "wet_tile":  # OOD: novel colour AND novel regular structure
        base = np.array([148, 158, 172], np.float32)
        tex = _band_noise(rng, size, 30)
        img = base[None, None] * (0.86 + 0.26 * tex[..., None])
        pitch = int(rng.integers(24, 34))
        off = int(rng.integers(0, pitch))
        img[:, off::pitch, :] *= 0.72  # grout lines
        img[off::pitch, :, :] *= 0.72

    elif terrain == "loose_gravel":  # OOD: novel colour and coarse speckle
        base = np.array([120, 112, 104], np.float32)
        tex = 0.8 * _band_noise(rng, size, 6) + 0.2 * _band_noise(rng, size, 40)
        img = base[None, None] * (0.72 + 0.62 * tex[..., None])
        stones = (_band_noise(rng, size, 7) > 0.72).astype(np.float32)
        img += 45.0 * stones[..., None]

    else:
        raise ValueError(f"unknown terrain: {terrain}")

    img *= float(rng.uniform(0.85, 1.15))  # global lighting jitter
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def roi_mask(size: int = IMAGE_SIZE) -> np.ndarray:
    """Fixed trapezoid over the ground the vehicle is about to drive onto.

    Fixed (not per-sample) on purpose: it makes the frozen-backbone features
    cacheable, which is what keeps the multi-seed x multi-mode sweep cheap.
    """
    mask = np.zeros((size, size), np.uint8)
    top_y = int(0.55 * size)
    half_top = int(0.20 * size)
    poly = np.array(
        [
            [size // 2 - half_top, top_y],
            [size // 2 + half_top, top_y],
            [int(0.95 * size), size - 1],
            [int(0.05 * size), size - 1],
        ],
        np.int32,
    )
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_slip(terrain: str, gear: int, rng: np.random.Generator) -> float:
    mean = TERRAIN_SLIP[terrain][gear]
    noisy = mean + rng.normal(0.0, TERRAIN_SIGMA[terrain])
    return float(np.clip(noisy, SLIP_MIN, SLIP_MAX))


def generate_split(
    split: str,
    n: int,
    seed: int = 0,
    terrains: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Manifest for one split.  Images are re-rendered from `seed_img` on demand."""
    if terrains is None:
        terrains = OOD_TERRAINS if split == "ood" else TRAIN_TERRAINS

    rng = np.random.default_rng(seed + _SPLIT_SEED_BASE[split])

    t_idx = rng.integers(0, len(terrains), size=n)
    # Uniform gear coverage: the slip head must be accurate at every gear, since
    # selection queries all three counterfactually.
    gears = rng.integers(0, NUM_GEARS, size=n)

    names = [terrains[i] for i in t_idx]
    slips = np.array([sample_slip(t, int(g), rng) for t, g in zip(names, gears)], np.float32)
    curves = np.array([TERRAIN_SLIP[t] for t in names], np.float32)
    seeds_img = _SPLIT_SEED_BASE[split] + np.arange(n, dtype=np.int64)

    return {
        "terrain_idx": t_idx.astype(np.int64),
        "terrain_names": np.array(terrains, dtype=object),
        "gear": gears.astype(np.int64),
        "slip": slips,
        "slip_curve": curves,
        "seed_img": seeds_img,
    }


def truth_dict() -> dict:
    return {
        "gears": GEARS,
        "terrain_slip_mean": TERRAIN_SLIP,
        "terrain_slip_sigma": TERRAIN_SIGMA,
        "train_terrains": TRAIN_TERRAINS,
        "ood_terrains": OOD_TERRAINS,
        "note": (
            "slip[terrain][gear] is the TRUE mean slip. Observed slip = "
            "clip(mean + N(0, sigma[terrain]), 0, 1). Heteroscedastic on purpose."
        ),
    }


def generate_all(
    out_dir: str | Path,
    sizes: dict[str, int] | None = None,
    seed: int = 0,
) -> Path:
    """Write train/cal/test/ood manifests + artifacts/mock_truth.json.

    `cal` is drawn i.i.d. from the same distribution as `train` and is disjoint
    from `test`.  That exchangeability is exactly what the conformal guarantee
    rests on.
    """
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, n in sizes.items():
        np.savez(out_dir / f"{split}.npz", **generate_split(split, n, seed=seed))

    truth_path = out_dir / "mock_truth.json"
    truth_path.write_text(json.dumps(truth_dict(), indent=2))
    return out_dir


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/mock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = generate_all(args.out, seed=args.seed)
    print(f"wrote mock data -> {path}")
