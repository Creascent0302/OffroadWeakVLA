"""Datasets, feature cache and language augmentation.

Three things live here:

1. `DataSource`  -- the abstract data interface.  `MockSource` implements it now;
   a real-robot adapter (Phase 2) implements the same methods and nothing
   downstream changes.

2. `TerraGripDataset` / `FeatureDataset` -- torch datasets.  Because the backbone
   is FROZEN, phi_vis for a given sample never changes, so we compute it once and
   train the small heads on cached features.  Mathematically identical, ~100x
   faster, which is what makes the 3 modes x 5 seeds x 4 scw sweep cheap.

3. `LanguageAugmenter` -- the piece the original spec was missing.  Training data
   has no language, so we SYNTHESISE it: every sample, every epoch, draws a random
   instruction; its risk budget (tau, lam) turns the true slip curve into an
   ORACLE gear label.  That is what lets e2e/hybrid be genuine VLAs -- they see
   (image, instruction) and must produce a gear -- while the slip head stays
   language-free, because physics is language-invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from constants import IMAGENET_MEAN, IMAGENET_STD
from data import mock_generator as mock
from data.labels import best_gear_batch
from data.schema import Sample
from language import INTENTS, LanguageEncoder, RISK_TABLE, instruction_pool


# --------------------------------------------------------------------------
# Data source (the Phase-2 extension point)
# --------------------------------------------------------------------------
@runtime_checkable
class DataSource(Protocol):
    def __len__(self) -> int: ...

    def get(self, i: int) -> Sample:
        """Full sample, including the (possibly expensive) image."""

    def labels(self, i: int) -> dict:
        """Cheap label-only view: gear, slip, slip_curve, terrain.  No image."""

    @property
    def terrains(self) -> list[str]: ...


class MockSource:
    """Reads a manifest written by mock_generator and re-renders images on demand."""

    def __init__(self, root: str | Path, split: str):
        data = np.load(Path(root) / f"{split}.npz", allow_pickle=True)
        self._terrains = [str(t) for t in data["terrain_names"]]
        self.terrain_idx = data["terrain_idx"]
        self.gear = data["gear"]
        self.slip = data["slip"]
        self.slip_curve = data["slip_curve"]
        self.seed_img = data["seed_img"]
        self._roi = mock.roi_mask()

    def __len__(self) -> int:
        return len(self.gear)

    @property
    def terrains(self) -> list[str]:
        return list(self._terrains)

    def terrain_of(self, i: int) -> str:
        return self._terrains[int(self.terrain_idx[i])]

    def labels(self, i: int) -> dict:
        return {
            "gear": int(self.gear[i]),
            "slip": float(self.slip[i]),
            "slip_curve": self.slip_curve[i].astype(np.float32),
            "terrain": self.terrain_of(i),
        }

    def get(self, i: int) -> Sample:
        lab = self.labels(i)
        return Sample(
            image=mock.render_terrain(lab["terrain"], int(self.seed_img[i])),
            roi_mask=self._roi,
            gear=lab["gear"],
            slip=lab["slip"],
            slip_curve=lab["slip_curve"],
            meta={"terrain": lab["terrain"], "seed": int(self.seed_img[i]), "index": i},
        )

    # Convenience: whole-split label arrays, used all over eval.
    def all_labels(self) -> dict:
        return {
            "gear": np.asarray(self.gear, np.int64),
            "slip": np.asarray(self.slip, np.float32),
            "slip_curve": np.asarray(self.slip_curve, np.float32),
            "terrain": np.array([self.terrain_of(i) for i in range(len(self))], dtype=object),
        }


# --------------------------------------------------------------------------
# Torch datasets
# --------------------------------------------------------------------------
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float, ImageNet-normalised."""
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return (x - _MEAN) / _STD


def _label_item(src: DataSource, i: int) -> dict:
    lab = src.labels(i)
    return {
        "gear": torch.tensor(lab["gear"], dtype=torch.long),
        "slip": torch.tensor(lab["slip"], dtype=torch.float32),
        "slip_curve": torch.from_numpy(np.asarray(lab["slip_curve"], np.float32)),
        "terrain": lab["terrain"],
        "index": i,
    }


class TerraGripDataset(Dataset):
    """Raw-image dataset.  Builds the feature cache; also used by image-space tests."""

    def __init__(self, source: DataSource):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, i: int) -> dict:
        s = self.source.get(i)
        item = _label_item(self.source, i)
        item["image"] = to_tensor(s.image)
        item["roi"] = torch.from_numpy(s.roi_mask.copy())
        return item


class FeatureDataset(Dataset):
    """Cached-phi dataset.  Identical labels, no backbone in the loop."""

    def __init__(self, source: DataSource, phi: torch.Tensor):
        if len(phi) != len(source):
            raise ValueError(f"stale feature cache: {len(phi)} feats vs {len(source)} samples")
        self.source = source
        self.phi = phi

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, i: int) -> dict:
        item = _label_item(self.source, i)
        item["phi"] = self.phi[i]
        return item


def collate(batch: list[dict]) -> dict:
    keys = ("image", "roi", "gear", "slip", "slip_curve", "phi")
    out = {k: torch.stack([b[k] for b in batch]) for k in keys if k in batch[0]}
    out["terrain"] = [b["terrain"] for b in batch]
    out["index"] = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return out


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,  # rendering is cheap and this keeps runs bit-exact
        generator=g if shuffle else None,
        drop_last=False,
    )


# --------------------------------------------------------------------------
# Language augmentation  (the missing half of the VLA)
# --------------------------------------------------------------------------
class LanguageAugmenter:
    """Attaches a synthetic instruction -- and therefore a gear label -- to a batch.

    The dataset has no language.  We create it: draw an intent, draw one of its
    paraphrases, decode its risk budget, and turn the sample's TRUE slip curve
    into the gear an oracle would pick under that budget.

    Every mode sees exactly the same (image, instruction) input, so the A/B/C
    comparison is fair.  Only the *route* from input to gear differs.
    """

    def __init__(self, encoder: LanguageEncoder, paraphrase_split: str = "train", seed: int = 0):
        self.encoder = encoder
        self.paraphrase_split = paraphrase_split
        self.rng = np.random.default_rng(seed)
        self.pool = {i: instruction_pool(i, paraphrase_split) for i in INTENTS}

    @property
    def dim(self) -> int:
        return self.encoder.dim

    def _pack(self, texts: list[str], intents: list[str], slip_curve) -> dict:
        slip_curve = np.asarray(slip_curve, np.float32)
        budgets = [RISK_TABLE[i] for i in intents]
        alpha = np.array([b.alpha for b in budgets], np.float32)
        tau = np.array([b.tau for b in budgets], np.float32)
        lam = np.array([b.lam for b in budgets], np.float32)
        return {
            "text": texts,
            "intent": intents,
            "lang": torch.from_numpy(self.encoder.encode(texts)),
            "alpha": torch.from_numpy(alpha),
            "tau": torch.from_numpy(tau),
            "lam": torch.from_numpy(lam),
            "best_gear": torch.from_numpy(best_gear_batch(slip_curve, tau, lam)),
        }

    def random(self, slip_curve) -> dict:
        """One random instruction per sample.  Used during training."""
        n = len(slip_curve)
        intents = [str(x) for x in self.rng.choice(INTENTS, size=n)]
        texts = [str(self.rng.choice(self.pool[i])) for i in intents]
        return self._pack(texts, intents, slip_curve)

    def fixed(self, intent: str, slip_curve, text: str | None = None) -> dict:
        """The same intent for every sample.  Used at evaluation."""
        n = len(slip_curve)
        chosen = text if text is not None else self.pool[intent][0]
        return self._pack([chosen] * n, [intent] * n, slip_curve)


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
