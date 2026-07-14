"""M0: synthetic data -> Dataset -> DataLoader, end to end."""

import json

import numpy as np
import torch

from constants import IMAGE_SIZE, NUM_GEARS
from data import mock_generator as mock
from data.dataset import LanguageAugmenter, TerraGripDataset, make_loader
from data.labels import best_gear, best_gear_batch
from language import RISK_TABLE, LanguageEncoder


def test_truth_file_written(mock_root):
    truth = json.loads((mock_root / "mock_truth.json").read_text())
    assert truth["terrain_slip_mean"]["mud"] == [0.55, 0.30, 0.12]
    assert truth["ood_terrains"] == ["wet_tile", "loose_gravel"]


def test_splits_and_ood(sources):
    train_terr = set(sources["train"].terrains)
    assert train_terr == set(mock.TRAIN_TERRAINS)
    # The OOD split must contain a terrain the model has never trained on.
    assert set(sources["ood"].terrains).isdisjoint(train_terr)


def test_cal_matches_train_distribution(sources):
    """Conformal validity needs cal ~ train (exchangeable) and cal disjoint from test."""
    assert set(sources["cal"].terrains) == set(sources["train"].terrains)
    assert set(sources["cal"].seed_img).isdisjoint(set(sources["test"].seed_img))
    assert set(sources["train"].seed_img).isdisjoint(set(sources["cal"].seed_img))


def test_render_is_deterministic():
    a = mock.render_terrain("mud", 7)
    b = mock.render_terrain("mud", 7)
    c = mock.render_terrain("mud", 8)
    assert a.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_terrains_are_visually_distinct():
    """A frozen backbone can only work if the terrains differ in pixel space."""
    means = {t: mock.render_terrain(t, 3).reshape(-1, 3).mean(0) for t in mock.TERRAIN_SLIP}
    names = list(means)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gap = np.abs(means[names[i]] - means[names[j]]).max()
            assert gap > 8.0, f"{names[i]} and {names[j]} look identical"


def test_slip_noise_is_heteroscedastic():
    """The whole point of the conformal layer: noise depends on terrain."""
    rng = np.random.default_rng(0)
    std = {
        t: np.std([mock.sample_slip(t, 1, rng) for _ in range(4000)])
        for t in ("concrete", "mud")
    }
    assert std["concrete"] < 0.05 < std["mud"]


def test_sample_and_loader_shapes(sources):
    src = sources["train"]
    s = src.get(0)
    assert s.image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert s.roi_mask.shape == (IMAGE_SIZE, IMAGE_SIZE) and s.roi_mask.dtype == bool
    assert s.roi_mask.any() and not s.roi_mask.all()
    assert s.gear in range(NUM_GEARS)
    assert 0.0 <= s.slip <= 1.0
    assert s.slip_curve.shape == (NUM_GEARS,)

    batch = next(iter(make_loader(TerraGripDataset(src), batch_size=8, shuffle=False)))
    assert batch["image"].shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["roi"].shape == (8, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["gear"].shape == (8,) and batch["gear"].dtype == torch.long
    assert batch["slip"].shape == (8,)
    assert batch["slip_curve"].shape == (8, NUM_GEARS)
    assert len(batch["terrain"]) == 8


def test_best_gear_rule():
    # every gear inside budget -> take the cheapest (smallest contact area)
    assert best_gear([0.05, 0.04, 0.04], tau=0.15, lam=1.0) == 0
    # only L inside budget
    assert best_gear([0.55, 0.30, 0.12], tau=0.15, lam=1.0) == 2
    # nothing inside budget -> fall back to the largest gear
    assert best_gear([0.9, 0.9, 0.9], tau=0.15, lam=1.0) == 2
    # lam trades slip against contact area among acceptable gears
    assert best_gear([0.30, 0.05, 0.04], tau=0.35, lam=0.0) == 0
    assert best_gear([0.30, 0.05, 0.04], tau=0.35, lam=10.0) == 1


def test_best_gear_batch_matches_scalar():
    rng = np.random.default_rng(1)
    curves = np.sort(rng.random((64, NUM_GEARS)), axis=1)[:, ::-1].copy()  # slip decreasing in gear
    taus = rng.uniform(0.05, 0.6, 64)
    lams = rng.uniform(0.0, 1.0, 64)
    got = best_gear_batch(curves, taus, lams)
    want = [best_gear(c, t, l) for c, t, l in zip(curves, taus, lams)]
    assert list(got) == want


def test_language_augmenter_makes_oracle_labels(sources):
    """The missing half of the VLA: instruction -> risk budget -> gear label."""
    src = sources["train"]
    aug = LanguageAugmenter(LanguageEncoder("hash"), "train", seed=0)
    curves = src.slip_curve[:32]

    rnd = aug.random(curves)
    assert rnd["lang"].shape == (32, aug.dim)
    assert rnd["best_gear"].shape == (32,)
    assert set(rnd["intent"]) <= set(RISK_TABLE)

    # The same terrain gets a different gear under a different instruction:
    # that is what makes this a language-conditioned task at all.
    mud = np.array([mock.TERRAIN_SLIP["mud"]] * 4, np.float32)
    careful = aug.fixed("careful", mud)["best_gear"]
    fast = aug.fixed("fast", mud)["best_gear"]
    assert (careful == 2).all(), "careful on mud must pick L"  # tau=0.15 admits only L
    assert (fast == 1).all(), "fast on mud must pick M"
