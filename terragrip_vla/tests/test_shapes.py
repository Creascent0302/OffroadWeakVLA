"""M1: the frozen perception front-end."""

import pytest
import torch

from constants import IMAGE_SIZE, NUM_GEARS
from data.dataset import TerraGripDataset, make_loader
from models.perception import Perception


@pytest.fixture(scope="module")
def perception():
    return Perception("dummy")


def test_phi_shape_and_no_grad(perception, sources):
    batch = next(iter(make_loader(TerraGripDataset(sources["train"]), 6, shuffle=False)))
    phi = perception.encode(batch["image"], batch["roi"])
    assert phi.shape == (6, perception.dim)
    assert torch.isfinite(phi).all()
    # Frozen: no gradient must ever flow back into the backbone.
    assert not phi.requires_grad
    assert all(not p.requires_grad for p in perception.parameters())


def test_backbone_stays_in_eval_mode(perception):
    perception.train(True)
    assert not perception.net.training, "backbone must stay frozen/eval even in train()"


def test_roi_actually_masks(perception):
    """phi must depend on the ROI, otherwise the pooling is a no-op."""
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    lower = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    upper = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    lower[:, IMAGE_SIZE // 2 :, :] = True
    upper[:, : IMAGE_SIZE // 2, :] = True
    assert not torch.allclose(
        perception.encode(img, lower), perception.encode(img, upper), atol=1e-5
    )


def test_empty_roi_does_not_nan(perception):
    img = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    empty = torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    assert torch.isfinite(perception.encode(img, empty)).all()


def test_non_multiple_of_patch_rejected(perception):
    img = torch.randn(1, 3, 100, 100)
    roi = torch.ones(1, 100, 100, dtype=torch.bool)
    with pytest.raises(ValueError, match="multiple of patch"):
        perception.encode(img, roi)


def test_phi_separates_terrains(perception, sources):
    """A frozen encoder is only useful if terrains land in different places."""
    src = sources["train"]
    feats: dict[str, list] = {}
    ds = TerraGripDataset(src)
    for i in range(96):
        item = ds[i]
        phi = perception.encode(item["image"][None], item["roi"][None])[0]
        feats.setdefault(item["terrain"], []).append(phi)

    centroids = {t: torch.stack(v).mean(0) for t, v in feats.items() if len(v) >= 5}
    assert len(centroids) >= 2
    names = list(centroids)
    within = torch.stack([torch.stack(feats[t]).std(0).mean() for t in names]).mean()
    between = torch.stack(
        [
            (centroids[a] - centroids[b]).abs().mean()
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        ]
    ).mean()
    assert between > within, "terrains are not separable in phi space"


@pytest.mark.slow
def test_real_dinov2_dim():
    """Only meaningful with network access; skipped otherwise."""
    try:
        p = Perception("dinov2_vitb14")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"dinov2 unavailable: {exc}")
    assert p.dim == 768
    img = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    roi = torch.ones(2, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    assert p.encode(img, roi).shape == (2, 768)
    assert NUM_GEARS == 3
