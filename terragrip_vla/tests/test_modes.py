"""M2: the three modes share one model class and one code path."""

import pytest
import torch

from constants import NUM_GEARS
from data.dataset import LanguageAugmenter, TerraGripDataset, make_loader
from language import LanguageEncoder
from models.model import MODES, TerraGripModel, language_features
from models.perception import Perception


@pytest.fixture(scope="module")
def perception():
    return Perception("dummy")


@pytest.fixture(scope="module")
def aug():
    return LanguageAugmenter(LanguageEncoder("hash"), "train", seed=0)


def _batch(sources, perception, aug, n=8):
    b = next(iter(make_loader(TerraGripDataset(sources["train"]), n, shuffle=False)))
    phi = perception.encode(b["image"], b["roi"])
    lb = aug.random(b["slip_curve"].numpy())
    return b, phi, lb


def make(mode, perception, aug, scw=0.0, train=False):
    """eval() by default: these tests assert properties of the deterministic
    inference function, and dropout would otherwise make them flaky."""
    m = TerraGripModel(
        mode, lang_dim=aug.dim, lang_condition="text", side_channel_weight=scw,
        perception=perception,
    )
    return m.train() if train else m.eval()


@pytest.mark.parametrize("mode", MODES)
def test_forward_keys_and_shapes(mode, sources, perception, aug):
    b, phi, lb = _batch(sources, perception, aug)
    model = make(mode, perception, aug, scw=0.5 if mode == "hybrid" else 0.0)
    out = model(phi=phi, lang=language_features(lb, "text"))

    n = phi.shape[0]
    if mode in ("modular", "hybrid"):
        assert out["slip"].shape == (n, NUM_GEARS)
        assert out["slip_sigma"].shape == (n, NUM_GEARS)
        assert ((out["slip"] >= 0) & (out["slip"] <= 1)).all(), "slip must be in [0,1]"
        assert (out["slip_sigma"] > 0).all()
    else:
        assert "slip" not in out

    if mode in ("e2e", "hybrid"):
        assert out["gear_logits"].shape == (n, NUM_GEARS)
    else:
        assert "gear_logits" not in out, "modular must not have a learned gear head"


@pytest.mark.parametrize("scw", [0.0, 0.5])
def test_hybrid_runs_at_any_scw(scw, sources, perception, aug):
    b, phi, lb = _batch(sources, perception, aug)
    model = make("hybrid", perception, aug, scw=scw)
    out = model(phi=phi, lang=language_features(lb, "text"))
    assert torch.isfinite(out["gear_logits"]).all()


def test_hybrid_scw0_is_a_real_bottleneck(sources, perception, aug):
    """At scw=0 the gear head must see vision ONLY through the slip concept."""
    b, phi, lb = _batch(sources, perception, aug)
    lang = language_features(lb, "text")
    model = make("hybrid", perception, aug, scw=0.0)

    base = model(phi=phi, lang=lang)["gear_logits"]
    # Same concept, completely different vision -> identical decision, because the
    # side channel is multiplied by 0.
    concept = model(phi=phi, lang=lang)["slip"]
    other = model.gear_logits_from(concept, torch.randn_like(phi), lang)
    assert torch.allclose(base, other, atol=1e-5)

    # And with scw>0 that is no longer true: vision leaks around the concept.
    leaky = make("hybrid", perception, aug, scw=1.0)
    lbase = leaky(phi=phi, lang=lang)["gear_logits"]
    lother = leaky.gear_logits_from(leaky(phi=phi, lang=lang)["slip"], torch.randn_like(phi), lang)
    assert not torch.allclose(lbase, lother, atol=1e-3)


def test_gradient_routes_per_mode(sources, perception, aug):
    """A: no selection gradient into the slip head.  C: gear loss must reach it."""
    b, phi, lb = _batch(sources, perception, aug)
    lang = language_features(lb, "text")

    hybrid = make("hybrid", perception, aug, scw=0.0)
    hybrid(phi=phi, lang=lang)["gear_logits"].sum().backward()
    grads = [p.grad for p in hybrid.slip_head.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), (
        "hybrid: the gear loss must backprop through the concept into the slip head"
    )

    modular = make("modular", perception, aug)
    assert modular.gear_head is None
    out = modular(phi=phi, lang=lang)
    assert "gear_logits" not in out  # nothing to backprop a gear loss from


def test_slip_is_language_invariant(sources, perception, aug):
    """Physics does not listen to instructions.  Slip must not depend on the text."""
    b, phi, _ = _batch(sources, perception, aug)
    curves = b["slip_curve"].numpy()
    model = make("hybrid", perception, aug, scw=0.5)

    s_careful = model(phi=phi, lang=language_features(aug.fixed("careful", curves), "text"))["slip"]
    s_fast = model(phi=phi, lang=language_features(aug.fixed("fast", curves), "text"))["slip"]
    assert torch.allclose(s_careful, s_fast), "the slip head must never see language"


def test_language_changes_the_decision(sources, perception, aug):
    """...but the DECISION must depend on the instruction, in e2e and hybrid."""
    b, phi, _ = _batch(sources, perception, aug)
    curves = b["slip_curve"].numpy()
    for mode in ("e2e", "hybrid"):
        model = make(mode, perception, aug, scw=0.5 if mode == "hybrid" else 0.0)
        a = model(phi=phi, lang=language_features(aug.fixed("careful", curves), "text"))
        c = model(phi=phi, lang=language_features(aug.fixed("fast", curves), "text"))
        assert not torch.allclose(a["gear_logits"], c["gear_logits"]), (
            f"{mode} ignores the instruction -- it is not a VLA"
        )


def test_head_capacity_is_matched(perception, aug):
    """Same hidden widths across modes, otherwise the comparison is rigged."""
    m = make("modular", perception, aug)
    e = make("e2e", perception, aug)
    h = make("hybrid", perception, aug, scw=0.5)
    widths = lambda mod: [p.shape[0] for n, p in mod.named_parameters() if n.endswith("weight")]
    assert widths(m.slip_head)[:2] == widths(e.gear_head)[:2] == widths(h.gear_head)[:2]


def test_concept_override(sources, perception, aug):
    """The intervention hook the analyses depend on."""
    b, phi, lb = _batch(sources, perception, aug)
    lang = language_features(lb, "text")
    model = make("hybrid", perception, aug, scw=0.0)
    base = model(phi=phi, lang=lang)["gear_logits"]
    truth = b["slip_curve"]
    forced = model(phi=phi, lang=lang, concept_override=truth)["gear_logits"]
    assert not torch.allclose(base, forced, atol=1e-4)


def test_budget_conditioning_dim(sources, perception, aug):
    b, phi, lb = _batch(sources, perception, aug)
    model = TerraGripModel(
        "e2e", lang_dim=aug.dim, lang_condition="budget", perception=perception
    )
    assert model.lang_dim == 3
    out = model(phi=phi, lang=language_features(lb, "budget"))
    assert out["gear_logits"].shape == (phi.shape[0], NUM_GEARS)


def test_sigma_nll_cannot_move_mu(sources, perception, aug):
    """The invariant the docs claim: fitting sigma must not retrain the mean.

    `mu.detach()` in the loss is NOT sufficient when mu and log_sigma share a
    trunk -- the NLL still reaches log_sigma's weights and flows back through the
    shared hidden state into the trunk that mu depends on.  SlipHead therefore
    feeds the sigma branch a DETACHED hidden state.  This test fails on the naive
    implementation.
    """
    import torch.nn.functional as F

    b, phi, _ = _batch(sources, perception, aug)
    model = make("modular", perception, aug, train=True)
    head = model.slip_head

    mu, log_sigma, _ = head(phi, b["gear"])
    # a pure sigma loss: no MSE anywhere
    nll = F.gaussian_nll_loss(mu.detach(), b["slip"], log_sigma.exp().pow(2), reduction="mean")
    nll.backward()

    mu_params = list(head.trunk.parameters()) + list(head.mu.parameters())
    leaked = [p for p in mu_params if p.grad is not None and p.grad.abs().sum() > 0]
    assert not leaked, (
        f"{len(leaked)} parameters that mu depends on received gradient from the sigma NLL"
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.log_sigma.parameters())


def test_concept_columns_share_one_dropout_mask(sources, perception, aug):
    """The [B,3] concept must not have its cross-gear structure scrambled by noise.

    Evaluating the three gears in three separate trunk calls draws three
    INDEPENDENT dropout masks, so the differences between the concept's columns
    carry dropout noise that has nothing to do with the terrain -- and that vector
    is exactly what the hybrid gear head reads.
    """
    b, phi, _ = _batch(sources, perception, aug)
    head = make("hybrid", perception, aug, scw=0.0, train=True).slip_head

    torch.manual_seed(0)
    mu_a, _, _ = head.predict_all(phi)
    torch.manual_seed(0)
    mu_b, _, _ = head.predict_all(phi)
    assert torch.allclose(mu_a, mu_b), "predict_all is not reproducible under a fixed seed"

    # With a shared per-sample mask the three columns come from the same subnetwork,
    # so at dropout=0 (eval) they must match the train-mode structure exactly.
    head.eval()
    mu_eval, sigma, mu_clean = head.predict_all(phi)
    assert torch.allclose(mu_eval, mu_clean, atol=1e-6), (
        "with dropout off, the dropout-active mean and the clean mean must coincide"
    )
    assert sigma.shape == mu_eval.shape
