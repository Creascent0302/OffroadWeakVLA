"""M4: the conformal guarantee, tested as a guarantee.

Both routes must empirically cover at their nominal rate:
    regression (A/C) : P(true slip <= upper)      >= 1 - alpha
    classification(B): P(best_gear in APS set)    >= 1 - alpha

The synthetic generator below deliberately mirrors the mock data: a predictor
that is right on average, with terrain-dependent noise.  Testing against that
rather than a trained checkpoint keeps the test fast, deterministic, and about
the MATH rather than about how well a particular net fitted.
"""

import numpy as np
import pytest

from conformal.aps import GearConformal, aps_calibrate, aps_coverage, aps_score, aps_set
from conformal.split_conformal import SlipConformal, empirical_coverage, quantile

ALPHAS = [0.05, 0.10, 0.20]
SIGMAS = {"concrete": 0.02, "grass": 0.05, "mud": 0.15, "sand": 0.10}
MEANS = {"concrete": 0.05, "grass": 0.12, "mud": 0.32, "sand": 0.24}
TERRAINS = list(SIGMAS)


def _draw(n, rng):
    """(mu_hat, sigma_hat, y) from a calibrated-but-noisy predictor."""
    t = rng.choice(TERRAINS, n)
    mu = np.array([MEANS[x] for x in t])
    sigma = np.array([SIGMAS[x] for x in t])
    y = np.clip(mu + rng.normal(0, sigma), 0, 1)
    # The head is imperfect: its mean is slightly off and its sigma slightly wrong.
    mu_hat = mu + rng.normal(0, 0.01, n)
    sigma_hat = sigma * rng.uniform(0.85, 1.15, n)
    return mu_hat, sigma_hat, y, t


@pytest.mark.parametrize("score", ["absolute", "normalized"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_regression_coverage(score, alpha):
    """Averaged over draws, coverage must not fall below the nominal level."""
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(20):
        mu_c, sg_c, y_c, _ = _draw(1000, rng)
        mu_t, sg_t, y_t, _ = _draw(1000, rng)

        cal = SlipConformal(score=score).fit(mu_c, y_c, [alpha], sigma=sg_c)
        # (N,) -> (N,1) so we can reuse the (B,3) upper() API on a single column.
        upper = cal.upper(mu_t[:, None], alpha, sigma=sg_t[:, None])[:, 0]
        covs.append(empirical_coverage(y_t, upper))

    covs = np.array(covs)
    assert covs.mean() >= 1 - alpha - 0.01, f"mean coverage {covs.mean():.3f} < {1-alpha}"
    assert covs.min() >= 1 - alpha - 0.05, f"worst-case coverage {covs.min():.3f} too low"


def test_normalized_score_is_what_saves_modular():
    """The reason `normalized` is the default, as an executable claim.

    With `absolute`, one global quantile must cover mud (sigma=0.15), so the bound
    on concrete (sigma=0.02) is far too wide, the acceptable set under a careful
    instruction (tau=0.15) is EMPTY, and modular degenerates to "always gear L" --
    even though the predictor here is essentially perfect.
    """
    rng = np.random.default_rng(1)
    mu_c, sg_c, y_c, _ = _draw(4000, rng)
    tau_careful, alpha = 0.15, 0.05

    concrete_mu = np.array([[MEANS["concrete"]]])
    concrete_sg = np.array([[SIGMAS["concrete"]]])

    absolute = SlipConformal("absolute").fit(mu_c, y_c, [alpha])
    normalized = SlipConformal("normalized").fit(mu_c, y_c, [alpha], sigma=sg_c)

    up_abs = absolute.upper(concrete_mu, alpha)[0, 0]
    up_nrm = normalized.upper(concrete_mu, alpha, sigma=concrete_sg)[0, 0]

    assert up_abs > tau_careful, "the absolute-score collapse is the thing we claim"
    assert up_nrm < tau_careful, "the normalized score must keep concrete certifiable"


def test_quantile_finite_sample_correction():
    s = np.arange(1, 11, dtype=float)  # n = 10
    # k = ceil(11 * 0.9) = 10 -> the 10th smallest
    assert quantile(s, 0.10) == 10.0
    # k = ceil(11 * 0.8) = 9
    assert quantile(s, 0.20) == 9.0
    # alpha too small for n: cannot certify -> vacuous, not overconfident
    assert quantile(s, 0.01) == np.inf
    assert quantile(np.array([]), 0.1) == np.inf


def test_per_gear_calibration():
    rng = np.random.default_rng(2)
    mu, sg, y, _ = _draw(3000, rng)
    gear = rng.integers(0, 3, len(y))
    cal = SlipConformal("normalized", per_gear=True).fit(mu, y, [0.1], sigma=sg, gear=gear)
    assert cal.q(0.1).shape == (3,)


def test_conformal_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    mu, sg, y, _ = _draw(500, rng)
    cal = SlipConformal("normalized").fit(mu, y, ALPHAS, sigma=sg)
    path = cal.save(tmp_path / "conformal_Q.json")
    back = SlipConformal.load(path)
    assert back.score == cal.score
    for a in ALPHAS:
        assert np.allclose(back.q(a), cal.q(a))


# --------------------------------------------------------------------------
# APS (classification path)
# --------------------------------------------------------------------------
def _draw_probs(n, rng, sharpness=2.0):
    true = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), true] += sharpness  # a decent but fallible classifier
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), true


@pytest.mark.parametrize("randomized", [False, True])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_aps_coverage(alpha, randomized):
    """Both APS variants must cover. The randomised one covers TIGHTLY."""
    rng = np.random.default_rng(0)
    covs = []
    for i in range(20):
        pc, tc = _draw_probs(1000, rng)
        pt, tt = _draw_probs(1000, rng)
        cal = GearConformal(randomized=randomized, seed=i).fit(pc, tc, [alpha])
        covs.append(cal.coverage(pt, tt, alpha))
    covs = np.array(covs)
    assert covs.mean() >= 1 - alpha - 0.01, f"mean APS coverage {covs.mean():.3f}"
    assert covs.min() >= 1 - alpha - 0.05


def test_randomized_aps_is_much_tighter_on_a_confident_classifier():
    """Why randomised is the default.

    A confident classifier makes every deterministic calibration score ~= p_top,
    so q lands near 1.0 and every test set needs 2-3 classes to reach it. The sets
    become useless and the downstream policy is forced to be maximally
    conservative for no statistical reason.
    """
    rng = np.random.default_rng(0)
    pc, tc = _draw_probs(2000, rng, sharpness=6.0)  # very confident, ~always right
    pt, tt = _draw_probs(2000, rng, sharpness=6.0)
    alpha = 0.05

    det = GearConformal(randomized=False, seed=0).fit(pc, tc, [alpha])
    rnd = GearConformal(randomized=True, seed=0).fit(pc, tc, [alpha])

    size_det = np.mean([len(s) for s in det.sets(pt, alpha)])
    size_rnd = np.mean([len(s) for s in rnd.sets(pt, alpha)])

    assert size_det > 1.5, "the deterministic pathology should reproduce"
    assert size_rnd < 1.2, "randomised APS should give near-singleton sets"
    assert det.coverage(pt, tt, alpha) >= 1 - alpha - 0.02
    assert rnd.coverage(pt, tt, alpha) >= 1 - alpha - 0.03


def test_deterministic_aps_set_grows_with_the_threshold():
    """S = { y : E(y) <= q }, where E(y) is the cumulative mass up to and INCLUDING y.

    So for p = [0.8, 0.15, 0.05] the scores are E = [0.80, 0.95, 1.00].  The set may
    be empty -- that is the correct conformal set, not a bug: a threshold below the
    top class's own mass certifies nothing, and the selector reads it as an
    abstention.
    """
    p = np.array([0.8, 0.15, 0.05])
    assert aps_set(p, 0.5) == []  # certifies nothing
    assert aps_set(p, 0.8) == [0]
    assert aps_set(p, 0.9) == [0]
    assert aps_set(p, 0.95) == [0, 1]
    assert aps_set(p, 1.0) == [0, 1, 2]


def test_aps_score_definition():
    p = np.array([0.6, 0.3, 0.1])
    assert aps_score(p, 0) == pytest.approx(0.6)
    assert aps_score(p, 1) == pytest.approx(0.9)
    assert aps_score(p, 2) == pytest.approx(1.0)


def test_gear_conformal_wrapper():
    rng = np.random.default_rng(4)
    pc, tc = _draw_probs(800, rng)
    cal = GearConformal().fit(pc, tc, ALPHAS)
    assert cal.q(0.05) >= cal.q(0.20)  # a stricter alpha needs a bigger threshold
    sets = cal.sets(pc[:10], 0.1)
    assert all(len(s) >= 1 for s in sets)
