"""M4: the selector, including the semantic direction language must obey."""

import numpy as np
import pytest

from constants import CONTACT_AREA, FALLBACK_GEAR
from conformal.aps import GearConformal, aps_set
from conformal.select import select_e2e, select_modular, select_point
from data.mock_generator import TERRAIN_SLIP
from language import RISK_TABLE, RiskBudget

CAREFUL, NORMAL, FAST = RISK_TABLE["careful"], RISK_TABLE["normal"], RISK_TABLE["fast"]


def test_empty_acceptable_set_falls_back_and_flags():
    upper = np.array([[0.9, 0.8, 0.7]])  # nothing fits a tau of 0.15
    sel = select_modular(upper, CAREFUL)
    assert sel.gear[0] == FALLBACK_GEAR
    assert bool(sel.low_conf[0]) is True


def test_certified_gear_is_the_cheapest_acceptable_one():
    upper = np.array([[0.10, 0.08, 0.05]])  # all three fit tau=0.15
    sel = select_modular(upper, CAREFUL, slip_point=upper)
    assert sel.gear[0] == 0  # smallest contact area
    assert not sel.low_conf[0]


def test_lambda_trades_slip_against_effort():
    point = np.array([[0.30, 0.05, 0.04]])
    loose = RiskBudget(alpha=0.2, tau=0.35, lam=0.0)  # ignore slip -> cheapest gear
    greedy = RiskBudget(alpha=0.2, tau=0.35, lam=10.0)  # slip dominates -> bigger gear
    assert select_modular(point, loose, point).gear[0] == 0
    assert select_modular(point, greedy, point).gear[0] == 1


def test_language_direction_modular():
    """careful -> conservative -> LARGER gear.  fast -> efficient -> SMALLER gear."""
    upper = np.array([TERRAIN_SLIP[t] for t in ("concrete", "grass", "mud", "sand")])
    area = lambda sel: np.mean([CONTACT_AREA[g] for g in sel.gear])

    a_careful = area(select_modular(upper, CAREFUL, upper))
    a_normal = area(select_modular(upper, NORMAL, upper))
    a_fast = area(select_modular(upper, FAST, upper))

    assert a_careful >= a_normal >= a_fast
    assert a_careful > a_fast, "the instruction must actually move the decision"


def test_stricter_tau_never_picks_a_smaller_gear():
    """Monotonicity: tightening the slip budget can only make the choice safer."""
    rng = np.random.default_rng(0)
    upper = np.sort(rng.uniform(0, 0.6, (200, 3)))[:, ::-1].copy()  # slip falls with gear
    prev = None
    for tau in (0.05, 0.15, 0.25, 0.35, 0.5):
        b = RiskBudget(alpha=0.1, tau=tau, lam=0.0)
        g = select_modular(upper, b, upper).gear
        if prev is not None:
            areas = np.array([CONTACT_AREA[x] for x in g])
            prev_areas = np.array([CONTACT_AREA[x] for x in prev])
            assert (areas <= prev_areas).all(), "a looser tau must never get MORE conservative"
        prev = g


# --------------------------------------------------------------------------
# e2e path
# --------------------------------------------------------------------------
def test_singleton_set_is_confident():
    sel = select_e2e([[0]])
    assert sel.gear[0] == 0 and not sel.low_conf[0]


def test_ambiguous_set_resolves_conservatively_by_default():
    safe = select_e2e([[0, 1]], ambiguity="safe")
    eff = select_e2e([[0, 1]], ambiguity="efficient")
    assert safe.gear[0] == 1 and eff.gear[0] == 0  # the largest vs the cheapest
    assert bool(safe.low_conf[0]) and bool(eff.low_conf[0])


def test_empty_set_is_an_abstention():
    """Randomised APS may return nothing. That is a refusal, not a gear."""
    sel = select_e2e([[]])
    assert sel.gear[0] == FALLBACK_GEAR and bool(sel.low_conf[0])


def test_safe_resolution_gives_e2e_a_traction_guarantee():
    """If APS covers the true best gear, `safe` never picks a gear that slips more.

    chosen = max(S) >= best  =>  slip(chosen) <= slip(best) <= tau,
    because slip decreases with contact area.  This is what makes B's guarantee
    the same SHAPE as A's, and hence the comparison fair.
    """
    rng = np.random.default_rng(1)
    n, alpha = 2000, 0.10
    true = rng.integers(0, 3, n)
    logits = rng.normal(0, 1.0, (n, 3))
    logits[np.arange(n), true] += 1.5
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)

    cal = GearConformal(randomized=True, seed=0).fit(p[:1000], true[:1000], [alpha])
    sets = cal.sets(p[1000:], alpha)
    sel = select_e2e(sets, ambiguity="safe")
    t = true[1000:]

    covered = np.array([t[i] in s for i, s in enumerate(sets)])
    assert covered.mean() >= 1 - alpha - 0.03
    # Wherever the set covered the truth, the chosen gear is at least as safe.
    assert (sel.gear[covered] >= t[covered]).all()


def test_point_baseline_has_no_fallback():
    """VisionOnlyNoConformal must never raise low_conf -- that is the whole point."""
    point = np.array([[0.9, 0.8, 0.7], [0.10, 0.08, 0.05]])
    sel = select_point(point, CAREFUL)
    assert not sel.low_conf.any()
    assert sel.gear[0] == 2  # nothing fits -> least-slipping gear, but uncertified
    assert sel.gear[1] == 0


def test_shapes_and_dtypes():
    upper = np.random.default_rng(0).uniform(0, 0.5, (17, 3))
    sel = select_modular(upper, NORMAL, upper)
    assert sel.gear.shape == (17,) and sel.gear.dtype == np.int64
    assert sel.low_conf.shape == (17,) and sel.low_conf.dtype == bool
    assert set(np.unique(sel.gear)) <= {0, 1, 2}


# --------------------------------------------------------------------------
# Invariants that the adversarial review showed were silently broken
# --------------------------------------------------------------------------
def test_lambda_actually_bites_on_the_real_risk_table():
    """lambda must change at least one real decision, or it is a decorative knob.

    With the raw cost `area + lam*slip` it changed NOTHING at any value in
    RISK_TABLE: the contact-area gap between gears is 0.5, while an acceptable gear
    has slip <= tau <= 0.35, so lam*slip could never span the gap.  gear_cost
    normalises slip by tau, which makes the two terms commensurate.
    """
    from data.labels import best_gear

    changed = sum(
        best_gear(TERRAIN_SLIP[t], b.tau, b.lam) != best_gear(TERRAIN_SLIP[t], b.tau, 0.0)
        for t in TERRAIN_SLIP
        for b in RISK_TABLE.values()
    )
    assert changed >= 1, "lambda changes no decision anywhere -- it is a dead knob"


def test_empty_aps_set_does_not_force_the_max_effort_gear():
    """An empty randomised-APS set is the coin flip u > q, not evidence of danger.

    Resolving it to gear L would make alpha act on e2e with the OPPOSITE sign to
    modular (bigger alpha -> more abstentions -> more effort, while modular gets
    cheaper), which corrupts every effort and language-direction comparison.
    """
    probs = np.array([[0.9, 0.07, 0.03]])
    sel = select_e2e([[]], probs=probs)
    assert sel.gear[0] == 0, "an abstention must fall back to the point estimate, not to L"
    assert bool(sel.low_conf[0]) and bool(sel.info["abstained"][0])
