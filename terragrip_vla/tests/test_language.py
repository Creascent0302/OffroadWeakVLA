"""M5: the language layer, and the semantic direction it must never violate."""

import numpy as np
import pytest

from language import (
    ANCHORS,
    INSTRUCTIONS,
    INTENTS,
    KEYWORDS,
    RISK_TABLE,
    LanguageEncoder,
    interpret,
    keyword_hits,
    match_intent,
)


def test_risk_table_is_monotone():
    """careful -> fast must relax every knob, or 'careful' would not be careful."""
    c, n, f = RISK_TABLE["careful"], RISK_TABLE["normal"], RISK_TABLE["fast"]
    assert c.alpha < n.alpha < f.alpha  # careful demands a wider (safer) bound
    assert c.tau < n.tau < f.tau  # careful tolerates less slip
    assert c.lam > n.lam >= f.lam  # careful cares more about slip than effort


def test_heldout_phrases_contain_no_keyword():
    """Otherwise the 'generalisation' eval would be solved by the keyword lookup."""
    for intent in INTENTS:
        for phrase in INSTRUCTIONS[intent]["heldout"]:
            assert not keyword_hits(phrase), f"held-out phrase {phrase!r} leaks a keyword"


def test_train_phrases_hit_exactly_one_intent():
    """No ambiguity: a training phrase must not match two intents' keywords."""
    for intent in INTENTS:
        for phrase in INSTRUCTIONS[intent]["train"]:
            assert keyword_hits(phrase) == {intent}, f"{phrase!r} -> {keyword_hits(phrase)}"


def test_train_and_heldout_are_disjoint():
    for intent in INTENTS:
        tr = set(INSTRUCTIONS[intent]["train"])
        ho = set(INSTRUCTIONS[intent]["heldout"])
        assert tr and ho and tr.isdisjoint(ho)


def test_keyword_path_resolves_training_phrases():
    """Every training phrase must be resolvable offline, without the encoder."""
    for intent in INTENTS:
        for phrase in INSTRUCTIONS[intent]["train"]:
            assert match_intent(phrase, backend="hash") == intent, phrase


def test_interpret_returns_the_right_budget():
    assert interpret("be careful", backend="hash") is RISK_TABLE["careful"]
    assert interpret("hurry up", backend="hash") is RISK_TABLE["fast"]
    assert interpret("proceed normally", backend="hash") is RISK_TABLE["normal"]


def test_encoder_is_deterministic_and_cached():
    enc = LanguageEncoder("hash")
    a = enc.encode(["be careful", "go fast"])
    b = enc.encode(["be careful", "go fast"])
    assert a.shape == (2, enc.dim)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


@pytest.mark.slow
def test_minilm_resolves_heldout_paraphrases():
    """The semantic path: no keywords, so this can only work through MiniLM.

    This is the ability modular gets for free and e2e has to learn.
    """
    try:
        LanguageEncoder("minilm").encode(["warmup"])
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"MiniLM unavailable: {exc}")

    correct = 0
    total = 0
    for intent in INTENTS:
        for phrase in INSTRUCTIONS[intent]["heldout"]:
            total += 1
            correct += int(match_intent(phrase, backend="minilm") == intent)
    assert correct / total >= 0.6, f"MiniLM resolved only {correct}/{total} held-out phrases"


@pytest.mark.slow
def test_anchors_are_semantically_separated():
    try:
        enc = LanguageEncoder("minilm")
        v = enc.encode([ANCHORS[i] for i in INTENTS])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MiniLM unavailable: {exc}")
    sim = v @ v.T
    off = sim[~np.eye(len(INTENTS), dtype=bool)]
    assert off.max() < 0.95, "the intent anchors are nearly identical sentences"
