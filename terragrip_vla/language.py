"""Language layer: instruction -> risk budget, and instruction -> embedding.

Design principle (this is the heart of how language enters a VLA whose training
labels contain no language):

    Physics is language-invariant.  How much a track slips on mud in gear L is a
    fact about the world; saying "be careful" does not change it.  So the slip
    head NEVER sees language.

    Preference is language-dependent.  Which gear you *should* pick given the
    predicted slip depends on how much risk you are willing to take.  So language
    enters at the decision, as a risk budget (alpha, tau, lambda):

        alpha  : conformal miss rate.   smaller = wider slip upper bound = safer
        tau    : slip budget.           smaller = stricter = safer
        lambda : efficiency weight on slip when several gears are acceptable

`modular` consumes the budget analytically through the conformal selector.
`e2e` / `hybrid` consume the raw instruction embedding and must LEARN the same
mapping.  Both receive the identical input (image, instruction), which is what
makes the three-way comparison fair.
"""

# ============================================================================
# 【中文导读】语言层 —— 这是“训练数据里没有语言，怎么做 VLA”的答案所在。
#
#   核心原则：物理与语言无关，偏好与语言有关。
#     · 泥地在 L 档打滑多少，是客观事实。你说“小心点”并不会让它少滑。
#       ⇒ slip 头永远看不到语言。
#     · 但“该选哪个档”取决于你愿意承担多少风险。
#       ⇒ 语言在【决策处】进入，形式是风险预算 (alpha, tau, lambda)：
#           alpha  保形失效率。越小 → 打滑上界越宽 → 越保守
#           tau    可容忍的打滑上限。越小 → 越严格 → 越保守
#           lambda 在“都可接受”的档位之间，用多大权重拿打滑换省力
#
#   三种模式都吃同一份输入 (图像, 指令)，只是【路径】不同：
#     modular  → interpret() 把文本解析成预算，交给保形选择器【解析式】使用
#     e2e/hybrid → 吃冻结 MiniLM 的句向量，必须自己【学】出同一个映射
#
#   指令池分 train / heldout 两半。heldout 里的句子刻意不含任何关键词，
#   只能靠语义最近邻解析 —— 这就是“指令泛化”实验的立足点。
# ============================================================================


from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Risk budgets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskBudget:
    """(alpha, tau, lambda) triple decoded from an instruction."""

    alpha: float
    tau: float
    lam: float

    def vector(self) -> np.ndarray:
        return np.array([self.alpha, self.tau, self.lam], dtype=np.float32)


# careful -> conservative -> larger gear ; fast -> efficient -> smaller gear.
# Monotone by construction: alpha and tau increase, lambda decreases, as we move
# from careful to fast.  `test_language.py` asserts this.
# lambda is now on the normalised scale (see data/labels.gear_cost): the slip term
# is lam * slip/tau, in [0, 1] for any acceptable gear, so it is commensurate with
# the 0.5 contact-area gap between gears. On the old raw scale lambda could not
# change a single decision at any value in this table.
RISK_TABLE: dict[str, RiskBudget] = {
    "careful": RiskBudget(alpha=0.05, tau=0.15, lam=3.0),
    "normal": RiskBudget(alpha=0.10, tau=0.25, lam=0.5),
    "fast": RiskBudget(alpha=0.20, tau=0.35, lam=0.0),
}

INTENTS: list[str] = list(RISK_TABLE)

# --------------------------------------------------------------------------
# Instruction pool
# --------------------------------------------------------------------------
# `train` paraphrases are shown during training.  `heldout` paraphrases are used
# only at evaluation, to test instruction generalisation.  Held-out phrases
# deliberately contain NO keyword from KEYWORDS, so they can only be resolved by
# the semantic nearest-neighbour path.  test_language.py enforces that.

INSTRUCTIONS: dict[str, dict[str, list[str]]] = {
    "careful": {
        "train": [
            "be careful",
            "drive cautiously",
            "go slow, the cargo is fragile",
            "prioritize safety above all else",
        ],
        "heldout": [
            "there is delicate equipment on board",
            "we absolutely cannot afford to get stuck here",
            "treat this stretch like a minefield",
        ],
    },
    "normal": {
        "train": [
            "proceed normally",
            "standard driving",
            "keep a normal pace",
            "a regular traverse, nothing unusual",
        ],
        "heldout": [
            "just do what you always do",
            "no particular constraints on this run",
            "an ordinary stretch, nothing to note",
        ],
    },
    "fast": {
        "train": [
            "go fast",
            "hurry up",
            "be efficient",
            "save energy",
        ],
        "heldout": [
            "we are running out of battery",
            "time is critical, do not waste a second",
            "get there as soon as you possibly can",
        ],
    },
}

# Keyword fast path for `interpret`.  Kept deliberately tight, so that:
#   * every TRAIN phrase hits exactly one intent (offline, no encoder needed);
#   * every HELD-OUT phrase hits none, forcing the semantic path.
# test_language.py enforces both, which is what stops the "instruction
# generalisation" experiment from being silently solved by a substring match.
KEYWORDS: dict[str, list[str]] = {
    "careful": ["careful", "caution", "cautious", "slow", "safe", "safety", "fragile", "gentle"],
    "normal": ["normal", "standard", "routine", "regular"],
    "fast": ["fast", "hurry", "quick", "efficient", "efficiency", "energy", "speed", "rush"],
}


def keyword_hits(text: str) -> set[str]:
    """Which intents' keywords appear in `text`.  Empty -> the semantic path runs."""
    low = text.lower()
    return {i for i, words in KEYWORDS.items() if any(w in low for w in words)}

# Canonical phrase used as the semantic anchor of each intent for NN matching.
ANCHORS: dict[str, str] = {
    "careful": "drive carefully and safely, traction matters more than speed",
    "normal": "drive normally with a balanced trade-off",
    "fast": "drive fast and efficiently, save energy",
}


def instruction_pool(intent: str, split: str) -> list[str]:
    """Paraphrases of `intent`. split is 'train' or 'heldout'."""
    return list(INSTRUCTIONS[intent][split])


def all_instructions(split: str) -> list[tuple[str, str]]:
    """[(instruction_text, intent), ...] for the given paraphrase split."""
    return [(text, intent) for intent in INTENTS for text in instruction_pool(intent, split)]


def budget_of(intent: str) -> RiskBudget:
    return RISK_TABLE[intent]


# --------------------------------------------------------------------------
# Frozen sentence encoder
# --------------------------------------------------------------------------

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384
HASH_DIM = 384


class LanguageEncoder:
    """Frozen sentence encoder.  Never trained.

    backend='minilm' : real semantics (needed for held-out paraphrase tests).
    backend='hash'   : deterministic non-semantic embedding.  Offline/CI only;
                       it cannot generalise across paraphrases, by construction.
    """

    def __init__(self, backend: str = "minilm"):
        if backend not in ("minilm", "hash"):
            raise ValueError(f"unknown language backend: {backend}")
        self.backend = backend
        self.dim = MINILM_DIM if backend == "minilm" else HASH_DIM
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _lazy_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MINILM_NAME)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    @staticmethod
    def _hash_embed(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**32)
        v = np.random.default_rng(seed).normal(size=HASH_DIM).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, texts: list[str]) -> np.ndarray:
        """[N, dim] float32, L2-normalised.  Cached per unique string."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if self.backend == "hash":
                new = np.stack([self._hash_embed(t) for t in missing])
            else:
                new = self._lazy_model().encode(
                    missing, convert_to_numpy=True, normalize_embeddings=True
                )
            for t, v in zip(missing, np.asarray(new, dtype=np.float32)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])


@functools.lru_cache(maxsize=4)
def get_encoder(backend: str = "minilm") -> LanguageEncoder:
    return LanguageEncoder(backend)


# --------------------------------------------------------------------------
# interpret: instruction -> risk budget
# --------------------------------------------------------------------------


def match_intent(instruction: str, backend: str = "minilm") -> str:
    """Resolve an instruction to one of INTENTS.

    1. keyword hit (cheap, offline);
    2. otherwise nearest neighbour of the frozen sentence embedding against the
       canonical anchor of each intent.
    """
    hits = keyword_hits(instruction)
    if len(hits) == 1:
        return hits.pop()
    # 0 hits (a novel paraphrase) or >1 hits (an ambiguous one) -> ask the encoder.

    enc = get_encoder(backend)
    anchors = [ANCHORS[i] for i in INTENTS]
    vecs = enc.encode(anchors + [instruction])
    sims = vecs[:-1] @ vecs[-1]
    return INTENTS[int(np.argmax(sims))]


def interpret(instruction: str, backend: str = "minilm") -> RiskBudget:
    """instruction -> (alpha, tau, lambda).  This is the whole 'L' of modular."""
    return RISK_TABLE[match_intent(instruction, backend)]
