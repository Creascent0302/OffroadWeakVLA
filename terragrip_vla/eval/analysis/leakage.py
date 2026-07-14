"""Leakage: does the hybrid decision really go THROUGH the slip concept?

A concept bottleneck is only a bottleneck if the decision cannot get the
information any other way.  `side_channel_weight` (scw) is the knob that opens a
path around it, and this file measures how much traffic takes that path.

Three quantities, all measured on the gear decision:

    acc_full        accuracy with the predicted concept and the side channel
    acc_shuffled    accuracy with the concept SHUFFLED across the batch, side
                    channel untouched.  If the decision survives this, it was not
                    using the concept.
    acc_lang_only   the floor.  Not 1/3!  Knowing only the instruction already
                    tells you the marginal best gear ("careful" -> L is common),
                    so a shuffled-concept model still beats chance.  Dividing by
                    the wrong floor is the easiest way to fake a leakage curve.

        leakage = (acc_shuffled - acc_lang_only) / (acc_full - acc_lang_only)

    0 -> the concept is load-bearing (a true bottleneck)
    1 -> the concept is decorative; vision reaches the decision around it

And the dual measurement:

    intervention_gain = acc(true concept) - acc_full

    How much better the decision gets when the concept is CORRECTED.  A model
    that ignores its concept cannot be helped by fixing it, so a high leakage
    must come with a low intervention gain.  The two are independent estimates of
    the same thing, which is why we report both.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from eval.metrics import EvalData
from eval.plotting import set_fig_dir, SLOT, save, setup
from eval.policies import run_model
from language import INTENTS, RISK_TABLE
from runtime import ROOT, build_context, checkpoint_path, load_cfg, load_model


def language_only_accuracy(data: EvalData) -> float:
    """Best possible accuracy using the instruction and nothing else.

    For each instruction, predict the most common oracle gear over the split.
    This is the floor any language-conditioned model clears without looking.
    """
    accs = []
    for intent in INTENTS:
        oracle = data.oracle(RISK_TABLE[intent])
        majority = np.bincount(oracle, minlength=3).argmax()
        accs.append(float((oracle == majority).mean()))
    return float(np.mean(accs))


def _acc(model, ctx, data, aug, split, concept: str, rng=None) -> float:
    """Mean gear accuracy over instructions, with the concept doctored as asked."""
    phi = ctx.phi[split]
    curves = ctx.sources[split].slip_curve
    accs = []

    for intent in INTENTS:
        budget = RISK_TABLE[intent]
        lang = aug.fixed(intent, curves)
        oracle = data.oracle(budget)

        override = None
        if concept == "true":
            override = data.slip_curve.astype(np.float32)
        elif concept == "shuffled":
            base = run_model(model, phi, lang, ctx.device).slip_mu
            override = base[rng.permutation(len(base))]
        elif concept != "predicted":
            raise ValueError(concept)

        out = run_model(model, phi, lang, ctx.device, concept_override=override)
        accs.append(float((out.probs.argmax(1) == oracle).mean()))

    return float(np.mean(accs))


def leakage_for(model, ctx, aug, split: str = "test", seed: int = 0) -> dict:
    data = EvalData.from_source(ctx.sources[split])
    rng = np.random.default_rng(seed)

    floor = language_only_accuracy(data)
    full = _acc(model, ctx, data, aug, split, "predicted")
    shuf = _acc(model, ctx, data, aug, split, "shuffled", rng)
    true = _acc(model, ctx, data, aug, split, "true")

    # headroom = how much of the decision is driven by VISION at all (through any
    # path). Leakage is the share of that which survives destroying the concept.
    #
    # If headroom is ~0 the model is answering from the instruction alone: there is
    # no vision-derived performance, so "how much of it bypasses the concept" is
    # 0/0, not 1. Reporting 1.0 here (as the first version did) would paint a model
    # that ignores vision entirely as maximally leaky.
    headroom = full - floor
    degenerate = headroom <= 1e-6
    leakage = float("nan") if degenerate else float(np.clip((shuf - floor) / headroom, 0.0, 1.0))

    # shuf < floor is legitimate and informative: destroying the concept leaves the
    # head WORSE than not looking at all, i.e. it genuinely depends on the concept.
    return {
        "scw": float(model.scw),
        "split": split,
        "acc_lang_only": floor,
        "acc_full": full,
        "acc_shuffled": shuf,
        "acc_true_concept": true,
        "headroom": float(headroom),
        "leakage": leakage,
        "leakage_undefined": bool(degenerate),
        "intervention_gain": float(true - full),
    }


# --------------------------------------------------------------------------
def figure_leakage(rows: list[dict], ood_rows: list[dict] | None = None) -> str:
    """The scw axis is CATEGORICAL, not linear.

    The sweep is log-spaced ({0, .003, .01, .03, .1, .3, 1}) because that is where
    the transition lives -- leakage is 0.00 at scw=0 and already 0.75 at scw=0.1.
    Plotting those on a linear axis crushes every interesting point into the left
    margin. Evenly spaced categories put the transition where it can be read.
    """
    setup()
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["scw"])
    x = np.arange(len(rows))
    ticks = [("0" if r["scw"] == 0 else f"{r['scw']:g}") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # (a) leakage -----------------------------------------------------------
    ax = axes[0]
    leak = [r["leakage"] for r in rows]
    ax.plot(x, leak, "-o", color=SLOT["hybrid"])
    ax.set_ylim(-0.08, 1.15)
    ax.set_xticks(x, ticks)
    ax.set_xlabel("side_channel_weight  (log-spaced, shown evenly)")
    ax.set_ylabel("leakage")
    ax.set_title("(a) opening the side channel leaks")
    for xi, r in zip(x, rows):
        v = r["leakage"]
        ax.annotate("n/a" if np.isnan(v) else f"{v:.2f}", (xi, 0 if np.isnan(v) else v),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#52514e")
    ax.annotate("0 = concept is load-bearing\n1 = vision goes around it",
                xy=(0.03, 0.80), xycoords="axes fraction", fontsize=7.5, color="#a8a7a1")

    # (b) accuracy: all three series are accuracies, so one axis is legal ----
    ax = axes[1]
    ax.plot(x, [r["acc_true_concept"] for r in rows], "-s", color=SLOT["oracle"],
            label="concept corrected")
    ax.plot(x, [r["acc_full"] for r in rows], "-o", color=SLOT["hybrid"], label="as trained")
    ax.plot(x, [r["acc_lang_only"] for r in rows], "--", color="#a8a7a1",
            label="instruction only (floor)")
    ax.set_xticks(x, ticks)
    ax.set_xlabel("side_channel_weight")
    ax.set_ylabel("gear accuracy")
    # NOT "the gap intervention closes": at scw=0 the corrected concept scores BELOW
    # the trained model. The learned head calibrated itself to its own predictor's
    # biases, so the TRUE slip curve is out of distribution *for the head*. That is
    # a real property of a learned bottleneck, and it is an argument for modular's
    # analytic selector, which cannot be miscalibrated this way.
    ax.set_title("(b) a learned head trusts its OWN concept, not the true one")
    ax.legend(loc="lower right")

    d0 = rows[0]["acc_true_concept"] - rows[0]["acc_full"]
    if d0 < -0.01:
        ax.annotate(
            f"at scw=0, feeding the TRUE slip curve\nmakes it {abs(d0):.2f} WORSE",
            xy=(0.30, 0.10), xycoords="axes fraction", fontsize=7.5, color=SLOT["e2e"],
        )

    # (c) the trade-off, on the terrain where it actually costs you ----------
    ax = axes[2]
    if ood_rows:
        o = sorted(ood_rows, key=lambda r: r["scw"])
        ax.plot(leak, [r["acc_full"] for r in o], "-s", color=SLOT["e2e"], label="OOD terrain")
    ax.plot(leak, [r["acc_full"] for r in rows], "-o", color=SLOT["hybrid"],
            label="in-distribution")

    # Label only the ends and the knee: labelling all 7 collides at leakage 0 and 1.
    keep = {0, len(rows) - 1}
    keep |= {min(range(len(rows)), key=lambda i: abs(rows[i]["leakage"] - 0.5))}
    for i in sorted(keep):
        v = rows[i]["leakage"]
        if np.isnan(v):
            continue
        ax.annotate(f"scw={ticks[i]}", (v, rows[i]["acc_full"]),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#52514e")
    ax.set_xlim(-0.08, 1.12)
    ax.set_xlabel("leakage")
    ax.set_ylabel("gear accuracy")
    ax.set_title("(c) interpretability / performance trade-off")
    ax.legend(loc="center left")

    return str(save(fig, "A1_leakage"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--train-missing", action="store_true",
                    help="train any hybrid checkpoint the sweep needs but cannot find")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    ctx = build_context(cfg)
    set_fig_dir(ROOT / cfg.artifacts_dir / "figures")
    aug = ctx.augmenter(str(cfg.language.paraphrase_split), seed=0)
    seed = int(cfg.train.seed)

    rows, ood_rows = [], []
    for scw in [float(s) for s in cfg.eval.scw_sweep]:
        path = checkpoint_path(cfg, "hybrid", scw, seed)
        if not path.exists():
            if not args.train_missing:
                print(f"[skip] missing {path.name} (rerun with --train-missing)")
                continue
            from training.train import run_training

            sub = load_cfg(args.config, list(args.overrides) + ["mode=hybrid",
                                                                f"side_channel_weight={scw}"])
            run_training(sub, ctx=ctx, progress=False)

        model = load_model(path, ctx)
        rows.append(leakage_for(model, ctx, aug, "test", seed))
        ood_rows.append(leakage_for(model, ctx, aug, "ood", seed))
        print(json.dumps(rows[-1], indent=2))

    if not rows:
        print("nothing to plot")
        return

    out = ROOT / cfg.artifacts_dir / f"leakage_seed{seed}.json"
    out.write_text(json.dumps({"test": rows, "ood": ood_rows}, indent=2))
    print(f"\nwrote {out}\nwrote {figure_leakage(rows, ood_rows)}")


if __name__ == "__main__":
    main()
