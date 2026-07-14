"""Test-time intervention: can you FIX the policy by correcting its concept?

This is the practical payoff of a concept bottleneck, and it is a capability e2e
structurally does not have -- there is nothing inside it to correct.

Two interventions, both at test time, no retraining:

    oracle          replace the whole predicted slip curve with the true one.
                    Upper bound: how much is the decision losing to a wrong
                    concept?

    proprioceptive  the vehicle has actually DRIVEN one gear and measured its
                    slip. Overwrite only that entry of the concept, and set its
                    uncertainty to ~0 since it is now observed rather than
                    predicted. This is the realistic one -- it is exactly the
                    signal a real robot gets for free, every step.

Reported on the in-distribution test split AND on the OOD terrain, because the
interesting claim is that the concept keeps being a valid handle even where the
predictor has degraded -- and that opening the side channel destroys that.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from conformal.select import select_e2e, select_modular
from eval.metrics import EvalData, score_selection
from eval.plotting import set_fig_dir, SLOT, save, setup
from eval.policies import run_model
from eval.run_eval import calibrate_gear, calibrate_slip
from language import INTENTS, RISK_TABLE
from runtime import ROOT, build_context, checkpoint_path, load_cfg, load_model

SIGMA_OBSERVED = 1e-3  # a measured value carries (almost) no predictive uncertainty


def _concepts(out, data: EvalData, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(slip_mu, slip_sigma) after the requested intervention."""
    mu = out.slip_mu.astype(np.float64).copy()
    sg = out.slip_sigma.astype(np.float64).copy()

    if kind == "none":
        return mu, sg

    if kind == "oracle":
        return data.slip_curve.astype(np.float64), np.full_like(sg, SIGMA_OBSERVED)

    if kind == "proprio":
        rows = np.arange(len(data))
        g = data.gear_driven.astype(int)
        mu[rows, g] = data.slip_driven  # what we actually measured, at the gear we drove
        sg[rows, g] = SIGMA_OBSERVED
        return mu, sg

    raise ValueError(kind)


def intervene(model, ctx, aug, cfg, split: str, slip_cal, gear_cal) -> dict:
    """Score the policy under none / proprioceptive / oracle concept correction."""
    data = EvalData.from_source(ctx.sources[split])
    curves = ctx.sources[split].slip_curve
    rows: dict[str, dict] = {}

    for kind in ("none", "proprio", "oracle"):
        per_intent = {}
        for intent in INTENTS:
            budget = RISK_TABLE[intent]
            lang = aug.fixed(intent, curves)
            out = run_model(model, ctx.phi[split], lang, ctx.device)
            mu, sg = _concepts(out, data, kind)

            if model.mode == "modular":
                upper = slip_cal.upper(mu, budget.alpha, sigma=sg)
                sel = select_modular(upper, budget, slip_point=mu)
            else:  # hybrid: the concept is an INPUT to the gear head, so re-forward
                out2 = run_model(
                    model, ctx.phi[split], lang, ctx.device,
                    concept_override=mu.astype(np.float32),
                )
                sel = select_e2e(
                    gear_cal.sets(out2.probs, budget.alpha), probs=out2.probs, ambiguity="safe"
                )

            per_intent[intent] = score_selection(sel, data, budget)

        rows[kind] = {
            k: float(np.nanmean([v[k] for v in per_intent.values()]))
            for k in ("gear_acc", "violation", "energy", "success", "fallback_rate")
        }

    base = rows["none"]
    return {
        "scw": float(model.scw),
        "split": split,
        "by_intervention": rows,
        "gain_proprio": rows["proprio"]["gear_acc"] - base["gear_acc"],
        "gain_oracle": rows["oracle"]["gear_acc"] - base["gear_acc"],
        "violation_drop_proprio": base["violation"] - rows["proprio"]["violation"],
        "violation_drop_oracle": base["violation"] - rows["oracle"]["violation"],
    }


# --------------------------------------------------------------------------
def figure_intervention(report: dict) -> str:
    setup()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) modular: how much does correcting the concept buy, in-dist vs OOD
    ax = axes[0]
    mod = report.get("modular", {})
    kinds = ["none", "proprio", "oracle"]
    labels = ["as deployed", "proprioceptive\n(measured gear)", "oracle\n(true curve)"]
    width = 0.36
    for i, (split, color) in enumerate([("test", SLOT["modular"]), ("ood", SLOT["e2e"])]):
        if split not in mod:
            continue
        vals = [mod[split]["by_intervention"][k]["gear_acc"] for k in kinds]
        ax.bar(np.arange(3) + (i - 0.5) * width, vals, width * 0.92, color=color,
               label={"test": "in-distribution", "ood": "OOD terrain"}[split],
               edgecolor=plt.rcParams["axes.facecolor"], linewidth=2)
        for x, v in zip(np.arange(3) + (i - 0.5) * width, vals):
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#52514e")
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("gear accuracy")
    ax.set_title("(a) modular: correcting the concept fixes the decision")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.08)

    # (b) hybrid: intervention gain dies as the side channel opens
    ax = axes[1]
    hyb = report.get("hybrid", [])
    if hyb:
        hyb = sorted(hyb, key=lambda r: r["scw"])
        scw = [r["scw"] for r in hyb]
        ax.plot(scw, [r["test"]["gain_oracle"] for r in hyb], "-o",
                color=SLOT["hybrid"], label="in-distribution")
        ax.plot(scw, [r["ood"]["gain_oracle"] for r in hyb], "-s",
                color=SLOT["e2e"], label="OOD terrain")
        ax.axhline(0, color="#e6e5e1", lw=1)
        ax.set_xlabel("side_channel_weight")
        ax.set_ylabel("gain in gear accuracy from oracle intervention")
        ax.set_title("(b) hybrid: leakage destroys intervenability")
        ax.legend(loc="best")
    ax.annotate("e2e has no concept to intervene on -- not plottable",
                xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                fontsize=7.5, color="#a8a7a1")

    return str(save(fig, "A3_intervention"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    ctx = build_context(cfg)
    set_fig_dir(ROOT / cfg.artifacts_dir / "figures")
    aug = ctx.augmenter(str(cfg.language.paraphrase_split), seed=0)
    seed = int(cfg.train.seed)
    alphas = [float(a) for a in cfg.alpha_buckets]

    report: dict = {}

    # --- modular ---
    path = checkpoint_path(cfg, "modular", 0.0, seed)
    if path.exists():
        m = load_model(path, ctx)
        slip_cal = calibrate_slip(m, ctx, aug, alphas, cfg)
        report["modular"] = {
            s: intervene(m, ctx, aug, cfg, s, slip_cal, None) for s in ("test", "ood")
        }

    # --- hybrid, across the leakage knob ---
    hyb = []
    for scw in [float(s) for s in cfg.eval.scw_sweep]:
        p = checkpoint_path(cfg, "hybrid", scw, seed)
        if not p.exists():
            continue
        h = load_model(p, ctx)
        slip_cal = calibrate_slip(h, ctx, aug, alphas, cfg)
        gear_cal = calibrate_gear(h, ctx, aug)
        hyb.append(
            {
                "scw": scw,
                **{s: intervene(h, ctx, aug, cfg, s, slip_cal, gear_cal)
                   for s in ("test", "ood")},
            }
        )
    if hyb:
        report["hybrid"] = hyb

    # --- e2e: the limitation, recorded rather than silently omitted ---
    report["e2e"] = {
        "intervenable": False,
        "reason": (
            "e2e has no physical concept in its forward pass, so there is nothing a "
            "measured slip could be written into. Proprioception can only be used by "
            "retraining. This is a capability gap, not a tuning gap."
        ),
    }

    out = ROOT / cfg.artifacts_dir / f"intervention_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2, default=float))

    if "modular" in report:
        for split in ("test", "ood"):
            r = report["modular"][split]
            print(
                f"modular [{split:4s}]  acc {r['by_intervention']['none']['gear_acc']:.3f}"
                f" -> proprio {r['by_intervention']['proprio']['gear_acc']:.3f}"
                f" -> oracle {r['by_intervention']['oracle']['gear_acc']:.3f}"
            )
    for r in report.get("hybrid", []):
        print(
            f"hybrid  [scw={r['scw']:<4g}] oracle gain: "
            f"test {r['test']['gain_oracle']:+.3f}   ood {r['ood']['gain_oracle']:+.3f}"
        )

    print(f"\nwrote {out}\nwrote {figure_intervention(report)}")


if __name__ == "__main__":
    main()
