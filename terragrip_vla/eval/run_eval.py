"""M6: calibrate, evaluate, and draw E1 (the Pareto figure).

    python -m eval.run_eval                    # every checkpoint found for seed 0
    python -m eval.run_eval train.seed=1

Calibration asymmetry worth noticing (it is a result, not an implementation
detail):

    the slip bound is calibrated ONCE and serves every instruction, because the
    slip head never saw language;

    APS must be re-calibrated PER INSTRUCTION, because the label it covers --
    best_gear -- changes when the instruction changes.

So modular extends to a novel risk budget for free; e2e needs a fresh
calibration set for every new one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from conformal.aps import GearConformal, aps_calibrate
from conformal.split_conformal import SlipConformal, empirical_coverage
from eval.baselines import model_free_baselines
from eval.metrics import EvalData, aggregate, per_terrain, score_selection, slip_fidelity
from eval.policies import (
    APSPolicy,
    ArgmaxPolicy,
    ConformalSlipPolicy,
    ModelOutputs,
    Policy,
    PointSlipPolicy,
    run_model,
)
from eval.plotting import set_fig_dir, BASELINE_INK, color_of, label_point, save, setup
from language import INTENTS, RISK_TABLE
from runtime import ROOT, build_context, checkpoint_path, load_cfg, load_model

MODES = ("modular", "e2e", "hybrid")


# --------------------------------------------------------------------------
# Model outputs, one forward pass per instruction
# --------------------------------------------------------------------------
def outputs_per_intent(model, ctx, split: str, aug, want_features: bool = False) -> dict:
    curves = ctx.sources[split].slip_curve
    phi = ctx.phi[split]
    return {
        intent: run_model(
            model, phi, aug.fixed(intent, curves), ctx.device, want_features=want_features
        )
        for intent in INTENTS
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibrate_slip(model, ctx, aug, alphas: list[float], cfg) -> SlipConformal:
    """Language-free: one calibration for every instruction, present and future."""
    out = run_model(model, ctx.phi["cal"], aug.fixed("normal", ctx.sources["cal"].slip_curve),
                    ctx.device)
    src = ctx.sources["cal"]
    driven = np.arange(len(src)), np.asarray(src.gear, int)
    return SlipConformal(
        score=str(cfg.conformal.score), per_gear=bool(cfg.conformal.per_gear)
    ).fit(
        mu=out.slip_mu[driven],
        y=np.asarray(src.slip, np.float64),
        alphas=alphas,
        sigma=out.slip_sigma[driven],
        gear=np.asarray(src.gear, int),
    )


def calibrate_gear(model, ctx, aug) -> GearConformal:
    """Per instruction: the class APS must cover (best_gear) depends on the budget.

    alpha identifies the instruction one-to-one in RISK_TABLE, so a single
    GearConformal can hold them all -- but each Q was fitted on ITS OWN
    instruction's calibration pass.
    """
    alphas = [b.alpha for b in RISK_TABLE.values()]
    assert len(set(alphas)) == len(alphas), "alpha must identify the intent uniquely"

    cal = EvalData.from_source(ctx.sources["cal"])
    gc = GearConformal(randomized=True, seed=0)
    for i, (intent, budget) in enumerate(RISK_TABLE.items()):
        out = run_model(
            model, ctx.phi["cal"], aug.fixed(intent, ctx.sources["cal"].slip_curve), ctx.device
        )
        gc.Q[budget.alpha] = aps_calibrate(
            out.probs, cal.oracle(budget), budget.alpha, gc.randomized, gc._rng(i + 1)
        )
    return gc


# --------------------------------------------------------------------------
# Coverage: the guarantee, measured
# --------------------------------------------------------------------------
def coverage_report(model, ctx, aug, slip_cal, gear_cal, split: str = "test") -> dict:
    src = ctx.sources[split]
    data = EvalData.from_source(src)
    rep: dict = {}

    if slip_cal is not None:
        out = run_model(model, ctx.phi[split], aug.fixed("normal", src.slip_curve), ctx.device)
        driven = np.arange(len(src)), np.asarray(src.gear, int)
        for a in sorted(slip_cal.Q):
            upper = slip_cal.upper(out.slip_mu, a, sigma=out.slip_sigma)[driven]
            rep[f"slip_coverage@{a:g}"] = empirical_coverage(data.slip_driven, upper)
            rep[f"slip_upper_width@{a:g}"] = float((upper - out.slip_mu[driven]).mean())

    if gear_cal is not None:
        for intent, budget in RISK_TABLE.items():
            out = run_model(model, ctx.phi[split], aug.fixed(intent, src.slip_curve), ctx.device)
            rep[f"aps_coverage@{budget.alpha:g}({intent})"] = gear_cal.coverage(
                out.probs, data.oracle(budget), budget.alpha
            )
            sets = gear_cal.sets(out.probs, budget.alpha)
            rep[f"aps_setsize@{budget.alpha:g}({intent})"] = float(
                np.mean([len(s) for s in sets])
            )
    return rep


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_policies(
    policies: list[Policy],
    outs: dict[str, ModelOutputs],
    data: EvalData,
    with_terrain: bool = False,
) -> dict:
    """{policy: {per-intent metrics, aggregated metrics, per-terrain}}"""
    results: dict[str, dict] = {}
    for p in policies:
        by_intent = {}
        terr = {}
        for intent in INTENTS:
            budget = RISK_TABLE[intent]
            sel = p.select(outs[intent], data, budget)
            by_intent[intent] = score_selection(sel, data, budget)
            if with_terrain:
                terr[intent] = per_terrain(sel, data, budget)
        results[p.name] = {"by_intent": by_intent, **aggregate(by_intent)}
        if with_terrain:
            results[p.name]["per_terrain"] = terr
    return results


def policies_for(model, slip_cal, gear_cal, cfg) -> list[Policy]:
    """The decision routes a given mode actually supports."""
    amb = str(cfg.get("select", {}).get("ambiguity", "safe"))
    out: list[Policy] = []
    if model.mode == "modular":
        out += [ConformalSlipPolicy(slip_cal, "modular"), PointSlipPolicy()]
    elif model.mode == "e2e":
        out += [APSPolicy(gear_cal, amb, "e2e"), ArgmaxPolicy()]
    elif model.mode == "hybrid":
        out += [APSPolicy(gear_cal, amb, "hybrid")]
    return out


def evaluate_model(model, ctx, cfg, aug, split: str = "test", with_terrain: bool = True) -> dict:
    alphas = [float(a) for a in cfg.alpha_buckets]
    slip_cal = calibrate_slip(model, ctx, aug, alphas, cfg) if model.has_concept else None
    gear_cal = calibrate_gear(model, ctx, aug) if model.mode in ("e2e", "hybrid") else None

    data = EvalData.from_source(ctx.sources[split])
    outs = outputs_per_intent(model, ctx, split, aug)

    res = {
        "mode": model.mode,
        "scw": model.scw,
        "policies": evaluate_policies(
            policies_for(model, slip_cal, gear_cal, cfg), outs, data, with_terrain
        ),
        "coverage": coverage_report(model, ctx, aug, slip_cal, gear_cal, split),
    }
    if model.has_concept:
        res["concept"] = slip_fidelity(outs["normal"].slip_mu, data)
        res["conformal_Q"] = slip_cal.to_dict()
    if gear_cal is not None:
        res["aps_Q"] = gear_cal.to_dict()
    return res, slip_cal, gear_cal


def evaluate_baselines(ctx, split: str = "test", with_terrain: bool = True) -> dict:
    data = EvalData.from_source(ctx.sources[split])
    dummy = {i: ModelOutputs() for i in INTENTS}  # baselines look at no model output
    return evaluate_policies(model_free_baselines(), dummy, data, with_terrain)


# --------------------------------------------------------------------------
# E1: the Pareto figure
# --------------------------------------------------------------------------
def figure_e1(rows: dict[str, dict], name: str = "E1_pareto") -> Path:
    """Slip vs effort, and violation vs effort.  Lower-left is better in both."""
    setup()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    panels = [
        ("slip_realized_mean", "mean slip at the chosen gear", "traction"),
        ("violation", f"rate of slip > tau", "safety"),
    ]
    for ax, (key, ylab, tag) in zip(axes, panels):
        for name_, m in rows.items():
            c = color_of(name_)
            is_model = name_ in ("modular", "e2e", "hybrid")
            ax.scatter(
                m["energy"], m[key],
                s=110 if is_model else 60,
                color=c,
                edgecolor="white",
                linewidth=1.6 if is_model else 1.0,
                zorder=3 if is_model else 2,
                marker="o" if is_model else "s",
                label=name_,
            )
            label_point(ax, m["energy"], m[key], name_, c)

        ax.set_xlabel("effort proxy  (mean contact area)")
        ax.set_ylabel(ylab)
        ax.set_title(f"E1 · {tag}: lower-left is better")
        ax.margins(x=0.22, y=0.18)

    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle(
        "Averaged over the three instructions. Fixed-gear baselines bound the region; "
        "the oracle knows the true slip curve.",
        y=1.03, fontsize=8, color="#52514e",
    )
    return save(fig, name)


# --------------------------------------------------------------------------
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

    rows: dict[str, dict] = {}
    report: dict = {"seed": seed, "models": {}, "baselines": {}}

    for mode in MODES:
        scw = float(cfg.side_channel_weight) if mode == "hybrid" else 0.0
        path = checkpoint_path(cfg, mode, scw, seed)
        if not path.exists():
            print(f"[skip] no checkpoint for {mode} (looked for {path.name})")
            continue

        model = load_model(path, ctx)
        res, slip_cal, _ = evaluate_model(model, ctx, cfg, aug)
        report["models"][mode] = res

        primary = mode  # the policy named after the mode is its headline route
        rows[primary] = res["policies"][primary]
        for extra in res["policies"]:
            if extra != primary:
                rows[extra] = res["policies"][extra]

        if slip_cal is not None and mode == "modular":
            slip_cal.save(ROOT / cfg.artifacts_dir / "conformal_Q.json")

    base = evaluate_baselines(ctx)
    report["baselines"] = base
    rows.update(base)

    out_path = ROOT / cfg.artifacts_dir / f"eval_seed{seed}.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    fig = figure_e1(rows)

    # ---- console table ----
    cols = ["energy", "slip_realized_mean", "violation", "success", "gear_acc",
            "fallback_rate", "lang_spread"]
    print(f"\n{'policy':<26}" + "".join(f"{c:>13}" for c in cols))
    print("-" * (26 + 13 * len(cols)))
    for name_, m in sorted(rows.items(), key=lambda kv: kv[1]["energy"]):
        print(f"{name_:<26}" + "".join(f"{m.get(c, float('nan')):>13.3f}" for c in cols))

    print("\ncoverage (must be >= 1 - alpha):")
    for mode, res in report["models"].items():
        for k, v in res["coverage"].items():
            print(f"  {mode:<9} {k:<28} {v:.3f}")

    for mode, res in report["models"].items():
        if "concept" in res:
            c = res["concept"]
            print(
                f"\nslip fidelity [{mode}]: MAE {c['slip_mae']:.4f} "
                f"(Bayes floor {c['slip_mae_bayes']:.4f}, ratio {c['slip_mae_ratio']:.2f}x)"
            )

    print(f"\nwrote {out_path}\nwrote {fig}")


if __name__ == "__main__":
    main()
