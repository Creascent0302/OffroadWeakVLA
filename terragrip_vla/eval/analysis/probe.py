"""Probe: did the end-to-end policy secretly compute traction anyway?

Freeze the trained e2e model, take its PENULTIMATE layer, and fit a LINEAR probe
from those features to slip at each gear.  A linear probe is deliberately weak: if
it succeeds, the information was already sitting there, linearly decodable.

    high R^2 -> the black box did learn an implicit traction model.  The concept is
                real; e2e simply refuses to show it to you.
    low  R^2 -> it took a shortcut (terrain identity -> gear) and never represented
                traction at all.

TWO THINGS THIS FILE IS CAREFUL ABOUT, because the first version of it was
misleading on both counts:

1. THE PROBE GETS THE SAME SUPERVISION THE SLIP HEAD GOT.
   The first version fitted the probe on the NOISE-FREE true slip curve, then
   compared its MAE to the slip head's -- but the slip head only ever sees a NOISY
   measurement at the ONE gear it drove.  That is a strictly easier problem, and it
   made the probe look better than the explicit concept for no real reason.  Here
   the probe is fitted per gear, on the noisy observed slip, using only the samples
   actually driven at that gear.  Identical information, so the comparison means
   something.

2. THE CONTROLS ARE REPORTED, AND THEY CAN INVALIDATE THE EXPERIMENT.
   ceiling: probe the frozen DINOv2 features directly -- all the slip information
            the representation ever had.
   floor  : probe an UNTRAINED gear head -- what a random projection of those same
            features already gives you for free.
   If floor ~= ceiling, the probe CANNOT discriminate: slip is decodable from
   anything, so a high R^2 on the trained model proves nothing.  We compute
   `headroom` and `probe_score` = (trained - floor) / (ceiling - floor) and say so
   out loud.  On this synthetic benchmark the terrains are linearly separable and
   the probe is close to saturated -- that is a limitation of the mock, and hiding
   it would be the easiest way to publish a false claim.

The instruction is held FIXED while probing, so the features vary only with the
image and the probe measures vision, not language.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from constants import GEAR_NAMES, GEARS
from eval.metrics import EvalData, bayes_slip_mae
from eval.plotting import set_fig_dir, SLOT, save, setup
from eval.policies import run_model
from runtime import ROOT, build_context, build_model, checkpoint_path, load_cfg, load_model

PROBE_INTENT = "normal"  # fixed: the probe must measure vision, not language
ALPHAS = np.logspace(-3, 4, 15)
SATURATION_LIMIT = 0.05  # ceiling - floor below this => the probe cannot discriminate


def _features(model, ctx, split: str, aug) -> np.ndarray:
    out = run_model(
        model,
        ctx.phi[split],
        aug.fixed(PROBE_INTENT, ctx.sources[split].slip_curve),
        ctx.device,
        want_features=True,
    )
    return out.gear_feat


def fit_probe(x_tr, src_tr, x_te, data_te: EvalData) -> dict:
    """Linear probe with EXACTLY the slip head's supervision.

    For each gear g: fit on the training samples that were DRIVEN at g, targeting
    the NOISY measured slip.  Evaluate against the true mean curve on test, which is
    the same yardstick the explicit slip head is measured with.
    """
    gear_tr = np.asarray(src_tr.gear, int)
    slip_tr = np.asarray(src_tr.slip, np.float64)

    pred = np.zeros((len(x_te), len(GEARS)), np.float64)
    for g in GEARS:
        m = gear_tr == g
        probe = RidgeCV(alphas=ALPHAS).fit(x_tr[m], slip_tr[m])
        pred[:, g] = probe.predict(x_te)

    y_true = data_te.slip_curve  # true mean slip per gear
    res = {
        "r2_overall": float(r2_score(y_true, pred, multioutput="variance_weighted")),
        "mae_overall": float(np.abs(pred - y_true).mean()),
    }
    for g in GEARS:
        res[f"r2_gear_{GEAR_NAMES[g]}"] = float(r2_score(y_true[:, g], pred[:, g]))
        res[f"mae_gear_{GEAR_NAMES[g]}"] = float(np.abs(pred[:, g] - y_true[:, g]).mean())
    for t in sorted(set(data_te.terrain)):
        m = data_te.terrain == t
        res[f"mae_{t}"] = float(np.abs(pred[m] - y_true[m]).mean())
    return res


def explicit_concept(model, ctx, aug, data_te: EvalData) -> dict:
    """The modular / hybrid slip head, scored on exactly the same yardstick."""
    mu = run_model(
        model,
        ctx.phi["test"],
        aug.fixed(PROBE_INTENT, ctx.sources["test"].slip_curve),
        ctx.device,
    ).slip_mu.astype(np.float64)

    y_true = data_te.slip_curve
    res = {
        "r2_overall": float(r2_score(y_true, mu, multioutput="variance_weighted")),
        "mae_overall": float(np.abs(mu - y_true).mean()),
    }
    for g in GEARS:
        res[f"r2_gear_{GEAR_NAMES[g]}"] = float(r2_score(y_true[:, g], mu[:, g]))
        res[f"mae_gear_{GEAR_NAMES[g]}"] = float(np.abs(mu[:, g] - y_true[:, g]).mean())
    for t in sorted(set(data_te.terrain)):
        m = data_te.terrain == t
        res[f"mae_{t}"] = float(np.abs(mu[m] - y_true[m]).mean())
    return res


def probe_report(cfg, ctx, aug, seed: int) -> dict:
    src_tr = ctx.sources["train"]
    data_te = EvalData.from_source(ctx.sources["test"])
    report: dict = {}

    # ceiling: the frozen representation itself
    report["frozen_dinov2"] = fit_probe(
        ctx.phi["train"].numpy(), src_tr, ctx.phi["test"].numpy(), data_te
    )

    # floor: a random projection of the same features
    untrained = build_model(cfg, ctx, mode="e2e").eval()
    report["e2e_untrained"] = fit_probe(
        _features(untrained, ctx, "train", aug), src_tr,
        _features(untrained, ctx, "test", aug), data_te,
    )

    # the question
    path = checkpoint_path(cfg, "e2e", 0.0, seed)
    if path.exists():
        e2e = load_model(path, ctx)
        report["e2e_trained"] = fit_probe(
            _features(e2e, ctx, "train", aug), src_tr,
            _features(e2e, ctx, "test", aug), data_te,
        )
    else:
        print(f"[skip] no e2e checkpoint ({path.name})")

    # the explicit concept, for contrast
    for mode in ("modular", "hybrid"):
        scw = float(cfg.side_channel_weight) if mode == "hybrid" else 0.0
        p = checkpoint_path(cfg, mode, scw, seed)
        if p.exists():
            report[f"{mode}_explicit"] = explicit_concept(load_model(p, ctx), ctx, aug, data_te)

    # --- is the probe even able to tell anything apart? ---
    floor = report["e2e_untrained"]["r2_overall"]
    ceiling = report["frozen_dinov2"]["r2_overall"]
    headroom = ceiling - floor
    diag = {
        "floor_r2_untrained": floor,
        "ceiling_r2_frozen": ceiling,
        "headroom": headroom,
        "saturated": bool(headroom < SATURATION_LIMIT),
        "bayes_slip_mae": bayes_slip_mae(data_te),
    }
    # Only meaningful when there is headroom to normalise by. With floor ~= ceiling
    # the ratio explodes (measured: 4.37 off a headroom of -0.002) and would read as
    # a spectacular result. Refuse to compute it.
    if "e2e_trained" in report and headroom > SATURATION_LIMIT:
        diag["probe_score"] = float((report["e2e_trained"]["r2_overall"] - floor) / headroom)
    if diag["saturated"]:
        diag["warning"] = (
            "PROBE SATURATED: an UNTRAINED head already decodes slip nearly as well as the "
            "frozen backbone itself, so a high R^2 on the trained model is NOT evidence that "
            "e2e learned traction -- slip is decodable from any projection of these features. "
            "The mock terrains are linearly separable; this experiment only becomes "
            "informative on visually ambiguous real terrain."
        )
    report["diagnostics"] = diag
    return report


def figure_probe(report: dict) -> str:
    setup()
    import matplotlib.pyplot as plt

    order = [
        ("e2e_untrained", "e2e\nuntrained\n(floor)", "#a8a7a1"),
        ("e2e_trained", "e2e\ntrained\n(implicit)", SLOT["e2e"]),
        ("hybrid_explicit", "hybrid\n(concept)", SLOT["hybrid"]),
        ("modular_explicit", "modular\n(concept)", SLOT["modular"]),
        ("frozen_dinov2", "frozen\nDINOv2\n(ceiling)", SLOT["oracle"]),
    ]
    order = [(k, lab, c) for k, lab, c in order if k in report]
    diag = report["diagnostics"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    xs = np.arange(len(order))
    vals = [report[k]["r2_overall"] for k, _, _ in order]
    ax.bar(xs, vals, width=0.62, color=[c for _, _, c in order],
           edgecolor=plt.rcParams["axes.facecolor"], linewidth=2)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color="#52514e")
    ax.axhline(diag["floor_r2_untrained"], ls="--", lw=1.2, color="#a8a7a1")
    ax.axhline(diag["ceiling_r2_frozen"], ls="--", lw=1.2, color=SLOT["oracle"])
    ax.set_xticks(xs, [lab for _, lab, _ in order])
    ax.set_ylabel("probe $R^2$ for slip")
    ax.set_ylim(min(0.0, min(vals) - 0.05), 1.06)
    ax.set_title("(a) is traction linearly decodable?")

    ax = axes[1]
    width = 0.8 / max(1, len(order))
    gears = [GEAR_NAMES[g] for g in GEARS]
    for i, (k, lab, c) in enumerate(order):
        v = [report[k][f"mae_gear_{g}"] for g in gears]
        ax.bar(np.arange(3) + i * width - 0.4 + width / 2, v, width * 0.9, color=c,
               label=lab.replace("\n", " "),
               edgecolor=plt.rcParams["axes.facecolor"], linewidth=2)
    ax.set_xticks(np.arange(3), [f"gear {g}" for g in gears])
    ax.set_ylabel("slip MAE vs the true mean curve")
    ax.set_title("(b) how accurately, per gear")
    ax.legend(loc="upper left", ncol=2, fontsize=7)

    note = (
        f"headroom (ceiling - floor) = {diag['headroom']:.3f}. "
        + ("PROBE SATURATED: an untrained head already decodes slip, so a high R^2 "
           "here proves nothing. See the docstring."
           if diag["saturated"] else "Probe has headroom; the comparison is meaningful.")
    )
    fig.suptitle(note, y=1.04, fontsize=8,
                 color=("#e34948" if diag["saturated"] else "#52514e"))
    return str(save(fig, "A2_probe"))


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

    report = probe_report(cfg, ctx, aug, seed)
    (ROOT / cfg.artifacts_dir / f"probe_seed{seed}.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'model':<20}{'R^2':>9}{'slip MAE':>11}")
    print("-" * 40)
    for k, v in report.items():
        if k == "diagnostics":
            continue
        print(f"{k:<20}{v['r2_overall']:>9.3f}{v['mae_overall']:>11.4f}")

    d = report["diagnostics"]
    print(f"\nBayes slip MAE floor          : {d['bayes_slip_mae']:.4f}")
    print(f"probe headroom (ceiling-floor): {d['headroom']:.3f}")
    if "probe_score" in d:
        print(f"normalised probe score        : {d['probe_score']:.3f}")
    if d["saturated"]:
        print(f"\n!! {d['warning']}")

    print(f"\nwrote {figure_probe(report)}")


if __name__ == "__main__":
    main()
