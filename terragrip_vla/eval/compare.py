"""M8: the controlled A/B/C comparison.

Everything is held fixed except `mode` (and, for C, `side_channel_weight`):
the same data splits, the same frozen backbone, the same feature cache, the same
optimiser / epochs / lr, matched head widths, the same alphas, the same
counterfactual noise draw.  Then it is run over >= 5 seeds and reported as
mean +- std.

Three regimes, because in-distribution alone would be misleading -- the task is
deterministic given (terrain, instruction), so a black box saturates it:

    in-distribution   test terrains, TRAINING paraphrases
    instruction gen.  test terrains, HELD-OUT paraphrases (never seen in training,
                      and containing no keyword, so they can only be resolved
                      semantically).  modular re-derives the risk budget
                      analytically; e2e has to interpolate in embedding space.
    OOD terrain       a terrain that appears in no split used for training.

Both routes read alpha from the same shared language layer (`interpret`), so the
only thing being compared is what happens AFTER the instruction is understood.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from conformal.select import select_e2e, select_modular, select_point
from eval.metrics import EvalData, aggregate, mean_std, score_selection, slip_fidelity
from eval.plotting import set_fig_dir, SLOT, color_of, save, setup
from eval.policies import run_model
from eval.run_eval import calibrate_gear, calibrate_slip
from language import INTENTS, RISK_TABLE, instruction_pool, interpret
from runtime import (
    ROOT,
    build_context,
    checkpoint_path,
    load_cfg,
    load_model,
)

MODES = ("modular", "e2e", "hybrid")
REGIMES = ("in_dist", "instruction_gen", "ood")

HEADLINE = [
    "energy", "slip_realized_mean", "violation", "violation_mean", "success",
    "gear_acc", "fallback_rate", "violation_when_confident",
    "lang_spread", "lang_monotone",
]


# --------------------------------------------------------------------------
def _texts(paraphrase_split: str) -> list[tuple[str, str]]:
    return [(t, i) for i in INTENTS for t in instruction_pool(i, paraphrase_split)]


def evaluate_regime(
    model, ctx, cfg, aug, split: str, paraphrase_split: str, slip_cal, gear_cal
) -> dict:
    """Score one model on one (terrain split, paraphrase split).

    The decision budget is ALWAYS re-derived from the raw instruction text via the
    shared language layer -- never handed the ground-truth intent.  On held-out
    paraphrases that means it can be wrong, which is exactly what we want to
    measure.  The oracle label used for scoring still uses the TRUE intent, since
    that is what the operator actually asked for.
    """
    data = EvalData.from_source(ctx.sources[split])
    curves = ctx.sources[split].slip_curve

    by_intent: dict[str, list[dict]] = {i: [] for i in INTENTS}

    for text, true_intent in _texts(paraphrase_split):
        true_budget = RISK_TABLE[true_intent]  # what the operator meant -> the label
        decided = interpret(text, backend=str(cfg.language.encoder))  # what we inferred

        lang = aug.fixed(true_intent, curves, text=text)

        # The model is handed ONLY what the language layer inferred from the raw
        # text -- never the ground-truth budget. Without this, a run with
        # `language.condition: budget` would feed e2e the true (alpha, tau, lam)
        # and its instruction-generalisation score would be free.
        n = len(curves)
        for key, val in (("alpha", decided.alpha), ("tau", decided.tau), ("lam", decided.lam)):
            lang[key] = torch.full((n,), float(val), dtype=torch.float32)

        out = run_model(model, ctx.phi[split], lang, ctx.device)

        if model.mode == "modular":
            upper = slip_cal.upper(out.slip_mu, decided.alpha, sigma=out.slip_sigma)
            sel = select_modular(upper, decided, slip_point=out.slip_mu)
        else:
            sel = select_e2e(
                gear_cal.sets(out.probs, decided.alpha), probs=out.probs, ambiguity="safe"
            )

        by_intent[true_intent].append(score_selection(sel, data, true_budget))

    # average the paraphrases within each intent, then aggregate across intents
    per_intent = {
        i: {k: float(np.nanmean([r[k] for r in rows]))
            for k in rows[0] if isinstance(rows[0][k], float)}
        for i, rows in by_intent.items()
    }
    res = aggregate(per_intent)

    if model.has_concept:
        mu = run_model(model, ctx.phi[split], aug.fixed("normal", curves), ctx.device).slip_mu
        res.update(slip_fidelity(mu, data))
    return res


def evaluate_seed(cfg, ctx, seed: int, train_missing: bool) -> dict:
    aug = ctx.augmenter(str(cfg.language.paraphrase_split), seed=seed)
    alphas = [float(a) for a in cfg.alpha_buckets]
    out: dict[str, dict] = {}

    for mode in MODES:
        scw = float(cfg.side_channel_weight) if mode == "hybrid" else 0.0
        path = checkpoint_path(cfg, mode, scw, seed)

        if not path.exists():
            if not train_missing:
                print(f"[skip] {path.name}")
                continue
            from training.train import run_training

            # Derive the training config from THIS run's cfg, never from a fresh
            # `default`. The earlier version reloaded the default config and applied
            # only (mode, scw, seed), silently dropping every other override -- so
            # `--train-missing` with any custom backbone / data size / artifacts_dir
            # trained the models under DEFAULT settings and then evaluated them under
            # yours. That is a controlled comparison quietly evaluating models that
            # were never trained the way you asked.
            sub = copy.deepcopy(cfg)
            sub.mode = mode
            sub.side_channel_weight = scw
            sub.train.seed = seed
            run_training(sub, ctx=ctx, progress=False)

        model = load_model(path, ctx)
        slip_cal = calibrate_slip(model, ctx, aug, alphas, cfg) if model.has_concept else None
        gear_cal = calibrate_gear(model, ctx, aug) if model.mode != "modular" else None

        out[mode] = {
            "in_dist": evaluate_regime(model, ctx, cfg, aug, "test", "train", slip_cal, gear_cal),
            "instruction_gen": evaluate_regime(
                model, ctx, cfg, aug, "test", "heldout", slip_cal, gear_cal
            ),
            "ood": evaluate_regime(model, ctx, cfg, aug, "ood", "train", slip_cal, gear_cal),
        }
        print(
            f"  seed {seed} {mode:<8} "
            f"in-dist acc {out[mode]['in_dist']['gear_acc']:.3f} | "
            f"heldout acc {out[mode]['instruction_gen']['gear_acc']:.3f} | "
            f"ood acc {out[mode]['ood']['gear_acc']:.3f}"
        )
    return out


# --------------------------------------------------------------------------
def build_table(runs: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        for regime in REGIMES:
            per_seed = [r[mode][regime] for r in runs.values() if mode in r]
            if not per_seed:
                continue
            stats = mean_std(per_seed)
            row = {"mode": mode, "regime": regime, "seeds": len(per_seed)}
            for k in HEADLINE + ["slip_mae", "slip_mae_ratio"]:
                if k in stats:
                    row[k] = stats[k]
                    row[f"{k}_std"] = stats[f"{k}_std"]
            rows.append(row)
    return pd.DataFrame(rows)


def figure_compare(df: pd.DataFrame) -> str:
    setup()
    import matplotlib.pyplot as plt

    panels = [
        ("gear_acc", "gear accuracy", True),
        ("violation", "rate of slip > tau", False),
        ("energy", "effort proxy", False),
        ("lang_spread", "instruction spread\n(careful effort - fast effort)", True),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16.5, 4.2))

    x = np.arange(len(REGIMES))
    width = 0.26

    for ax, (key, ylab, higher_better) in zip(axes, panels):
        for i, mode in enumerate(MODES):
            sub = df[df["mode"] == mode].set_index("regime")
            vals = [sub.loc[r, key] if r in sub.index else np.nan for r in REGIMES]
            errs = [sub.loc[r, f"{key}_std"] if r in sub.index else 0.0 for r in REGIMES]
            ax.bar(
                x + (i - 1) * width, vals, width * 0.9,
                yerr=errs, capsize=3, color=SLOT[mode], label=mode,
                edgecolor=plt.rcParams["axes.facecolor"], linewidth=2,
                error_kw={"ecolor": "#52514e", "lw": 1},
            )
        ax.set_xticks(x, ["in-dist", "held-out\ninstruction", "OOD\nterrain"])
        ax.set_ylabel(ylab)
        arrow = "higher is better" if higher_better else "lower is better"
        ax.set_title(f"{key}  ({arrow})")

    axes[0].legend(loc="lower left", ncol=3)
    fig.suptitle(
        "A/B/C, mean +- std over seeds. Only `mode` differs: same data, same frozen "
        "backbone, same optimiser, matched head widths, same alphas.",
        y=1.04, fontsize=8, color="#52514e",
    )
    return str(save(fig, "A4_compare"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--train-missing", action="store_true",
                    help="train any (mode, seed) whose checkpoint is absent")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    ctx = build_context(cfg)
    set_fig_dir(ROOT / cfg.artifacts_dir / "figures")
    seeds = [int(s) for s in cfg.eval.seeds]

    runs = {}
    for seed in seeds:
        print(f"seed {seed}")
        runs[seed] = evaluate_seed(cfg, ctx, seed, args.train_missing)

    df = build_table(runs)
    if df.empty:
        print("\nno checkpoints found for any (mode, seed). Rerun with --train-missing.")
        return

    out_dir = ROOT / cfg.artifacts_dir
    df.to_csv(out_dir / "compare_table.csv", index=False)
    (out_dir / "compare_raw.json").write_text(json.dumps(runs, indent=2, default=float))

    # markdown main table
    lines = ["| mode | regime | gear acc | violation | effort | slip MAE / Bayes | lang spread |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        mae = (f"{r['slip_mae_ratio']:.2f}x +- {r['slip_mae_ratio_std']:.2f}"
               if "slip_mae_ratio" in r and not pd.isna(r.get("slip_mae_ratio")) else "n/a")
        lines.append(
            f"| {r['mode']} | {r['regime']} | "
            f"{r['gear_acc']:.3f} ± {r['gear_acc_std']:.3f} | "
            f"{r['violation']:.3f} ± {r['violation_std']:.3f} | "
            f"{r['energy']:.3f} ± {r['energy_std']:.3f} | {mae} | "
            f"{r['lang_spread']:.3f} ± {r['lang_spread_std']:.3f} |"
        )
    table_md = "\n".join(lines)
    (out_dir / "compare_table.md").write_text(table_md)

    print("\n" + table_md)
    print(f"\nwrote {out_dir/'compare_table.csv'}")
    print(f"wrote {figure_compare(df)}")


if __name__ == "__main__":
    main()
