"""框架效果验证 —— 把“方案期望的效果”逐条变成可断言的检查。

这个脚本回答的不是“代码能不能跑”（那是 pytest 的活），而是：
**跑出来的东西，是不是方案里想要的那个东西。**

它会（必要时先训练缺失的 checkpoint，然后）逐条检查：

  数据层   异方差噪声 / cal 与 train 同分布 / OOD 地形训练未见
  感知层   冻结、无梯度、地形在特征空间可分
  概念层   modular 的 slip MAE 是否逼近 Bayes 噪声底
  保形层   回归路径与分类路径的经验覆盖率是否 >= 1 - alpha   ← 硬保证
  语言层   语言是否真的改变 oracle 标签；三种模式是否都“听指令”；
           slip 头是否对语言不变（物理与语言无关）
  决策层   三模式是否优于固定档 / 纯反应式（帕累托）
  分析层   leakage 是否随 scw 从 0 升到 1；概念干预是否真的改善决策；
           探针实验是否饱和（饱和 = 该实验在本 mock 上证明不了任何事）
  泛化     OOD 地形 / 留出改写指令上，modular 是否比 e2e 更稳

用法:
    python scripts/verify_framework.py --config small     # 小而真，几分钟
    python scripts/verify_framework.py --config default   # 正式规模
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformal.select import select_e2e, select_modular  # noqa: E402
from constants import CONTACT_AREA  # noqa: E402
from data.labels import best_gear  # noqa: E402
from data.mock_generator import (  # noqa: E402
    OOD_TERRAINS,
    TERRAIN_SIGMA,
    TERRAIN_SLIP,
    TRAIN_TERRAINS,
)
from eval.baselines import model_free_baselines  # noqa: E402
from eval.metrics import EvalData, aggregate, score_selection, slip_fidelity  # noqa: E402
from eval.policies import ModelOutputs, run_model  # noqa: E402
from eval.run_eval import calibrate_gear, calibrate_slip  # noqa: E402
from language import INTENTS, RISK_TABLE, instruction_pool, interpret  # noqa: E402
from models.model import language_features  # noqa: E402
from runtime import (  # noqa: E402
    build_context,
    build_model,
    checkpoint_path,
    load_cfg,
    load_model,
)

# --------------------------------------------------------------------------
# 结果记录
# --------------------------------------------------------------------------
ROWS: list[tuple[str, str, str, str]] = []  # (分组, 检查项, 状态, 实测)


def check(group: str, name: str, ok: bool | None, detail: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    ROWS.append((group, name, status, detail))


def report() -> int:
    w = max(len(r[1]) for r in ROWS) + 2
    last = None
    print()
    for group, name, status, detail in ROWS:
        if group != last:
            print(f"\n── {group} " + "─" * (68 - len(group)))
            last = group
        mark = {"PASS": "✓", "FAIL": "✗", "INFO": "·"}[status]
        print(f"  {mark} {name:<{w}} {detail}")

    fails = [r for r in ROWS if r[2] == "FAIL"]
    print("\n" + "=" * 74)
    n_pass = sum(r[2] == "PASS" for r in ROWS)
    print(f"  {n_pass} 项通过 · {len(fails)} 项失败 · "
          f"{sum(r[2] == 'INFO' for r in ROWS)} 项仅供参考")
    for _, name, _, detail in fails:
        print(f"    ✗ {name}: {detail}")
    print("=" * 74)
    return 1 if fails else 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small")
    ap.add_argument("--no-train", action="store_true", help="缺 checkpoint 时不自动训练")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    ctx = build_context(cfg)
    seed = int(cfg.train.seed)
    alphas = [float(a) for a in cfg.alpha_buckets]
    aug = ctx.augmenter(str(cfg.language.paraphrase_split), seed=seed)

    print(f"config={args.config}  backbone={cfg.backbone}  device={ctx.device}")
    print(f"data={dict(cfg.data.sizes)}  alphas={alphas}")

    # ---------------- 训练缺失的模型 ----------------
    from training.train import run_training
    import copy

    models = {}
    scw_sweep = [float(s) for s in cfg.eval.scw_sweep]
    needed = [("modular", 0.0), ("e2e", 0.0)] + [("hybrid", s) for s in scw_sweep]

    for mode, scw in needed:
        p = checkpoint_path(cfg, mode, scw, seed)
        if not p.exists():
            if args.no_train:
                print(f"[skip] 缺少 {p.name}")
                continue
            sub = copy.deepcopy(cfg)
            sub.mode, sub.side_channel_weight, sub.train.seed = mode, scw, seed
            print(f"  训练 {mode} scw={scw:g} ...", flush=True)
            run_training(sub, ctx=ctx, progress=False)
        models[(mode, scw)] = load_model(p, ctx)

    modular = models.get(("modular", 0.0))
    e2e = models.get(("e2e", 0.0))
    hybrid0 = models.get(("hybrid", 0.0))

    data_te = EvalData.from_source(ctx.sources["test"])
    data_ood = EvalData.from_source(ctx.sources["ood"])
    curves_te = ctx.sources["test"].slip_curve

    # ==================================================================
    # 1. 数据层
    # ==================================================================
    g = "1. 数据层 (mock)"
    sig = [TERRAIN_SIGMA[t] for t in TRAIN_TERRAINS]
    check(g, "打滑噪声是异方差的", max(sig) / min(sig) >= 3,
          f"sigma 从 {min(sig)} (concrete) 到 {max(sig)} (mud), 比值 {max(sig)/min(sig):.1f}x")

    check(g, "cal 与 train 同分布", set(ctx.sources["cal"].terrains) == set(TRAIN_TERRAINS),
          "保形保证的前提: cal 必须与 train 可交换")
    check(g, "cal 与 test 样本不重叠",
          set(ctx.sources["cal"].seed_img).isdisjoint(set(ctx.sources["test"].seed_img)),
          f"cal {len(ctx.sources['cal'])} 条, test {len(ctx.sources['test'])} 条")
    check(g, "OOD 地形训练未见",
          set(OOD_TERRAINS).isdisjoint(set(TRAIN_TERRAINS)),
          f"OOD = {OOD_TERRAINS}, 训练 = {TRAIN_TERRAINS}")

    # ==================================================================
    # 2. 感知层 (冻结)
    # ==================================================================
    g = "2. 感知层 (冻结 DINOv2)"
    check(g, "骨干无可训练参数",
          all(not p.requires_grad for p in ctx.perception.parameters()),
          f"{cfg.backbone}, D={ctx.perception.dim}")

    from sklearn.linear_model import LogisticRegression

    tr, te = ctx.sources["train"], ctx.sources["test"]
    ttr = np.array([tr.terrain_of(i) for i in range(len(tr))])
    tte = np.array([te.terrain_of(i) for i in range(len(te))])
    acc = LogisticRegression(max_iter=2000).fit(ctx.phi["train"].numpy(), ttr).score(
        ctx.phi["test"].numpy(), tte
    )
    check(g, "地形在 phi 空间线性可分", acc > 0.9,
          f"线性探针地形分类准确率 = {acc:.3f}  (信息在, 剩下的就是头会不会用)")

    # ==================================================================
    # 3. 概念层
    # ==================================================================
    g = "3. 概念层 (slip 预测)"
    if modular:
        mu = run_model(modular, ctx.phi["test"], aug.fixed("normal", curves_te), ctx.device).slip_mu
        fid = slip_fidelity(mu, data_te)
        ratio = fid["slip_mae_ratio"]
        check(g, "modular 的 slip MAE 逼近 Bayes 噪声底", ratio <= 1.35,
              f"MAE {fid['slip_mae']:.4f} / Bayes底 {fid['slip_mae_bayes']:.4f} = {ratio:.2f}x "
              f"(阈值 1.35x; 规格里的 <0.06 在噪声底之下, 不可达)")
    if hybrid0 and modular:
        muh = run_model(hybrid0, ctx.phi["test"], aug.fixed("normal", curves_te),
                        ctx.device).slip_mu
        rh = slip_fidelity(muh, data_te)["slip_mae_ratio"]
        check(g, "联合训练会扭曲概念 (预期现象, 非 bug)", None,
              f"hybrid {rh:.2f}x vs modular {ratio:.2f}x —— CE 梯度把 'slip' 拉向利于选档的值")

    # ==================================================================
    # 4. 保形层 —— 硬保证
    # ==================================================================
    g = "4. 保形层 (统计保证)"
    slip_cal = calibrate_slip(modular, ctx, aug, alphas, cfg) if modular else None
    gear_cal = calibrate_gear(e2e, ctx, aug) if e2e else None

    if slip_cal:
        out = run_model(modular, ctx.phi["test"], aug.fixed("normal", curves_te), ctx.device)
        driven = np.arange(len(te)), np.asarray(te.gear, int)
        for a in alphas:
            up = slip_cal.upper(out.slip_mu, a, sigma=out.slip_sigma)[driven]
            cov = float((data_te.slip_driven <= up + 1e-12).mean())
            tol = 3 * np.sqrt(a * (1 - a) / len(te))  # 有限样本波动 3 sigma
            check(g, f"回归覆盖率 @ alpha={a}", cov >= 1 - a - tol,
                  f"实测 {cov:.3f}  >=  目标 {1-a:.2f} - {tol:.3f}(采样波动)")

    if gear_cal:
        for intent, b in RISK_TABLE.items():
            o = run_model(e2e, ctx.phi["test"], aug.fixed(intent, curves_te), ctx.device)
            cov = gear_cal.coverage(o.probs, data_te.oracle(b), b.alpha)
            tol = 3 * np.sqrt(b.alpha * (1 - b.alpha) / len(te))
            check(g, f"APS 覆盖率 @ alpha={b.alpha} ({intent})", cov >= 1 - b.alpha - tol,
                  f"实测 {cov:.3f}  >=  目标 {1-b.alpha:.2f} - {tol:.3f}")

    # ==================================================================
    # 5. 语言层 —— 你指出的那个缺口
    # ==================================================================
    g = "5. 语言层 (VLA 的另一半)"
    tbl = {t: [best_gear(TERRAIN_SLIP[t], b.tau, b.lam) for b in RISK_TABLE.values()]
           for t in TRAIN_TERRAINS + OOD_TERRAINS}
    n_varies = sum(len(set(v)) > 1 for v in tbl.values())
    check(g, "语言真的改变 oracle 标签", n_varies >= 3,
          f"{n_varies}/{len(tbl)} 个地形上, 换一条指令就换一个最优档 "
          + str({t: "".join("SML"[x] for x in v) for t, v in tbl.items()}))

    lam_bites = sum(
        best_gear(TERRAIN_SLIP[t], b.tau, b.lam) != best_gear(TERRAIN_SLIP[t], b.tau, 0.0)
        for t in TERRAIN_SLIP for b in RISK_TABLE.values()
    )
    check(g, "lambda 不是死旋钮", lam_bites >= 1,
          f"lambda 改变了 {lam_bites} 个决策 (旧的原始尺度下恒为 0)")

    if modular:
        by_intent = {}
        for intent in INTENTS:
            b = RISK_TABLE[intent]
            o = run_model(modular, ctx.phi["test"], aug.fixed(intent, curves_te), ctx.device)
            up = slip_cal.upper(o.slip_mu, b.alpha, sigma=o.slip_sigma)
            by_intent[intent] = score_selection(
                select_modular(up, b, slip_point=o.slip_mu), data_te, b)
        m = aggregate(by_intent)
        check(g, "modular 语义方向正确 (小心→大档, 快→小档)",
              m["lang_monotone"] == 1.0 and m["lang_spread"] > 0.05,
              f"接地面积: careful {m['energy_careful']:.3f} >= normal "
              f"{m['energy_normal']:.3f} >= fast {m['energy_fast']:.3f}, 跨度 {m['lang_spread']:.3f}")

    if e2e and hybrid0:
        for name, mdl in (("e2e", e2e), ("hybrid", hybrid0)):
            a = run_model(mdl, ctx.phi["test"], aug.fixed("careful", curves_te), ctx.device)
            c = run_model(mdl, ctx.phi["test"], aug.fixed("fast", curves_te), ctx.device)
            diff = float(np.abs(a.probs - c.probs).mean())
            check(g, f"{name} 确实是 VLA (决策依赖指令)", diff > 1e-3,
                  f"careful vs fast 的档位概率平均差异 = {diff:.4f}")

    if hybrid0:
        a = run_model(hybrid0, ctx.phi["test"], aug.fixed("careful", curves_te), ctx.device)
        c = run_model(hybrid0, ctx.phi["test"], aug.fixed("fast", curves_te), ctx.device)
        same = float(np.abs(a.slip_mu - c.slip_mu).max())
        check(g, "slip 头对语言不变 (物理与语言无关)", same < 1e-6,
              f"careful vs fast 的 slip 预测最大差异 = {same:.2e}  (必须严格为 0)")

    # ==================================================================
    # 6. 决策层 —— 帕累托
    # ==================================================================
    g = "6. 决策层 (对比基线)"
    rows = {}
    if modular:
        rows["modular"] = m
    if e2e:
        bi = {}
        for intent in INTENTS:
            b = RISK_TABLE[intent]
            o = run_model(e2e, ctx.phi["test"], aug.fixed(intent, curves_te), ctx.device)
            sel = select_e2e(gear_cal.sets(o.probs, b.alpha), probs=o.probs, ambiguity="safe")
            bi[intent] = score_selection(sel, data_te, b)
        rows["e2e"] = aggregate(bi)

    dummy = {i: ModelOutputs() for i in INTENTS}
    for p in model_free_baselines():
        bi = {i: score_selection(p.select(dummy[i], data_te, RISK_TABLE[i]), data_te,
                                 RISK_TABLE[i]) for i in INTENTS}
        rows[p.name] = aggregate(bi)

    def dominated(a, b):  # a 是否被 b 支配 (violation 和 energy 都不更好)
        return rows[b]["violation"] <= rows[a]["violation"] and rows[b]["energy"] <= rows[a]["energy"]

    for name in ("modular", "e2e"):
        if name not in rows:
            continue
        bad = [b for b in ("fixed_M", "reactive_only", "random") if dominated(name, b)]
        check(g, f"{name} 不被 固定档/反应式 支配", not bad,
              f"violation {rows[name]['violation']:.3f}, energy {rows[name]['energy']:.3f}"
              + (f"  被 {bad} 支配!" if bad else "  (帕累托前沿上)"))

    check(g, "打滑-能耗权衡的两端 (定标用)", None,
          f"fixed_S: violation {rows['fixed_S']['violation']:.3f} energy 0.000 | "
          f"fixed_L: violation {rows['fixed_L']['violation']:.3f} energy 1.000 | "
          f"oracle: violation {rows['oracle']['violation']:.3f} energy {rows['oracle']['energy']:.3f}")

    # ==================================================================
    # 7. 分析套件
    # ==================================================================
    g = "7. 分析套件 (论文核心)"
    from eval.analysis.leakage import leakage_for

    leak = {}
    for scw in scw_sweep:
        mdl = models.get(("hybrid", scw))
        if mdl:
            leak[scw] = leakage_for(mdl, ctx, aug, "test", seed)

    if 0.0 in leak and max(leak) > 0:
        l0 = leak[0.0]["leakage"]
        l1 = leak[max(leak)]["leakage"]
        ok = (np.isnan(l0) or l0 < 0.35) and (np.isnan(l1) or l1 > 0.65)
        check(g, "leakage: scw=0 是真瓶颈, scw 变大则泄漏", ok,
              "  ".join(f"scw={k:g}→{v['leakage']:.2f}" for k, v in sorted(leak.items())))
        check(g, "  (仅语言基线 = leakage 的分母)", None,
              f"acc_lang_only = {leak[0.0]['acc_lang_only']:.3f} (不是 1/3!), "
              f"acc_full = {leak[0.0]['acc_full']:.3f}")

    if modular:
        from eval.analysis.intervention import intervene

        iv = intervene(modular, ctx, aug, cfg, "test", slip_cal, None)
        check(g, "概念干预能改善决策 (modular 可干预)", iv["gain_oracle"] > 0.05,
              f"gear_acc: 原始 {iv['by_intervention']['none']['gear_acc']:.3f} → "
              f"本体感觉 {iv['by_intervention']['proprio']['gear_acc']:.3f} → "
              f"oracle {iv['by_intervention']['oracle']['gear_acc']:.3f}")
        check(g, "  e2e 无此接口 (结构性局限)", None,
              "e2e 前向里没有物理概念, 实测 slip 无处可写入 —— 只能重训")

    from eval.analysis.probe import probe_report

    pr = probe_report(cfg, ctx, aug, seed)
    d = pr["diagnostics"]
    check(g, "探针实验是否可信 (饱和检测)", None,
          f"地板(未训练头) R²={d['floor_r2_untrained']:.3f}, 天花板(冻结骨干) "
          f"R²={d['ceiling_r2_frozen']:.3f}, headroom={d['headroom']:.3f} "
          + ("→ 已饱和, 该实验在本 mock 上证明不了任何事(代码已警告)"
             if d["saturated"] else "→ 有区分度, 结论可用"))

    # ==================================================================
    # 8. 泛化
    # ==================================================================
    g = "8. 泛化 (OOD / 留出指令)"
    if modular and e2e:
        accs = {}
        for name, mdl in (("modular", modular), ("e2e", e2e)):
            bi = {}
            for intent in INTENTS:
                b = RISK_TABLE[intent]
                o = run_model(mdl, ctx.phi["ood"], aug.fixed(intent, ctx.sources["ood"].slip_curve),
                              ctx.device)
                if name == "modular":
                    up = slip_cal.upper(o.slip_mu, b.alpha, sigma=o.slip_sigma)
                    sel = select_modular(up, b, slip_point=o.slip_mu)
                else:
                    sel = select_e2e(gear_cal.sets(o.probs, b.alpha), probs=o.probs)
                bi[intent] = score_selection(sel, data_ood, b)
            accs[name] = aggregate(bi)
        check(g, "OOD 地形上 modular 优于 e2e",
              accs["modular"]["gear_acc"] > accs["e2e"]["gear_acc"],
              f"gear_acc: modular {accs['modular']['gear_acc']:.3f} vs "
              f"e2e {accs['e2e']['gear_acc']:.3f}")

        # 留出改写句：预算必须由 interpret() 从原始文本推断
        spread = {}
        for name, mdl in (("modular", modular), ("e2e", e2e)):
            bi = {i: [] for i in INTENTS}
            for intent in INTENTS:
                for text in instruction_pool(intent, "heldout"):
                    dec = interpret(text, backend=str(cfg.language.encoder))
                    lang = aug.fixed(intent, curves_te, text=text)
                    o = run_model(mdl, ctx.phi["test"], lang, ctx.device)
                    if name == "modular":
                        up = slip_cal.upper(o.slip_mu, dec.alpha, sigma=o.slip_sigma)
                        sel = select_modular(up, dec, slip_point=o.slip_mu)
                    else:
                        sel = select_e2e(gear_cal.sets(o.probs, dec.alpha), probs=o.probs)
                    bi[intent].append(score_selection(sel, data_te, RISK_TABLE[intent]))
            per = {i: {k: float(np.mean([r[k] for r in v])) for k in v[0] if isinstance(v[0][k], float)}
                   for i, v in bi.items()}
            spread[name] = aggregate(per)["lang_spread"]
        check(g, "留出改写句上 modular 仍然听指令", spread["modular"] > spread["e2e"],
              f"指令跨度(careful能耗 - fast能耗): modular {spread['modular']:.3f} vs "
              f"e2e {spread['e2e']:.3f}   (越接近 0 = 越不听指令)")

    return report()


if __name__ == "__main__":
    sys.exit(main())
