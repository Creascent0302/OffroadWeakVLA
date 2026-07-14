"""Per-mode losses.  One function, switched by `mode` -- no parallel scripts.

    modular : MSE on the slip of the gear that was actually driven.
              This is the ONLY mode whose label is available from a single
              traversal: you drove gear g, you measured slip. Self-supervised.

    e2e     : cross-entropy against the ORACLE best_gear.
              That label needs slip at gears you did NOT drive, i.e.
              counterfactual knowledge. Strictly stronger supervision.

    hybrid  : both. The CE gradient flows through the slip concept into the slip
              head, which is exactly what "jointly trained bottleneck" means.

The sigma term: mu is fitted by MSE alone. Enforcing that takes more than a
`mu.detach()` in the loss -- see models/slip_head.py, which detaches the hidden
state the sigma branch reads, because a shared trunk lets the NLL reach mu's
parameters even when mu itself is detached.
"""

# ============================================================================
# 【中文导读】按 mode 切换的损失。一个函数，不写三份平行脚本。
#   modular : MSE(实际所驾档位的 slip)  ← 唯一只需单次通行就能拿到的标签，真·自监督
#   e2e     : CE(oracle best_gear)     ← 需要没走过的档位的 slip，即反事实监督
#   hybrid  : 两者相加，CE 梯度穿过概念回流到 slip 头（这才叫“联合训练的瓶颈”）
#
#   sigma 的 NLL：mu 只由 MSE 训练。但光在损失里写 mu.detach() 是【不够的】——
#   mu 和 log_sigma 共享 trunk，梯度会绕回去。真正的隔离在 models/slip_head.py 里
#   （sigma 分支读的是 detach 过的隐状态）。
# ============================================================================


from __future__ import annotations

import torch
import torch.nn.functional as F


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B,3], idx [B] -> [B]"""
    return x.gather(1, idx.long().view(-1, 1)).squeeze(1)


def slip_loss(out: dict, batch: dict, sigma_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    gear = batch["gear"]
    y = batch["slip"].to(out["slip"].dtype)

    mu = _gather(out["slip"], gear)
    mse = F.mse_loss(mu, y)

    logs = {"slip_mse": float(mse.detach()), "slip_mae": float((mu - y).abs().mean().detach())}
    total = mse

    if sigma_weight > 0.0 and "slip_sigma" in out:
        sigma = _gather(out["slip_sigma"], gear)

        # The NLL is fitted against the DROPOUT-FREE mean, not the dropout-active
        # one. With dropout active the residual (y - mu) carries
        # terrain_noise + dropout_noise, and sigma would absorb both -- inflating
        # the conformal bound worst on exactly the low-noise terrain where a tight
        # bound is the whole point. `slip_clean` already carries no gradient, and
        # `slip_sigma` is computed from a detached hidden state, so this term
        # cannot move mu. See models/slip_head.py.
        target_mu = _gather(out.get("slip_clean", out["slip"]), gear).detach()
        nll = F.gaussian_nll_loss(target_mu, y, sigma.pow(2), full=False, reduction="mean")
        total = total + sigma_weight * nll
        logs["sigma_nll"] = float(nll.detach())
        logs["sigma_mean"] = float(sigma.mean().detach())

    return total, logs


def gear_loss(out: dict, lang: dict) -> tuple[torch.Tensor, dict]:
    logits = out["gear_logits"]
    target = lang["best_gear"].to(logits.device)
    ce = F.cross_entropy(logits, target)
    acc = (logits.argmax(-1) == target).float().mean()
    return ce, {"gear_ce": float(ce.detach()), "gear_acc": float(acc.detach())}


def compute_loss(
    mode: str,
    out: dict,
    batch: dict,
    lang: dict,
    beta: float = 1.0,
    sigma_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Returns (loss to optimise, logs).

    logs["monitor"] is what early stopping watches, and it is NOT the total loss.
    The Gaussian NLL that fits sigma is around -2 while the slip MSE is around
    0.008, so the total is dominated by a term that says nothing about how well
    the model predicts slip.  Early stopping on it halts training while the actual
    task is still improving.  `monitor` is the task loss with the NLL removed.
    """
    if mode == "modular":
        loss, logs = slip_loss(out, batch, sigma_weight)
        logs["monitor"] = logs["slip_mse"]
        return loss, logs

    if mode == "e2e":
        loss, logs = gear_loss(out, lang)
        logs["monitor"] = logs["gear_ce"]
        return loss, logs

    if mode == "hybrid":
        ce, l1 = gear_loss(out, lang)
        sl, l2 = slip_loss(out, batch, sigma_weight)
        logs = {**l1, **l2}
        logs["monitor"] = l1["gear_ce"] + beta * l2["slip_mse"]
        return ce + beta * sl, logs

    raise ValueError(f"unknown mode: {mode}")
