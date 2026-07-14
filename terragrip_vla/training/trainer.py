"""Training loop.  Identical for all three modes; only the loss differs.

Two details that matter for the science:

1. The validation split is carved out of TRAIN, never from `cal`.  `cal` is used
   for nothing but conformal calibration.  If early stopping peeked at `cal`, the
   calibration residuals would no longer be exchangeable with test and the
   coverage guarantee would quietly stop holding.

2. Validation instructions are drawn ONCE and reused every epoch, so the val loss
   is comparable across epochs (the training instructions are re-drawn every
   epoch -- that is the language augmentation).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from data.dataset import LanguageAugmenter, make_loader
from models.model import TerraGripModel, language_features
from runtime import set_seed, to_device
from training.losses import compute_loss


@dataclass
class History:
    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")


def split_train_val(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(val_frac * n)))
    return perm[n_val:], perm[:n_val]


class Trainer:
    def __init__(self, model: TerraGripModel, cfg, aug: LanguageAugmenter, device, logger=None):
        self.model = model
        self.cfg = cfg
        self.aug = aug
        self.device = device
        self.logger = logger
        self.mode = model.mode
        self.lang_condition = model.lang_condition

        self.opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
        self.use_amp = device.type == "cuda" and str(cfg.train.precision) == "bf16"

    # ------------------------------------------------------------------
    def _forward_loss(self, batch: dict, lang: dict) -> tuple[torch.Tensor, dict]:
        phi = batch["phi"].to(self.device)
        lang = to_device(lang, self.device)
        batch = {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        out = self.model(phi=phi, lang=language_features(lang, self.lang_condition, self.device))
        return compute_loss(
            self.mode,
            out,
            batch,
            lang,
            beta=float(self.cfg.train.beta),
            sigma_weight=float(self.cfg.train.sigma_weight),
        )

    def _epoch(self, loader, train: bool, fixed_lang: list[dict] | None = None) -> dict:
        self.model.train(train)
        totals: dict[str, float] = {}
        n_batches = 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for i, batch in enumerate(loader):
                lang = (
                    fixed_lang[i]
                    if fixed_lang is not None
                    else self.aug.random(batch["slip_curve"].numpy())
                )
                if self.use_amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, logs = self._forward_loss(batch, lang)
                    loss = loss.float()
                else:
                    loss, logs = self._forward_loss(batch, lang)

                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 5.0)
                    self.opt.step()

                logs["loss"] = float(loss.detach())
                for k, v in logs.items():
                    totals[k] = totals.get(k, 0.0) + v
                n_batches += 1

        return {k: v / max(1, n_batches) for k, v in totals.items()}

    # ------------------------------------------------------------------
    def fit(self, dataset, val_frac: float = 0.1, progress: bool = True) -> History:
        cfg = self.cfg
        set_seed(int(cfg.train.seed))

        tr_idx, va_idx = split_train_val(len(dataset), val_frac, int(cfg.train.seed))
        train_loader = make_loader(
            Subset(dataset, tr_idx), int(cfg.train.batch_size), True, int(cfg.train.seed)
        )
        val_loader = make_loader(Subset(dataset, va_idx), int(cfg.train.batch_size), False)

        # Freeze the validation instructions once: a moving target would make the
        # early-stopping signal noisy for reasons unrelated to the model.
        val_aug = LanguageAugmenter(self.aug.encoder, self.aug.paraphrase_split, seed=12345)
        fixed_lang = [val_aug.random(b["slip_curve"].numpy()) for b in val_loader]

        hist = History()
        best_state = copy.deepcopy(self.model.state_dict_heads())
        bad_epochs = 0

        bar = range(int(cfg.train.epochs))
        if progress:
            bar = tqdm(bar, desc=f"train[{self.mode}]", leave=False)

        for epoch in bar:
            tr = self._epoch(train_loader, train=True)
            va = self._epoch(val_loader, train=False, fixed_lang=fixed_lang)
            hist.train.append(tr)
            hist.val.append(va)

            if self.logger is not None:
                self.logger.log(
                    {**{f"train/{k}": v for k, v in tr.items()},
                     **{f"val/{k}": v for k, v in va.items()},
                     "epoch": epoch}
                )

            # Watch the TASK loss, not the total: see losses.compute_loss.
            score = va.get("monitor", va["loss"])
            if score < hist.best_val - 1e-5:
                hist.best_val = score
                hist.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict_heads())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(cfg.train.patience):
                    break

        self.model.load_state_dict_heads(best_state)  # always return the best, not the last
        self.model.eval()
        return hist
