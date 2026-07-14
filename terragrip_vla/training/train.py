"""Training entry point.

    python -m training.train                          # modular, seed 0
    python -m training.train mode=e2e train.seed=1
    python -m training.train mode=hybrid side_channel_weight=0.5

`run_training(cfg)` is the same code path, callable in-process, which is how
eval/compare.py trains the whole grid without shelling out.
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from data.dataset import LanguageAugmenter
from runtime import (
    build_context,
    build_model,
    checkpoint_path,
    save_model,
    set_seed,
)
from training.trainer import Trainer


class _WandbLogger:
    def __init__(self, cfg: DictConfig, name: str):
        import wandb

        self.run = wandb.init(
            project=cfg.wandb.project, name=name, config=OmegaConf.to_container(cfg, resolve=True)
        )

    def log(self, d: dict) -> None:
        self.run.log(d)

    def finish(self) -> None:
        self.run.finish()


def run_training(cfg: DictConfig, ctx=None, progress: bool = True) -> dict:
    """Train one (mode, scw, seed) and write its checkpoint.  Returns a summary."""
    mode = str(cfg.mode)
    scw = float(cfg.side_channel_weight)
    seed = int(cfg.train.seed)

    ctx = ctx or build_context(cfg)

    # Seed AFTER the context is built. build_context loads DINOv2 through
    # torch.hub, which consumes the global torch RNG -- so seeding before it made
    # the head's initial weights depend on whether the backbone happened to be
    # cached, i.e. the same train.seed gave different models on different machines.
    set_seed(seed)
    model = build_model(cfg, ctx)
    # The language augmenter is seeded per run, so different seeds see different
    # instruction draws -- language augmentation is part of what a seed varies.
    aug = ctx.augmenter(str(cfg.language.paraphrase_split), seed=seed)

    logger = None
    if bool(cfg.wandb.enabled):
        logger = _WandbLogger(cfg, name=f"{mode}_scw{scw:g}_seed{seed}")

    trainer = Trainer(model, cfg, aug, ctx.device, logger=logger)
    hist = trainer.fit(ctx.dataset("train"), progress=progress)

    path = checkpoint_path(cfg, mode, scw, seed)
    save_model(
        model,
        path,
        cfg,
        extra={"seed": seed, "best_epoch": hist.best_epoch, "best_val": hist.best_val},
    )

    if logger is not None:
        logger.finish()

    summary = {
        "mode": mode,
        "scw": scw,
        "seed": seed,
        "epochs_run": len(hist.train),
        "best_epoch": hist.best_epoch,
        "best_val_loss": hist.best_val,
        "final_train": hist.train[-1] if hist.train else {},
        "final_val": hist.val[hist.best_epoch] if hist.val else {},
        "checkpoint": str(path),
    }
    if progress:
        print(json.dumps(summary, indent=2, default=float))
    return summary


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
