from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train CrossKey from a Hydra configuration.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Resolved training configuration.
    """
    seed_everything(cfg.seed, workers=True)

    run_dir = Path(HydraConfig.get().run.dir).resolve()
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = hydra.utils.instantiate(cfg.logger, save_dir=run_dir)

    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    main()
