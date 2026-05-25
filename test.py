import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch

torch.set_float32_matmul_precision("medium")
OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])
OmegaConf.register_new_resolver("eval", lambda x: eval(x))


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    """
    Entry point for training the model.
    """
    if cfg.get("model") is None:
        raise ValueError("Model configuration is missing, please specify a model to train.")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    loggers: list[Logger] = [hydra.utils.instantiate(logger) for logger in cfg.get("loggers", {}).values()]

    # Log hyperparameters
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    # Initialize the model
    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: list[L.Callback] = [hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()]

    # Initialize the trainer
    trainer: L.Trainer = hydra.utils.instantiate(cfg.get("trainer"), callbacks=callbacks, logger=loggers)
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=False)


if __name__ == "__main__":
    main()
