import logging
import os, tempfile
from pathlib import Path

import hydra
import torch

torch.set_float32_matmul_precision("medium")
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.models import get_builder
from src.utils import DataCompose

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="finetune_multistep_pangu")
def main(cfg: DictConfig) -> None:
    hydra_oup_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) # type: ignore
    log.info(f"Working directory: {Path.cwd()}")
    log.info(f"Output directory: {hydra_oup_dir}")

    # lightning ddp strategy doesn't need manual seed
    # https://github.com/Lightning-AI/pytorch-lightning/issues/12986
    # seed_everything(1000)

    # prevent access to non-existing keys
    OmegaConf.set_struct(cfg, True)

    # prepare data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, **cfg.data, **cfg.lightning)
    data_manager.setup("test")

    # model
    model_builder = get_builder(cfg.model.model_name)(
        hydra_oup_dir,
        data_list,
        image_shape=data_manager.image_shape,
        **cfg.model,
        **cfg.lightning,
    )
    model = model_builder.build_model(data_manager.test_dataloader(), datamodule=data_manager)

    # trainer
    wandb_logger = model_builder.wandb_logger()
    wandb_logger.watch(model, log="all")
    trainer = model_builder.build_trainer(wandb_logger)

    r  = int(os.environ.get("RANK", -1))
    lr = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[pre-fit rank {r}] local_rank={lr} current_device={torch.cuda.current_device()}", flush=True)
    print("SLURM_PROCID", os.environ.get("SLURM_PROCID"))
    print("SLURM_LOCALID", os.environ.get("SLURM_LOCALID"))
    print("SLURM_NTASKS", os.environ.get("SLURM_NTASKS"))
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.current_device()", torch.cuda.current_device())

    # Directly inspect the backbone.
    p = next(model.backbone_model.parameters(), None)
    b = next(model.backbone_model.buffers(), None)
    if p is not None:
        print(f"[pre-fit rank {r}] backbone first param device = {p.device}", flush=True)
    if b is not None:
        print(f"[pre-fit rank {r}] backbone first buffer device = {b.device}", flush=True)

    # start training
    trainer.fit(
        model,
        data_manager,
        ckpt_path=getattr(cfg.lightning, "resume_from_checkpoint", None),
        weights_only=False,
    )


if __name__ == "__main__":
    print("TMPDIR env:", os.environ.get("TMPDIR"))
    print("tempfile.gettempdir():", tempfile.gettempdir())
    print("exists:", os.path.exists(tempfile.gettempdir()))
    main()
