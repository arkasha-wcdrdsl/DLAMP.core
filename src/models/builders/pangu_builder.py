from os import environ
from pathlib import Path

import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch.utils.data import DataLoader

from ...const import CHECKPOINT_DIR
from ...utils import DataCompose, convert_hydra_dir_to_timestamp
from .. import PanguModel
from ..callbacks import LogPredictionSamplesCallback
from ..lightning_modules import PanguLightningModule
from .base_builder import BaseBuilder

__all__ = ["PanguBuilder"]


class PanguBuilder(BaseBuilder):
    def __init__(self, hydra_dir: Path, data_list: list[DataCompose], **kwargs):
        super().__init__(**kwargs)

        if "loss" not in self.kwargs:
            self.kwargs.loss = "L1"

        self.multi_step_forecast: int = 1
        self.model_native_interval_h: int = 1
        self.add_time_features = True
        self.detach_every_step = False
        self.sequence_mode = False

        if "multi_step_forecast" in kwargs:
            self.multi_step_forecast: int = kwargs['multi_step_forecast']
            assert "output_itv" in kwargs and kwargs['output_itv'] is not None, \
                "output_itv must be provided to Datamodule in kwargs for multi-step forecast."
            oup_itv_h = getattr(kwargs['output_itv'], "hours", 1)
            self.model_native_interval_h: int = getattr(kwargs['model_native_interval'], "hours", 1)
            assert self.multi_step_forecast == oup_itv_h//self.model_native_interval_h, \
                f"multi_step_forecast ({kwargs['multi_step_forecast']}) must be equal to output_itv hours ({kwargs['output_itv']['hours']}) divided by model_native_interval hours ({kwargs['model_native_interval']['hours']})"
            if "add_time_features" in kwargs:
                self.add_time_features = kwargs['add_time_features']
            if self.multi_step_forecast > 1:
                self.info_log(f"Will do multi-step forward: {self.multi_step_forecast} steps with interval of {self.model_native_interval_h} hours")
            self.detach_every_step = kwargs.get("detach_every_step", False)
            self.sequence_mode = kwargs.get("sequence_mode", False)

        self.pressure_levels: list = DataCompose.get_all_levels(
            data_list, only_upper=True, to_str=True
        )
        self.upper_vars: list = DataCompose.get_all_vars(
            data_list, only_upper=True, to_str=True
        )
        self.surface_vars: list = DataCompose.get_all_vars(
            data_list, only_surface=True, to_str=True
        )

        self.time_stamp = convert_hydra_dir_to_timestamp(hydra_dir)

        self.info_log(f"Input Image Shape: {self.kwargs.image_shape}")
        self.info_log(f"Patch Size: {self.kwargs.patch_size}")
        self.info_log(f"Window Size: {self.kwargs.window_size}")

    def _backbone_model(self) -> nn.Module:
        sfc_input_ch = (
            len(self.surface_vars) + 4
            if self.kwargs.add_time_features
            else len(self.surface_vars)
        )
        return PanguModel(
            image_shape=self.kwargs.image_shape,
            patch_size=self.kwargs.patch_size,
            window_size=self.kwargs.window_size,
            upper_levels=len(self.pressure_levels),
            upper_channels=len(self.upper_vars),
            surface_input_channels=sfc_input_ch,
            surface_output_channels=len(self.surface_vars),
            embed_dim=self.kwargs.embed_dim,
            heads=self.kwargs.heads,
            depths=self.kwargs.depths,
            max_drop_path_ratio=self.kwargs.max_drop_path_ratio,
            dropout_rate=self.kwargs.dropout_rate,
            smoothing_kernel_size=self.kwargs.smoothing_kernel_size,
            segmented_smooth_boundary_width=self.kwargs.segmented_smooth_boundary_width,
            post_process_size=self.kwargs.post_process_size,
        )

    def build_model(
        self,
        test_dataloader: DataLoader | None = None,
        predict_iters: int | None = None,
        datamodule: LightningDataModule | None = None,
    ) -> LightningModule:
        if self.kwargs.finetune_with_checkpoint is not None:
            return PanguLightningModule.load_from_checkpoint(
                self.kwargs.finetune_with_checkpoint,
                weights_only=False,
                strict=False,
                test_dataloader=test_dataloader,
                backbone_model=self._backbone_model(),
                datamodule=datamodule,
                upper_var_weights=None,
                surface_var_weights=None,
                surface_alpha=self.kwargs.surface_alpha,
                pressure_levels=self.pressure_levels,
                upper_vars=self.upper_vars,
                surface_vars=self.surface_vars,
                optim_config=self.kwargs.optim_config,
                lr_schedule=self.kwargs.lr_schedule,
                loss=self.kwargs.loss,
                image_shape=self.kwargs.image_shape,
                predict_iters=predict_iters,
                multi_step_forecast=self.multi_step_forecast,
                model_native_interval_h=self.model_native_interval_h,
                detach_every_step=self.detach_every_step,
                sequence_mode = self.sequence_mode,
            )
        return PanguLightningModule(
            test_dataloader=test_dataloader,
            backbone_model=self._backbone_model(),
            datamodule=datamodule,
            upper_var_weights=None,
            surface_var_weights=None,
            surface_alpha=self.kwargs.surface_alpha,
            pressure_levels=self.pressure_levels,
            upper_vars=self.upper_vars,
            surface_vars=self.surface_vars,
            optim_config=self.kwargs.optim_config,
            lr_schedule=self.kwargs.lr_schedule,
            loss=self.kwargs.loss,
            image_shape=self.kwargs.image_shape,
            predict_iters=predict_iters,
            multi_step_forecast=self.multi_step_forecast,
            model_native_interval_h=self.model_native_interval_h,
            detach_every_step=self.detach_every_step,
            sequence_mode = self.sequence_mode,
        )

    def build_trainer(self, logger) -> Trainer:
        num_gpus = (
            torch.cuda.device_count()
            if self.kwargs.num_gpus is None
            else self.kwargs.num_gpus
        )
        strategy = getattr(self.kwargs, "strategy", "auto")

        callbacks: list[Callback] = []
        callbacks.append(LearningRateMonitor())
        callbacks.extend(self.checkpoint_callback())
        if self.kwargs.log_image_every_n_steps or self.kwargs.log_image_every_n_epoch:
            callbacks.append(
                LogPredictionSamplesCallback(
                    self.kwargs.log_image_every_n_steps,
                    self.kwargs.log_image_every_n_epoch,
                    self.sequence_mode,
                    self.multi_step_forecast,
                    self.model_native_interval_h,
                )
            )
        if self.kwargs.early_stop_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss_epoch", patience=self.kwargs.early_stop_patience
                )
            )

        node_count = environ.get('SLURM_NNODES')
        if node_count is not None:
            node_count = int(node_count)
        else:
            node_count = 1
        print("trainer config:")
        print(f"  slurm_nodes: {node_count}")
        print(f"  num_gpus: {num_gpus}")
        print(f"  strategy: {strategy}")

        return Trainer(
            # accumulate_grad_batches=4,
            # gradient_clip_val=10,
            # gradient_clip_algorithm="norm",
            num_sanity_val_steps=2,
            benchmark=True,
            fast_dev_run=self.kwargs.fast_dev_run,  # use n batch(es) to fast run through train/valid, no checkpoint, no max_epoch
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=self.kwargs.log_every_n_steps,  # only affect train_loss
            # -1: infinite epochs, None: default 1000 epochs
            max_epochs=getattr(self.kwargs, "max_epochs", None),
            # If min_steps > 0, max_epoch must be valid. And min_steps is prior to early stopping
            min_steps=getattr(self.kwargs, "min_steps", -1),
            limit_train_batches=getattr(self.kwargs, "limit_train_batches", None),
            limit_val_batches=getattr(self.kwargs, "limit_val_batches", None),
            accelerator="gpu",
            num_nodes=node_count,
            devices=num_gpus,
            strategy=strategy,
            callbacks=callbacks,
            # profiler=AdvancedProfiler(
            #     dirpath="./profiler", filename=f"{self.__class__.__name__}"
            # ),
            precision=self.kwargs.precision,
        )

    def checkpoint_callback(self) -> list[ModelCheckpoint]:
        metric_string = '{epoch:03d}-{val_loss_epoch:.4f}'
        model_label = self.kwargs.get("model_label", "")
        if model_label:
            model_label = f"_{model_label}"
        filename=f"{self.kwargs.model_name}{model_label}_{self.time_stamp}-{metric_string}"
        callbacks = [
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                filename=filename,
                save_top_k=5,
                verbose=True,
                monitor="val_loss_epoch",
                mode="min",
                save_last=False,
            )
        ]
        if self.kwargs.save_last:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=CHECKPOINT_DIR,
                    filename=filename,
                    verbose=True,
                    save_last='link',
                    monitor=None,
                )
            )
        return callbacks


    def wandb_logger(self, save_dir: str = "./logs") -> WandbLogger:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return WandbLogger(
            save_dir=save_dir,
            log_model=False,  # log W&B artifacts
            project="my-awesome-project",
            name=self.kwargs.model_name + f"_{self.time_stamp}",
            offline=True,
        )
