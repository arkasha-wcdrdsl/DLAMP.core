import math
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.grads import grad_norm
from torch.utils.data import DataLoader
from tqdm import trange
from datetime import datetime, timedelta
import logging

from src.utils import TimeUtil, group_tensor_pairs, ungroup_tensor_pairs
from ..model_utils import get_scheduler_with_warmup

__all__ = ["PanguLightningModule"]

log = logging.getLogger(__name__)


# TODO: weighted MAE loss
class PanguLightningModule(L.LightningModule):
    def __init__(
            self, *, test_dataloader, backbone_model, loss="L1", datamodule=None,
            image_shape: tuple[int, int]|None = None,
            multi_step_forecast: int = 1,
            addtime_features: bool = True,
            model_native_interval_h: int = 1,
            **kwargs
        ):
        super().__init__()

        self.image_shape = image_shape
        self.multi_step_forecast = multi_step_forecast
        self.add_time_features = addtime_features
        self.model_native_interval = timedelta(hours=model_native_interval_h)
        self.detach_every_step = kwargs.get("detach_every_step", False)
        self.sequence_mode = kwargs.get("sequence_mode", False)

        assert (self.sequence_mode or multi_step_forecast < 2 or not addtime_features) or image_shape, f"Must provide image_shape when multi_step_forecast>1 and addtime_features is True."

        # generate decaying weight for multi-horizon losses
        # which is like [0.67, 0.33] for 2-step, [0.5, 0.33, 0.17] for 3-step, and sum to 1
        self.step_weights = [2*(multi_step_forecast-i)/(multi_step_forecast**2+multi_step_forecast) for i in range(multi_step_forecast)]

        if datamodule is not None:
            self.train_size = getattr(datamodule, "train_size", None)
            self.batch_size = getattr(datamodule, "batch_size", None)

        self.save_hyperparameters(ignore=["test_dataloader", "backbone_model", "datamodule"])
        if datamodule is not None:
            del datamodule

        self._test_dataloader: DataLoader = test_dataloader
        self.backbone_model: nn.Module = backbone_model

        if kwargs["upper_var_weights"] is None or kwargs["surface_var_weights"] is None:
            self.weighted_loss = False
            if loss == "L1":
                self.criterion = nn.L1Loss(reduction="mean")
            elif loss == "RMSLAE":
                from src.models.loss_fn.rmslae import RootMeanSquaredLogAbsoluteError
                self.criterion = RootMeanSquaredLogAbsoluteError()
            elif loss == "MAECRPS":
                from src.models.loss_fn.spl import MAECRPS
                self.criterion = MAECRPS(do_mae=True)
            elif loss == "SPECTRUM":
                from src.models.loss_fn.spectrum import SpectrumLoss
                self.criterion = SpectrumLoss(num_bins=16, H=224, W=224)
            else:
                raise ValueError(f"Unsupported loss: {loss}")
            upper_var_weights_tensor = None
            surface_var_weights_tensor = None
        self.register_buffer("upper_var_weights", upper_var_weights_tensor)
        self.register_buffer("surface_var_weights", surface_var_weights_tensor)


    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Performs a forward pass.

        The model consumes a flat sequence of tensors that represent one or more
        (upper_air, surface) pairs. The inputs must alternate in the following order:

            upper_air_0, surface_0,
            upper_air_1, surface_1,
            ...

        When operating in non-sequence mode, exactly one pair is expected
        (i.e., two tensors). When operating in sequence mode, multiple pairs may be
        provided. In all cases, the number of input tensors must be even.

        Args:
            *inputs (torch.Tensor):
                Variable-length sequence of input tensors alternating between
                "upper_air" and "surface" for each step.

                Expected shapes per step:
                    - upper_air: Tensor of shape [B, Z, H, W, C]
                    - surface:   Tensor of shape [B, 1, H, W, C]

        Returns:
            tuple[torch.Tensor, ...]:
                A flat tuple of output tensors alternating between "upper_air" and
                "surface" for each forecast step, preserving the same ordering:

                    upper_air_out_0, surface_out_0,
                    upper_air_out_1, surface_out_1,
                    ...

                The number of returned tensors is always even.
        """
        if len(inputs) % 2 != 0:
            raise ValueError(
                f"Expected even number of input tensors (upper/surface pairs), "
                f"but got {len(inputs)}."
            )

        actual_steps = len(inputs) // 2
        # sanity_checking = bool(getattr(self, "trainer", None)) and bool(getattr(self.trainer, "sanity_checking", False))
        assert actual_steps == self.multi_step_forecast or not self.sequence_mode, f"Number of input pairs ({actual_steps}) does not match multi_step_forecast ({self.multi_step_forecast}) when sequence_mode is True."

        data = group_tensor_pairs(*inputs)

        input_upper = inputs[0]
        input_surface = inputs[1]
        outputs: list[torch.Tensor] = []

        log.debug(f"model device: {self.device}")
        log.debug(f"Surface Data type: {type(input_surface)}, device: {input_surface.device}, shape: {input_surface.shape}")
        log.debug(f"Upper Data type: {type(input_upper)}, device: {input_upper.device}, shape: {input_upper.shape}")

        if not torch.is_tensor(input_upper):
            log.warning(f"Converting input_upper to tensor...")
            input_upper = torch.as_tensor(input_upper).to(self.device)
        if not torch.is_tensor(input_surface):
            log.warning(f"Converting input_surface to tensor...")
            input_surface = torch.as_tensor(input_surface).to(self.device)

        # recover datetime from input_surface if time features are added
        if not self.sequence_mode and self.add_time_features and self.multi_step_forecast > 1:
            # check condictions
            log.debug(f"add_time_features: {self.add_time_features}, multi_step_forecast: {self.multi_step_forecast}")
            time_feature: torch.Tensor = input_surface[...,-4:]
            input_dt: list[datetime] = TimeUtil.recover_datetime_from_time_features(time_feature.cpu().numpy())
            log.debug(f"Recovered datetime from input_surface: {input_dt}")

        for step in range(self.multi_step_forecast):
            log.debug(f"Forecast step {step+1}/{self.multi_step_forecast} starting...")

            if step > 0:
                if self.detach_every_step:
                    input_upper = input_upper.detach()
                    input_surface = input_surface.detach()

                # add DoY and ToD (1, lv=1, h, w, c+4)
                if self.add_time_features:
                    if self.sequence_mode:
                        # directly use the time features from input for each step if in sequence_mode and input sequence length is correct, to avoid error accumulation from time feature recovery
                        time_features = data[step]["surface"][...,-4:]
                    else:
                        assert input_dt is not None, "input_dt must be provided for recover time features"
                        log.debug(f"Updated datetime for step {step+1}: {input_dt}")
                        assert self.image_shape is not None, "image_shape must be provided for recover time features"
                        # increment time for the next step
                        input_dt = [dt+self.model_native_interval for dt in input_dt]
                        time_features = torch.tensor(TimeUtil.create_time_features(input_dt, self.image_shape)).to(self.device)
                    log.debug(f"Time features shape: {time_features.shape}")
                    log.debug(f"Input surface shape: {input_surface.shape}")
                    input_surface = torch.cat([input_surface, time_features], dim=-1)

            output_upper: torch.Tensor
            output_surface: torch.Tensor
            output_upper, output_surface = self.backbone_model(input_upper, input_surface)

            if self.sequence_mode:
                outputs.extend([output_upper, output_surface])

            input_upper, input_surface = output_upper, output_surface
            # log.debug(f"[DEBUG] Forecast step {step+1}/{self.multi_step_forecast} done.")
            # log.debug(f"[DEBUG] Output upper shape: {output_upper.shape}, surface shape: {output_surface.shape}")
        if not outputs:
            outputs.extend([output_upper, output_surface])
        return tuple(outputs)


    def _infer_total_train_steps(self) -> int:
        trainer = self.trainer
        if trainer is None:
            return 0

        # 1) If max_steps is explicitly specified, use it directly (most robust).
        if getattr(trainer, "max_steps", -1) not in (-1, None):
            return int(trainer.max_steps)

        # 2) Otherwise estimate it from the dataset size and effective batch size.
        # train_size and batch_size must be available from the DataManager.
        # If they come from the DataManager, expose them through hparams or the datamodule.
        if self.train_size is None or self.batch_size is None:
            msg = f"Need train_size and batch_size to infer total steps without touching dataloader,\nwhich train_size={self.train_size}, batch_size={self.batch_size}"
            raise RuntimeError(msg)

        world_size = getattr(trainer, "world_size", 1)
        accum = getattr(trainer, "accumulate_grad_batches", 1)
        max_epochs = getattr(trainer, "max_epochs", 1)

        effective_batch = self.batch_size * world_size * accum
        steps_per_epoch = math.ceil(self.train_size / effective_batch)

        # Apply limit_train_batches here when it is configured as a float or integer.
        limit = getattr(trainer, "limit_train_batches", 1.0)
        if isinstance(limit, float):
            steps_per_epoch = math.ceil(steps_per_epoch * limit)
        elif isinstance(limit, int):
            steps_per_epoch = min(steps_per_epoch, limit)

        return int(steps_per_epoch * max_epochs)


    def configure_optimizers(self):
        # set optimizer
        optimizer = getattr(torch.optim, self.hparams['optim_config'].name)(
            self.parameters(), **self.hparams['optim_config'].args
        )

        total_steps = self._infer_total_train_steps()  # Use the robust estimate.

        # set learning rate schedule
        lr_schedule_name = self.hparams['lr_schedule'].name  # "cosine"
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR = get_scheduler_with_warmup(
            optimizer,
            schedule_type=lr_schedule_name,
            training_steps=total_steps,
            **self.hparams['lr_schedule'].args,  # warmup_steps=1000
        )
        @rank_zero_only
        def _print():
            print(f'total_steps used: {total_steps}')
        _print()

        interval = "epoch" if lr_schedule_name in ["linear_decay"] else "step"
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": interval,
            "frequency": 1,
            "name": "customized_lr",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


    def on_fit_start(self) -> None:
        @rank_zero_only
        def _print():
            log.debug(f'self.global_step = {self.global_step}')      # Expected to be 0 during fine-tuning.
            log.debug(f'self.current_epoch = {self.current_epoch}')  # Expected to be 0 during fine-tuning.
            w: torch.Tensor = self.backbone_model.patch_recover.conv_upper.weight # type: ignore

            log.debug(f"[CHECK] weight mean: {w.mean().item()}")
            log.debug(f"[CHECK] weight std : {w.std().item()}")
        _print()
        return super().on_fit_start()


    def _calc_total_loss(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the total loss for a single forecast step.

        Args:
            output (dict): Model output containing 'upper_air' and 'surface'.
            target (dict): Target data containing 'upper_air' and 'surface'.

        Returns:
            torch.Tensor: The combined loss value.
        """
        if self.weighted_loss:
            raise NotImplementedError("Weighted loss is not implemented yet.")

        loss_upper = self.criterion(output["upper_air"], target["upper_air"])
        # loss_upper = var_loss_upper.mean()
        # remove time feature from target["surface"] if exists
        if output["surface"].shape[-1] != target["surface"].shape[-1]:
            target["surface"] = target["surface"][...,:-4]
        loss_surface = self.criterion(output["surface"], target["surface"])
        # loss_surface = var_loss_surface.mean()

        # self._temp_var_upper = var_loss_upper
        # self._temp_var_surface = var_loss_surface

        return loss_upper + loss_surface * self.hparams['surface_alpha']


    def common_step(self, *data: dict[str, torch.Tensor]):
        """
        Runs a shared step used by training/validation/testing.

        This method typically performs:
            1) forward pass to obtain model outputs,
            2) loss computation (e.g., total loss),
            3) auxiliary metrics (e.g., MAE) per component.

        Each argument in `data` represents one step and must contain both keys
        "upper_air" and "surface".

        Args:
            *data (dict[str, torch.Tensor]):
                Variable-length sequence of step dictionaries with the structure:

                    {
                        "upper_air": Tensor of shape [B, Z, H, W, C],
                        "surface":   Tensor of shape [B, 1, H, W, C],
                    }
        """
        # all data in the shape of (B, Z, H, W, C)
        assert len(data) == self.multi_step_forecast+1 or not self.sequence_mode, f"Expected number of input data dictionaries is ##{self.multi_step_forecast+1}## when sequence_mode is True, but got ##{len(data)}##. Check your config."
        outputs_flat: tuple[torch.Tensor, ...] = self(*ungroup_tensor_pairs(*data[:-1]))
        outputs = group_tensor_pairs(*outputs_flat)
        assert len(outputs) == self.multi_step_forecast or not self.sequence_mode, f"Expected number of outputs from the model is ##{self.multi_step_forecast}## when sequence_mode is True, but got ##{len(outputs)}##. Check your config."

        # generate decaying weight for multi-step losses, sum to 1
        total_loss = sum([self._calc_total_loss(outputs[i], data[i+1]) * self.step_weights[i] for i in range(len(outputs))])
        oup_upper, oup_surface = outputs[-1]['upper_air'], outputs[-1]['surface']
        # var_loss_upper = self._temp_var_upper
        # var_loss_surface = self._temp_var_surface

        return total_loss, (oup_upper, oup_surface)#, (var_loss_upper, var_loss_surface)#, (mae_upper, mae_surface)

    def training_step(self, batch, batch_idx):
        # inp_data, target = batch
        loss, _ = self.common_step(*batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        # self.log_mae_for_each_element(
        #     "train", self.hparams['pressure_levels'], self.hparams['upper_vars'], mae_upper
        # )
        # self.log_mae_for_each_element(
        #     "train", ["Surface"], self.hparams['surface_vars'], mae_surface
        # )
        return loss


    def validation_step(self, batch, batch_idx):
        # inp_data, target = batch
        loss, _ = self.common_step(*batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        # self.log_mae_for_each_element(
        #     "val", self.hparams['pressure_levels'], self.hparams['upper_vars'], mae_upper
        # )
        # self.log_mae_for_each_element(
        #     "val", ["Surface"], self.hparams['surface_vars'], mae_surface
        # )
        # log crps for every variables
        return loss


    def predict_step(self, batch, batch_idx):
        inp_data, target = batch
        for _ in trange(self.hparams['predict_iters'], desc=f"Predict batch {batch_idx}"):
            upper, surface = self(inp_data["upper_air"], inp_data["surface"])[-2:]
        return (
            inp_data["upper_air"],
            inp_data["surface"],
            target["upper_air"],
            target["surface"],
            upper,
            surface,
        )


    @staticmethod
    def get_product_mapping():
        # check `self.predict_step()` for the order
        return {
            "input_upper": 0,
            "input_surface": 1,
            "target_upper": 2,
            "target_surface": 3,
            "output_upper": 4,
            "output_surface": 5,
        }

    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.backbone_model, norm_type=2)
    #     self.log(
    #         name="gradient_2norm", value=norms["grad_2.0_norm_total"], on_step=True
    #     )
    #     norms.pop("grad_2.0_norm_total")
    #     self.log_dict(norms, on_step=True)


    def log_mae_for_each_element(
        self, prefix: str, lv_names: list[str], var_names: list[str], mae: torch.Tensor
    ):
        for i, pl in enumerate(lv_names):
            for j, var in enumerate(var_names):
                self.log(
                    f"{prefix}_facl/{var}_{pl}",
                    mae[i, j],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def test_dataloader(self) -> DataLoader:
        """
        Load the test dataset from external `LightningDataModule`.

        The reason doing so is that the `test_dataloader` is not accessible during
        the `trainer.fit()` loop, but we need the `test_dataloader` to record the
        images in `LogPredictionSamplesCallback`.
        """
        return self._test_dataloader
