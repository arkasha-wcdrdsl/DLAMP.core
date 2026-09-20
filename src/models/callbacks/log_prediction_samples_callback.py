from datetime import datetime

import lightning as L
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import Callback

from visual import VizGeneral

from ...const import DATA_CONFIG_PATH
from ...datasets import CustomDataset
from ...standardization import destandardization
from ...utils import DataCompose, ungroup_tensor_pairs


class LogPredictionSamplesCallback(Callback):
    def __init__(
            self, log_image_every_n_steps: int|None = None,
            log_image_every_n_epoch: int|None = None,
            sequence_mode: bool = False,
            multi_step_forecast: int = 1,
            model_native_interval_h: int = 1,
            ):
        super().__init__()

        self.sequence_mode = sequence_mode,
        self.multi_step_forecast = multi_step_forecast
        self.model_native_interval_h = model_native_interval_h

        self.plot_per_steps = log_image_every_n_steps
        self.plot_per_epochs = log_image_every_n_epoch
        self.painter = VizGeneral()
        self.global_logged_record = 0
        self.already_load_data_for_plot = False

        # load config
        with open(DATA_CONFIG_PATH, "r") as stream:
            data_config = yaml.safe_load(stream)
        data_list = DataCompose.from_config(data_config["train_data"])
        self.p_levels = DataCompose.get_all_levels(data_list, only_upper=True)
        self.upp_vars = DataCompose.get_all_vars(data_list, only_upper=True)
        self.sfc_vars = DataCompose.get_all_vars(data_list, only_surface=True)

        #
        self.log_cases_input_tensors: list[list[dict[str, torch.Tensor]]] = []
        self.log_target_imgs = []
        self.log_pred_imgs = []


    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return

        if self.already_load_data_for_plot:
            # skip this hook after epoch0
            return

        epoch = trainer.current_epoch

        # choose cases from `src.const.EVAL_CASES`
        case_dates = [
            datetime(2022, 9, 12),
            # datetime(2022, 10, 16),
        ]

        ds: CustomDataset = pl_module.test_dataloader().dataset
        dc_lat, dc_lon = DataCompose.from_config({"Lat": ["NoRule"], "Lon": ["NoRule"]})
        self.data_lat = ds._data_gnrt.yield_data(ds._init_time_list[0], dc_lat, use_Kth_hour_pred=ds.use_Kth_hour_pred)
        self.data_lon = ds._data_gnrt.yield_data(ds._init_time_list[0], dc_lon, use_Kth_hour_pred=ds.use_Kth_hour_pred)
        cases = [ds[ds.get_internal_index_from_dt(case)] for case in case_dates]

        # load axis
        for data in cases:
            inputs, target = data[:-1], data[-1]

            tensor_inputs: list[dict[str, torch.Tensor]] = []
            for input in inputs:
                # Input Data: (lv, H, W, C) -> (1, lv, H, W, C)
                step: dict[str, torch.Tensor] = {}
                for k in input.keys():
                    assert k in ["upper_air", "surface"]
                    step[k] = torch.from_numpy(input[k][None]).to(pl_module.device)
                tensor_inputs.append(step)
            self.log_cases_input_tensors.append(tensor_inputs)  # torch.Tensor

            # Target Data
            target_upp = destandardization(target["upper_air"])  # (lv, H, W, C)
            target_sfc = destandardization(target["surface"])  # (lv, H, W, C)
            fig_gt = self.painter.plot(self.data_lon, self.data_lat, target_upp, target_sfc, self.p_levels, self.upp_vars, self.sfc_vars, epoch)
            self.log_target_imgs.append(wandb.Image(fig_gt[0]))

        self.already_load_data_for_plot = True


    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not trainer.is_global_zero:
            return

        # prioritize trainer.strategy.model since it is wrapped by FSDP, while pl_module is the original model
        model = getattr(trainer.strategy, "model", None) or getattr(trainer, "model", None) or pl_module
        model.eval()

        global_step = trainer.global_step
        epoch = trainer.current_epoch

        if self.plot_per_epochs:
            if epoch != 0 and epoch - self.global_logged_record < self.plot_per_epochs:
                return
            else:
                self.global_logged_record = epoch
        elif self.plot_per_steps:
            if global_step != 0 and global_step - self.global_logged_record < self.plot_per_steps:
                return
            else:
                self.global_logged_record = global_step
        else:
            return

        wandb_logger = trainer.logger.experiment # type: ignore

        # no step slider for Table: https://github.com/wandb/wandb/issues/1826
        # table = wandb.Table(columns=["case ID", "pred", "target"])
        for idx, case in enumerate(self.log_cases_input_tensors):
            with torch.no_grad():
                oup_upp, oup_sfc = model(*ungroup_tensor_pairs(*case))[-2:]
            oup_upp = destandardization(oup_upp.cpu().numpy())  # (B, 1, H, W, C)
            oup_sfc = destandardization(oup_sfc.cpu().numpy())  # (B, 1, H, W, C)
            fig_pd, _ = self.painter.plot(self.data_lon, self.data_lat, oup_upp, oup_sfc, self.p_levels, self.upp_vars, self.sfc_vars, epoch)
            self.log_pred_imgs.append(wandb.Image(fig_pd))

            # table.add_data(idx, wandb.Image(fig_pd), wandb.Image(fig_gt))

        if epoch == 0:
            wandb_logger.log(
                {"ground truth": self.log_target_imgs, "predictions": self.log_pred_imgs}
            )
        else:
            wandb_logger.log(
                {"predictions": self.log_pred_imgs}
            )
        # wandb_logger.log({"prediction_table": table})

        self.log_pred_imgs.clear()
