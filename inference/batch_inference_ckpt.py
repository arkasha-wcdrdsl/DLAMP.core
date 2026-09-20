import logging
from datetime import datetime, timedelta
import os
import multiprocessing

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import trange

from src.models.lightning_modules import PanguLightningModule
from src.models.model_utils import get_builder
from src.utils import TimeUtil

from .infer_utils import prediction_postprocess
from .inference_base import InferenceBase

log = logging.getLogger(__name__)


class BatchInferenceCkpt(InferenceBase):
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        """
        This class can only inference by checkpoint.
        """
        super().__init__(cfg, eval_cases)

    def _setup(self):
        model_builder = get_builder(self.cfg.model.model_name)(
            "predict",
            self.data_list,
            image_shape=self.data_manager.image_shape,
            add_time_features=self.cfg.data.add_time_features,
            **self.cfg.model,
            **self.cfg.lightning,
        )

        # === NEW: Automatically select the device. ===
        # This could also be read from cfg, such as cfg.inference.device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === NEW: Enable multiple CPU threads when running on CPU. ===
        if self.device.type == "cpu":
            n_cores = multiprocessing.cpu_count() // 2
            # The value can be capped if needed, for example min(8, n_cores).
            if n_cores > 1:
                torch.set_num_threads(n_cores)
                torch.set_num_interop_threads(max(1, n_cores // 2))
                # Optionally set environment variables for MKL / OpenMP as well.
                os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
                os.environ.setdefault("MKL_NUM_THREADS", str(n_cores))
            log.info(
                f"Using CPU with {n_cores} threads "
                f"(torch.get_num_threads={torch.get_num_threads()})"
            )
        else:
            log.info(f"Using device: {self.device}")

        self.pl_module = PanguLightningModule.load_from_checkpoint(
            checkpoint_path=self.cfg.inference.best_ckpt,
            test_dataloader=None,
            backbone_model=model_builder._backbone_model(),
            weights_only=False,
        )

        # === CHANGED: Move everything to self.device instead of calling .cuda() manually. ===
        self.pl_module.to(self.device)
        self.pl_module.eval()

    def infer(self, bdy_swap_method: dict | None = None):
        """
        Perform batch inference using PyTorch (CPU/GPU).

        When running on CPU, PyTorch will use multiple threads/cores as
        configured in __init__ via torch.set_num_threads().
        """
        data_loader = self.data_manager.predict_dataloader()
        interval = self.output_itv // self.data_itv
        predict_iters = (self.showcase_length - 1) * interval

        ret = []
        for batch_id, (input, target) in enumerate(data_loader):
            # === CHANGED: Move inputs to the selected device. ===
            inp_upper = input["upper_air"].to(self.device)
            inp_surface = input["surface"].to(self.device)

            # auto-regression
            tmp_upper, tmp_sfc = [], []
            for step in trange(predict_iters, desc=f"Infer batch {batch_id}"):
                with torch.inference_mode():
                    inp_upper, inp_surface = self.pl_module(inp_upper, inp_surface)

                # Move back to CPU for NumPy operations, time features, and boundary swapping.
                inp_upper_np = inp_upper.detach().cpu().numpy()
                inp_surface_np = inp_surface.detach().cpu().numpy()

                if (step + 1) % interval == 0:
                    tmp_upper.append(inp_upper_np.copy())
                    tmp_sfc.append(inp_surface_np.copy())

                curr_time = self.init_time[batch_id] + timedelta(hours=step + 1)

                # === Process time features with NumPy. ===
                if self.cfg.data.add_time_features:
                    time_features = TimeUtil.create_time_features(
                        [curr_time], inp_surface_np.shape[2:4]
                    )  # (H, W, 4)
                    # time_features = np.expand_dims(time_features, axis=(0, 1))
                    inp_surface_np = np.concatenate(
                        (inp_surface_np, time_features), axis=-1
                    )

                # === Process boundary swapping with NumPy as well. ===
                if bdy_swap_method:
                    inp_upper_np = self._boundary_swapping(
                        inp_upper_np,
                        curr_time,
                        bdy_swap_method["name"],
                        bdy_swap_method["n_of_grid"],
                    )
                    inp_surface_np = self._boundary_swapping(
                        inp_surface_np,
                        curr_time,
                        bdy_swap_method["name"],
                        bdy_swap_method["n_of_grid"],
                    )

                # Convert back to torch and move to the device for the next autoregressive step.
                inp_upper = torch.from_numpy(inp_upper_np).to(self.device)
                inp_surface = torch.from_numpy(inp_surface_np).to(self.device)

            # post-process 1, shape = (1, lv, H, W, c) or (Seq, lv, H, W, c)
            tmp_upper = np.concatenate(tmp_upper, axis=0)
            tmp_sfc = np.concatenate(tmp_sfc, axis=0)
            ret.append(
                (
                    input["upper_air"].cpu().numpy(),
                    input["surface"].cpu().numpy(),
                    target["upper_air"].cpu().numpy(),
                    target["surface"].cpu().numpy(),
                    tmp_upper,
                    tmp_sfc,
                )
            )

        # post-process 2, shape = (B, lv, H, W , c) or (B, Seq, lv, H, W, c)
        mapping = PanguLightningModule.get_product_mapping()
        predictions = prediction_postprocess(ret, mapping)
        for product_type, tensor in predictions.items():
            setattr(self, product_type, tensor)

        log.info(f"Batch inference finished at {datetime.now()}")
