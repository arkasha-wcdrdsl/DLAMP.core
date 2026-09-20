import torch
import torch.nn as nn
import logging

from .crps import PerVariableCRPS

class MAECRPS(nn.Module):

    def __init__(self, do_mae=False, target_idx=[7, 8, 9, 10, 11, 12], crps_weight=0.5):
        """
        Args:
            do_mae (bool):
                True -> specific levels using MAE + CRPS, others using only MAE
                False -> all level using CRPS (default)
            target_idx (list): which levels using MAE + CRPS (be careful if Z is not 13 levels)
            crps_weight (float): Weight of CRPS and MAE. Default: 0.5.
        """
        super().__init__()
        self.weighted = crps_weight
        self.register_buffer('target_idx', torch.tensor(target_idx, dtype=torch.long))

        var_idx = [2, 3, 4, 5, 6]
        if do_mae:
            self.register_buffer('var_idx', torch.tensor(var_idx, dtype=torch.long), persistent=False)
        else:
            self.var_idx = None

        self.loss_crps = PerVariableCRPS(do_mae=do_mae, target_idx=target_idx)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized MAECRPS with weight={crps_weight}, log_interval={100}")

    def forward(self, prediction, target):
        """
        Input: (B, Z, H, W, C)
        Output: (Z, C) -> each var in each levels
        """
        B, Z, H, W, C = prediction.shape

        Loss_mae = torch.abs(prediction - target).mean(dim=(0, 2, 3))
        Loss_crps= self.loss_crps(prediction, target)

        self._log_magnitude(Loss_mae, Loss_crps, Z)
        mae_weights = torch.ones_like(Loss_mae)

        if self.var_idx is not None and self.target_idx is not None and Z != 1:
            # Use PyTorch meshgrid to index the Z and C dimensions together.
            z_mesh, c_mesh = torch.meshgrid(self.target_idx, self.var_idx, indexing='ij')
            mae_weights[z_mesh, c_mesh] = 1.0 - self.weighted
        elif self.target_idx is not None and Z != 1:
            # If no specific var_idx is provided, reduce all selected Z levels as before.
            mae_weights.index_fill_(0, self.target_idx, (1.0 - self.weighted))
        else:
            mae_weights = 1 - self.weighted

        # 10 * Loss_crps: since mae is L1-loss and crps is almost L2-loss
        # 5  * Loss_crps: since the results has too much high frequency problem
        Loss_pervariables = mae_weights * Loss_mae + self.weighted * Loss_crps * 5

        return Loss_pervariables.mean()

    def _log_magnitude(self, loss_mae, loss_crps, Z):
        with torch.no_grad():
            mae_val = loss_mae.mean().item()

            if Z > 1:
                active_crps = loss_crps.index_select(0, self.target_idx)
                crps_val = active_crps.mean().item()
            else:
                crps_val = loss_crps.mean().item()

            ratio = mae_val / (crps_val + 1e-8)

            self.logger.info(
                f"Magnitude Check: "
                f"MAE={mae_val:.5f} | "
                f"CRPS={crps_val:.5f} | "
                f"Ratio(MAE/CRPS)={ratio:.1f}"
            )
