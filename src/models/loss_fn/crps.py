import torch
import torch.nn as nn


class CRPS(nn.Module):
    def __init__(self, integral_number: int = 1000):
        super().__init__()
        self.number = integral_number

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return self._calculate_crps(prediction.flatten(), target.flatten())

    def _calculate_crps(self, prediction: torch.Tensor, target: torch.Tensor):
        min_val = torch.min(torch.min(prediction), torch.min(target))
        max_val = torch.max(torch.max(prediction), torch.max(target))

        x = torch.linspace(min_val, max_val, self.number, device=prediction.device)
        x = x.to(prediction.dtype)

        cdf_prediction = self._calculate_cdf(x, prediction)
        cdf_target = self._calculate_cdf(x, target)
        diff = torch.abs(cdf_prediction - cdf_target)

        return torch.trapz(diff**2, x)

    def _calculate_cdf(self, x: torch.Tensor, data: torch.Tensor):
        # use sigmoid to approximate the cdf, since genuine method:
        # return torch.mean((data.unsqueeze(1) <= x.unsqueeze(0)).float(), dim=0)
        # is not continuous.
        return torch.mean(
            torch.sigmoid((x.unsqueeze(0) - data.unsqueeze(1)) * 1000), dim=0
        )


class PerVariableCRPS(nn.Module):
    def __init__(self, do_mae=False, target_idx=[7, 8, 9, 10, 11, 12]):
        super().__init__()
        if do_mae:
            self.register_buffer('target_idx', torch.tensor(target_idx, dtype=torch.long))
        else:
            self.target_idx = None

        var_idx = [2, 3, 4, 5, 6]
        if do_mae:
            self.register_buffer('var_idx', torch.tensor(var_idx, dtype=torch.long), persistent=False)
        else:
            self.var_idx = None

    def forward(self, prediction, target):
        """
        Input: (B, Z, H, W, C)
        Output: (Z, C) -> each var in each levels avg CRPS
        """
        B, Z, H, W, C = prediction.shape

        # who will be the fast
        pred_flat = prediction.permute(0, 1, 4, 2, 3).reshape(B, Z, C, -1)
        targ_flat = target.permute(0, 1, 4, 2, 3).reshape(B, Z, C, -1)

        if Z == 1:

            pred_sort, _ = torch.sort(pred_flat, dim=-1)
            with torch.no_grad():
                targ_sort, _ = torch.sort(targ_flat, dim=-1)

            diff_sq = (pred_sort - targ_sort)**2
            return diff_sq.mean(dim=(0, 3))

        else:

            if self.target_idx is not None:
                # 1000hPa-500hPa
                pred_calc = pred_flat.index_select(1, self.target_idx)
                targ_calc = targ_flat.index_select(1, self.target_idx)

            if self.var_idx is not None:
                pred_calc = pred_calc.index_select(2, self.var_idx)
                targ_calc = targ_calc.index_select(2, self.var_idx)

            # Compare only the variables in the same levels
            pred_sort, _ = torch.sort(pred_calc, dim=-1)

            with torch.no_grad():
                targ_sort, _ = torch.sort(targ_calc, dim=-1)

            # Squared Error (L2-Wasserstein)
            # Shape: (B, Z, C, H*W)
            diff_sq = (pred_sort - targ_sort)**2

            # return: (Z, C)
            crps_per_var = diff_sq.mean(dim=(0, 3))
            # print(crps_per_var.shape)
            full_crps = prediction.new_zeros(Z, C)

            if self.target_idx is not None and self.var_idx is not None:
                temp_crps = prediction.new_zeros(len(self.target_idx), C)
                # print(temp_crps.shape, temp_crps.max().item(), temp_crps.min().item())
                temp_crps.index_copy_(1, self.var_idx, crps_per_var)
                # print(temp_crps.shape, temp_crps.max().item(), temp_crps.min().item())
                full_crps.index_copy_(0, self.target_idx, temp_crps)
                # print(full_crps.shape, full_crps.max().item(), full_crps.min().item())

            elif self.target_idx is not None:
                full_crps.index_copy_(0, self.target_idx, crps_per_var)

            elif self.var_idx is not None:
                full_crps.index_copy_(1, self.var_idx, crps_per_var)

            else:
                full_crps = crps_per_var

            return full_crps

class L1CRPS(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        sort_p, _ = torch.sort(torch.flatten(prediction))
        sort_t, _ = torch.sort(torch.flatten(target))
        dx = torch.abs(sort_p - sort_t)
        loss = torch.sum(dx) / torch.numel(target)

        return loss
