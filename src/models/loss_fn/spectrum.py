import torch
import torch.nn as nn
import math
import logging
logger = logging.getLogger(__name__)

class SpectrumLoss(nn.Module):

    def __init__(self, num_bins: int = 16, H: int = 224, W: int = 224):
        super().__init__()
        self.num_bins = num_bins
        self.log_every_n_steps = 20000
        self._step_count       = 0
        self.register_buffer('_bin_idx', torch.empty(0, dtype=torch.long))
        self.register_buffer('_S_count', torch.empty(0, dtype=torch.float32))

        # Build the frequency at first
        self._build_bins(H, W)

    @torch.no_grad()
    def _build_bins(self, H: int, W: int):
        """
        Pre-compute bin_idx and S_count for given H, W.
        Results are stored as buffers (static after init).
        """
        freq_y = torch.fft.fftfreq(H)                              # (H,)
        freq_x = torch.fft.rfftfreq(W)                             # (W//2+1,)
        fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        radius = torch.sqrt(fy**2 + fx**2).flatten()               # (H*(W//2+1),)

        bins    = torch.linspace(0, 0.5, self.num_bins + 1)
        bin_idx = torch.bucketize(radius, bins[1:-1])              # (H*(W//2+1),)

        S_count = torch.zeros(self.num_bins, dtype=torch.float32)
        S_count.scatter_add_(0, bin_idx, torch.ones(len(bin_idx)))

        self._bin_idx = bin_idx                                    # (H*(W//2+1),)
        self._S_count = S_count                                    # (num_bins,)


    def radial_power_spectrum_2D(self, x):
        '''
        calculate power spectrum with different bins
        input:
            - x: (B, Z, H, W, C)
            - num_bins: How many bins are used in this PSD loss function. Default is 16.
        '''
        B, Z, H, W, C = x.shape
        BCZ = B * C * Z

        x_flat     = x.permute(0, 4, 1, 2, 3).reshape(BCZ, H, W)   # (BCZ, H, W)
        fft        = torch.fft.rfft2(x_flat, norm='ortho')         # (BCZ, H, W//2+1)
        power      = fft.real**2 + fft.imag**2                     # (BCZ, H, W//2+1)
        power_flat = power.reshape(BCZ, -1)                        # (BCZ, H*(W//2+1))

        S_sum = torch.zeros(BCZ, self.num_bins, dtype=x.dtype, device=x.device)
        S_sum.scatter_add_(
            1,
            self._bin_idx.unsqueeze(0).expand(BCZ, -1),
            power_flat
        )

        S = S_sum / self._S_count.clamp(min=1).unsqueeze(0)        # (BCZ, num_bins)
        return S.reshape(B, C, Z, self.num_bins)                   # (B, C, Z, num_bins)

    @torch.no_grad()
    def _log_magnitude(self, loss_mae, loss_spectrum, loss_spectrum_bias):
        mae_val      = loss_mae.item()
        spectrum_val = loss_spectrum.item()
        bias_val     = loss_spectrum_bias.item()

        ratio_mae_spectrum = mae_val / (spectrum_val + 1e-8)
        ratio_mae_bias     = mae_val / (bias_val     + 1e-8)

        logger.info(
            f"[Step {self._step_count}] "
            f"MAE={mae_val:.5f} | "
            f"SPECTRUM={spectrum_val:.5f} | "
            f"SPECTRUM_BIAS={bias_val:.5f} | "
            f"Ratio(MAE/SPECTRUM)={ratio_mae_spectrum:.2f} | "
            f"Ratio(MAE/BIAS)={ratio_mae_bias:.2f}"
        )

    def forward(self, prediction, target):
        """
        MAE spectrum loss
        input:
            - pred: (B, Z, H, W, C)
            - target: (B, Z, H, W, C)
        """
        B, Z, H, W, C = prediction.shape
        if Z > 1:
            pred_spec = prediction[:, 8:, :, :, 1:]  # (B, 5, H, W, C-1)
            targ_spec = target[:, 8:, :, :, 1:]
        else:
            pred_spec = prediction
            targ_spec = target

        S_pred = self.radial_power_spectrum_2D(pred_spec)
        S_targ = self.radial_power_spectrum_2D(targ_spec)

        loss_spectrum      = (S_pred - S_targ).abs().mean()
        loss_spectrum_bias = (S_pred[..., -1] - S_targ[..., -1]).abs().mean()
        loss_MAE           = (prediction - target).abs().mean()

        self._step_count += 1
        if self._step_count % self.log_every_n_steps == 0:
            self._log_magnitude(loss_MAE, loss_spectrum / 10, loss_spectrum_bias * 100)

        return (loss_MAE + loss_spectrum / 10 + loss_spectrum_bias * 100)