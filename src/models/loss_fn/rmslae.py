import torch
import torch.nn as nn


class RMSLAE(nn.Module):
    r"""Compute `root mean squared logarithmic absolute error`_ (RMSLAE).

    .. math:: \text{RMSLAE} = \sqrt{\frac{1}{N}\sum_i^N (\log_e(1 + |y_i|) - \log_e(1 + |\hat{y_i}|))^2}
    """

    def __init__(self, eps: float = 1e-12, smooth_abs: float = 0.0):
        """
        eps: Avoid gradient spikes or NaNs caused by sqrt(0).
        smooth_abs: When >0, replace abs(x) with sqrt(x^2 + smooth_abs) for smoother behavior at zero.
        """
        super().__init__()
        self.eps = eps
        self.smooth_abs = smooth_abs

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: {preds.shape} vs {target.shape}")

        if self.smooth_abs > 0:
            ap = torch.sqrt(preds * preds + self.smooth_abs)
            at = torch.sqrt(target * target + self.smooth_abs)
        else:
            ap = torch.abs(preds)
            at = torch.abs(target)

        diff = torch.log1p(ap) - torch.log1p(at)
        mse = (diff * diff).mean()
        return torch.sqrt(mse + self.eps)
