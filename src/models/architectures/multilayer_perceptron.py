import torch
import torch.nn as nn
import torch.fft

__all__ = ["MultilayerPerceptron"]


class MultilayerPerceptron(nn.Module):
    def __init__(self, dim: int, dropout_rate: float, reduce_dim: bool,
                 input_shape: tuple[int, int, int] = None, sigma: float = 0.8) -> None:
        super().__init__()
        oup_dim = dim // 2 if reduce_dim else dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, oup_dim)

        if input_shape is not None:
            print("Using FilteredGELU_Horizontal activation in MLP with input shape:", input_shape)
            self.activation = FilteredGELU_Horizontal(input_shape=input_shape, sigma=sigma)
        else:
            self.activation = nn.GELU()

        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x

class FilteredGELU_Horizontal(nn.Module):
    """
    Nonlinear activation module designed for 3D atmospheric models.
    Applies frequency-domain Gaussian low-pass filtering only along the horizontal
    dimensions (H, W), preserving physical gradients along the vertical dimension (Z).
    """
    def __init__(self, input_shape: tuple[int, int, int], sigma: float = 0.8):
        super().__init__()
        self.activation = nn.GELU()
        self.input_shape = input_shape

        Z, H, W = input_shape

        fh = torch.fft.fftfreq(H).view(-1, 1)  # Shape: (H, 1)
        fw = torch.fft.fftfreq(W).view(1, -1)  # Shape: (1, W)

        freq_radius = torch.sqrt(fh**2 + fw**2)

        max_radius = 0.5
        cutoff_ratio = 1.0
        cutoff_threshold = max_radius * cutoff_ratio

        # Build a mask: retain radii at or below the threshold and remove larger radii.
        mask = (freq_radius <= cutoff_threshold).float()

        # Reshape for broadcasting over (B, Z, H, W, C).
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1))

        cos_term = (torch.cos(2 * torch.pi * fh) - 1) + (torch.cos(2 * torch.pi * fw) - 1)
        G_hat = torch.exp(float(sigma ** 2) * cos_term)
        self.register_buffer('G_hat', G_hat.unsqueeze(0).unsqueeze(0).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.activation(x)

        B, L, C = v.shape
        Z, H, W = self.input_shape
        v_3d = v.reshape(B, Z, H, W, C)

        v_hat = torch.fft.fft2(v_3d, dim=(2, 3))
        filtered_v_hat = v_hat * self.mask
        filtered_v_hat = filtered_v_hat * self.G_hat
        output_3d = torch.fft.ifft2(filtered_v_hat, dim=(2, 3)).real

        del v_hat
        del filtered_v_hat
        del v_3d
        del v

        return output_3d.reshape(B, L, C)
