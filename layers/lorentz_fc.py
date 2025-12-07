import torch
import torch.nn as nn
from .lorentz import Lorentz
from einops import rearrange

class Lorentz_fully_connected(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        manifold: Lorentz = Lorentz(0.1),
        reset_params="eye",
        a_default=0.0,
        activation=nn.functional.relu,
        do_mlr = False,
    ):
        super().__init__()
        self.manifold = manifold
        self.U = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features + 1, out_features))
        self.reset_parameters(reset_params=reset_params, a_default=a_default)
        self.activation = activation

        self.do_mlr = do_mlr

    def reset_parameters(self, reset_params, a_default):
        in_features, out_features = self.U.shape
        if reset_params == "eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                print("not possible 'eye' initialization, defaulting to kaiming")
                with torch.no_grad():
                    self.U.data.copy_(
                        torch.randn(in_features, out_features)
                        * (2 * in_features * out_features) ** -0.5
                    )
            self.a.data.fill_(a_default)
        elif reset_params == "kaiming":
            with torch.no_grad():
                self.U.data.copy_(
                    torch.randn(in_features, out_features)
                    * (2 * in_features * out_features) ** -0.5
                )
            self.a.data.fill_(a_default)
        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

    def create_spacelike_vector(self):
        U_norm = self.U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k().sqrt() * U_norm * self.a
        time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * self.U
        return torch.cat([time, space], dim=0)

    def signed_dist2hyperplanes_scaled_angle(self, x):
        """Scale the distances by scaling the angle (implicitly)"""
        V = self.create_spacelike_vector()
        sqrt_k = self.manifold.k().sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)

    def signed_dist2hyperplanes_scaled_dist(self, x):
        """Scale the distances by scaling the total distance (explicitly)"""
        V = self.create_spacelike_vector()
        V_norm = self.manifold.normL(V.transpose(0, 1)).transpose(0, 1)
        sqrt_k = self.manifold.k().sqrt()
        return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V / V_norm))

    def compute_output_space(self, x):
        V = self.create_spacelike_vector()
        return self.activation(x @ V)

    def forward(self, x):
        if self.do_mlr:
            return self.mlr(x)
        output_space = self.compute_output_space(x)
        return self.manifold.projection_space_orthogonal(output_space)

    def forward_cache(self, x):
        output_space = self.activation(x @ self.V_auxiliary)
        return self.manifold.projection_space_orthogonal(output_space)

    def mlr(self, x):
        return self.signed_dist2hyperplanes_scaled_angle(x)


class Lorentz_Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 manifold: Lorentz = Lorentz(1.0),
                 stride=1,
                 padding=0,
                 bias=True,):
        """
        Lorentz fully connected layer applied in a convolutional manner.
        
        Args:
            in_channels: Number of input channels (excluding time dimension).
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            manifold: Instance of the Lorentz manifold.
            stride: Stride of the convolution.
            padding: Padding for the convolution (only 'same' supported)
            bias: Whether to include a bias term (not used here).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.manifold = manifold
        self.lin = Lorentz_fully_connected(
            in_features=in_channels * kernel_size * kernel_size,
            out_features=out_channels,
            manifold=manifold,
        )

    def forward(self, x: torch.Tensor):
        if self.padding == "same":
            pad = self.kernel_size // 2
            x = nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)  # Pad height and width
            time_comp = torch.sqrt(1.0 + (x[:, 1:, :, :] ** 2).sum(dim=1, keepdim=True))
            x = torch.cat([time_comp, x[:, 1:, :, :]], dim=1)
        batch_size, in_channels, height, width = x.shape
        x_unfolded = x.unfold(dimension=2, size=self.kernel_size, step=self.stride).unfold(dimension=3, size=self.kernel_size, step=self.stride)  # (B, (C+1), H_out, W_out, k, k)
        x_unfolded = rearrange(x_unfolded, 'b c h w k1 k2 -> b (h w) (k1 k2) c')  # (B, L, k*k, (C+1))
        x_unfolded = self.manifold.direct_concat(x_unfolded)  # (B, L, C*k*k + 1)
        
        y = self.lin(x_unfolded)  # (B, L, out_channels)
        y = y.transpose(1, 2)  # (B, out_channels, L)
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        y = y.contiguous().view(
            batch_size,
            -1,
            out_height,
            out_width,
        )  # (B, out_channels, H_out, W_out)
        return y
    

