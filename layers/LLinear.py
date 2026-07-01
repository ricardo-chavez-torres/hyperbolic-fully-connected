import torch
import torch.nn as nn
import torch.nn.functional as F
from .lorentz import Lorentz
from .ilnn import ILNNLinear
from einops import rearrange
import math

class LorentzFullyConnectedOurs(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        manifold: Lorentz = Lorentz(0.1),
        reset_params="eye",
        a_default=0.5,
        activation=nn.functional.relu,
        do_mlr = False,
        mlr_init: str | None = None,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.manifold = manifold
        self.use_weight_norm = use_weight_norm
        in_features = in_features - 1
        out_features = out_features - 1
        if self.use_weight_norm:
            self.v = nn.Parameter(torch.randn(in_features, out_features))
            self.g = nn.Parameter(torch.ones(out_features))
            self.U = None
        else:
            self.U = nn.Parameter(torch.randn(in_features, out_features))
            self.v = None
            self.g = None
        self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features + 1, out_features))
        
        self.activation = activation
        self.do_mlr = do_mlr
        if do_mlr:
            reset_params = mlr_init if mlr_init is not None else "mlr"
        self.reset_parameters(reset_params=reset_params, a_default=a_default)

    def get_U(self):
        if not self.use_weight_norm:
            return self.U
        v_norm = self.v.norm(dim=0, keepdim=True).clamp(min=1e-8)
        g_pos = F.softplus(self.g)
        return g_pos.unsqueeze(0) * self.v / v_norm

    def reset_parameters(self, reset_params, a_default):
        if self.use_weight_norm:
            in_features, out_features = self.v.shape
            # Initialize direction randomly
            nn.init.kaiming_normal_(self.v)
            # Initialize magnitude based on desired init scheme
            self.g.data.fill_((1.0 / (in_features + out_features)) ** 0.5)
            self.a.data.fill_(a_default)
            return

        in_features, out_features = self.U.shape
        if reset_params == "eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))

        elif reset_params == "xavier":
            std = (1.0 / (in_features + out_features)) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)
        elif reset_params == "lorentz_kaiming":
            std = (1.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)

        elif reset_params == "kaiming":
            # For Lorentz models: divide std by 0.5 to account for time coordinate
            std = (2.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)

        elif reset_params == "mlr":
            std = (5.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)

        elif reset_params == "mlr_eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))

        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

        self.a.data.fill_(a_default)

    def create_spacelike_vector(self):
        U = self.get_U()
        U_norm = U.norm(dim=0, keepdim=True).clamp(1e-10)
        # Clamp the sinh/cosh argument to prevent overflow
        
        U_norm_sqrt_k_b = (self.manifold.k().sqrt() * self.a / U_norm).clamp(-100, 100)
        time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * U
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
    
    def compute_V_auxiliary(self):
        self.V_auxiliary = torch.nn.Parameter(self.create_spacelike_vector())


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """
    def __init__(
            self, 
            manifold: Lorentz, 
            num_features: int, 
            num_classes: int
        ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes + 1,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes + 1, num_features-2), pad=(1,0), value=1)) # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1/self.manifold.k().sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK*self.a)*norm_z)
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        # Clamp sqrt argument to prevent NaN from numerical errors
        # Mathematically this equals ||z||^2 (since cosh^2 - sinh^2 = 1)
        beta_sq = -w_t**2 + torch.norm(w_s, dim=-1)**2
        beta = torch.sqrt(beta_sq.clamp(min=1e-10))
        alpha = -w_t*x.narrow(-1, 0, 1) + (torch.cosh(sqrt_mK*self.a)*torch.inner(x.narrow(-1, 1, x.shape[-1]-1), self.z))

        # Clamp beta to prevent division by zero
        d = self.manifold.k().sqrt()*torch.abs(torch.asinh(sqrt_mK*alpha/beta.clamp(min=1e-6)))  # Distance to hyperplane
        logits = torch.sign(alpha)*beta*d

        return logits
        
    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class LorentzFullyConnectedTheirs(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: Lorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False,
            init_method=None,  # Added for compatibility
            reset_params=None,  # Added for compatibility
            a_default=None,  # Added for compatibility
            activation=None,  # Added for compatibility
            do_mlr=False,  # Added for compatibility
            mlr_init: str | None = None,  # Added for compatibility
            use_weight_norm: bool = False,  # Added for compatibility
            **kwargs,
        ):

        super(LorentzFullyConnectedTheirs, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x = self.weight(x)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k() + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k().sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


def resolve_lorentz_fc_class(variant: str):
    """Resolve Lorentz FC implementation by variant name.

    Supported values:
        - "ours": LorentzFullyConnectedOurs
        - "theirs": LorentzFullyConnectedTheirs
        - "ilnn": ILNNLinear (Point-to-Hyperplane Lorentz FC, ICLR 2026)
    """
    key = (variant or "ours").lower()
    if key in {"ours", "new", "default"}:
        return LorentzFullyConnectedOurs
    if key in {"theirs", "old", "legacy"}:
        return LorentzFullyConnectedTheirs
    if key in {"ilnn", "plfc"}:
        return ILNNLinear
    raise ValueError(f"Unknown Lorentz FC variant: {variant}")


# Backwards-compatible alias
LorentzFullyConnected = LorentzFullyConnectedOurs
