import torch
import torch.nn as nn
from .lorentz import Lorentz

class Lorentz_fully_connected(nn.Module):
    def __init__(self, in_features, out_features, manifold: Lorentz = Lorentz(0.1), reset_params="eye", a_default=0.0, activation=nn.functional.relu):
        super().__init__()
        self.manifold = manifold
        self.U = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features)) # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features+1, out_features))
        self.reset_parameters(reset_params=reset_params, a_default=a_default)
        self.activation = activation
    
    def reset_parameters(self, reset_params, a_default):
        in_features, out_features = self.U.shape
        if reset_params == "eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                print("not possible 'eye' initialization, defaulting to kaiming")
                with torch.no_grad():
                    self.U.data.copy_(torch.randn(in_features, out_features) * (2 * in_features * out_features) ** -0.5)
            self.a.data.fill_(a_default)
        elif reset_params == "kaiming":
            with torch.no_grad():
                self.U.data.copy_(torch.randn(in_features, out_features) * (2 * in_features * out_features) ** -0.5)
            self.a.data.fill_(a_default)
        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

    def create_spacelike_vector(self):
        U_norm  = self.U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k().sqrt() * U_norm * self.a
        time = - U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * self.U
        return torch.cat([space, time], dim=0)
    
    def signed_dist2hyperplanes_scaled_angle(self, x):
        """ Scale the distances by scaling the angle (implicitly)"""
        V = self.create_spacelike_vector()
        sqrt_k = self.manifold.k().sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)

    def signed_dist2hyperplanes_scaled_dist(self, x):
        """ Scale the distances by scaling the total distance (explicitly)"""
        V = self.create_spacelike_vector()
        V_norm = self.manifold.normL(V.transpose(0, 1)).transpose(0, 1)
        sqrt_k = self.manifold.k().sqrt()
        return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V/V_norm))
    
    def compute_output_space(self, x):
        V = self.create_spacelike_vector()
        return self.activation(x @ V)

    def forward(self, x):
        output_space = self.compute_output_space(x)
        return self.manifold.projection_space_orthogonal(output_space)
    
    def forward_cache(self, x):
        output_space = self.activation(x @ self.V_auxiliary)
        return self.manifold.projection_space_orthogonal(output_space)

    def mlr(self, x):
        return self.signed_dist2hyperplanes_scaled_angle(x)
    
    
