import torch
import torch.nn as nn
import torch.nn.functional as F


def project(x: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5)
    maxnorm = torch.where(c.gt(0), maxnorm, c.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


class Poincare(nn.Module):
    def __init__(self, c: float = 0.1, requires_grad=False):
        super().__init__()
        # Use expm1 for stability with small curvature
        k_value = torch.log(torch.expm1(torch.tensor(c))) # Inverse of softplus
        self.c_softplus_inv = nn.Parameter(k_value, requires_grad=requires_grad)

    def c(self):
        return F.softplus(self.c_softplus_inv)

    def forward(self, x):
        return project(self.expmap0(x), self.c(), dim=-1)

    def expmap0(self, x):
        sqrt_c = self.c() ** 0.5
        norm_x_c_sqrt = x.norm(dim=-1, keepdim=True).clamp(min=1e-15) * sqrt_c
        return project(torch.tanh(norm_x_c_sqrt / 2) * x / norm_x_c_sqrt, self.c(), dim=-1)

    def logmap0(self, y):
        y_norm_c_sqrt = y.norm(dim=-1, keepdim=True).clamp_min(1e-15) * self.c().sqrt()
        return 2 * torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt

    def radius(self):
        """
        Returns the radius of the Poincare ball.

        The radius is defined as 1/sqrt(c), where c is the curvature of the Poincare model.
        """
        return 1 / self.c().sqrt()

    def lorentz_to_poincare(self, y_lorentz: torch.Tensor):
        """
        Converts points from the Lorentz hyperboloid model to the Poincaré ball model.
        The conversion assumes both models share the same curvature parameter c > 0.

        Args:
            y_lorentz: Point(s) in the Lorentz model. Expected shape [..., dim + 1].
                       The first dimension is the time-like component.

        Returns:
            Point(s) in the Poincare model. Shape [..., dim].
        """
        c = self.c()
        sqrt_c = c.sqrt()

        # Split the Lorentz point into time and space components
        y_time = y_lorentz[..., :1]
        y_space = y_lorentz[..., 1:]

        # Denominator for the conversion formula (stereographic projection)
        # Add epsilon for numerical stability
        denom = 1 + sqrt_c * y_time + 1e-9

        # Compute the Poincare point
        return y_space / denom


class Poincare_distance2hyperplanes(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        manifold: Poincare = Poincare(0.1),
        reset_params="eye",
    ):
        super().__init__()
        self.manifold = manifold
        self.Z = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features))
        self.reset_parameters(reset_params)

    def reset_parameters(self, reset_params="eye"):
        if reset_params == "eye":
            in_features, out_features = self.Z.shape
            if in_features <= out_features:
                with torch.no_grad():
                    self.Z.data.copy_(1 / 2 * torch.eye(in_features, out_features))
            else:
                print("not possible 'eye' initialization, defaulting to kaiming")
                with torch.no_grad():
                    self.Z.data.copy_(
                        torch.randn(in_features, out_features)
                        * (2 * in_features * out_features) ** -0.5
                    )
            self.a.data.fill_(0.0)
        elif reset_params == "kaiming":
            in_features, out_features = self.Z.shape
            with torch.no_grad():
                self.Z.data.copy_(
                    torch.randn(in_features, out_features)
                    * (2 * in_features * out_features) ** -0.5
                )
            self.a.data.fill_(0.0)
        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

    def forward(self, x):
        sqrt_c = self.manifold.c().sqrt()
        lambda_x_c = 2 / (1 - self.manifold.c() * x.pow(2).sum(dim=-1, keepdim=True))
        Z_norm = self.Z.norm(dim=0, keepdim=True)
        scores = (
            2
            / sqrt_c
            * Z_norm
            * torch.asinh(
                lambda_x_c
                * torch.matmul(sqrt_c * x, self.Z / Z_norm)
                * torch.cosh(2 * self.a * sqrt_c)
                - (lambda_x_c - 1) * torch.sinh(2 * sqrt_c * self.a)
            )
        )
        return scores


class Poincare_dist2Poincare(nn.Module):
    def __init__(self, manifold: Poincare = Poincare(0.1)):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        sqrt_c = self.manifold.c().sqrt()
        w = (1 / sqrt_c) * torch.sinh(sqrt_c * x)
        return w / (
            1 + torch.sqrt(1 + self.manifold.c() * (w**2).sum(dim=-1, keepdim=True))
        )


class Poincare_linear(nn.Module):
    def __init__(self, in_features, out_features, manifold: Poincare = Poincare(0.1)):
        super().__init__()
        self.manifold = manifold
        self.get_scores = Poincare_distance2hyperplanes(
            in_features, out_features, manifold
        )
        self.get_point = Poincare_dist2Poincare(manifold)

    def forward(self, x, clip_poincare=True):
        scores = self.get_scores(x)
        points = self.get_point(scores)
        if clip_poincare:
            points = project(points, self.manifold.c(), dim=-1)
        return points


class PoincareActivation(nn.Module):
    def __init__(
        self, activation=nn.functional.relu, manifold: Poincare = Poincare(0.1)
    ):
        super().__init__()
        self.manifold = manifold
        self.activation = activation

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.activation(x)
        x = self.manifold.expmap0(x)
        return x
