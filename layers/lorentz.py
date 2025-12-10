from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

class Lorentz(nn.Module):
    """Lorentz model for hyperbolic geometry.
    The Lorentz model is defined as the hyperboloid in Minkowski space.
    The manifold is defined by the equation:
        -x_0^2 + x_1^2 + ... + x_n^2 = -1/k"""

    def __init__(
        self, k: float = 0.1, requires_grad=False, constraining_strategy=nn.Identity()
    ):
        super().__init__()
        k_value = torch.log(torch.exp(torch.tensor(k)) - 1)
        self.c_softplus_inv = nn.Parameter(k_value, requires_grad=requires_grad)
        self.constraining_strategy = constraining_strategy

    def k(self):
        """Returns the negative curvature of the Lorentz model."""
        return F.softplus(self.c_softplus_inv)

    def forward(self, x):
        return self.expmap0(x)

    def normL(self, x, metric=None):
        if metric is None:
            metric = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        metric[0] = -1
        return (x * x * metric).sum(dim=-1, keepdim=True).sqrt()

    def expmap0(self, x):
        """
        Maps tangent vectors from the origin of the tangent space T_0 H^n_k
        to the Lorentz hyperboloid H^n_k.
        Handles the case where the input vector norm is zero.
        """
        sqrt_k = self.k() ** 0.5
        norm_x = torch.norm(x, dim=-1, keepdim=True)

        eps = 1e-9
        is_zero = norm_x < eps

        sqrt_k_norm_x = norm_x * sqrt_k
        time = 1.0 / sqrt_k * torch.cosh(sqrt_k_norm_x)

        factor = torch.where(
            is_zero, torch.ones_like(norm_x), torch.sinh(sqrt_k_norm_x) / sqrt_k_norm_x
        )
        space = factor * x
        return torch.cat([time, space], dim=-1)

    def logmap0(self, y):
        """
        Logarithmic map from the origin for the Lorentz model.

        Args:
            y: Point on the hyperboloid

        Returns:
            Tangent vector at the origin that maps to y under expmap0
        """
        k = self.k()
        sqrt_k = k**0.5

        y_time = y[..., :1]  # First component (time)
        y_space = y[..., 1:]  # Remaining components (space)

        # A small epsilon to avoid instability when y is close to the origin.
        eps = 1e-9

        # Calculate the factor based on the formula
        # arccosh(sqrt(k) * y_time) / sqrt((sqrt(k) * y_time)^2 - 1)
        # The argument to sqrt can be negative due to floating point errors, so clamp at 0.
        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denominator_sqrt = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=eps))

        factor = torch.acosh(sqrt_k * y_time) / denominator_sqrt

        # Compute the tangent vector (0, y_space) scaled by the factor
        return factor * y_space
    
    def logmap(self, base_point: torch.Tensor, x: torch.Tensor):
        """
        Logarithmic map for the Lorentz model at an arbitrary base point.
        
        Args:
            base_point: [batch_size, dim]
            x: [batch_size, m, dim]

        Returns:
            Tangent vector at base_point that maps to x under expmap: [batch_size, m, dim]
        """
        if base_point.dim() < x.dim():
            base_point = base_point.unsqueeze(-2)  # [batch, 1, dim]
        k = self.k()

        inner = -base_point[..., 0] * x[..., 0] + torch.sum(base_point[..., 1:] * x[..., 1:], dim=-1)
        
        eps = 1e-6 
        k_inner = torch.clamp(-k * inner, min=1.0 + eps) 

        denominator = torch.sqrt(k_inner.pow(2) - 1)
        factor = torch.acosh(k_inner) / denominator
        
        # Replace NaNs/Infs at the singularity with 1.0
        # (The limit of x/sinh(x) as x->0 is 1)
        is_close = denominator < eps
        factor = torch.where(is_close, torch.ones_like(factor), factor)

        return factor.unsqueeze(-1) * (x - k_inner.unsqueeze(-1) * base_point)
    
    def expmap(self, base_point: torch.Tensor, v: torch.Tensor):
        if base_point.dim() == 2 and v.dim() == 3:
            base_point = base_point.unsqueeze(1)
        # 2. Compute Norm and Argument
        # normL returns [B, M, 1]
        sqrt_k = self.k().sqrt()
        norm_v = self.normL(v)
        arg = sqrt_k * norm_v

        # 3. Compute Coefficients safely
        # left_term: cosh(sqrt(k)|v|) * base_point
        # We assume base_point is on the manifold, so its norm is -1/k.
        left_coeff = torch.cosh(arg)
        
        # right_term: sinh(sqrt(k)|v|) / (sqrt(k)|v|) * v
        # Handle the singularity where arg -> 0
        eps = 1e-8
        mask_zero = arg < eps
        
        # Standard formula
        sinh_term = torch.sinh(arg) / arg
        
        # Apply limit: if arg is small, sinh(x)/x approx 1
        right_coeff = torch.where(mask_zero, torch.ones_like(sinh_term), sinh_term)

        # 4. Combine
        # [B, M, 1] * [B, 1, D] + [B, M, 1] * [B, M, D]
        return left_coeff * base_point + right_coeff * v
    
    def parallel_transport(self, base_point: torch.Tensor, tangent_vec: torch.Tensor, to_point: torch.Tensor):
        """
        Parallel transport with broadcasting support.
        
        Args:
            base_point:  [batch, dim]      (e.g., [10, 9])
            tangent_vec: [batch, m, dim]   (e.g., [10, 12, 9])
            to_point:    [batch, dim]      (e.g., [10, 9])
            
        Returns:
            transported_vec: [batch, m, dim]
        """
        # 1. Align dimensions for broadcasting
        # We unsqueeze dim 1 so shapes become [Batch, 1, Dim] to match [Batch, M, Dim]
        if base_point.dim() == 2 and tangent_vec.dim() == 3:
            x = base_point.unsqueeze(1)
            v = tangent_vec
            y = to_point.unsqueeze(1)
        else:
            # Fallback for when all inputs are [Batch, Dim]
            x, v, y = base_point, tangent_vec, to_point

        # 2. Lorentz Inner Product <to_point, tangent_vec>_L
        # y: [B, 1, D], v: [B, M, D] -> y*v: [B, M, D] -> sum: [B, M, 1]
        yLv = (-y[..., 0] * v[..., 0]).unsqueeze(-1) + torch.sum(y[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        
        # 3. Lorentz Inner Product <base_point, to_point>_L
        # u: [B, 1, D], y: [B, 1, D] -> u*y: [B, 1, D] -> sum: [B, 1, 1]
        xLy = (-x[..., 0] * y[..., 0]).unsqueeze(-1) + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        
        # 4. Scaling Coefficient
        numerator = yLv
        
        denominator = 1/self.k() - xLy 

        # 5. Compute Transport
        # Broadcasts: [B, M, 1] * ([B, 1, D] + [B, 1, D]) -> [B, M, D]
        return v + (numerator / denominator) * (x + y)
    
    def projection_space_orthogonal(self, x):
        """
        Projects a point onto the Lorentz model orthogonally from the space dimensions.

        Args:
            x: Point in the Lorentz model dim [batch_size, dim]
        Returns:
            Projected point dim [batch_size, dim]
        """
        return torch.cat(
            [torch.sqrt(1 / self.k() + x.pow(2).sum(-1, keepdim=True)), x], dim=-1
        )

    def poincare_to_lorentz(self, x_poincare: torch.Tensor):
        """
        Converts points from the Poincaré ball model to the Lorentz hyperboloid model.
        The conversion assumes both models share the same curvature parameter k > 0.
        """
        k = self.k()
        sqrt_k = k.sqrt()

        # Calculate the squared Euclidean norm of the Poincaré points
        x_norm_sq = torch.sum(x_poincare * x_poincare, dim=-1, keepdim=True)

        # Denominator for the conversion formula
        # Add epsilon for numerical stability
        denom = 1 - k * x_norm_sq + 1e-9

        # Time component of the Lorentz point
        time_component = (1 / sqrt_k) * (1 + k * x_norm_sq) / denom

        # Space components of the Lorentz point
        space_components = (2 * x_poincare) / denom

        # Concatenate time and space to form the Lorentz point
        return torch.cat([time_component, space_components], dim=-1)
    
    def direct_concat(self, xs: torch.Tensor):
        """
        Perform Lorentz direct concatenation as described by Qu, E., et al. (https://openreview.net/forum?id=NQi9U0YLW3)

        Args:
            xs: list of tensors to concatenate. Assumes the last dimension is the dimension along which the tensor lies on the manifold, and the second last dimension is the dimension to concatenate over.

        
        """
        # Input check
        time_sq = xs.narrow(dim=-1, start=0, length=1) ** 2
        space_sq = xs.narrow(dim=-1, start=1, length=xs.size(-1)-1) ** 2
        lorentz_norm_sq = -time_sq + space_sq.sum(dim=-1, keepdim=True)
        target_norm = -1.0 / self.k()
        assert torch.allclose(lorentz_norm_sq, target_norm, atol=1e-3), f"Input tensors do not lie on the Lorentz manifold. Mean deviation: {torch.abs(lorentz_norm_sq - target_norm).mean().item()}"

        time = torch.sqrt(((xs[..., 0]) ** 2).sum(dim=-1, keepdim=True) - (xs.shape[-2] - 1) / self.k())
        space = xs[..., 1:].reshape(*xs.shape[:-2], -1)
        out = torch.cat([time, space], dim=-1)
        assert torch.allclose(-out[..., 0]**2 + (out[..., 1:]**2).sum(dim=-1), -1.0 / self.k(), atol=1e-3), "Output tensor does not lie on the Lorentz manifold."
        return out
    
    def lorentz_midpoint(self, xs: torch.Tensor, weights: Optional[torch.Tensor] = None):
        """
        Compute the Lorentz midpoint of a set of points on the Lorentz manifold.

        Args:
            xs: Tensor of shape (..., N, D) where N is the number of points and D is the dimension of the Lorentz manifold.
            weights: Optional tensor of shape (M, N) representing the weights for each point. If None, equal weights are assumed.

        Returns:
            Tensor of shape (..., D) representing the Lorentz midpoint.
        """
        time_sq = xs.narrow(dim=-1, start=0, length=1) ** 2
        space_sq = xs.narrow(dim=-1, start=1, length=xs.size(-1)-1) ** 2
        lorentz_norm_sq = -time_sq + space_sq.sum(dim=-1, keepdim=True)
        target_norm = -1.0 / self.k()
        assert torch.allclose(lorentz_norm_sq, target_norm, atol=1e-3), f"Input tensors do not lie on the Lorentz manifold. Mean deviation: {torch.abs(lorentz_norm_sq - target_norm).mean().item()}"

        if weights is None:
            numerator = xs.sum(dim=-2)
        else:
            numerator = weights @ xs
        
        time_sq = numerator.narrow(dim=-1, start=0, length=1) ** 2
        space_sq = numerator.narrow(dim=-1, start=1, length=numerator.size(-1)-1) ** 2
        lorentz_norm_sq = time_sq - space_sq.sum(dim=-1, keepdim=True)
        denominator = lorentz_norm_sq.sqrt()
        denominator = denominator * self.k().sqrt()
        out = numerator / denominator
        assert torch.allclose(-out[..., 0]**2 + (out[..., 1:]**2).sum(dim=-1), -1.0 / self.k(), atol=1e-3), "Output tensor does not lie on the Lorentz manifold."
        return out
    
    # def distL(self, x, y):
    #     """
    #     Squared Lorentzian distance.
    #     Args:
    #          x: [batch, N, dim]
    #          y: [batch, M, dim]
    #     Returns:
    #          dist: [batch, N, M]
    #     """
    #     xLy = x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    #     return -2 / self.k() + 2 * xLy
    
    # def dist(self, x, y, keepdim=False):
    #     """
    #     Hyperbolic Geodesic Distance (required for HNN++ Attention).
        
    #     Args:
    #         x: [batch, N, D] (Queries)
    #         y: [batch, M, D] (Keys)
            
    #     Returns:
    #         dist: [batch, N, M]
    #     """
    #     if len(x.shape) == 1:
    #         x = x.unsqueeze(0)  # Add batch dimension if missing
    #     if len(y.shape) == 1:
    #         y = y.unsqueeze(0)  # Add batch dimension if missing
    #     # 1. Lorentz Inner Product (with broadcasting for pairwise computation)
    #     # x: (B, N, 1, D)
    #     # y: (B, 1, M, D)
    #     x = x.unsqueeze(-2)
    #     y = y.unsqueeze(-3)
        
    #     # -x0*y0 + x_space*y_space
    #     # Note: We use the analytical formula directly rather than creating a massive intermediate tensor
    #     time_prod = x[..., 0] * y[..., 0]
    #     space_prod = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        
    #     # This is -<x, y>_L
    #     neg_inner = time_prod - space_prod
        
    #     # 2. Clamping for Numerical Stability
    #     # -<x, y>_L must be >= 1/k. 
    #     # Floating point errors can push it slightly below, causing NaNs in acosh.
    #     min_val = 1.0 / self.k() + 1e-7
    #     neg_inner = torch.clamp(neg_inner, min=min_val)
        
    #     # 3. Geodesic Distance Formula: 1/sqrt(k) * acosh( -k * <x, y>_L )
    #     sqrt_k = self.k().sqrt()
    #     dist = (1.0 / sqrt_k) * torch.acosh(self.k() * neg_inner)
        
    #     if not keepdim:
    #         dist = dist.squeeze(-1) # Remove the singleton dim from the inner product sum if it persisted
            
    #     return dist