import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gc  # Import the garbage collector module at the top of your script
import csv


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


class Lorentz_fully_connected(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        manifold: Lorentz = Lorentz(0.1),
        reset_params="eye",
        activation=nn.functional.relu,
    ):
        super().__init__()
        self.manifold = manifold
        self.U = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features + 1, out_features))
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self, reset_params="eye"):
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
            self.a.data.fill_(0.0)
        elif reset_params == "kaiming":
            with torch.no_grad():
                self.U.data.copy_(
                    torch.randn(in_features, out_features)
                    * (2 * in_features * out_features) ** -0.5
                )
            self.a.data.fill_(0.0)
        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

    def create_spacelike_vector(self):
        U_norm = self.U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k().sqrt() * U_norm * self.a
        time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * self.U
        return torch.cat([space, time], dim=0)

    def signed_dist2hyperplanes_scaled_angle(self, x):
        """Scale the distances by scaling the angle (implicitly)"""
        V = self.create_spacelike_vector()
        sqrt_k = self.manifold.k().sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)

    def signed_dist2hyperplanes_scaled_dist(self, x):
        """Scale the distances by scaling the total distance (explicitly)"""
        V = self.create_spacelike_vector()
        V_norm = self.manifold.normL(V)
        sqrt_k = self.manifold.k().sqrt()
        return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V / V_norm))

    def forward(self, x):
        V = self.create_spacelike_vector()
        output_space = self.activation(x @ V)
        return self.manifold.projection_space_orthogonal(output_space)

    def forward_cache(self, x):
        output_space = self.activation(x @ self.V_auxiliary)
        return self.manifold.projection_space_orthogonal(output_space)

    def mlr(self, x):
        return self.signed_dist2hyperplanes_scaled_angle(x)

    def mlr_cache(self, x):
        V = self.V_auxiliary
        sqrt_k = self.manifold.k().sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)


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
        k_value = torch.log(torch.exp(torch.tensor(c)) - 1)
        self.c_softplus_inv = nn.Parameter(k_value, requires_grad=requires_grad)

    def c(self):
        return F.softplus(self.c_softplus_inv)

    def forward(self, x):
        return project(self.expmap0(x), self.c(), dim=-1)

    def expmap0(self, x):
        sqrt_c = self.c() ** 0.5
        norm_x_c_sqrt = x.norm(dim=-1, keepdim=True).clamp(min=1e-15) * sqrt_c
        return project(torch.tanh(norm_x_c_sqrt) * x / norm_x_c_sqrt, self.c(), dim=-1)

    def logmap0(self, y):
        y_norm_c_sqrt = y.norm(dim=-1, keepdim=True).clamp_min(1e-15) * self.c().sqrt()
        return torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt

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
        lambda_x_c = 2 / (1 - self.manifold.c() * x.norm(dim=-1, keepdim=True) ** 2)
        Z_norm = self.Z.norm(dim=0, keepdim=True)
        scores = (
            2
            / sqrt_c
            * Z_norm
            * torch.asinh(
                lambda_x_c
                * torch.inner(sqrt_c * x, (self.Z / Z_norm).T)
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


class BdeirLorentzMLR(nn.Module):
    """Multinomial logistic regression (MLR) in the Lorentz model"""

    def __init__(self, num_features: int, num_classes: int, manifold: Lorentz):
        super().__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(
            torch.zeros(
                num_classes,
            )
        )
        self.z = torch.nn.Parameter(
            F.pad(torch.zeros(num_classes, num_features - 2), pad=(1, 0), value=1)
        )  # z should not be (0,0)

        self.init_weights()

    def forward(self, x, scale_displacement: bool = False):
        # Hyperplane
        if not scale_displacement:
            sqrt_mK = self.manifold.k().sqrt()
            norm_z = torch.norm(self.z, dim=-1)
            w_t = torch.sinh(sqrt_mK * self.a) * norm_z
            w_s = torch.cosh(sqrt_mK * self.a.view(-1, 1)) * self.z
            beta = torch.sqrt(-(w_t**2) + torch.norm(w_s, dim=-1) ** 2)
            alpha = -w_t * x.narrow(-1, 0, 1) + (
                torch.cosh(sqrt_mK * self.a)
                * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z)
            )

            d = (
                1 / sqrt_mK * torch.abs(torch.asinh(sqrt_mK * alpha / beta))
            )  # Distance to hyperplane
            logits = torch.sign(alpha) * beta * d

            return logits
        else:
            sqrt_mK = self.manifold.k().sqrt()
            norm_z = torch.norm(self.z, dim=-1)
            sqrtk_a_znorm = sqrt_mK * self.a * norm_z
            w_t = torch.sinh(sqrtk_a_znorm) * norm_z
            w_s = torch.cosh(sqrtk_a_znorm.view(-1, 1)) * self.z

            beta = torch.sqrt(-(w_t**2) + torch.norm(w_s, dim=-1) ** 2)
            alpha = -w_t * x.narrow(-1, 0, 1) + (
                torch.cosh(sqrtk_a_znorm)
                * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z)
            )

            d = (
                1 / sqrt_mK * torch.abs(torch.asinh(sqrt_mK * alpha / beta))
            )  # Distance to hyperplane
            logits = torch.sign(alpha) * beta * d

            return logits

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class ChenLinear(nn.Module):
    """Linear layer in the Lorentz model, as described in Chen et al. (2020)."""

    def __init__(
        self,
        manifold: Lorentz,
        in_features,
        out_features,
        bias=False,
        init_scale=None,
        learn_scale=False,
        normalize=False,
    ):
        super().__init__()
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
            self.scale = nn.Parameter(
                torch.ones(()) * init_scale, requires_grad=learn_scale
            )
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
            unit_length = x_space / torch.sqrt(square_norm)
            x_space = scale * unit_length

            x_time = torch.sqrt(scale**2 + 1 / self.manifold.k() + 1e-5)
            x_time = x_time.masked_fill(mask, 1 / self.manifold.k().sqrt())

            mask = torch.logical_not(mask)
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.projection_space_orthogonal(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


# # Examples of input shapes and outputs shapes:
# Lorentz_fully_connected(10, 5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10+1] and outputs [batch_size, 5+1]
# BdeirLorentzMLR(num_features=10, num_classes=5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
# ChenLinear(manifold=Lorentz(0.1), in_features=10, out_features=5) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
# Poincare_linear(10, 5, manifold=Poincare(0.1)) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]

# =============================================================================
# Timing Function (with minor cleanup)
# =============================================================================


def time_operation_cuda(model, input_tensor, method_name="forward"):
    """
    Measures the forward and backward pass time for a given model and input
    using CUDA events for accurate GPU timing. Removes outliers and calculates
    mean and standard deviation.

    Args:
        model (torch.nn.Module): The model to benchmark. Must be on a CUDA device.
        input_tensor (torch.Tensor): The input data. Must be on a CUDA device.
        method_name (str): The name of the method to call on the model.

    Returns:
        tuple: (mean_forward, std_forward, mean_backward, std_backward) in milliseconds.
    """
    if not next(model.parameters()).is_cuda or not input_tensor.is_cuda:
        raise ValueError(
            "Model and input_tensor must be on a CUDA device for accurate timing."
        )

    # --- Warm-up runs ---
    for _ in range(10):
        output = getattr(model, method_name)(input_tensor)
        if output.requires_grad:
            grad_tensor = torch.randn_like(output)
            output.backward(grad_tensor, retain_graph=False)
            del grad_tensor
        del output
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # --- Timing runs ---
    forward_times = []
    backward_times = []
    num_runs = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        # --- Forward Pass Timing ---
        start_event.record()
        output = getattr(model, method_name)(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event))

        # --- Backward Pass Timing (if applicable) ---
        if output.requires_grad:
            grad_tensor = torch.randn_like(output)
            start_event.record()
            output.backward(grad_tensor, retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            backward_times.append(start_event.elapsed_time(end_event))
            del grad_tensor

        del output  # Clean up the output tensor in the loop

    def get_stats_without_outliers(times):
        if not times:
            return 0.0, 0.0
        times_np = np.array(times)
        q1, q3 = np.percentile(times_np, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_times = times_np[(times_np >= lower_bound) & (times_np <= upper_bound)]
        if len(filtered_times) == 0:
            filtered_times = times_np
        return np.mean(filtered_times), np.std(filtered_times)

    fw_mean, fw_std = get_stats_without_outliers(forward_times)
    bw_mean, bw_std = get_stats_without_outliers(backward_times)

    return fw_mean, fw_std, bw_mean, bw_std


# =============================================================================
# Main Experiment Function (Corrected)
# =============================================================================


def main():
    """
    Main function to run the benchmarking experiment with robust memory
    management and dynamic results saving.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- Experiment Configurations ---
    input_dims = [2**i for i in range(4, 15)]
    output_dims = [2**i for i in range(4, 15)]
    batch_dims = [2**i for i in range(1, 10)]

    max_input = max(input_dims)
    max_output = max(output_dims)
    max_batch = max(batch_dims)

    configs = []

    # Vary input_dims, fix output_dims and batch_dims
    configs += [(i, max_output, max_batch) for i in input_dims]
    # Vary output_dims, fix input_dims and batch_dims
    configs += [(max_input, o, max_batch) for o in output_dims]
    # Vary batch_dims, fix input_dims and output_dims
    configs += [(max_input, max_output, b) for b in batch_dims]

    # input_dims = [2**i for i in range(4,10)]
    # output_dims = [2**i for i in range(4,10)]
    # batch_dims = [2**i for i in range(1,10)]
    # configs = [(i,o,b) for i in input_dims for o in output_dims for b in batch_dims]

    # input_dims_large = [2**i for i in range(10,15)]
    # output_dims_large = [2**i for i in range(10,15)]
    # batch_dims_large = [2**i for i in range(1,3)]
    # configs += [(i,o,b) for i in input_dims_large for o in output_dims_large for b in batch_dims_large]

    # --- Setup for Dynamic CSV Writing ---
    output_filename = "exp1/runtime_results.csv"
    fieldnames = [
        "Model",
        "In",
        "Out",
        "Batch",
        "Compiled",
        "Fwd Mean (ms)",
        "Fwd Std (ms)",
        "Bwd Mean (ms)",
        "Bwd Std (ms)",
    ]

    # Write header. 'w' mode overwrites the file if it exists.
    with open(output_filename, "w", newline="") as f_output:
        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

    lorentz_manifold = Lorentz().to(device)
    poincare_manifold = Poincare().to(device)
    len_configs = len(configs)

    for index_run, (in_dim, out_dim, batch_size) in enumerate(configs):
        print(
            f"\n--- Running Config: In={in_dim}, Out={out_dim}, Batch={batch_size} ---"
        )
        print(
            f"  Index: {index_run + 1}/{len_configs} | {100 * (index_run + 1) / len_configs:.2f}% complete"
        )

        try:
            # --- Prepare Synthetic Data ---
            base_tangent_data = torch.randn(batch_size, in_dim, device=device)
            poincare_input = poincare_manifold.expmap0(
                base_tangent_data.clone()
            ).detach()
            poincare_input.requires_grad = True
            lorentz_input = lorentz_manifold.expmap0(base_tangent_data.clone()).detach()
            lorentz_input.requires_grad = True
            euclidean_input = base_tangent_data.clone().detach()
            euclidean_input.requires_grad = True
            del base_tangent_data

        except torch.cuda.OutOfMemoryError as e:
            print(
                f"  ERROR preparing data for this config: {e}. Skipping to next config."
            )
            # Clean up any partial allocations
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- Models to Test (defined inside the loop to be fresh for each config) ---
        models_to_test = {
            "Lorentz_FC (MLR)": (
                Lorentz_fully_connected(in_dim, out_dim, manifold=lorentz_manifold),
                lorentz_input,
                "mlr",
            ),
            "Lorentz_FC cached (MLR)": (
                Lorentz_fully_connected(in_dim, out_dim, manifold=lorentz_manifold),
                lorentz_input,
                "mlr_cache",
            ),
            "Lorentz_FC + ReLU": (
                Lorentz_fully_connected(
                    in_dim, out_dim, manifold=lorentz_manifold, activation=F.relu
                ),
                lorentz_input,
                "forward",
            ),
            # "Lorentz_FC + GELU": (Lorentz_fully_connected(in_dim, out_dim, manifold=lorentz_manifold, activation=F.gelu), lorentz_input, 'forward'),
            "Lorentz_FC cached + ReLU": (
                Lorentz_fully_connected(
                    in_dim, out_dim, manifold=lorentz_manifold, activation=F.relu
                ),
                lorentz_input,
                "forward_cache",
            ),
            # "Lorentz_FC cached + GELU": (Lorentz_fully_connected(in_dim, out_dim, manifold=lorentz_manifold, activation=F.gelu), lorentz_input, 'forward_cache'),
            "BdeirLorentzMLR": (
                BdeirLorentzMLR(in_dim + 1, out_dim, manifold=lorentz_manifold),
                lorentz_input,
                "forward",
            ),
            "ChenLinear (bias=F, norm=F)": (
                ChenLinear(
                    manifold=lorentz_manifold,
                    in_features=in_dim + 1,
                    out_features=out_dim + 1,
                    bias=False,
                    normalize=False,
                ),
                lorentz_input,
                "forward",
            ),
            "ChenLinear (bias=T, norm=T)": (
                ChenLinear(
                    manifold=lorentz_manifold,
                    in_features=in_dim + 1,
                    out_features=out_dim + 1,
                    bias=True,
                    normalize=True,
                    learn_scale=True,
                ),
                lorentz_input,
                "forward",
            ),
            "Poincare_Dist2Hyperplanes": (
                Poincare_distance2hyperplanes(
                    in_dim, out_dim, manifold=poincare_manifold
                ),
                poincare_input,
                "forward",
            ),
            "Poincare_Linear": (
                Poincare_linear(in_dim, out_dim, manifold=poincare_manifold),
                poincare_input,
                "forward",
            ),
            "PoincareActivation (ReLU)": (
                PoincareActivation(activation=nn.ReLU(), manifold=poincare_manifold),
                poincare_input,
                "forward",
            ),
            # "PoincareActivation (GELU)": (PoincareActivation(activation=nn.GELU(), manifold=poincare_manifold), poincare_input, 'forward'),
            "Euclidean Linear": (
                nn.Linear(in_dim, out_dim),
                euclidean_input,
                "forward",
            ),
            "Euclidean Linear no bias": (
                nn.Linear(in_dim, out_dim, bias=False),
                euclidean_input,
                "forward",
            ),
            "Euclidean Linear + ReLU": (
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()),
                euclidean_input,
                "forward",
            ),
            "Euclidean Linear no bias + ReLU": (
                nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU()),
                euclidean_input,
                "forward",
            ),
            # "Euclidean Linear + GELU": (nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU()), euclidean_input, 'forward'),
        }

        for name, (model_instance, input_data, method) in models_to_test.items():
            for compiled in [False]:
                print(f"  Testing: {name} | Compiled: {compiled}")

                model = None  # Ensure model is defined for the finally block
                result_row = {
                    "Model": name,
                    "In": in_dim,
                    "Out": out_dim,
                    "Batch": batch_size,
                    "Compiled": compiled,
                }

                try:
                    model = model_instance.to(device)

                    if compiled:
                        try:
                            if method != "forward":
                                setattr(
                                    model, method, torch.compile(getattr(model, method))
                                )
                            else:
                                model = torch.compile(model)
                        except Exception as e:
                            print(f"    Could not compile {name}: {e}")
                            # Skip this iteration if compilation fails
                            continue

                    current_input = input_data.clone().detach()
                    current_input.requires_grad = True

                    fw_mean, fw_std, bw_mean, bw_std = time_operation_cuda(
                        model, current_input, method_name=method
                    )

                    result_row.update(
                        {
                            "Fwd Mean (ms)": fw_mean,
                            "Fwd Std (ms)": fw_std,
                            "Bwd Mean (ms)": bw_mean,
                            "Bwd Std (ms)": bw_std,
                        }
                    )

                except Exception as e:
                    print(f"    ERROR running {name} (Compiled: {compiled}): {e}")
                    result_row.update(
                        {
                            "Fwd Mean (ms)": "ERROR",
                            "Fwd Std (ms)": "ERROR",
                            "Bwd Mean (ms)": "ERROR",
                            "Bwd Std (ms)": "ERROR",
                        }
                    )

                finally:
                    # 💾 DYNAMICALLY SAVE RESULT: Append the result for this run to the CSV.
                    with open(output_filename, "a", newline="") as f_output:
                        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
                        writer.writerow(result_row)

                    # 🧹 MEMORY CLEANUP: Crucial for preventing OOM errors.
                    del model
                    if "current_input" in locals():
                        del current_input
                    gc.collect()
                    torch.cuda.empty_cache()

        # --- Clean up data tensors at the end of the config loop ---
        del poincare_input
        del lorentz_input
        del euclidean_input
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

    print("\n\n--- Benchmark Finished ---")
    print(f"Results have been saved to {output_filename}")


if __name__ == "__main__":
    main()
