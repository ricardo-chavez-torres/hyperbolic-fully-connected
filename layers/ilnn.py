import torch
import torch.nn as nn

from .lorentz import Lorentz


class ILNNLinear(nn.Module):
    """Point-to-Hyperplane Lorentz fully connected layer (ILNN, ICLR 2026).

    Baseline re-implementation of the ``PointToHyperplaneLorentzFC`` layer from
    Long et al., "ILNN" (https://github.com/Longchentong/ILNN,
    ``lib/lorentz/layers/LFC.py``), adapted to this repo's :class:`Lorentz`
    manifold and feature-dimension conventions.

    Like the rest of this codebase, ``in_features``/``out_features`` include the
    time component, and curvature is exposed through ``manifold.k()`` (positive),
    with the hyperboloid constraint ``-x_0^2 + ... + x_n^2 = -1/k``. The original
    code uses a negative curvature ``K`` and works with ``sqrt(-K)``; under the
    mapping ``K = -k`` this is exactly ``manifold.k().sqrt()``, and ``1/(-K)`` is
    ``1/manifold.k()`` (recovered here via ``projection_space_orthogonal``).

    The layer learns hyperplanes parameterised by a spatial direction ``z`` and a
    bias ``a``. It computes the signed geodesic distance to each hyperplane
    (the ``asinh`` formula, identical in spirit to the Lorentz MLR of Bdeir et
    al. / Chen et al.), then maps that distance back onto the hyperboloid via
    ``sinh`` followed by orthogonal projection. When ``do_mlr=True`` the signed
    distances are returned directly as logits.

    Note: the optional learned gyro-bias from the reference implementation
    (``share.share_b``, which depends on the ILNN GyroBN package) is intentionally
    omitted to keep this baseline self-contained. It is off by default in the
    reference configs.
    """

    def __init__(
        self,
        in_features,
        out_features,
        manifold: Lorentz = Lorentz(0.1),
        do_mlr: bool = False,
        normalize: bool = False,
        eps: float = 1e-9,
        dropout: float = 0.0,
        init_scale=None,
        learn_scale: bool = False,
        # --- Accepted for drop-in compatibility with the other FC variants. ---
        # ILNN uses its own parameterisation / initialisation and intrinsic
        # nonlinearity, so these are ignored (kept so the layer can be selected
        # via resolve_lorentz_fc_class in the ResNet/CIFAR pipeline).
        activation=None,
        reset_params=None,
        a_default=None,
        mlr_init: str | None = None,
        use_weight_norm: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert in_features >= 2, "input must be [t, x_1, ..., x_n]"

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.do_mlr = do_mlr
        self.normalize = normalize
        self.eps = eps
        self.dropout = dropout

        # Hyperplane parameters: direction z and offset a (one per output unit).
        self.z = nn.Parameter(torch.empty(out_features - 1, in_features - 1))  # (m, n)
        self.a = nn.Parameter(torch.zeros(out_features - 1))  # (m,)

        if init_scale is not None:
            self.scale = nn.Parameter(
                torch.ones(()) * init_scale, requires_grad=learn_scale
            )
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.z, mean=0.0, std=0.02)
        nn.init.zeros_(self.a)

    def signed_distances(self, x):
        """Signed geodesic distances from ``x`` to each learned hyperplane."""
        sqrt_k = self.manifold.k().sqrt()  # sqrt(-K)
        inv_sqrt_k = 1.0 / sqrt_k

        z, a = self.z, self.a
        if self.training and self.dropout > 1e-8:
            z = z * torch.bernoulli(torch.full_like(z, 1.0 - self.dropout))
            a = a * torch.bernoulli(torch.full_like(a, 1.0 - self.dropout))

        x_t = x[..., 0]  # (...,) time
        x_s = x[..., 1:]  # (..., n) space

        norm_z = torch.linalg.norm(z, dim=-1)  # (m,)
        cosh_term = torch.cosh(sqrt_k * a)  # (m,)
        sinh_term = torch.sinh(sqrt_k * a)  # (m,)

        # alpha = cosh(sqrt(-K) a) <z, x_s> - sinh(sqrt(-K) a) ||z|| x_t
        z_dot_xs = torch.einsum("mk,...k->...m", z, x_s)  # (..., m)
        alpha = cosh_term * z_dot_xs - sinh_term * norm_z * x_t.unsqueeze(-1)

        # beta = sqrt(||cosh z||^2 - (sinh ||z||)^2) = ||z|| (cosh^2 - sinh^2 = 1)
        beta = torch.sqrt((cosh_term * norm_z) ** 2 - (sinh_term * norm_z) ** 2 + self.eps)

        ratio = sqrt_k * alpha / beta
        # v = (1/sqrt(-K)) sign(alpha) beta |asinh(sqrt(-K) alpha / beta)|
        v = inv_sqrt_k * torch.sign(alpha) * beta * torch.abs(torch.asinh(ratio))
        return v.clamp(min=-10.0, max=10.0)

    def forward(self, x):
        v = self.signed_distances(x)
        if self.do_mlr:
            return v

        sqrt_k = self.manifold.k().sqrt()
        inv_sqrt_k = 1.0 / sqrt_k
        # Spatial coords w = (1/sqrt(-K)) sinh(sqrt(-K) v); time from the constraint.
        w = inv_sqrt_k * torch.sinh((sqrt_k * v).clamp(-100, 100))
        y = self.manifold.projection_space_orthogonal(w)

        if self.normalize:
            y = self._normalize(y)
        return y

    def _normalize(self, y):
        """Optional Chen-style renormalisation of the output point."""
        y_space = y.narrow(-1, 1, y.shape[-1] - 1)

        scale = y.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
        square_norm = (y_space * y_space).sum(dim=-1, keepdim=True)

        mask = square_norm <= 1e-10
        square_norm = square_norm.masked_fill(mask, 1.0)
        unit_length = y_space / torch.sqrt(square_norm)
        y_space = scale * unit_length

        y_time = torch.sqrt(scale**2 + 1 / self.manifold.k() + 1e-5)
        y_time = y_time.masked_fill(mask, 1 / self.manifold.k().sqrt())

        y_space = y_space * (~mask)
        return torch.cat([y_time, y_space], dim=-1)

    def mlr(self, x):
        return self.signed_distances(x)
