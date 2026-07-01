"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(Lorentz(c), in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg, linear_variant='standard'):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear_variant = linear_variant
        if linear_variant == 'ours':
            self.linear = HypLinearOurs(manifold=manifold, in_features=in_features, out_features=out_features)
        elif linear_variant == 'chen':
            self.linear = HypLinearChen(manifold=manifold, in_features=in_features, out_features=out_features)
        elif linear_variant == 'ilnn':
            self.linear = HypLinearILNN(manifold=manifold, in_features=in_features, out_features=out_features)
        else:
            self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypLinearOurs(nn.Module):
    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        a_default=0.0,
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
        self.reset_parameters(a_default=a_default)

    def get_U(self):
        if not self.use_weight_norm:
            return self.U
        v_norm = self.v.norm(dim=0, keepdim=True).clamp(min=1e-8)
        g_pos = F.softplus(self.g)
        return g_pos.unsqueeze(0) * self.v / v_norm

    def reset_parameters(self, a_default):
        if self.use_weight_norm:
            in_features, out_features = self.v.shape
            # Initialize direction randomly
            nn.init.kaiming_normal_(self.v)
            # Initialize magnitude based on desired init scheme
            self.g.data.fill_((1.0 / (in_features + out_features)) ** 0.5)
            self.a.data.fill_(a_default)
            return

        in_features, out_features = self.U.shape
        std = (1.0 / (in_features + out_features)) ** 0.5
        with torch.no_grad():
            self.U.data.normal_(0, std)

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
        return x @ V
        # return self.activation(x @ V)

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



class HypLinearChen(nn.Module):
    """
    Lorentz linear layer (Chen et al. 2020), adapted for the HGCN pipeline.
    Applies nn.Linear, then reprojects space components onto the hyperboloid.
    """

    def __init__(
        self,
        manifold,
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
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        if init_scale is not None:
            self.scale = nn.Parameter(
                torch.ones(()) * init_scale, requires_grad=learn_scale
            )
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

        self.activation = nn.ReLU()

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

            mask = mask == False  # noqa: E712
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.projection_space_orthogonal(x_space)

        return self.activation(x)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)
        if self.weight.bias is not None:
            nn.init.constant_(self.weight.bias, 0)


class HypLinearILNN(nn.Module):
    """Point-to-Hyperplane Lorentz linear layer (ILNN, ICLR 2026), adapted for HGCN.

    Baseline re-implementation of ``PointToHyperplaneLorentzFC`` from
    Long et al., "ILNN" (https://github.com/Longchentong/ILNN). Computes the
    signed geodesic distance to learned hyperplanes (the ``asinh`` formula) and
    maps it back onto the hyperboloid via ``sinh`` + orthogonal projection.

    Uses the same manifold API (``k()``, ``projection_space_orthogonal``) as
    ``HypLinearOurs``, so the curvature convention is shared with the rest of the
    pipeline. The optional gyro-bias from the reference implementation is omitted
    (off by default there) to keep this self-contained.
    """

    def __init__(self, manifold, in_features, out_features, eps=1e-9):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.z = nn.Parameter(torch.empty(out_features - 1, in_features - 1))  # (m, n)
        self.a = nn.Parameter(torch.zeros(out_features - 1))  # (m,)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.z, mean=0.0, std=0.02)
        nn.init.zeros_(self.a)

    def forward(self, x):
        sqrt_k = self.manifold.k().sqrt()  # sqrt(-K)
        inv_sqrt_k = 1.0 / sqrt_k

        x_t = x[..., 0]
        x_s = x[..., 1:]

        norm_z = torch.linalg.norm(self.z, dim=-1)
        cosh_term = torch.cosh(sqrt_k * self.a)
        sinh_term = torch.sinh(sqrt_k * self.a)

        z_dot_xs = torch.einsum('mk,...k->...m', self.z, x_s)
        alpha = cosh_term * z_dot_xs - sinh_term * norm_z * x_t.unsqueeze(-1)
        beta = torch.sqrt((cosh_term * norm_z) ** 2 - (sinh_term * norm_z) ** 2 + self.eps)

        ratio = sqrt_k * alpha / beta
        v = inv_sqrt_k * torch.sign(alpha) * beta * torch.abs(torch.asinh(ratio))
        v = v.clamp(min=-10.0, max=10.0)

        w = inv_sqrt_k * torch.sinh((sqrt_k * v).clamp(-100, 100))
        return self.manifold.projection_space_orthogonal(w)


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
