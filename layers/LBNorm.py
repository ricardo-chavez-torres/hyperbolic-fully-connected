from .lorentz import Lorentz
import torch
import torch.nn as nn

class LorentzBatchNorm(nn.Module):
    def __init__(self, manifold: Lorentz, num_features: int):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold
        self.eps = 1e-5
        self.num_features = num_features

        self.beta = nn.Parameter(torch.zeros(1, num_features + 1))
        self.beta.data[..., 0] = 1 / self.manifold.k().sqrt()  # Initialize beta at origin

        self.gamma = nn.Parameter(torch.ones((1,)))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.running_mean.data[0] = 1 / self.manifold.k().sqrt()
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        input_shape = x.shape

        x_permuted = x.permute(0, 2, 3, 1)
        permuted_shape = x_permuted.shape


        x_flat = x_permuted.reshape(-1, self.num_features + 1)  # (B*H*W, D)


        self.beta.data[..., 0] = torch.sqrt(1.0/self.manifold.k() + self.beta.data[..., 1:].pow(2).sum(dim=-1))

        if self.training:
            batch_mean = self.manifold.lorentz_midpoint(x_flat)  # (D)

            # Transport batch to origin (centering the batch)
            x_T = self.manifold.logmap(base_point=batch_mean, x=x_flat)  # (B*H*W, D)
            origin = torch.zeros_like(batch_mean)  # (D)
            origin[..., 0] = 1 / self.manifold.k().sqrt()
            x_T = self.manifold.parallel_transport(base_point=batch_mean, tangent_vec=x_T, to_point=origin)  # (B*H*W, D)

            # Compute Frechet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0, 1))  
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)
            
            # print(-self.beta.data[..., 0]**2 + torch.sum(self.beta.data[..., 1:]**2))

            x_T = x_T * (self.gamma / (var + self.eps))

            x_T = self.manifold.parallel_transport(base_point=origin, tangent_vec=x_T, to_point=self.beta)  # (B*H*W, D)
            output = self.manifold.expmap(base_point=self.beta, v=x_T)  # (B*H*W, D)

            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), batch_mean.detach().unsqueeze(0)), dim=0)

                self.running_mean.copy_(self.manifold.logmap0(self.manifold.lorentz_midpoint(means, weights=torch.tensor(((1 - momentum), momentum), dtype=means.dtype, device=means.device))))
                self.running_var.copy_((1-momentum)*self.running_var + momentum*var.detach())
        else:
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(base_point=running_mean, x=x_flat)
            origin = torch.zeros_like(running_mean)
            origin[..., 0] = self.manifold.k().sqrt()
            x_T = self.manifold.parallel_transport(base_point=running_mean, tangent_vec=x_T, to_point=origin)

            x_T = x_T * (self.gamma / (self.running_var + self.eps))

            x_T = self.manifold.parallel_transport(base_point=origin, tangent_vec=x_T, to_point=self.beta)
            output = self.manifold.expmap(base_point=self.beta, v=x_T)

        output = output.view(permuted_shape)
        output = output.permute(0, 3, 1, 2)

        return output