import torch
from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from layers import Lorentz_fully_connected, Lorentz

class LorentzMHA(nn.Module):
    def __init__(self, dim, num_heads, manifold):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.manifold = manifold
        
        self.q_projs = nn.ModuleList([Lorentz_fully_connected(dim, dim, manifold, activation=nn.Identity()) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([Lorentz_fully_connected(dim, dim, manifold, activation=nn.Identity()) for _ in range(num_heads)])
        self.v_projs = nn.ModuleList([Lorentz_fully_connected(dim, dim, manifold, activation=nn.Identity()) for _ in range(num_heads)])
        
        self.beta = nn.Parameter(torch.ones(num_heads))
        self.gamma = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x, y=None):
        if y is None: y = x
        
        head_outputs = []
        for h in range(self.num_heads):
            Q = self.q_projs[h](x) 
            K = self.k_projs[h](y) 
            V = self.v_projs[h](y) 
            
            dists = self.manifold.dist(Q, K) 
            scores = -self.beta[h] * dists - self.gamma[h]
            attn_weights = F.softmax(scores, dim=-1)
            
            head_out = self.manifold.lorentz_midpoint(V, attn_weights)
            head_outputs.append(head_out)
            
        stacked = torch.stack(head_outputs, dim=-2) 
        concat_out = self.manifold.direct_concat(stacked)
        return concat_out

class LorentzSAB(nn.Module):
    def __init__(self, dim, num_heads, manifold):
        super().__init__()
        self.mha = LorentzMHA(dim, num_heads, manifold)
        
        # Direct concatenation of H heads of dim D (spatial):
        # Resulting spatial dimension is H * D
        concat_spatial_dim = num_heads * dim
        
        self.output_proj = Lorentz_fully_connected(concat_spatial_dim, dim, manifold, activation=nn.Identity())

    def forward(self, x):
        z = self.mha(x, x)
        return self.output_proj(z)

class LorentzPMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, manifold):
        super().__init__()
        self.manifold = manifold
        self.mha = LorentzMHA(dim, num_heads, manifold)
        
        # Seeds defined in tangent space (R^dim), then mapped to manifold
        self.seed_params = nn.Parameter(torch.randn(1, num_seeds, dim))
        
        concat_spatial_dim = num_heads * dim
        self.output_proj = Lorentz_fully_connected(concat_spatial_dim, dim, manifold, activation=nn.Identity())

    def forward(self, x):
        batch_size = x.size(0)
        # expmap0 expects spatial tangent vector, maps R^dim -> H^dim
        seeds = self.manifold.expmap0(self.seed_params.repeat(batch_size, 1, 1))
        
        z = self.mha(seeds, x)
        return self.output_proj(z)

class HyperbolicMoGClustering(nn.Module):
    def __init__(self, in_dim, out_dim, num_clusters, manifold, hidden_dim=32):
        super().__init__()
        self.manifold = manifold
        
        # in_dim is the Euclidean spatial dim (2). 
        # expmap0 will map this to H^2 (dim 3). 
        # input_proj expects the spatial dimension of the manifold (2) as in_features.
        self.input_proj = Lorentz_fully_connected(in_dim, hidden_dim, manifold, activation=nn.Identity())
        
        self.sab1 = LorentzSAB(hidden_dim, 4, manifold)
        self.sab2 = LorentzSAB(hidden_dim, 4, manifold)
        self.pma = LorentzPMA(hidden_dim, 4, num_clusters, manifold)
        
        # FIX: logmap0 already returns just the spatial components. Do not slice again.
        self.to_tangent = lambda x: manifold.logmap0(x)
        
        # Final Euclidean layers
        self.fc_pi = nn.Linear(hidden_dim, 1)
        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        self.fc_sigma = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [Batch, N, 2]
        
        # 1. Lift to Manifold
        # expmap0 treats inputs as tangent vectors at origin. 
        # x has dim 2, so it maps to H^2 (tensor dim 3)
        x_hyp = self.manifold.expmap0(x) 
        
        # 2. Encode
        x_emb = self.input_proj(x_hyp) # [B, N, hidden_dim+1]
        
        z = self.sab1(x_emb)
        z = self.sab2(z)
        
        clusters = self.pma(z) # [B, K, hidden_dim+1]
        
        # 3. Decode
        clusters_tan = self.to_tangent(clusters) # [B, K, hidden_dim]
        
        pi_logits = self.fc_pi(clusters_tan).squeeze(-1)
        mu = self.fc_mu(clusters_tan)
        sigma = F.softplus(self.fc_sigma(clusters_tan)) + 1e-6
        
        return pi_logits, mu, sigma

# ---------------------------------------------------------------------------------------
# UTILS & TRAINING
# ---------------------------------------------------------------------------------------

def generate_mog_batch(batch_size, num_points, num_components=4, dim=2):
    mix_logits = torch.randn(batch_size, num_components)
    mix = Categorical(logits=mix_logits)
    
    means = torch.randn(batch_size, num_components, dim, dtype=torch.float64) * 2.0
    stds = torch.rand(batch_size, num_components, dim, dtype=torch.float64) + 0.5
    
    comp = Independent(Normal(means, stds), 1)
    gmm = MixtureSameFamily(mix, comp)
    
    x = gmm.sample((num_points,)) 
    x = x.transpose(0, 1)        
    
    return x, {"means": means, "stds": stds, "weights": mix_logits.softmax(dim=-1)}

def mog_nll_loss(x, pi_logits, mu, sigma):
    batch, n, d = x.shape
    x_expanded = x.unsqueeze(2)
    mu_expanded = mu.unsqueeze(1)
    sigma_expanded = sigma.unsqueeze(1)
    
    var = sigma_expanded ** 2
    log_prob_comp = -0.5 * ((x_expanded - mu_expanded)**2 / var).sum(-1) 
    log_prob_comp -= sigma_expanded.log().sum(-1)
    log_prob_comp -= 0.5 * d * torch.log(torch.tensor(2 * 3.14159))
    
    log_pi = F.log_softmax(pi_logits, dim=-1).unsqueeze(1)
    log_prob_points = torch.logsumexp(log_pi + log_prob_comp, dim=-1)
    return -log_prob_points.mean()

# Setup
torch.set_default_dtype(torch.float64)
manifold = Lorentz(k=1.0)
model = HyperbolicMoGClustering(in_dim=2, out_dim=2, num_clusters=4, manifold=manifold, hidden_dim=32).double().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Starting training...")
model.train()

for step in range(2001):
    x_euclid, _ = generate_mog_batch(batch_size=32, num_points=200)
    x_euclid = x_euclid.double().cuda()
    
    pi, mu, sigma = model(x_euclid)
    loss = mog_nll_loss(x_euclid, pi, mu, sigma)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to avoid explosion in hyperbolic operations

    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")