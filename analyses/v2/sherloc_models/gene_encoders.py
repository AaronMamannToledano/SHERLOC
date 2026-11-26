"""
Gene-level longitudinal encoders for SHERLOC's ctDNA-dynamics arm.

Input : x (n, G, T) gene-level VAF trajectories (already batch-normalised per gene).
Output: (n, G * k) gene representations fed to the Cox layer (k = out_per_gene; 1 except "conv2").
All modules share their parameters across genes (as SHERLOC's 1-D temporal convolution), except the
learned gene-identity embedding of "gene_transformer". Every gene keeps its own output unit, so each
gene's representation still has a single interpretable Cox coefficient.

  conv              SHERLOC: y_g = sigmoid(sum_t w_t x_{g,t} + b)  (fixed temporal weights)
  conv2             two shared temporal kernels per gene (as the panel-level arm)
  mlp               shared nonlinear temporal kernel on [x_1..x_T, x_2-x_1, ..., x_T-x_{T-1}]
  gru               shared GRU over the T timepoints of each gene, last hidden state -> scalar
  attn_pool         temporal attention pooling: patient- and gene-specific weights over timepoints,
                    alpha_{g,t} = softmax_t(v . tanh(W [x_t, dx_t] + p_t)); y = sigmoid(u . sum_t alpha h_t)
  temporal_attn     self-attention across timepoint tokens (shared), mean-pooled -> scalar
  gene_transformer  each gene trajectory is a token (shared temporal projection + gene-identity embedding);
                    self-attention across genes models co-evolution of mutations -> one scalar per gene
  gene_set          attention pooling across genes into K=2 patient-level gene-dynamics features
transform_gene_input: raw / log / clonal-fraction inputs (clonal = VAF / patient max VAF at that timepoint).
"""
import math

import torch
from torch import nn


def _deltas(x):
    return x[..., 1:] - x[..., :-1]


class GeneConv(nn.Module):
    out_per_gene = 1

    def __init__(self, T, **_):
        super().__init__()
        self.lin = nn.Linear(T, 1)

    def forward(self, x):
        return torch.sigmoid(self.lin(x)).squeeze(-1)


class GeneConv2(nn.Module):
    out_per_gene = 2

    def __init__(self, T, **_):
        super().__init__()
        self.lin = nn.Linear(T, 2)

    def forward(self, x):
        return torch.sigmoid(self.lin(x)).reshape(x.shape[0], -1)


class GeneMLP(nn.Module):
    out_per_gene = 1

    def __init__(self, T, d=8, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * T - 1, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        z = torch.cat((x, _deltas(x)), dim=-1)
        return torch.sigmoid(self.net(z)).squeeze(-1)


class GeneGRU(nn.Module):
    out_per_gene = 1

    def __init__(self, T, d=4, **_):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=d, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        n, G, T = x.shape
        h, _ = self.gru(x.reshape(n * G, T, 1))
        return torch.sigmoid(self.out(h[:, -1, :])).reshape(n, G)


class GeneAttnPool(nn.Module):
    """Additive (Bahdanau) attention over timepoints; exposes the attention weights."""
    out_per_gene = 1

    def __init__(self, T, d=8, **_):
        super().__init__()
        self.inp = nn.Linear(2, d)                       # [x_t, dx_t]
        self.pos = nn.Parameter(torch.zeros(T, d))       # learned timepoint embedding
        self.v = nn.Linear(d, 1, bias=False)
        self.u = nn.Linear(d, 1)
        self.last_alpha = None

    def forward(self, x):
        dx = torch.cat((torch.zeros_like(x[..., :1]), _deltas(x)), dim=-1)
        h = torch.tanh(self.inp(torch.stack((x, dx), dim=-1)) + self.pos)    # (n, G, T, d)
        alpha = torch.softmax(self.v(h).squeeze(-1), dim=-1)                # (n, G, T)
        self.last_alpha = alpha.detach()
        ctx = (alpha.unsqueeze(-1) * h).sum(-2)                              # (n, G, d)
        return torch.sigmoid(self.u(ctx)).squeeze(-1)


class GeneTemporalAttn(nn.Module):
    out_per_gene = 1

    def __init__(self, T, d=8, **_):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.1)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.d = d

    def forward(self, x):
        n, G, T = x.shape
        e = self.inp(x.reshape(n * G, T, 1)) + self.pos                      # (nG, T, d)
        att = torch.softmax(self.q(e) @ self.k(e).transpose(1, 2) / math.sqrt(self.d), dim=-1)
        e = self.norm(e + att @ self.v(e))
        return torch.sigmoid(self.out(e.mean(1))).reshape(n, G)


class GeneTransformer(nn.Module):
    """Attention across genes (co-evolution); token = shared temporal projection + gene embedding."""
    out_per_gene = 1

    def __init__(self, T, n_genes, d=8, **_):
        super().__init__()
        self.inp = nn.Linear(2 * T - 1, d)
        self.gene_emb = nn.Parameter(torch.randn(n_genes, d) * 0.1)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.d = d
        self.last_att = None

    def forward(self, x):
        e = self.inp(torch.cat((x, _deltas(x)), dim=-1)) + self.gene_emb    # (n, G, d)
        att = torch.softmax(self.q(e) @ self.k(e).transpose(1, 2) / math.sqrt(self.d), dim=-1)
        self.last_att = att.detach()
        e = self.norm(e + att @ self.v(e))
        return torch.sigmoid(self.out(e)).squeeze(-1)


class GeneSetAttn(nn.Module):
    """Set encoder over genes: each gene trajectory -> h_g (shared MLP) + gene-identity embedding;
    K learned queries attend over genes -> K patient-level gene-dynamics features (fewer Cox weights)."""
    def __init__(self, T, n_genes, d=8, k=2, **_):
        super().__init__()
        self.out_total = k
        self.h = nn.Sequential(nn.Linear(T, d), nn.ReLU(), nn.Linear(d, d))
        self.gene_emb = nn.Parameter(torch.randn(n_genes, d) * 0.1)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.1)
        self.val = nn.Linear(d, 1)
        self.d = d
        self.last_att = None

    def forward(self, x):
        h = self.h(x) + self.gene_emb                                   # (n, G, d)
        att = torch.softmax(h @ self.queries.T / math.sqrt(self.d), dim=1)   # (n, G, K) over genes
        self.last_att = att.detach()
        v = self.val(h)                                                 # (n, G, 1)
        return torch.sigmoid((att * v).sum(1))                          # (n, K)


GENE_ENCODERS = {
    "conv": GeneConv, "conv2": GeneConv2, "mlp": GeneMLP, "gru": GeneGRU,
    "attn_pool": GeneAttnPool, "temporal_attn": GeneTemporalAttn, "gene_transformer": GeneTransformer,
    "gene_set": GeneSetAttn,
}


def transform_gene_input(x, mode, eps=0.05):
    """x: raw gene VAFs (n, G, T). Returns the encoder input (n, G, T') before batch normalisation.
    raw | log | clonal (VAF / patient max VAF over tracked genes at t, 0 if none) | raw_clonal | log_clonal
    (the last two concatenate both channels along the time axis -> T' = 2T)."""
    if mode in (None, "raw"):
        return x
    lg = torch.log(x.clamp_min(0) + eps)
    mx = x.max(dim=1, keepdim=True).values
    cl = torch.where(mx > 0, x / mx.clamp_min(1e-8), torch.zeros_like(x))
    return {"log": lg, "clonal": cl, "raw_clonal": torch.cat((x, cl), -1),
            "log_clonal": torch.cat((lg, cl), -1)}[mode]
