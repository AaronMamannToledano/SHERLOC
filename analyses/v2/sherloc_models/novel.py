"""
Novel SHERLOC variants for short longitudinal ctDNA (2-4 timepoints), small cohorts, and
patient-level interpretability. All share the SHERLOC interface:

    forward(x_gene (n, G, T), x_agg (n, P, T), x_clinical (n, 1, C), x_sb (n, 2, T)) -> (n, d_output)

SHERLOC_NAM   Neural additive model. One small "shape network" per longitudinal ctDNA feature
              maps its trajectory (BL, ..., last) to a scalar risk contribution; static covariates
              and the pretrained genomic-interaction representation enter linearly. The log-risk is
              the exact sum of per-feature contributions -> per-patient explanation.
              monotone_last=True: the contribution of each ctDNA feature is non-decreasing in its
              last (on-treatment) value, f(x) = g(earlier values) + h_mono(last value): more ctDNA on
              treatment can never lower predicted risk.
SHERLOC_Proto Trajectory prototypes (keep_panel_arm=True: hybrid that also keeps SHERLOC's
              feature-specific panel arm). Panel-level trajectories are compared with K learnable
              prototype trajectories; the soft membership (softmax of -distance) feeds the Cox layer
              with the static covariates and the genomic representation. Each patient is described by
              the trajectory type(s) he/she resembles.
"""
import torch
from torch import nn
import torch.nn.functional as F


def _log(x, eps):
    return torch.log(x.clamp_min(0) + eps)


class _ShapeNet(nn.Module):
    """Unconstrained shape function R^c -> R (one per feature)."""
    def __init__(self, c, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


class _MonoNet(nn.Module):
    """Monotone non-decreasing R -> R: positive weights (softplus) + monotone activations."""
    def __init__(self, h):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(h, 1) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(h))
        self.w2 = nn.Parameter(torch.randn(1, h) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = torch.tanh(F.linear(x, F.softplus(self.w1), self.b1))
        return F.linear(z, F.softplus(self.w2), self.b2)


class SHERLOC_NAM(nn.Module):
    def __init__(self, d_input, d_agg_input, d_channel, d_output, d_other_input, device,
                 hidden=8, log_eps=0.05, use_gene_arm=True, monotone_last=False,
                 feature_dropout=0.1, output_sigmoid=True):
        super().__init__()
        self.use_gene_arm, self.monotone_last = use_gene_arm, monotone_last
        self.log_eps, self.output_sigmoid = log_eps, output_sigmoid
        n_temp = d_agg_input + (d_input if use_gene_arm else 0)
        self.n_temp, self.d_channel = n_temp, d_channel
        self.bn_temp = nn.BatchNorm1d(n_temp)                  # per-feature normalisation (over n and T)
        if monotone_last:
            self.free = nn.ModuleList([_ShapeNet(d_channel - 1, hidden) for _ in range(n_temp)])
            self.mono = nn.ModuleList([_MonoNet(hidden) for _ in range(n_temp)])
        else:
            self.shape = nn.ModuleList([_ShapeNet(d_channel, hidden) for _ in range(n_temp)])
        self.bn_static = nn.BatchNorm1d(d_other_input)
        self.lin_static = nn.Linear(d_other_input, 1, bias=False)
        self.bn_sb = nn.BatchNorm1d(d_channel * 2)
        self.lin_sb = nn.Linear(d_channel * 2, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.feature_dropout = feature_dropout

    def _temporal(self, x_gene, x_agg):
        xs = [x_agg] + ([x_gene] if self.use_gene_arm else [])
        x = torch.cat(xs, dim=1)                               # (n, F, T)
        if self.log_eps:
            x = _log(x, self.log_eps)
        return self.bn_temp(x)

    def contributions(self, x_gene, x_agg, x_clinical, x_sb):
        """Per-patient additive contributions to the pre-sigmoid log-risk.
        Returns (temporal (n, F), static (n, C), genomic (n, 2T), bias)."""
        x = self._temporal(x_gene, x_agg)
        if self.monotone_last:
            temp = [self.free[f](x[:, f, :-1]) + self.mono[f](x[:, f, -1:]) for f in range(self.n_temp)]
        else:
            temp = [self.shape[f](x[:, f, :]) for f in range(self.n_temp)]
        temp = torch.cat(temp, dim=1)
        s = self.bn_static(x_clinical.reshape(x_clinical.shape[0], -1))
        static = s * self.lin_static.weight                    # (n, C) per-covariate terms
        g = self.bn_sb(x_sb.reshape(x_sb.shape[0], -1))
        genomic = g * self.lin_sb.weight
        return temp, static, genomic, self.bias

    def forward(self, x_gene, x_agg, x_clinical, x_sb):
        temp, static, genomic, b = self.contributions(x_gene, x_agg, x_clinical, x_sb)
        if self.training and self.feature_dropout > 0:         # drop whole features (NAM regulariser)
            mask = (torch.rand_like(temp) > self.feature_dropout).float() / (1 - self.feature_dropout)
            temp = temp * mask
        eta = temp.sum(1, keepdim=True) + static.sum(1, keepdim=True) + genomic.sum(1, keepdim=True) + b
        return torch.sigmoid(eta) if self.output_sigmoid else eta


class SHERLOC_Proto(nn.Module):
    def __init__(self, d_input, d_agg_input, d_channel, d_output, d_other_input, device,
                 n_prototypes=4, log_eps=0.05, add_delta=True, temperature=1.0,
                 output_sigmoid=True, dropout=0.2, keep_panel_arm=False):
        super().__init__()
        self.keep_panel_arm = keep_panel_arm
        self.log_eps, self.add_delta, self.output_sigmoid = log_eps, add_delta, output_sigmoid
        c = d_channel + int(add_delta)
        self.d_traj = d_agg_input * c
        self.bn_traj = nn.BatchNorm1d(self.d_traj, affine=False)   # fixed scale -> prototypes interpretable
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, self.d_traj))
        self.log_tau = nn.Parameter(torch.tensor(float(temperature)).log())
        self.bn_other = nn.BatchNorm1d(d_other_input)
        self.bn_sb = nn.BatchNorm1d(d_channel * 2)
        self.dropout = nn.Dropout(dropout)
        d_panel = 0
        if keep_panel_arm:   # hybrid: SHERLOC's feature-specific temporal maps on raw panel features
            self.mlps = nn.ModuleList([nn.Linear(d_channel, 2) for _ in range(d_agg_input)])
            self.bn_agg = nn.BatchNorm1d(2 * d_agg_input)
            d_panel = 2 * d_agg_input
        self.fc = nn.Linear(n_prototypes + d_panel + d_other_input + d_channel * 2, d_output)

    def trajectory(self, x_agg):
        x = _log(x_agg, self.log_eps) if self.log_eps else x_agg
        if self.add_delta:
            x = torch.cat((x, (x[..., -1] - x[..., 0]).unsqueeze(-1)), dim=-1)
        return self.bn_traj(x.reshape(x.shape[0], -1))

    def membership(self, x_agg):
        z = self.trajectory(x_agg)
        d2 = torch.cdist(z, self.prototypes) ** 2
        return torch.softmax(-d2 / self.log_tau.exp(), dim=1)

    def forward(self, x_gene, x_agg, x_clinical, x_sb):
        m = self.membership(x_agg)
        s = self.bn_other(x_clinical.reshape(x_clinical.shape[0], -1))
        g = self.bn_sb(x_sb.reshape(x_sb.shape[0], -1))
        parts = [m]
        if self.keep_panel_arm:
            pa = torch.cat([self.mlps[f](x_agg[:, f, :]) for f in range(len(self.mlps))], dim=1)
            parts.append(self.bn_agg(pa))
        out = self.fc(self.dropout(torch.cat(parts + [s, g], dim=-1)))
        return torch.sigmoid(out) if self.output_sigmoid else out
