import torch
from torch.nn import Module

from sherloc_models.gene_encoders import GENE_ENCODERS, transform_gene_input


class SHERLOC_v2(Module):
    """
        SHERLOC with PADA-1 extensions. With the defaults
        (output_sigmoid=True, panel_mode="linear", panel_delta=False, gene_delta=False,
        use_gene_arm=True) it is identical to SHERLOC (same layers, same initialisation order).

        Extensions
        ----------
        output_sigmoid (bool): False = linear log-risk output.
        panel_mode ("linear" | "mlp"): feature-specific temporal map for each panel-level
            feature: Linear(c, 2) as in SHERLOC, or Linear(c, 4) -> ReLU -> Linear(4, 2).
        panel_delta (bool): add a change channel (last minus first timepoint) to every
            panel-level feature. With log-scale inputs this channel is the log-ratio C2/BL.
        gene_delta (bool): same change channel for the gene-level shared filter.
        use_gene_arm (bool): False drops the gene-level arm (ablation / simplification).
        gene_mode (str | None): None = SHERLOC's shared 1-D temporal convolution; otherwise one of
            sherloc_models.gene_encoders.GENE_ENCODERS (conv, conv2, mlp, gru, attn_pool, temporal_attn,
            gene_transformer, gene_set) as the gene-level longitudinal encoder.
        gene_input: raw | log | clonal | raw_clonal | log_clonal (see gene_encoders.transform_gene_input);
            two-channel inputs need gene_mode in (conv, conv2, mlp, gene_set).
        output_scale (float | "learn"): log-risk = scale * sigmoid(eta). SHERLOC uses 1, which bounds
            the hazard ratio between any two patients by e (~2.7) and compresses individual survival
            predictions; a larger scale keeps the output bounded (regularising) but realistic.

        Args (as SHERLOC):
            d_input, d_agg_input, d_channel, d_output, d_other_input, device.
    """
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_other_input: int,
                 device: str, output_sigmoid: bool = True, panel_mode: str = "linear",
                 panel_delta: bool = False, gene_delta: bool = False, use_gene_arm: bool = True,
                 output_scale=1.0, gene_mode=None, gene_hidden=8, gene_input="raw",
                 use_panel_arm=True, use_encoder_arm=True, n_event_tokens=0, event_mode="shared"):
        # n_event_tokens: the last n_event_tokens rows of the gene input are longitudinal gene-level event tokens
        # (functional class, co-mutation, emergence / clearance; build_event_tokens.py) with their own arm:
        # batch norm per token + temporal filter Linear(T, 1) + sigmoid, shared across tokens
        # (event_mode="shared") or one filter per token ("specific"); one Cox weight per token.
        super(SHERLOC_v2, self).__init__()
        self.n_event = int(n_event_tokens)
        d_input = d_input - self.n_event          # VAF genes only for the original gene arm

        self.d_input = d_input
        self.nb_agg = d_agg_input
        self.d_channel = d_channel
        self.output_sigmoid = output_sigmoid
        self.panel_delta = panel_delta
        self.gene_delta = gene_delta
        self.use_gene_arm = use_gene_arm
        if output_scale == "learn":
            self.log_scale = torch.nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_scale", torch.tensor([float(output_scale)]).log())
        c_gene = d_channel + int(gene_delta)
        c_panel = d_channel + int(panel_delta)

        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)
        # Gene-level arm: batch norm per gene + shared temporal filter
        self.bn_input = torch.nn.BatchNorm1d(d_input)
        self.gene_mode = gene_mode
        self.gene_input = gene_input
        t_in = d_channel * (2 if gene_input in ("raw_clonal", "log_clonal") else 1)
        if gene_mode is None:
            self.filter1 = torch.nn.Linear(c_gene, 1)
            gene_k = 1
        else:
            enc = GENE_ENCODERS[gene_mode]
            if gene_mode in ("mlp",) and t_in != d_channel:
                self.gene_enc = torch.nn.Sequential(torch.nn.Linear(t_in, gene_hidden), torch.nn.ReLU(),
                                                    torch.nn.Linear(gene_hidden, 1))
                self.gene_enc.out_per_gene = 1
                self._mlp_multi = True
            else:
                self.gene_enc = enc(T=t_in, n_genes=d_input, d=gene_hidden)
                self._mlp_multi = False
            gene_k = getattr(self.gene_enc, "out_per_gene", None)
        # Panel-level arm: one temporal map per feature
        if panel_mode == "mlp":
            self.mlps = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(c_panel, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
                for _ in range(d_agg_input)])
        else:
            self.mlps = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(c_panel, 2)) for _ in range(d_agg_input)])
        self.bn_agg_input = torch.nn.BatchNorm1d(d_agg_input * 2)
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        self.bn_SB = torch.nn.BatchNorm1d((d_channel * 2))
        if gene_mode == "gene_set":
            d_gene_out = self.gene_enc.out_total if use_gene_arm else 0
        else:
            d_gene_out = d_input * gene_k if use_gene_arm else 0
        self.use_panel_arm, self.use_encoder_arm = use_panel_arm, use_encoder_arm
        if self.n_event:
            self.bn_event = torch.nn.BatchNorm1d(self.n_event)
            self.event_mode = event_mode
            if event_mode == "specific":
                self.event_w = torch.nn.Parameter(torch.randn(self.n_event, d_channel) * 0.3)
                self.event_b = torch.nn.Parameter(torch.zeros(self.n_event))
            else:
                self.event_filter = torch.nn.Linear(d_channel, 1)
        self.fc_1 = torch.nn.Linear((d_channel * 2) * int(use_encoder_arm) + d_gene_out + self.n_event
                                    + int(2 * d_agg_input) * int(use_panel_arm) + d_other_input, d_output)

    @staticmethod
    def _add_delta(x):
        """x: (n, features, timepoints) -> append last - first timepoint as an extra channel."""
        return torch.cat((x, (x[..., -1] - x[..., 0]).unsqueeze(-1)), dim=-1)

    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
        parts = []
        if self.n_event:
            x_ev = self.bn_event(x[:, -self.n_event:, :])
            x = x[:, :-self.n_event, :]
            if self.event_mode == "specific":
                ev = torch.sigmoid((x_ev * self.event_w).sum(-1) + self.event_b)
            else:
                ev = torch.sigmoid(self.event_filter(x_ev)).squeeze(-1)
        if self.use_gene_arm:
            x = transform_gene_input(x, self.gene_input)
            x = self.bn_input(x)
            if self.gene_mode is None:
                if self.gene_delta:
                    x = self._add_delta(x)
                x = self.sigmoid(self.filter1(x))
            elif self._mlp_multi:
                x = torch.sigmoid(self.gene_enc(x)).squeeze(-1)
            else:
                x = self.gene_enc(x)
            parts.append(x.reshape(x.shape[0], -1))
        if self.n_event:
            parts.append(ev)

        if self.panel_delta:
            x_agg = self._add_delta(x_agg)
        x_temp = [self.mlps[f](x_agg[:, f, :]).unsqueeze(1) for f in range(self.nb_agg)]
        x_agg = torch.cat(x_temp, dim=1).reshape(x_agg.shape[0], -1)
        if self.use_panel_arm:
            parts.append(self.bn_agg_input(x_agg))

        x_clinical = self.bn_other_input(x_clinical.reshape(x_clinical.shape[0], -1))
        x_sb = self.bn_SB(x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1))
        parts += [x_clinical] + ([x_sb] if self.use_encoder_arm else [])

        x_new = self.dropout(torch.cat(parts, dim=-1))
        x_new = self.fc_1(x_new)
        if self.output_sigmoid:
            x_new = self.sigmoid(x_new) * self.log_scale.exp()
        return x_new
