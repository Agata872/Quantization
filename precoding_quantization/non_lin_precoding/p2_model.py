"""GNN used by the Prompt-2 ablation (p2_ablation.py): one backbone, switchable input design.

The six ablation variants differ ONLY in what the network reads and how it is trained, so every
variant uses this class with the same widths and the same 4-dim input layer:

  edge='raw'     z_mk = [Re h, Im h, 0, 0]                     Feys et al. (27), zero-padded to 4
  edge='polar4'  z_mk = [cos arg h, sin arg h, std log|h|/||H||, std log ||H||]   eq:tp_edge_init
  ref=False      z_m  = [0, 0, 0, 0],  z_k = [Re s, Im s, 0, 0]  Feys et al. (28)-(29), padded
  ref=True       z_m  = [Re y_lin, Im y_lin, 0, 0] / l_max,
                 z_k  = [Re s, Im s, Re r_lin, Im r_lin / (||H|| sqrt(Pt))]    eq:tp_ap_init/ue_init

A zero input column only removes the matching rows of the input-layer weights from the
computation, so 'raw' + ref=False is the Feys input layer d0=2 exactly, apart from the Xavier bound
of the remaining weights (sqrt(6/132) instead of sqrt(6/130), 0.8 %). The parameter count is the
same for all variants.

Weights are named exactly as in exp_cellfree_ablation.GNNv2, so a paper checkpoint (GNNv2,
input_mode='polar4' with the quantized-linear warm start) loads into P2GNN(edge='polar4',
ref=True) unchanged; p2_ablation.py --selftest checks that both give the same decisions.

Layer: GNN_layer_fast written with broadcasting instead of repeat_interleave/tile. Same math, same
parameter shapes and names; it only avoids materialising two (bs x MK x d) copies per layer.
"""
import torch
import torch.nn.functional as F
from torch import nn


class GNNLayerBcast(nn.Module):
    """model.GNN_layer_fast with broadcast node terms. Edge tensors are kept as bs x M x K x d."""

    def __init__(self, din, dout, outputlayer=False):
        super().__init__()
        self.outputlayer = outputlayer
        for name, shape in (('Wedge', (din, dout)), ('Wm', (din, dout)), ('Wk', (din, dout)),
                            ('Wself_m', (din, dout)), ('Wself_k', (din, dout)),
                            ('Wneigh_m', (dout, dout)), ('Wneigh_k', (dout, dout))):
            w = nn.Parameter(torch.zeros(*shape))
            nn.init.xavier_uniform_(w)
            setattr(self, name, w)

    def forward(self, z_mk, z_m, z_k):
        # z_mk: bs x M x K x din, z_m: bs x M x din, z_k: bs x K x din
        e = F.leaky_relu(z_mk @ self.Wedge + (z_m @ self.Wm)[:, :, None, :]
                         + (z_k @ self.Wk)[:, None, :, :])
        msg_k = e.mean(dim=1)                       # antennas -> users, bs x K x dout
        msg_m = e.mean(dim=2)                       # users -> antennas, bs x M x dout
        zm = z_m @ self.Wself_m + msg_m @ self.Wneigh_m
        if not self.outputlayer:
            zm = F.leaky_relu(zm)
        zk = F.leaky_relu(z_k @ self.Wself_k + msg_k @ self.Wneigh_k)
        return e, zm, zk


class P2GNN(nn.Module):
    def __init__(self, M, K, dl, nr_hidden_layers, bits, tau, levels, edge='polar4', ref=True,
                 feat_stats=None, pt=40.0, output_type='gumbel_softmax_hard'):
        super().__init__()
        assert edge in ('raw', 'polar4')
        self.M, self.K, self.bits = M, K, bits
        self.nr_out_levels = 2 ** bits
        self.tau, self.levels, self.pt = tau, levels, pt
        self.edge, self.ref = edge, ref
        self.output_type = output_type
        self.register_buffer('fstats', torch.tensor(feat_stats if feat_stats is not None
                                                    else [0., 1., 0., 1.]))
        self.din = 4
        self.input_layer = GNNLayerBcast(4, dl)
        self.hidden_layers = nn.ModuleList([GNNLayerBcast(dl, dl) for _ in range(nr_hidden_layers)])
        self.output_layer = GNNLayerBcast(dl, 2 * self.nr_out_levels, outputlayer=True)
        self.last_logits = None

    def features(self, H, s, y_ref=None):
        """H: bs x M x K, s: bs x K, y_ref: bs x M (quantized linear reference) or None."""
        bs = H.shape[0]
        nrm = torch.linalg.vector_norm(H.reshape(bs, -1), dim=1).clamp_min(1e-12)
        zero_mk = torch.zeros_like(H.real)
        if self.edge == 'raw':
            z_mk = torch.stack((H.real, H.imag, zero_mk, zero_mk), -1)
        else:
            hn = H / nrm[:, None, None]
            mag = hn.abs().clamp_min(1e-12)
            lg = (torch.log(mag) - self.fstats[0]) / self.fstats[1]
            ln = ((torch.log(nrm) - self.fstats[2]) / self.fstats[3])[:, None, None].expand_as(lg)
            z_mk = torch.stack((hn.real / mag, hn.imag / mag, lg, ln), -1)
        zero_k = torch.zeros_like(s.real)
        if self.ref:
            assert y_ref is not None
            r = torch.einsum('bmk,bm->bk', H, y_ref) / (nrm * self.pt ** 0.5).unsqueeze(-1)
            z_k = torch.stack((s.real, s.imag, r.real, r.imag), -1)
            a = self.levels.abs().max()
            zero_m = torch.zeros_like(y_ref.real)
            z_m = torch.stack((y_ref.real / a, y_ref.imag / a, zero_m, zero_m), -1)
        else:
            z_k = torch.stack((s.real, s.imag, zero_k, zero_k), -1)
            z_m = torch.zeros(bs, self.M, 4, device=H.device, dtype=z_mk.dtype)
        return z_mk, z_m, z_k

    def logits(self, H, s, y_ref=None):
        z_mk, z_m, z_k = self.features(H, s, y_ref)
        z_mk, z_m, z_k = self.input_layer(z_mk, z_m, z_k)
        for layer in self.hidden_layers:
            z_mk, z_m, z_k = layer(z_mk, z_m, z_k)
        _, z_m, _ = self.output_layer(z_mk, z_m, z_k)
        return z_m.reshape(-1, self.M, 2, self.nr_out_levels)

    def forward(self, H, s, y_ref=None):
        logits = self.logits(H, s, y_ref)
        self.last_logits = logits
        if self.output_type == 'gumbel_softmax_hard':
            oh = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        elif self.output_type == 'argmax':
            oh = torch.zeros_like(logits).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
        else:
            raise ValueError(self.output_type)
        lv = (oh * self.levels).sum(-1)
        return lv[:, :, 0] + 1j * lv[:, :, 1]
