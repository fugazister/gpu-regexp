"""ScanTagger: sparse hashed embeddings -> gated bidirectional scans -> 5-tap conv -> per-token roles.

Each weight read goes through int6 fake-quantization, so the model is trained as it ships.
web/program.js has to compute exactly this; eval/parity.mjs and eval/webgpu.mjs check that it does.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 31  # signed 6-bit symmetric


def fq(p):
    """Straight-through int6 rounding, per tensor."""
    scale = p.detach().abs().max().clamp(min=1e-8) / QMAX
    return p + ((p / scale).round() * scale - p).detach()


class Dense(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_out, d_in))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        return x @ fq(self.w).T + fq(self.b)


def scan(a, u, lengths, reverse=False):
    """h[t] = a[t] * h[t-1] + (1 - a[t]) * tanh-candidate u[t], over each sequence's real tokens."""
    B, T, D = a.shape
    outs = [None] * T
    h = torch.zeros(B, D, device=a.device)
    live = (torch.arange(T, device=a.device)[None] < lengths[:, None]).float()
    for t in (range(T - 1, -1, -1) if reverse else range(T)):
        m = live[:, t, None]
        h = m * (a[:, t] * h + (1 - a[:, t]) * u[:, t]) + (1 - m) * h
        outs[t] = h
    return torch.stack(outs, 1) * live[..., None]


class ScanLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.uf, self.af, self.ub, self.ab = Dense(d, d), Dense(d, d), Dense(d, d), Dense(d, d)
        self.proj = Dense(2 * d, d)

    def forward(self, x, lengths):
        f = scan(torch.sigmoid(self.af(x)), torch.tanh(self.uf(x)), lengths)
        b = scan(torch.sigmoid(self.ab(x)), torch.tanh(self.ub(x)), lengths, reverse=True)
        return x + self.proj(torch.cat([f, b], -1))


class ScanTagger(nn.Module):
    def __init__(self, n_rows, n_roles, d=24, layers=2, hidden=48):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, hidden=hidden, n_rows=n_rows, n_roles=n_roles)
        self.emb = nn.Parameter(torch.randn(n_rows, d) * 0.05)
        self.scans = nn.ModuleList(ScanLayer(d) for _ in range(layers))
        self.conv = nn.Parameter(torch.zeros(5, d))  # depthwise, one 5-tap filter per channel
        self.head1 = Dense(2 * d, hidden)
        self.head2 = Dense(hidden, n_roles)

    def forward(self, rows, lengths):
        """rows: [B, T, slots] of embedding row ids. Tokens past lengths[b] are ignored."""
        B, T, _ = rows.shape
        live = (torch.arange(T, device=rows.device)[None] < lengths[:, None]).float()[..., None]
        x = fq(self.emb)[rows].sum(2) * live
        for s in self.scans:
            x = s(x, lengths) * live
        w = fq(self.conv)
        pad = F.pad(x.transpose(1, 2), (2, 2))  # [B, D, T + 4]
        c = F.conv1d(pad, w.T[:, None, :], groups=x.shape[-1]).transpose(1, 2)
        z = (x + c) * live
        pool = z.sum(1) / lengths[:, None].float().clamp(min=1)
        h = F.relu(self.head1(torch.cat([z, pool[:, None].expand(-1, T, -1)], -1)))
        return self.head2(h)
