"""SOAR: signed orientation-aware residual for graph fraud detection.

Paper mapping (handmade.tex):
    energy_profiles            Eq. (profile)
    calibrate_orientation      Eqs. (orientation)--(reliability), optional (band-confidence)
    SOAR signed mix / fusion   Eqs. (hidden)--(h-sgn-concat)
    dual-path logits           Eq. (logits)
    directional_consistency    Eqs. (contrast)--(dc)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from data import (
    StreamV1RelationFn,
    apply_bernstein,
    as_spmm_adjacency,
    bernstein_coefficients,
    eval_bernstein_band,
    normalized_adjacency,
)

EPS_E = 1e-8
EPS_O = 1e-6
TAU_N = 20.0
VARIANTS = ("full", "orientation_off", "reliability_off", "signed_only", "magnitude")


@dataclass
class Orientation:
    o: torch.Tensor  # [R, K] in [-1, 1]
    rho: torch.Tensor  # [R] in [0, 1]
    resolved: torch.Tensor  # [R] bool
    band_confidence: torch.Tensor  # [R, K] in [0, 1]


def energy_profiles(
    x: torch.Tensor,
    edge_indices: list[torch.Tensor],
    S: int,
    *,
    stream_bands: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
    """Label-free band energy share ``q[N, R, K]``."""
    x = x.detach()
    profiles, activity, adjs = [], [], []
    k_bands = S + 1
    coeff = None
    if stream_bands:
        coeff = bernstein_coefficients(S, x.dtype).to(device=x.device, dtype=x.dtype)
    for edge in edge_indices:
        active = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        edge = torch.as_tensor(edge, dtype=torch.long, device=x.device)
        if edge.numel():
            active[edge.reshape(-1)] = True
        adj = normalized_adjacency(edge, x.shape[0], x.device, self_loops=False)
        if stream_bands:
            adj = as_spmm_adjacency(adj)
            cols = []
            for band in range(k_bands):
                piece = eval_bernstein_band(adj, x, S, band, coeff)
                cols.append(piece.square().sum(-1))
                del piece
            energy = torch.stack(cols, dim=1)
        else:
            bands = apply_bernstein(adj, x, S)
            energy = bands.square().sum(-1).transpose(0, 1)
        q = energy / energy.sum(-1, keepdim=True).clamp_min(EPS_E)
        q = torch.where(active[:, None], q, torch.full_like(q, 1.0 / k_bands))
        profiles.append(q)
        activity.append(active)
        adjs.append(adj)
    return torch.stack(profiles, dim=1), torch.stack(activity, dim=1), adjs


def calibrate_orientation(
    q: torch.Tensor,
    activity: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    tau_n: float = TAU_N,
) -> Orientation:
    """Train-only per-band orientation, relation reliability, and band confidence."""
    _, n_rel, n_bands = q.shape
    o = torch.zeros(n_rel, n_bands, dtype=q.dtype, device=q.device)
    rho = torch.zeros(n_rel, dtype=q.dtype, device=q.device)
    band_confidence = torch.zeros_like(o)
    resolved = torch.zeros(n_rel, dtype=torch.bool, device=q.device)
    y = labels.long()
    train = train_mask.bool()
    for relation in range(n_rel):
        mask = train & activity[:, relation]
        y_r = y[mask]
        q_r = q[mask, relation]
        n0 = int((y_r == 0).sum())
        n1 = int((y_r == 1).sum())
        if n0 < 2 or n1 < 2:
            continue
        mean0 = q_r[y_r == 0].mean(0)
        mean1 = q_r[y_r == 1].mean(0)
        pooled = q_r.std(0, unbiased=True).clamp_min(EPS_O)
        difference = mean1 - mean0
        o[relation] = torch.tanh(difference / (pooled + EPS_O))
        n_eff = 2.0 * n0 * n1 / (n0 + n1)
        support = min(n_eff / tau_n, 1.0)
        var0 = q_r[y_r == 0].var(0, unbiased=True)
        var1 = q_r[y_r == 1].var(0, unbiased=True)
        standard_error = (var0 / n0 + var1 / n1).sqrt().clamp_min(EPS_O)
        t_magnitude = difference.abs() / standard_error
        band_confidence[relation] = support * torch.erf(t_magnitude / math.sqrt(2.0))
        rho[relation] = (support * o[relation].abs()).mean()
        resolved[relation] = True
    return Orientation(
        o=o.detach(),
        rho=rho.detach(),
        resolved=resolved,
        band_confidence=band_confidence.detach(),
    )


def hidden_band_contrast(
    band_energy: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    activity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fraud-minus-normal hidden energy on training nodes."""
    contrasts, valid = [], []
    y = labels.long()
    for relation in range(band_energy.shape[1]):
        mask = train_mask.bool() & activity[:, relation]
        left = mask & (y == 0)
        right = mask & (y == 1)
        if bool(left.any()) and bool(right.any()):
            contrasts.append(
                band_energy[right, relation].mean(0) - band_energy[left, relation].mean(0)
            )
            valid.append(True)
        else:
            contrasts.append(band_energy.new_zeros(band_energy.shape[-1]))
            valid.append(False)
    return torch.stack(contrasts), torch.tensor(valid, device=band_energy.device)


def directional_consistency_loss(
    band_energy: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    activity: torch.Tensor,
    source_o: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Align hidden-band class contrast with stop-grad orientation."""
    target = source_o.detach()
    has_pos = bool((target > 1e-8).any())
    has_neg = bool((target < -1e-8).any())
    if not (has_pos and has_neg):
        return source_o.sum() * 0.0
    contrast, valid = hidden_band_contrast(band_energy, labels, train_mask, activity)
    weight = rho.detach() * valid.to(rho.dtype)
    weight = weight * (target.abs().sum(-1) > 1e-8).to(rho.dtype)
    per_rel = 1.0 - F.cosine_similarity(contrast, target, dim=-1, eps=1e-8)
    if float(weight.sum()) <= 0.0:
        return contrast.sum() * 0.0
    return (weight * per_rel).sum() / weight.sum()


class SOAR(nn.Module):
    """Orientation estimator is frozen; encoder, fusion, and readout are trained."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        S: int,
        orientation: Orientation,
        *,
        dropout: float = 0.5,
        lam: float = 1.0,
        variant: str = "full",
        fusion: str = "mean",
        encoder_layers: int = 1,
        num_classes: int = 1,
        band_confidence: bool = False,
        shared_head: bool = False,
        stream_bands: bool = False,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        if fusion not in {"mean", "concat"}:
            raise ValueError(f"unknown fusion {fusion}")
        if encoder_layers not in {1, 2}:
            raise ValueError("encoder_layers must be 1 or 2")
        if num_classes not in {1, 2}:
            raise ValueError("num_classes must be 1 or 2")
        self.S = int(S)
        self.K = S + 1
        self.lam = float(lam)
        self.variant = variant
        self.fusion = fusion
        self.num_classes = int(num_classes)
        self.band_confidence_enabled = bool(band_confidence)
        self.shared_head = bool(shared_head) or variant == "magnitude"
        self.stream_bands = bool(stream_bands)
        if encoder_layers == 1:
            self.encoder = nn.Linear(in_dim, hidden_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.dropout = nn.Dropout(dropout)
        self.neu_fuse = nn.Sequential(
            nn.Linear(hidden_dim * self.K, hidden_dim),
            nn.ReLU(),
        )
        relations = orientation.o.shape[0]
        neu_out = hidden_dim * relations if fusion == "concat" else hidden_dim
        self.head_neu = nn.Linear(neu_out, self.num_classes)
        self.head_sgn = None if self.shared_head else nn.Linear(neu_out, self.num_classes, bias=False)

        o = orientation.o.detach().float().clone()
        rho = orientation.rho.detach().float().clone()
        band_conf = orientation.band_confidence.detach().float().clone()
        if variant == "magnitude":
            o = o.abs()
        if variant == "orientation_off":
            rho = torch.zeros_like(rho)
        elif variant == "reliability_off":
            rho = orientation.resolved.to(rho.dtype)
        self.register_buffer("o", o)
        self.register_buffer("rho", rho)
        self.register_buffer("resolved", orientation.resolved.bool().clone())
        self.register_buffer("band_confidence", band_conf)

    def routed_orientation(self) -> torch.Tensor:
        if self.band_confidence_enabled:
            return self.o * self.band_confidence
        return self.o

    def _stream_v1_relation(
        self,
        adjacency: torch.Tensor | None,
        hidden: torch.Tensor,
        relation: int,
        routed_o: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacency = as_spmm_adjacency(adjacency)
        coeff = bernstein_coefficients(self.S, hidden.dtype).to(
            device=hidden.device, dtype=hidden.dtype
        )
        hidden_dim = hidden.shape[-1]
        weight = self.neu_fuse[0].weight
        bias = self.neu_fuse[0].bias
        if bias is None:
            bias = hidden.new_zeros(hidden_dim)
        if adjacency is None:
            bands = apply_bernstein(None, hidden, self.S)
            energy = bands.square().sum(-1).transpose(0, 1)
            share = energy / energy.sum(-1, keepdim=True).clamp_min(EPS_E)
            h_neu = self.neu_fuse(torch.cat(tuple(bands), dim=-1))
            mix = torch.einsum("k,knd->nd", routed_o[relation], bands)
            return h_neu, mix, share
        crow = adjacency.crow_indices()
        col = adjacency.col_indices()
        values = adjacency.values()
        neu_acc, mix, energy = StreamV1RelationFn.apply(
            hidden,
            crow,
            col,
            values,
            weight,
            bias,
            routed_o[relation].contiguous(),
            coeff,
        )
        share = energy / energy.sum(-1, keepdim=True).clamp_min(EPS_E)
        return torch.relu(neu_acc), mix, share

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        hidden = torch.relu(hidden)
        return self.dropout(hidden)

    def forward(
        self,
        x: torch.Tensor,
        adjacencies: list[torch.Tensor | None],
        activity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        routed_o = self.routed_orientation()
        if self.stream_bands:
            hidden = checkpoint(self.encode, x, use_reentrant=False)
        else:
            hidden = self.encode(x)
        neu_terms, sgn_terms, energies = [], [], []
        active = activity.to(hidden.dtype)
        for relation, adjacency in enumerate(adjacencies):
            if self.stream_bands:
                h_neu_r, mix_r, share = self._stream_v1_relation(
                    adjacency, hidden, relation, routed_o
                )
            else:
                bands = apply_bernstein(adjacency, hidden, self.S)
                energy = bands.square().sum(-1).transpose(0, 1)
                share = energy / energy.sum(-1, keepdim=True).clamp_min(EPS_E)
                h_neu_r = self.neu_fuse(torch.cat(tuple(bands), dim=-1))
                mix_r = torch.einsum("k,knd->nd", routed_o[relation], bands)
            energies.append(share)
            neu_terms.append(h_neu_r)
            sgn_terms.append(mix_r)
        neu = torch.stack(neu_terms, dim=1)
        mix = torch.stack(sgn_terms, dim=1)
        masked = neu * active.unsqueeze(-1)
        if self.fusion == "concat":
            h_neu = masked.flatten(1)
        else:
            denom = active.sum(1, keepdim=True).clamp_min(1.0)
            h_neu = masked.sum(1) / denom
        rho_w = active * self.rho[None, :]
        weighted_rel = mix * rho_w.unsqueeze(-1)
        if self.fusion == "concat":
            h_sgn = weighted_rel.flatten(1)
        elif mix.shape[1] == 1:
            h_sgn = mix[:, 0] * self.rho[0]
        else:
            rho_sum = rho_w.sum(1, keepdim=True)
            weighted = weighted_rel.sum(1)
            h_sgn = torch.where(
                rho_sum > 1e-8,
                weighted / rho_sum.clamp_min(1e-8),
                torch.zeros_like(weighted),
            )
        if self.variant == "signed_only":
            if self.shared_head:
                features = self.lam * h_sgn
                logits = self.head_neu(features)
            else:
                logits = self.lam * self.head_sgn(h_sgn)
            logit_neu = logits * 0
            logit_sgn = logits
        elif self.shared_head:
            features = h_neu + self.lam * h_sgn
            logits = self.head_neu(features)
            logit_neu = logits
            logit_sgn = logits * 0
        else:
            logit_neu = self.head_neu(h_neu)
            logit_sgn = self.lam * self.head_sgn(h_sgn)
            logits = logit_neu + logit_sgn
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
            logit_neu = logit_neu.squeeze(-1)
            logit_sgn = logit_sgn.squeeze(-1)
        return {
            "logits": logits,
            "logit_neu": logit_neu,
            "logit_sgn": logit_sgn,
            "routed_o": routed_o,
            "band_energy": torch.stack(energies, dim=1),
            "h_neu": h_neu,
            "h_sgn": h_sgn,
        }


def check_identities(
    model: SOAR,
    x: torch.Tensor,
    adjs: list[torch.Tensor | None],
    activity: torch.Tensor,
) -> dict[str, bool]:
    """Algebraic contracts used as smoke assertions, not paper results."""
    with torch.no_grad():
        out = model(x, adjs, activity)
        rho_zero = bool((model.rho.abs() <= 1e-12).all())
        signed_off = rho_zero and bool(out["h_sgn"].abs().max() <= 1e-6)
        orientation_off_ok = model.variant != "orientation_off" or signed_off
        confidence_ok = bool(
            torch.isfinite(model.band_confidence).all()
            and (model.band_confidence >= 0.0).all()
            and (model.band_confidence <= 1.0 + 1e-6).all()
        )
    return {
        "orientation_off_zeros_signed": orientation_off_ok,
        "finite_logits": bool(torch.isfinite(out["logits"]).all()),
        "band_confidence_is_bounded": confidence_ok,
    }
