"""Graph I/O, Bernstein spectral bank, and evaluation metrics for SOAR."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


MAT_RELATIONS = {
    "amazon": ("net_upu", "net_usu", "net_uvu"),
    "yelpchi": ("net_rur", "net_rtr", "net_rsr"),
}


@dataclass
class GraphData:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_indices: list[torch.Tensor]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sha256(*masks: torch.Tensor) -> str:
    if len(masks) != 3:
        raise ValueError("expected train, validation, and test masks")
    digest = hashlib.sha256()
    for name, mask in zip(("train", "validation", "test"), masks):
        indices = torch.where(mask.detach().cpu().bool())[0].long().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(indices.numpy().tobytes())
    return digest.hexdigest()


def validate_data(data: GraphData) -> None:
    if data.x.ndim != 2 or min(data.x.shape) < 1:
        raise ValueError("x must have shape (N, d) with N,d > 0")
    if data.y.shape != (data.num_nodes,):
        raise ValueError("y must have shape (N,)")
    labels = data.y.detach().cpu().long()
    if not set(labels.tolist()).issubset({-1, 0, 1}):
        raise ValueError("labels must be binary or masked as -1")
    if bool((labels == -1)[~data.test_mask].any()):
        raise ValueError("only held-out test labels may be masked as -1")
    if not data.edge_indices:
        raise ValueError("at least one relation is required")
    for edge_index in data.edge_indices:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("each edge_index must have shape (2, E)")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= data.num_nodes
        ):
            raise ValueError("edge_index contains an invalid node")
    masks = (data.train_mask, data.val_mask, data.test_mask)
    for mask in masks:
        if mask.dtype != torch.bool or mask.shape != (data.num_nodes,):
            raise ValueError("split masks must be boolean vectors of length N")
        if not bool(mask.any()):
            raise ValueError("each split must be nonempty")
    if any(bool((left & right).any()) for left, right in itertools.combinations(masks, 2)):
        raise ValueError("train, validation, and test masks must be disjoint")


def _trusted_torch_load(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_frozen_masks(path: Path, labels: torch.Tensor) -> tuple[torch.Tensor, ...]:
    payload = _trusted_torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError("mask file must contain a dictionary")
    label_hash = hashlib.sha256(
        labels.cpu().numpy().astype(np.int64, copy=False).tobytes()
    ).hexdigest()
    if payload.get("label_sha256") != label_hash:
        raise ValueError("mask label hash does not match the dataset")
    try:
        masks = tuple(
            torch.as_tensor(payload[name]).reshape(-1).bool()
            for name in ("train", "val", "test")
        )
    except KeyError as error:
        raise KeyError(f"mask file is missing {error.args[0]!r}") from error
    return masks


def _load_mat(path: Path) -> tuple[str, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    import scipy.io as scipy_io
    import scipy.sparse as scipy_sparse

    fields = {name for name, _, _ in scipy_io.whosmat(path)}
    if set(MAT_RELATIONS["amazon"]).issubset(fields):
        name = "amazon"
    elif set(MAT_RELATIONS["yelpchi"]).issubset(fields):
        name = "yelpchi"
    else:
        raise ValueError("MAT file is neither canonical Amazon nor YelpChi")
    required = ["features", "label", *MAT_RELATIONS[name]]
    missing = set(required) - fields
    if missing:
        raise KeyError(f"MAT file is missing fields: {sorted(missing)}")
    payload = scipy_io.loadmat(path, variable_names=required)
    raw_x = payload["features"]
    if scipy_sparse.issparse(raw_x):
        raw_x = raw_x.toarray()
    x = torch.from_numpy(np.asarray(raw_x, dtype=np.float32))
    y = torch.from_numpy(
        np.asarray(payload["label"]).reshape(-1).astype(np.int64, copy=False)
    )
    edges = []
    for relation in MAT_RELATIONS[name]:
        matrix = payload[relation]
        if not scipy_sparse.issparse(matrix):
            matrix = scipy_sparse.csr_matrix(matrix)
        matrix = matrix.maximum(matrix.T).tocsr()
        matrix.setdiag(0)
        matrix.eliminate_zeros()
        upper = scipy_sparse.triu(matrix, k=1, format="coo")
        edges.append(torch.tensor(np.vstack((upper.row, upper.col)), dtype=torch.long))
    return name, x, y.long(), edges


def _load_dgl(path: Path) -> tuple[str, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    try:
        from dgl.data.utils import load_graphs
    except ImportError as error:
        raise RuntimeError("DGL is required to load T-Finance/T-Social") from error
    graphs, _ = load_graphs(str(path))
    if len(graphs) != 1:
        raise ValueError("expected exactly one serialized DGL graph")
    graph = graphs[0]
    name = path.name.lower()
    x = graph.ndata["feature"].float()
    y = graph.ndata["label"]
    if name == "tfinance":
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("T-Finance labels must be one-hot with two columns")
        y = y.argmax(1)
    source, target = graph.edges()
    return name, x, y.reshape(-1).long(), [torch.stack((source, target)).long()]


def load_graph(
    data_path: str | Path,
    mask_path: str | Path,
    *,
    reveal_test: bool = False,
) -> GraphData:
    """Load data with test labels masked unless final evaluation opts in."""
    data_path, mask_path = Path(data_path), Path(mask_path)
    if not data_path.exists() or not mask_path.is_file():
        raise FileNotFoundError("data and frozen-mask files must both exist")
    if data_path.suffix.lower() == ".mat":
        name, x, y, edges = _load_mat(data_path)
    elif data_path.name.lower() in {"tfinance", "tsocial"}:
        name, x, y, edges = _load_dgl(data_path)
    else:
        raise ValueError("supported data are canonical .mat files or DGL T-Finance/T-Social")
    train, val, test = _load_frozen_masks(mask_path, y)
    visible_y = y if reveal_test else y.masked_fill(test, -1)
    data = GraphData(name, x, visible_y, edges, train, val, test)
    validate_data(data)
    return data


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Label-free transductive column standardization."""
    x = x.detach().float().clone()
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True, unbiased=False)
    valid = torch.isfinite(mean) & torch.isfinite(std) & (std > 1e-8)
    out = torch.zeros_like(x)
    out[:, valid[0]] = (x[:, valid[0]] - mean[:, valid[0]]) / std[:, valid[0]]
    out[~torch.isfinite(out)] = 0.0
    return out.detach()


def prepare_features(x: torch.Tensor, mode: str = "standard") -> torch.Tensor:
    """Apply a fixed label-free feature transform."""
    if mode == "raw":
        transformed = x.detach().float().clone()
        transformed[~torch.isfinite(transformed)] = 0.0
        return transformed
    if mode == "standard":
        transformed = x
    elif mode == "signed_log":
        transformed = torch.sign(x) * torch.log1p(x.abs())
    else:
        raise ValueError("feature mode must be 'raw', 'standard', or 'signed_log'")
    return standardize(transformed)


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    *,
    self_loops: bool = False,
) -> torch.Tensor | None:
    """Symmetric degree-normalized adjacency; isolated inverse degrees are 0."""
    edge = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge.numel() == 0:
        return None
    if int(edge.min()) < 0 or int(edge.max()) >= num_nodes:
        raise ValueError("edge_index contains an invalid node")
    lo = torch.minimum(edge[0], edge[1])
    hi = torch.maximum(edge[0], edge[1])
    keep = lo != hi
    keys = torch.unique(lo[keep] * num_nodes + hi[keep], sorted=True)
    if self_loops:
        diagonal = torch.arange(num_nodes, dtype=torch.long)
        keys = torch.unique(torch.cat((keys, diagonal * num_nodes + diagonal)), sorted=True)
    if not keys.numel():
        return None
    lo, hi = keys.div(num_nodes, rounding_mode="floor"), keys.remainder(num_nodes)
    off_diagonal = lo != hi
    if self_loops:
        diagonal = lo == hi
        indices = torch.stack(
            (
                torch.cat((lo[off_diagonal], hi[off_diagonal], lo[diagonal])),
                torch.cat((hi[off_diagonal], lo[off_diagonal], lo[diagonal])),
            )
        ).to(device)
    else:
        indices = torch.stack((torch.cat((lo, hi)), torch.cat((hi, lo)))).to(device)
    degree = torch.bincount(indices[0], minlength=num_nodes).to(torch.float32)
    inv = torch.zeros_like(degree)
    inv[degree > 0] = degree[degree > 0].pow(-0.5)
    values = inv[indices[0]] * inv[indices[1]]
    kwargs = {"device": device}
    if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
        with torch.sparse.check_sparse_tensor_invariants():
            normalized = torch.sparse_coo_tensor(
                indices, values, (num_nodes, num_nodes), check_invariants=True, **kwargs
            ).coalesce()
    else:
        normalized = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), **kwargs
        ).coalesce()
    return normalized


def bernstein_coefficients(S: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if S < 0:
        raise ValueError("S must be nonnegative")
    coeff = torch.zeros((S + 1, S + 1), dtype=torch.float64)
    for s in range(S + 1):
        for k in range(s, S + 1):
            coeff[s, k] = (
                math.comb(S, s) * math.comb(S - s, k - s) * (-1.0) ** (k - s) / 2.0**k
            )
    return coeff.to(dtype)


def apply_bernstein(
    adjacency: torch.Tensor | None, x: torch.Tensor, S: int
) -> torch.Tensor:
    """Stacked Bernstein bank ``Ψ_s(L̃) x``, shape ``[K, N, d]``."""
    coeff = bernstein_coefficients(S, x.dtype).to(device=x.device, dtype=x.dtype)
    if adjacency is None:
        return torch.stack([row.sum() * x for row in coeff])
    if adjacency.device != x.device:
        raise ValueError("adjacency and x must be on the same device")
    bands = [torch.zeros_like(x) for _ in range(S + 1)]
    current = x
    for k in range(S + 1):
        for s in range(k + 1):
            bands[s] = bands[s] + coeff[s, k] * current
        if k < S:
            current = current - torch.sparse.mm(adjacency, current)
    return torch.stack(bands)


def as_spmm_adjacency(adjacency: torch.Tensor | None) -> torch.Tensor | None:
    """CSR SpMM uses about half the COO footprint on T-Social."""
    if adjacency is None:
        return None
    if adjacency.layout == torch.sparse_csr:
        return adjacency
    if adjacency.is_sparse:
        return adjacency.coalesce().to_sparse_csr()
    return adjacency


def _bernstein_poly(adj: torch.Tensor, hidden: torch.Tensor, coeff_row: torch.Tensor) -> torch.Tensor:
    current = hidden
    acc = coeff_row[0] * current
    for k in range(1, int(coeff_row.numel())):
        current = current - torch.sparse.mm(adj, current)
        acc = acc + coeff_row[k] * current
    return acc


class StreamV1RelationFn(torch.autograd.Function):
    """Fold all bands into the neutral Linear and signed mix without stacking the bank."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        crow: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        neu_weight: torch.Tensor,
        neu_bias: torch.Tensor,
        orientation: torch.Tensor,
        coeff: torch.Tensor,
    ):
        n, hidden_dim = hidden.shape
        n_bands = int(orientation.numel())
        ctx.n = n
        ctx.hidden_dim = hidden_dim
        ctx.n_bands = n_bands
        ctx.save_for_backward(hidden, crow, col, values, neu_weight, orientation, coeff)
        adj = torch.sparse_csr_tensor(
            crow, col, values, size=(n, n), device=hidden.device, dtype=hidden.dtype
        )
        neu = hidden.new_zeros(n, hidden_dim)
        if neu_bias is not None:
            neu = neu + neu_bias
        mix = hidden.new_zeros(n, hidden_dim)
        energy = hidden.new_zeros(n, n_bands)
        for band in range(n_bands):
            piece = _bernstein_poly(adj, hidden, coeff[band])
            neu = neu + F.linear(
                piece,
                neu_weight[:, band * hidden_dim : (band + 1) * hidden_dim],
                None,
            )
            mix = mix + orientation[band] * piece
            energy[:, band] = piece.square().sum(-1)
            del piece
        ctx.mark_non_differentiable(energy)
        ctx.has_bias = neu_bias is not None
        return neu, mix, energy

    @staticmethod
    def backward(ctx, d_neu, d_mix, _d_energy):
        hidden, crow, col, values, neu_weight, orientation, coeff = ctx.saved_tensors
        n, hidden_dim, n_bands = ctx.n, ctx.hidden_dim, ctx.n_bands
        with torch.no_grad():
            if hidden.is_cuda:
                torch.cuda.empty_cache()
            adj = torch.sparse_csr_tensor(
                crow, col, values, size=(n, n), device=hidden.device, dtype=hidden.dtype
            )
            d_hidden = torch.zeros_like(hidden)
            d_weight = torch.zeros_like(neu_weight)
            d_orient = torch.zeros_like(orientation)
            d_bias = d_neu.sum(0) if ctx.has_bias else None
            for band in range(n_bands):
                piece = _bernstein_poly(adj, hidden, coeff[band])
                sl = slice(band * hidden_dim, (band + 1) * hidden_dim)
                d_weight[:, sl] = d_neu.transpose(0, 1) @ piece
                d_orient[band] = (d_mix * piece).sum()
                d_piece = d_neu @ neu_weight[:, sl].transpose(0, 1) + orientation[band] * d_mix
                d_hidden = d_hidden + _bernstein_poly(adj, d_piece, coeff[band])
                del piece, d_piece
        return d_hidden, None, None, None, d_weight, d_bias, d_orient, None


class BernsteinBandFn(torch.autograd.Function):
    """One Bernstein band with adjoint backward (no stored powers)."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        crow: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        coeff_row: torch.Tensor,
    ) -> torch.Tensor:
        n = hidden.shape[0]
        ctx.n = n
        ctx.save_for_backward(crow, col, values, coeff_row)
        adj = torch.sparse_csr_tensor(
            crow, col, values, size=(n, n), device=hidden.device, dtype=hidden.dtype
        )
        current = hidden
        acc = coeff_row[0] * current
        for k in range(1, int(coeff_row.numel())):
            current = current - torch.sparse.mm(adj, current)
            acc = acc + coeff_row[k] * current
        return acc

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        crow, col, values, coeff_row = ctx.saved_tensors
        n = ctx.n
        adj = torch.sparse_csr_tensor(
            crow, col, values, size=(n, n), device=grad.device, dtype=grad.dtype
        )
        current = grad
        acc = coeff_row[0] * current
        for k in range(1, int(coeff_row.numel())):
            current = current - torch.sparse.mm(adj, current)
            acc = acc + coeff_row[k] * current
        return acc, None, None, None, None


def eval_bernstein_band(
    adjacency: torch.Tensor | None,
    x: torch.Tensor,
    S: int,
    band: int,
    coeff: torch.Tensor | None = None,
) -> torch.Tensor:
    """One Bernstein band. Peak extra activations: current + accumulator."""
    if coeff is None:
        coeff = bernstein_coefficients(S, x.dtype).to(device=x.device, dtype=x.dtype)
    else:
        coeff = coeff.to(device=x.device, dtype=x.dtype)
        if tuple(coeff.shape) != (S + 1, S + 1):
            raise ValueError("coeff must have shape (S+1, S+1)")
    if not 0 <= int(band) <= S:
        raise ValueError("band index out of range")
    row = coeff[int(band)]
    if adjacency is None:
        return row.sum() * x
    if adjacency.device != x.device:
        raise ValueError("adjacency and x must be on the same device")
    adjacency = as_spmm_adjacency(adjacency)
    if adjacency.layout == torch.sparse_csr:
        return BernsteinBandFn.apply(
            x,
            adjacency.crow_indices(),
            adjacency.col_indices(),
            adjacency.values(),
            row.contiguous(),
        )
    current = x
    acc = row[0] * current
    for k in range(int(S)):
        current = current - torch.sparse.mm(adjacency, current)
        acc = acc + row[k + 1] * current
    return acc


def probabilities_and_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    y = labels.detach().cpu().reshape(-1).long()
    z = logits.detach()
    if not bool(torch.isfinite(z).all()):
        raise ValueError("logits must be finite")
    if z.ndim == 2 and z.shape[-1] == 2:
        if z.shape[0] != int(y.numel()):
            raise ValueError("logits and labels must be aligned")
        probability = torch.softmax(z, dim=-1)[:, 1].cpu().numpy()
    else:
        z = z.reshape(-1)
        if z.numel() != int(y.numel()):
            raise ValueError("logits and labels must be aligned")
        probability = torch.sigmoid(z).cpu().numpy()
    y_np = y.numpy().astype(np.int64)
    if not set(np.unique(y_np)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    return probability, y_np


def select_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Validation-only operational threshold on the BWGNN 19-point grid."""
    from sklearn.metrics import f1_score

    probability, y = probabilities_and_labels(logits, labels)
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.linspace(0.05, 0.95, 19)
    scores = np.asarray(
        [
            f1_score(y, probability >= value, average="macro", zero_division=0)
            for value in candidates
        ]
    )
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0.0))
    order = np.lexsort((candidates[best], np.abs(candidates[best] - 0.5)))
    return float(candidates[best[order[0]]])


def evaluate(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    probability, y = probabilities_and_labels(logits, labels)
    auroc = 0.5 if len(np.unique(y)) < 2 else float(roc_auc_score(y, probability))
    auprc = (
        float(y.mean())
        if len(np.unique(y)) < 2
        else float(average_precision_score(y, probability))
    )
    prediction = probability >= threshold
    tp = float(np.logical_and(prediction, y == 1).sum())
    tn = float(np.logical_and(~prediction, y == 0).sum())
    fp = float(np.logical_and(prediction, y == 0).sum())
    fn = float(np.logical_and(~prediction, y == 1).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auroc": auroc,
        "auprc": auprc,
        "gmean": float(np.sqrt(tpr * tnr)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "threshold": float(threshold),
    }
