"""Train SOAR on a frozen BWGNN split with validation-only selection.

Paper cells in configs/paper.json are selected by --dataset and --abundant/--scarce.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import as_spmm_adjacency, evaluate, load_graph, prepare_features, select_threshold
from protocol import (
    DATASETS,
    default_data_path,
    default_mask_path,
    ensure_masks,
    paper_cell,
)
from soar import (
    VARIANTS,
    SOAR,
    calibrate_orientation,
    check_identities,
    directional_consistency_loss,
    energy_profiles,
)

DEFAULT_GRAPH_MODE = {
    "yelpchi": "hetero",
    "amazon": "hetero",
    "tfinance": "homo",
    "tsocial": "homo",
}
ACTIVE_SOURCE_FILES = ("soar.py", "train.py", "data.py", "protocol.py")
PAPER = json.loads((_ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))

# Hyperparameters filled from the paper cell when the CLI leaves them unset.
_CELL_KEYS = (
    "S",
    "hidden",
    "dropout",
    "epochs",
    "patience",
    "lr",
    "weight_decay",
    "lam",
    "beta",
    "tau_n",
    "pos_weight_scale",
    "feature_mode",
    "profile_feature_mode",
    "graph_mode",
    "fusion",
    "encoder_layers",
    "num_classes",
    "variant",
    "shared_head",
    "band_confidence",
    "stream_bands",
)
_TEN_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ACTIVE_SOURCE_FILES
    }


def environment_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__)}
    for package in ("numpy", "scipy", "scikit-learn", "dgl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False


def supervised_loss(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[-1] == 2:
        ce_w = torch.tensor([1.0, float(pos_weight)], device=logits.device)
        return F.cross_entropy(logits, y.long(), weight=ce_w)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def score_split(logits: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, float], float]:
    threshold = select_threshold(logits, labels)
    return evaluate(logits, labels, threshold), threshold


def smoke(args: argparse.Namespace) -> dict[str, object]:
    """Synthetic identity checks plus a short optimization; not a paper result."""
    set_seed(args.seed)
    device = torch.device(args.device)
    nodes, S = 64, 2
    index = torch.arange(nodes, device=device)
    labels = (index % 2).long()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    signal = labels.float() * 2.0 - 1.0
    x = torch.stack(
        (
            signal + 0.08 * torch.randn(nodes, generator=generator, device=device),
            torch.sin(index.float() / 3.0),
            torch.cos(index.float() / 5.0),
            0.2 * torch.randn(nodes, generator=generator, device=device),
        ),
        dim=1,
    )
    edge_indices = []
    for shift in (1, 5):
        target = torch.roll(index, shifts=-shift)
        edge_indices.append(torch.stack((index, target)).cpu())
    q, activity, adjs = energy_profiles(x, edge_indices, S)
    train_mask = index < 48
    orientation = calibrate_orientation(q, activity, labels, train_mask)
    if not bool(orientation.resolved.all()):
        raise RuntimeError("smoke failed to resolve both relations")
    model = SOAR(
        x.shape[1],
        args.hidden,
        S,
        orientation,
        dropout=0.0,
        lam=1.0,
        variant="full",
    ).to(device)
    checks = check_identities(model, x, adjs, activity)
    y_train = labels[train_mask].float()
    pos_weight = ((y_train == 0).sum() / y_train.sum().clamp_min(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    last_loss = None
    for _ in range(min(args.epochs, 30)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x, adjs, activity)
        loss_sup = supervised_loss(out["logits"][train_mask], y_train, pos_weight)
        loss_dc = directional_consistency_loss(
            out["band_energy"], labels, train_mask, activity, out["routed_o"], model.rho
        )
        loss = loss_sup + 0.05 * loss_dc
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite smoke loss")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    off = SOAR(
        x.shape[1], args.hidden, S, orientation, dropout=0.0, variant="orientation_off"
    ).to(device)
    checks.update(check_identities(off, x, adjs, activity))
    if not all(checks.values()):
        raise RuntimeError(f"smoke identity checks failed: {checks}")
    return {
        "smoke": True,
        "checks": checks,
        "loss": last_loss,
        "resolved_relations": int(orientation.resolved.sum()),
    }


def train(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    started = time.monotonic()
    hashes = source_sha256()
    set_seed(args.seed)
    device = torch.device(args.device)
    data = load_graph(args.data, args.masks)
    stream_bands = bool(args.stream_bands)
    x_host = prepare_features(data.x, args.feature_mode)
    x = x_host if stream_bands else x_host.to(device)
    if args.profile_feature_mode == "same":
        profile_x = x
    else:
        profile_x = prepare_features(data.x, args.profile_feature_mode).to(
            x.device if stream_bands else device
        )
    edges = list(data.edge_indices)
    requested = str(args.graph_mode)
    expected = DEFAULT_GRAPH_MODE[data.name]
    graph_mode = expected if requested == "auto" else requested
    if graph_mode == "homo" and len(edges) > 1:
        edges = [torch.cat(edges, dim=1)]
    elif graph_mode not in {"hetero", "homo"}:
        raise ValueError(f"unknown graph_mode {requested}")
    q, activity, adjs = energy_profiles(
        profile_x, edges, args.S, stream_bands=stream_bands
    )
    if stream_bands:
        x = x_host.to(device)
        q = q.to(device)
        activity = activity.to(device)
        adjs = [as_device_adj(adj, device) for adj in adjs]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        activity = activity.to(device)
        adjs = [adj.to(device) if adj is not None else None for adj in adjs]
    calib_y = torch.full_like(data.y, -1)
    calib_y[data.train_mask] = data.y[data.train_mask]
    orientation = calibrate_orientation(
        q, activity, calib_y.to(device), data.train_mask.to(device), tau_n=args.tau_n
    )
    model = SOAR(
        x.shape[1],
        args.hidden,
        args.S,
        orientation,
        dropout=args.dropout,
        lam=args.lam,
        variant=args.variant,
        fusion=args.fusion,
        encoder_layers=args.encoder_layers,
        num_classes=args.num_classes,
        band_confidence=args.band_confidence,
        shared_head=args.shared_head,
        stream_bands=stream_bands,
    ).to(device)
    y_train = data.y[data.train_mask].float().to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    y_val = data.y[data.val_mask].to(device)
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    pos_weight = torch.tensor(args.pos_weight_scale * negatives / max(positives, 1), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, best_auc, best_epoch, best_state = -float("inf"), -float("inf"), -1, None
    history = []
    y_all = data.y.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x, adjs, activity)
        loss_sup = supervised_loss(out["logits"][train_mask], y_train, pos_weight)
        loss_dc = directional_consistency_loss(
            out["band_energy"],
            y_all,
            train_mask,
            activity,
            out["routed_o"],
            model.rho,
        )
        loss = loss_sup + args.beta * loss_dc
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if stream_bands and device.type == "cuda":
            del out
            torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            selected = model(x, adjs, activity)
            val_metrics, threshold = score_split(selected["logits"][val_mask], y_val)
        history.append(
            {
                "epoch": epoch,
                "sup": float(loss_sup.detach()),
                "dc": float(loss_dc.detach()),
                "total": float(loss.detach()),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_auroc": float(val_metrics["auroc"]),
                "val_threshold": float(threshold),
            }
        )
        val_f1, val_auc = val_metrics["macro_f1"], val_metrics["auroc"]
        if val_f1 > best_f1 + 1e-12 or (
            abs(val_f1 - best_f1) <= 1e-12 and val_auc > best_auc + 1e-12
        ):
            best_f1, best_auc, best_epoch = val_f1, val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if args.patience > 0 and best_epoch > 0 and epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no validation-selected state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        selected = model(x, adjs, activity)
    validation, threshold = score_split(selected["logits"][val_mask], y_val)
    result = {
        "schema": "soar-result-v1",
        "method": "SOAR",
        "variant": args.variant,
        "experiment_id": args.experiment_id,
        "config": _jsonable(vars(args)),
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": hashes,
        "environment": environment_versions(),
        "mask_sha256": hashlib.sha256(Path(args.masks).read_bytes()).hexdigest(),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "lam": args.lam,
        "beta": args.beta,
        "tau_n": args.tau_n,
        "S": args.S,
        "fusion": args.fusion,
        "graph_mode": graph_mode,
        "graph_mode_default": expected,
        "feature_mode": args.feature_mode,
        "profile_feature_mode": args.profile_feature_mode,
        "band_confidence": bool(args.band_confidence),
        "shared_head": bool(model.shared_head),
        "stream_bands": stream_bands,
        "dropout": args.dropout,
        "patience": args.patience,
        "epochs_run": len(history),
        "data": str(args.data),
        "masks": str(args.masks),
        "dataset": data.name,
        "orientation": {
            "o": orientation.o.cpu().tolist(),
            "rho": orientation.rho.cpu().tolist(),
            "band_confidence": orientation.band_confidence.cpu().tolist(),
            "resolved": orientation.resolved.cpu().tolist(),
        },
        "parameters": {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        "validation": validation,
        "history": history,
    }
    frozen = {
        "logits": selected["logits"].detach().cpu(),
        "test_mask": data.test_mask.detach().cpu(),
        "threshold": float(threshold),
    }
    return result, frozen


def as_device_adj(adj: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if adj is None:
        return None
    return as_spmm_adjacency(adj.to(device))


def finalize(
    args: argparse.Namespace,
    result: dict[str, object],
    frozen: dict[str, object],
) -> dict[str, object]:
    revealed = load_graph(args.data, args.masks, reveal_test=True)
    y_test = revealed.y[revealed.test_mask]
    mask = frozen["test_mask"]
    test = evaluate(frozen["logits"][mask], y_test, float(frozen["threshold"]))
    completed = {**result, "test": test}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    shared = PAPER["shared"]
    if args.hidden is None:
        args.hidden = int(shared["hidden"])
    if args.lr is None:
        args.lr = float(shared["lr"])
    if args.epochs is None:
        args.epochs = int(shared["epochs"])
    if args.device is None:
        args.device = default_device()


def resolve_cell(args: argparse.Namespace) -> str:
    """Fill paper-table hyperparameters from --dataset and --abundant/--scarce."""
    cell = paper_cell(args.dataset, args.scarce)
    spec = PAPER["cells"][cell]
    merged = {**PAPER["shared"], **spec}
    for key in _CELL_KEYS:
        if getattr(args, key, None) is None and key in merged:
            setattr(args, key, merged[key])
    if args.device is None:
        args.device = default_device()
    if not args.data:
        args.data = str(default_data_path(args.dataset, args.data_dir))
    else:
        args.data = str(args.data)
    if not args.masks:
        args.masks = str(default_mask_path(args.dataset, args.scarce, args.split_dir))
    else:
        args.masks = str(args.masks)
    if not args.experiment_id:
        args.experiment_id = f"paper-{cell}-{args.variant}"
    args.cell = cell
    return cell


def _seeds_for(args: argparse.Namespace) -> list[int]:
    if args.ten_seeds:
        return list(_TEN_SEEDS)
    return [int(args.seed)]


def _output_dir(args: argparse.Namespace, cell: str) -> Path:
    if args.output:
        path = Path(args.output)
        if path.suffix.lower() == ".json":
            return path.parent
        return path
    return _ROOT / "outputs" / cell


def _print_recipe(cell: str, args: argparse.Namespace, seeds: list[int]) -> None:
    spec = PAPER["cells"][cell]
    paper = spec.get("paper", {})
    n = len(seeds)
    seed_note = f"{n} runs (--10-seeds)" if args.ten_seeds else f"1 run (--seed {seeds[0]})"
    print(
        f"SOAR  cell={cell}  variant={args.variant}  {seed_note}  device={args.device}",
        flush=True,
    )
    print(
        "  "
        f"S={args.S}  lam={args.lam}  beta={args.beta}  tau_n={args.tau_n}  "
        f"fusion={args.fusion}  graph_mode={args.graph_mode}  "
        f"feature_mode={args.feature_mode}  encoder_layers={args.encoder_layers}  "
        f"num_classes={args.num_classes}  shared_head={bool(args.shared_head)}  "
        f"band_confidence={bool(args.band_confidence)}  stream_bands={bool(args.stream_bands)}",
        flush=True,
    )
    if paper:
        print(
            "  paper  "
            f"{paper['auroc']:.2f}±{paper['sd_auroc']:.2f} / "
            f"{paper['macro_f1']:.2f}±{paper['sd_macro_f1']:.2f}  "
            f"(n={paper.get('runs', '?')})",
            flush=True,
        )


def _metric_pair(payload: dict[str, object], split: str) -> tuple[float, float] | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    return 100.0 * float(block["auroc"]), 100.0 * float(block["macro_f1"])


def summarize_outputs(
    directory: Path, split: str, cell: str, variant: str | None = None
) -> dict[str, object]:
    rows = []
    pattern = f"{variant}_s*.json" if variant else "*_s*.json"
    for path in sorted(directory.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pair = _metric_pair(payload, split)
        if pair is None:
            continue
        auroc, macro_f1 = pair
        rows.append(
            {
                "file": path.name,
                "seed": payload.get("seed"),
                "variant": payload.get("variant"),
                "auroc": auroc,
                "macro_f1": macro_f1,
            }
        )
        print(
            f"  {path.name:32s}  seed={payload.get('seed')}  "
            f"{auroc:6.2f} / {macro_f1:6.2f}",
            flush=True,
        )
    if not rows:
        raise SystemExit(f"no {split} metrics in {directory}")
    auc = [row["auroc"] for row in rows]
    f1 = [row["macro_f1"] for row in rows]
    summary: dict[str, object] = {
        "cell": cell,
        "variant": variant,
        "split": split,
        "n": len(rows),
        "auroc_mean": float(statistics.mean(auc)),
        "macro_f1_mean": float(statistics.mean(f1)),
        "runs": rows,
        "paper": PAPER["cells"].get(cell, {}).get("paper"),
    }
    if len(rows) > 1:
        summary["auroc_sd"] = float(statistics.pstdev(auc))
        summary["macro_f1_sd"] = float(statistics.pstdev(f1))
        print(
            "mean±sd  "
            f"{summary['auroc_mean']:.2f}±{summary['auroc_sd']:.2f} / "
            f"{summary['macro_f1_mean']:.2f}±{summary['macro_f1_sd']:.2f}  "
            f"(n={len(rows)}, {split})",
            flush=True,
        )
    else:
        print(f"mean  {auc[0]:.2f} / {f1[0]:.2f}  ({split})", flush=True)
    paper = summary.get("paper") or {}
    if paper:
        print(
            "paper   "
            f"{paper['auroc']:.2f}±{paper.get('sd_auroc', 0):.2f} / "
            f"{paper['macro_f1']:.2f}±{paper.get('sd_macro_f1', 0):.2f}",
            flush=True,
        )
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SOAR. Dataset and --abundant/--scarce select the paper recipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python train.py --smoke
  python train.py --dataset yelpchi --abundant
  python train.py --dataset amazon --scarce --10-seeds
  python train.py --dataset tsocial --abundant --10-seeds --device cuda:0
  python train.py --dataset yelpchi --abundant --10-seeds --variant orientation_off
""",
    )
    parser.add_argument("--smoke", action="store_true", help="Synthetic identity checks; no dataset.")
    parser.add_argument("--dataset", choices=list(DATASETS))
    regime = parser.add_mutually_exclusive_group()
    regime.add_argument("--abundant", action="store_true", help="40%% training labels.")
    regime.add_argument(
        "--scarce",
        action="store_true",
        help="1%% training labels (T-Social: 0.01%%).",
    )
    parser.add_argument(
        "--10-seeds",
        dest="ten_seeds",
        action="store_true",
        help="Ten fixed-split reinitializations (paper protocol).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single-run seed when --10-seeds is off.")
    parser.add_argument("--variant", default=None, choices=list(VARIANTS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    parser.add_argument("--split-dir", type=Path, default=_ROOT / "splits")
    parser.add_argument("--data", help="Override the dataset file.")
    parser.add_argument("--masks", help="Override the frozen-split path.")
    parser.add_argument("--output", help="Output directory, or a .json path for a single run.")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the routed recipe and exit.")

    overrides = parser.add_argument_group("optional overrides (default: paper cell)")
    overrides.add_argument("--S", type=int, default=None)
    overrides.add_argument("--hidden", type=int, default=None)
    overrides.add_argument("--dropout", type=float, default=None)
    overrides.add_argument("--epochs", type=int, default=None)
    overrides.add_argument("--patience", type=int, default=None)
    overrides.add_argument("--lr", type=float, default=None)
    overrides.add_argument("--weight-decay", type=float, default=None)
    overrides.add_argument("--lam", type=float, default=None)
    overrides.add_argument("--beta", type=float, default=None)
    overrides.add_argument("--tau-n", type=float, default=None)
    overrides.add_argument("--pos-weight-scale", type=float, default=None)
    overrides.add_argument(
        "--feature-mode", default=None, choices=["raw", "standard", "signed_log"]
    )
    overrides.add_argument(
        "--profile-feature-mode",
        default=None,
        choices=["same", "raw", "standard", "signed_log"],
    )
    overrides.add_argument("--graph-mode", default=None, choices=["auto", "hetero", "homo"])
    overrides.add_argument("--fusion", default=None, choices=["mean", "concat"])
    overrides.add_argument("--encoder-layers", type=int, default=None, choices=[1, 2])
    overrides.add_argument("--num-classes", type=int, default=None, choices=[1, 2])
    overrides.add_argument(
        "--shared-head",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    overrides.add_argument(
        "--band-confidence",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    overrides.add_argument(
        "--stream-bands",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    args = parser.parse_args(argv)
    if args.smoke:
        return args
    if not args.dataset:
        parser.error("--dataset is required (or pass --smoke)")
    if not args.abundant and not args.scarce:
        parser.error("specify --abundant or --scarce")
    return args


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.smoke:
        _apply_smoke_defaults(args)
        payload = smoke(args)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload

    cell = resolve_cell(args)
    seeds = _seeds_for(args)
    out_dir = _output_dir(args, cell)
    _print_recipe(cell, args, seeds)

    if args.dry_run:
        planned = [
            str(out_dir / f"{args.variant}_s{seed}.json")
            if not (args.output and Path(args.output).suffix.lower() == ".json" and len(seeds) == 1)
            else str(args.output)
            for seed in seeds
        ]
        payload: dict[str, object] = {
            "cell": cell,
            "data": args.data,
            "masks": args.masks,
            "n_runs": len(seeds),
            "ten_seeds": bool(args.ten_seeds),
            "outputs": planned,
        }
        if not args.ten_seeds:
            payload["seed"] = seeds[0]
        print(json.dumps(payload, indent=2))
        return {"cell": cell, "n_runs": len(seeds), "dry_run": True}

    try:
        created = ensure_masks(args.data, args.dataset, args.scarce, args.masks)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    if created is not None:
        print(
            f"wrote frozen split {args.masks}  counts={created['counts']}  "
            f"regime={created['regime']}",
            flush=True,
        )

    split = "validation" if args.validation_only else "test"
    single_json = (
        Path(args.output)
        if args.output and Path(args.output).suffix.lower() == ".json" and len(seeds) == 1
        else None
    )
    last: dict[str, object] | None = None
    for seed in seeds:
        args.seed = int(seed)
        args.output = str(single_json if single_json is not None else out_dir / f"{args.variant}_s{seed}.json")
        destination = Path(args.output)
        if destination.is_file() and not args.overwrite:
            print(f"skip existing {destination}", flush=True)
            continue
        result, frozen = train(args)
        if args.validation_only:
            payload = result
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            payload = finalize(args, result, frozen)
        last = payload
        pair = _metric_pair(payload, split)
        if pair is None:
            print(f"seed {seed}  wrote {destination}", flush=True)
        else:
            auroc, macro_f1 = pair
            print(
                f"seed {seed}  {split}  {auroc:6.2f} / {macro_f1:6.2f}  "
                f"epoch={payload.get('best_epoch')}  -> {destination}",
                flush=True,
            )

    summary = summarize_outputs(out_dir, split, cell, variant=args.variant)
    if last is not None and len(seeds) == 1:
        print(json.dumps(last, indent=2, sort_keys=True, default=str))
    return summary


if __name__ == "__main__":
    main()
