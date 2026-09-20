"""Frozen BWGNN-style data split used by every method and control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


# BWGNN (Tang et al., ICML 2022, official main.py): the split seed is fixed.
SPLIT_SEED = 2
# Published regimes: train_ratio=0.4 abundant; train_ratio=0.01 scarce.
REGIME_RATIOS = {
    "abundant": 0.4,
    "scarce": 0.01,
    "tsocial-scarce": 0.0001,
}

DATASETS = ("yelpchi", "amazon", "tfinance", "tsocial")
DATA_FILENAMES = {
    "yelpchi": "YelpChi.mat",
    "amazon": "Amazon.mat",
    "tfinance": "tfinance",
    "tsocial": "tsocial",
}

ROOT = Path(__file__).resolve().parent


def label_sha256(labels: torch.Tensor) -> str:
    values = labels.detach().cpu().reshape(-1).long().numpy().astype(np.int64, copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def eligible_index(dataset: str, num_nodes: int) -> np.ndarray:
    """Return the published labeled universe; Amazon starts at node 3305."""
    name = dataset.lower()
    if name == "amazon":
        if num_nodes <= 3305:
            raise ValueError("Amazon must contain eligible nodes after index 3304")
        return np.arange(3305, num_nodes, dtype=np.int64)
    if name not in {"yelp", "yelpchi", "tfinance", "tsocial"}:
        raise ValueError(f"unsupported graph-fraud dataset: {dataset}")
    return np.arange(num_nodes, dtype=np.int64)


def normalize_regime(dataset: str, regime: str) -> str:
    """Map public --scarce/--abundant flags onto the frozen split regime.

    T-Social scarce is 0.01% labels (``tsocial-scarce``), not 1%.
    """
    name = dataset.lower()
    if name == "tsocial" and regime == "scarce":
        return "tsocial-scarce"
    if name != "tsocial" and regime == "tsocial-scarce":
        raise ValueError("tsocial-scarce is reserved for T-Social")
    if regime not in REGIME_RATIOS:
        raise ValueError(f"unknown regime: {regime}")
    return regime


def public_regime_name(dataset: str, scarce: bool) -> str:
    """User-facing split filename: abundant.pt / scarce.pt."""
    del dataset
    return "scarce" if scarce else "abundant"


def default_data_path(dataset: str, data_dir: str | Path | None = None) -> Path:
    root = Path(data_dir) if data_dir is not None else ROOT / "data"
    return root / DATA_FILENAMES[dataset.lower()]


def default_mask_path(
    dataset: str,
    scarce: bool,
    split_dir: str | Path | None = None,
) -> Path:
    root = Path(split_dir) if split_dir is not None else ROOT / "splits"
    return root / dataset.lower() / f"{public_regime_name(dataset, scarce)}.pt"


def paper_cell(dataset: str, scarce: bool) -> str:
    return f"{dataset.lower()}-{'scarce' if scarce else 'abundant'}"


def bwgnn_split(
    labels: torch.Tensor | np.ndarray,
    dataset: str,
    train_ratio: float,
    *,
    split_seed: int = SPLIT_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one stratified split; training restarts never alter it."""
    if split_seed != SPLIT_SEED:
        raise ValueError("the frozen split seed is 2 and cannot vary by run")
    expected = {0.4, 0.01, 0.0001}
    if float(train_ratio) not in expected:
        raise ValueError(f"train_ratio must be one of {sorted(expected)}")
    if dataset.lower() != "tsocial" and float(train_ratio) == 0.0001:
        raise ValueError("0.0001 is reserved for T-Social scarce")
    if dataset.lower() == "tsocial" and float(train_ratio) == 0.01:
        raise ValueError("T-Social scarce uses 0.0001, not 0.01")

    y = torch.as_tensor(labels).detach().cpu().reshape(-1).long().numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("the split requires both binary classes")
    index = eligible_index(dataset, len(y))
    # BWGNN (Tang et al., ICML 2022): both calls use random_state=2.
    idx_train, idx_rest, _, y_rest = train_test_split(
        index,
        y[index],
        train_size=float(train_ratio),
        stratify=y[index],
        random_state=SPLIT_SEED,
        shuffle=True,
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_rest,
        y_rest,
        test_size=0.67,
        stratify=y_rest,
        random_state=SPLIT_SEED,
        shuffle=True,
    )
    masks = []
    for values in (idx_train, idx_val, idx_test):
        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[torch.as_tensor(values, dtype=torch.long)] = True
        masks.append(mask)
    validate_split(*masks, labels=torch.from_numpy(y), dataset=dataset)
    return tuple(masks)  # type: ignore[return-value]


def validate_split(
    train: torch.Tensor,
    val: torch.Tensor,
    test: torch.Tensor,
    *,
    labels: torch.Tensor,
    dataset: str,
) -> None:
    masks = tuple(torch.as_tensor(value).reshape(-1).bool() for value in (train, val, test))
    y = torch.as_tensor(labels).reshape(-1).long()
    if any(mask.shape != y.shape for mask in masks):
        raise ValueError("split masks must align with labels")
    if bool((masks[0] & masks[1] | masks[0] & masks[2] | masks[1] & masks[2]).any()):
        raise ValueError("split masks overlap")
    eligible = torch.zeros(len(y), dtype=torch.bool)
    eligible[torch.from_numpy(eligible_index(dataset, len(y)))] = True
    if not torch.equal(masks[0] | masks[1] | masks[2], eligible):
        raise ValueError("split must partition exactly the eligible nodes")
    for mask in masks:
        if set(y[mask].tolist()) != {0, 1}:
            raise ValueError("every split must contain both classes")


def load_labels(data_path: str | Path, dataset: str) -> torch.Tensor:
    path = Path(data_path)
    name = dataset.lower()
    if name in {"yelp", "yelpchi", "amazon"}:
        import scipy.io as scipy_io

        labels = scipy_io.loadmat(path, variable_names=["label"])["label"]
        return torch.from_numpy(np.asarray(labels).reshape(-1).astype(np.int64, copy=False))
    if name in {"tfinance", "tsocial"}:
        try:
            from dgl.data.utils import load_graphs
        except ImportError as error:
            raise RuntimeError("DGL is required to read T-Finance/T-Social") from error
        graph = load_graphs(str(path))[0][0]
        labels = graph.ndata["label"]
        if name == "tfinance":
            if labels.ndim != 2 or labels.shape[1] != 2:
                raise ValueError("T-Finance labels must be one-hot with two columns")
            labels = labels.argmax(1)
        return labels.reshape(-1).long().cpu()
    raise ValueError(f"unsupported graph-fraud dataset: {dataset}")


def write_frozen_masks(
    data_path: str | Path,
    dataset: str,
    regime: str,
    output: str | Path,
) -> dict[str, object]:
    regime = normalize_regime(dataset, regime)
    ratio = REGIME_RATIOS[regime]
    labels = load_labels(data_path, dataset)
    train, val, test = bwgnn_split(labels, dataset, ratio)
    payload: dict[str, object] = {
        "schema": "soar-bwgnn-split-v1",
        "source": "Tang et al., ICML 2022 official protocol",
        "dataset": dataset.lower(),
        "regime": regime,
        "train_ratio": ratio,
        "split_seed": SPLIT_SEED,
        "label_sha256": label_sha256(labels),
        "train": train,
        "val": val,
        "test": test,
        "counts": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())},
        "class_counts": {
            split: [int(((labels == value) & mask).sum()) for value in (0, 1)]
            for split, mask in (("train", train), ("val", val), ("test", test))
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return payload


def ensure_masks(
    data_path: str | Path,
    dataset: str,
    scarce: bool,
    output: str | Path,
) -> dict[str, object] | None:
    """Write the frozen split if it is missing; return None when it already exists."""
    destination = Path(output)
    if destination.is_file():
        return None
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {data_path}\n"
            "Place the BWGNN files under data/ (see data/README.md)."
        )
    regime = normalize_regime(dataset, "scarce" if scarce else "abundant")
    return write_frozen_masks(data_path, dataset, regime, destination)


def _jobs(args: argparse.Namespace) -> list[tuple[str, bool]]:
    datasets = DATASETS if args.all else (args.dataset,)
    jobs: list[tuple[str, bool]] = []
    for dataset in datasets:
        if args.abundant and not args.scarce:
            jobs.append((dataset, False))
        elif args.scarce and not args.abundant:
            jobs.append((dataset, True))
        else:
            jobs.append((dataset, False))
            jobs.append((dataset, True))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write frozen BWGNN splits. T-Social --scarce is 0.01% labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python protocol.py --all
  python protocol.py --dataset yelpchi --abundant
  python protocol.py --dataset tsocial --scarce
""",
    )
    parser.add_argument("--dataset", choices=list(DATASETS))
    regime = parser.add_mutually_exclusive_group()
    regime.add_argument("--abundant", action="store_true", help="40%% training labels.")
    regime.add_argument(
        "--scarce",
        action="store_true",
        help="1%% training labels (T-Social: 0.01%%).",
    )
    parser.add_argument("--all", action="store_true", help="All four datasets, both regimes.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--data", help="Override the dataset file for a single --dataset run.")
    parser.add_argument("--output", help="Override the mask path for a single split.")
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("specify --dataset or --all")
    if args.all and args.dataset:
        parser.error("use either --all or --dataset, not both")
    if args.output and (args.all or (not args.abundant and not args.scarce)):
        parser.error("--output is only valid with one dataset and one of --abundant/--scarce")
    if args.data and args.all:
        parser.error("--data is only valid with --dataset")
    return args


def main() -> None:
    args = parse_args()

    summaries = []
    for dataset, scarce in _jobs(args):
        data_path = Path(args.data) if args.data else default_data_path(dataset, args.data_dir)
        mask_path = (
            Path(args.output)
            if args.output
            else default_mask_path(dataset, scarce, args.split_dir)
        )
        regime = normalize_regime(dataset, "scarce" if scarce else "abundant")
        payload = write_frozen_masks(data_path, dataset, regime, mask_path)
        row = {
            "dataset": dataset,
            "regime": regime,
            "output": str(mask_path),
            "counts": payload["counts"],
            "train_ratio": payload["train_ratio"],
        }
        summaries.append(row)
        print(
            f"{dataset:9s} {regime:15s}  "
            f"train={payload['counts']['train']:<7d} "
            f"val={payload['counts']['val']:<7d} "
            f"test={payload['counts']['test']:<7d}  -> {mask_path}",
            flush=True,
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
