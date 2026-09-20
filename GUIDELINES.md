# Reproduction guidelines

This repository reproduces the SOAR paper tables. It is not a general GNN library.
`--dataset` plus `--abundant` or `--scarce` selects the paper recipe; you should not
need a long flag list.

## Environment

- Python **3.10–3.12** (recommended). Python 3.13 can run YelpChi, Amazon, and `--smoke`.
- PyTorch ≥ 2.1. Install the [CUDA wheel](https://pytorch.org/get-started/locally/) that matches your driver **before** `requirements.txt` if you train on GPU.
- DGL ≥ 1.1 is required **only** for T-Finance and T-Social. Current DGL wheels do not import on Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --smoke
```

`--smoke` is a synthetic identity check. It is not a paper number.

## Data

Datasets are not redistributed. Download the BWGNN fraud graphs and place them here:

| Dataset | Path | Loader |
| --- | --- | --- |
| YelpChi | `data/YelpChi.mat` | SciPy |
| Amazon | `data/Amazon.mat` | SciPy |
| T-Finance | `data/tfinance` | DGL |
| T-Social | `data/tsocial` | DGL |

Source: [squareRoot3/Rethinking-Anomaly-Detection](https://github.com/squareRoot3/Rethinking-Anomaly-Detection).

## Public commands

```bash
python train.py --dataset yelpchi --abundant
python train.py --dataset yelpchi --scarce --10-seeds
python train.py --dataset amazon --scarce --10-seeds
python train.py --dataset tfinance --abundant --10-seeds
python train.py --dataset tsocial --scarce --10-seeds --device cuda:0
```

| Flag | Meaning |
| --- | --- |
| `--dataset {yelpchi,amazon,tfinance,tsocial}` | Graph |
| `--abundant` | 40% training labels |
| `--scarce` | 1% training labels; **T-Social is 0.01%** internally |
| `--10-seeds` | Ten fixed-split reinitializations (paper protocol) |
| `--variant` | `full` (default) or a matched control |
| `--dry-run` | Print the routed recipe; do not load data |

Hyperparameters, graph mode, feature transform, Amazon-scarce concat / two-class head, and T-Social streamed bands are filled from [`configs/paper.json`](configs/paper.json). Optional `--S`, `--lam`, `--beta`, … flags override a cell; they are not required for reproduction.

Inspect routing:

```bash
python train.py --dataset amazon --scarce --10-seeds --dry-run
```

## Frozen splits

The split seed is **2** and cannot vary. Training `--seed` / `--10-seeds` only reinitialize the model.

Splits are written on the first training run. To write all eight at once:

```bash
python protocol.py --all
```

Invariants:

- Both stratified draws use `random_state=2`.
- Amazon nodes `0–3304` stay in the graph but are ineligible for train / val / test.
- Abundant: 40% train. Scarce: 1% (T-Social 0.01%). Of the remainder, 33% val / 67% test.
- T-Social `--scarce` maps to regime `tsocial-scarce` (ratio `0.0001`). The public filename is still `splits/tsocial/scarce.pt`.
- Epoch and threshold are selected on **validation Macro-F1** over `linspace(0.05, 0.95, 19)`. Test labels are revealed only after that lock.
- Orientation statistics use **training labels only** and stay frozen during optimization.

Do not retune on test. Do not change `SPLIT_SEED`.

## Outputs

Files land in `outputs/<dataset>-<regime>/` (one JSON per reinitialization, plus `summary.json`).

```bash
python scripts/summarize.py outputs/yelpchi-abundant
```

Existing JSON is skipped unless you pass `--overwrite`.

Paper tables use ten reinitializations except T-Social (five abundant, four scarce). `--10-seeds` always launches ten runs. A single seed matching the table is not required; compare the aggregated mean ± SD.

## Ablations

Same recipe, different `--variant`:

| Variant | Control |
| --- | --- |
| `full` | SOAR |
| `magnitude` | unsigned \|o\|; shared head; DC off |
| `orientation_off` | no signed residual |
| `reliability_off` | unit weight on resolved relations |
| `signed_only` | drop the orientation-neutral path |

```bash
python train.py --dataset yelpchi --abundant --10-seeds --variant orientation_off
```

## Hardware

- YelpChi / Amazon / T-Finance: a single modern GPU is enough.
- T-Social: streamed Bernstein bands; plan for ~24 GB GPU memory.
- Graph modes in the main tables: YelpChi and Amazon **hetero**; T-Finance and T-Social **homo**.

## Scope

- Paper reproduction only.
- Amazon scarce is a disclosed configuration (concat fusion, shared two-class head, band-confidence), not a silent change of the method.
- Baselines in the paper are published references, not matched reruns in this repo.
- New methods belong in a fork. See [CONTRIBUTING.md](CONTRIBUTING.md).
