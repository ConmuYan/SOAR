<div align="center">

# SOAR

**S**igned **O**rientation-**A**ware **R**esidual for graph fraud detection

[![License](https://img.shields.io/badge/license-MIT-0B6E4F.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-3776AB.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.1-EE4C2C.svg)](https://pytorch.org)

[Overview](#overview) ·
[Results](#results) ·
[Setup](#setup) ·
[Reproduce](#reproduce) ·
[Guidelines](GUIDELINES.md) ·
[Citation](#citation)

<img src="manuscript/fig_motivation.png" width="88%" alt="Band importance does not specify class orientation on YelpChi.">

<sub>Frequency importance does not specify which class a relation–band supports.</sub>

</div>

Official implementation of

> **Beyond Band Importance: Signed Orientation-Aware Spectral Learning for Graph Fraud Detection**

Spectral fraud detectors typically ask *which frequencies are informative*. SOAR additionally asks **which class each relation–band favors**, how reliable that orientation is under limited labels, and how to keep the corresponding evidence from being mixed away during representation learning.

`--dataset` plus `--abundant` or `--scarce` selects the paper recipe in [`configs/paper.json`](configs/paper.json). Hyperparameters, graph mode, feature transform, Amazon-scarce concat / two-class head, and the T-Social 0.01% split are filled in automatically.

```bash
python train.py --dataset yelpchi --abundant
python train.py --dataset amazon --scarce --10-seeds
```

## Overview

SOAR estimates a class orientation for every relation–band from label-free spectral energy profiles and training labels, then injects that orientation **in-band** as a signed residual. A parallel orientation-neutral path retains spectral information when the estimated direction is uncertain. Directional consistency regularizes the hidden class contrast toward the fixed orientation reference.

| Paper object | Implementation |
| --- | --- |
| Energy profiles *q* | `soar.energy_profiles` |
| Orientation **o**, reliability *ρ* | `soar.calibrate_orientation` |
| In-band signed residual and dual-path readout | `soar.SOAR` |
| Directional consistency L<sub>DC</sub> | `soar.directional_consistency_loss` |
| Frozen BWGNN split | `protocol.py` |
| Validation-only epoch / threshold selection | `train.py` |

Matched controls (`--variant`):

| Variant | What it tests |
| --- | --- |
| `full` | SOAR |
| `magnitude` | unsigned \|**o**\|; shared head; DC off |
| `orientation_off` | no signed residual (*ρ* = 0) |
| `reliability_off` | unit weight on resolved relations |
| `signed_only` | drop the orientation-neutral path |

Method figure: [`manuscript/fig_framework.pdf`](manuscript/fig_framework.pdf).

## Results

AUROC / Macro-F1 (%), mean ± population SD over **fixed-split reinitializations**. These are the paper-table entries, not a claim that one seed must match them.

| Supervision | YelpChi | Amazon | T-Finance | T-Social |
| --- | ---: | ---: | ---: | ---: |
| 40% labels | 92.28±0.17 / 79.30±0.23 | 97.82±0.19 / 93.02±0.32 | 97.10±0.73 / 92.23±0.88 | 99.12±0.02 / 94.98±0.19 |
| 1% labels<sup>†</sup> | 78.21±0.60 / 67.07±0.39 | 91.83±1.05 / 90.05±0.64 | 95.05±0.68 / 90.75±0.28 | 94.34±0.98 / 85.13±0.54 |

<sup>†</sup>T-Social scarce uses 0.01% training labels. Paper tables use ten reinitializations except T-Social (five abundant, four scarce). `--10-seeds` always launches ten runs.

Baselines in the paper are **published references**, not matched reruns. Amazon scarce uses a disclosed recipe (concat fusion, shared two-class head, band-confidence refinement) that `--scarce` routes in automatically.

## Setup

Python **3.10–3.12** recommended. Install a CUDA build of [PyTorch](https://pytorch.org/get-started/locally/) that matches your driver **before** `requirements.txt` if you train on GPU. DGL is required only for T-Finance and T-Social and is skipped on Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --smoke
```

`--smoke` is a synthetic identity check, not a paper number. Full protocol: [GUIDELINES.md](GUIDELINES.md).

## Data

Datasets are **not** redistributed. Place the [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) fraud graphs under `data/`:

| Dataset | Path | Loader |
| --- | --- | --- |
| YelpChi | `data/YelpChi.mat` | SciPy |
| Amazon | `data/Amazon.mat` | SciPy |
| T-Finance | `data/tfinance` | DGL |
| T-Social | `data/tsocial` | DGL |

Splits are written on the first training run (split seed **2**, never a training seed):

```bash
python protocol.py --all
```

- Amazon nodes 0–3304 stay in the graph but are ineligible for train / val / test.
- Abundant: 40% train. Scarce: 1% (T-Social 0.01%). Remainder: 33% val / 67% test.
- Epoch and threshold are selected on **validation Macro-F1** (`linspace(0.05, 0.95, 19)`). Test labels are used only after that lock.

## Reproduce

```bash
python train.py --dataset yelpchi --abundant
python train.py --dataset yelpchi --scarce --10-seeds
python train.py --dataset amazon --scarce --10-seeds
python train.py --dataset tfinance --abundant --10-seeds
python train.py --dataset tsocial --scarce --10-seeds --device cuda:0
```

`--abundant` / `--scarce` pick the paper cell. `--10-seeds` runs the ten-reinitialization protocol. Default is a single run.

```bash
python train.py --dataset yelpchi --abundant --10-seeds --variant orientation_off
python train.py --dataset amazon --scarce --10-seeds --dry-run
```

Outputs land in `outputs/<dataset>-<regime>/`. After a multi-run job, `summary.json` reports mean ± population SD next to the paper numbers.

```bash
python scripts/summarize.py outputs/yelpchi-abundant
```

T-Social uses streamed Bernstein bands (plan for ~24 GB GPU memory). Graph mode and feature transforms are in the cell config, not extra flags.

## Repository

```text
train.py              public entry: --dataset --abundant/--scarce --10-seeds
soar.py               model, orientation, directional consistency
protocol.py           frozen BWGNN split (seed 2); T-Social --scarce → 0.01%
data.py               graph I/O, features, metrics
configs/paper.json    main-table recipes (auto-routed)
scripts/summarize.py  mean ± population SD over result JSON
GUIDELINES.md         reproduction protocol
CONTRIBUTING.md       issues / PRs
manuscript/           TeX, bibliography, figures, numeric archive
```

This is a paper-reproduction codebase, not a general-purpose GNN library. Optional `--S`, `--lam`, `--beta`, … flags override a cell; they are not required for reproduction.

## Citation

```bibtex
@article{tian2026soar,
  title   = {Beyond Band Importance: Signed Orientation-Aware Spectral Learning for Graph Fraud Detection},
  author  = {Tian, Chunwei and Meng, Qi},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Manuscript}
}
```

See also [`CITATION.cff`](CITATION.cff). Issues and PRs: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
