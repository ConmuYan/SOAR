# SOAR

Official code for **Beyond Band Importance: Signed Orientation-Aware Spectral Learning for Graph Fraud Detection**.

> **Reviewer:** please read **[REVIEW.md](./REVIEW.md)** first. It pins you to
> the submission snapshot (`v1.0`), maps every paper claim to the file that
> implements it, and gives the exact reproduction commands.

`--dataset` with `--abundant` or `--scarce` loads the paper recipe. `--10-seeds` runs the ten-reinitialization protocol.

## Environment

Python **3.10–3.12**. If you train on GPU, install a CUDA [PyTorch](https://pytorch.org/get-started/locally/) wheel **before** `requirements.txt`. DGL is required only for T-Finance and T-Social, and is skipped on Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --smoke
```

## Data

Datasets are not in this repository. Download the [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) fraud graphs and place them here:

| Dataset | Path |
| --- | --- |
| YelpChi | `data/YelpChi.mat` |
| Amazon | `data/Amazon.mat` |
| T-Finance | `data/tfinance` |
| T-Social | `data/tsocial` |

Frozen splits are written on the first training run (split seed **2**). To write all eight:

```bash
python protocol.py --all
```

`--scarce` on T-Social is 0.01% labels. Amazon nodes `0–3304` stay in the graph but are not used for train / val / test.

## Reproduce

```bash
python train.py --dataset yelpchi --abundant
python train.py --dataset yelpchi --scarce --10-seeds
python train.py --dataset amazon --scarce --10-seeds
python train.py --dataset tfinance --abundant --10-seeds
python train.py --dataset tsocial --scarce --10-seeds --device cuda:0
```

Default is one run. Ablations: `--variant {full,magnitude,orientation_off,reliability_off,signed_only}`. Print a recipe without training: `--dry-run`.

Outputs: `outputs/<dataset>-<regime>/`.

```bash
python scripts/summarize.py outputs/yelpchi-abundant
```

T-Social needs ~24 GB GPU memory.
