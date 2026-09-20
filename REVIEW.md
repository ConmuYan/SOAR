# Reviewer Guide — SOAR (TKDE submission)

> **Reviewer: please pin to tag `v1.0`.** The repository you are looking at
> may evolve on `main`; `v1.0` is the exact snapshot under review.

| Field | Value |
| --- | --- |
| Paper | *Beyond Band Importance: Signed Orientation-Aware Spectral Learning for Graph Fraud Detection* |
| Venue | IEEE Transactions on Knowledge and Data Engineering (TKDE) |
| Pinned commit | see `git rev-parse v1^{commit}` or the GitHub release page |
| Frozen split seed | `2` (BWGNN default, hard-coded in `protocol.py:16`) |
| License | MIT |
| Citation metadata | `CITATION.cff` |

## 60-second orientation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --smoke               # ~10 s on CPU; verifies the pipeline
python train.py --dataset yelpchi --abundant --dry-run   # prints the recipe, no training
```

That is enough to know whether the code runs in your environment. For the full
reproduction (≈4–6 h on a single GPU, dominated by T-Social), see the table
below.

## Map: paper claim → code

| Paper concept | Source location |
| --- | --- |
| Band-energy profiles `q[N, R, K]` | `soar.py:34` — `energy_profiles` |
| Per-band signed orientation `o` and reliability `ρ` | `soar.py:69` — `calibrate_orientation` |
| Band-confidence gating (optional) | `soar.py:91–96`, `soar.py:196` — `routed_orientation` |
| Dual-path hidden states `h_neu`, `h_sgn` | `soar.py:288–319` (`SOAR.forward`) |
| Dual-path logits, fusion (`mean` / `concat`) | `soar.py:320–346` |
| Variants (full / magnitude / orientation_off / reliability_off / signed_only) | `soar.py:38`, `soar.py:251–256` |
| Directional-consistency loss | `soar.py:127` — `directional_consistency_loss` |
| Loss wiring into training loop | `train.py:173`, `train.py:268` |
| Frozen BWGNN split (seed 2) | `protocol.py:16`, `protocol.py:91` |
| Smoke / algebraic-contract checks | `soar.py:356` — `check_identities`; `train.py --smoke` |
| Memory-efficient Bernstein SPMM | `data.py:267` — `bernstein_coefficients`; `data.py:309` — `_bernstein_poly`; `data.py:318` — `StreamV1RelationFn` |

The paper mapping is also stated at the top of `soar.py` (next to the module
docstring) and is kept in sync with the manuscript.

## Reproduction commands

These are the exact commands that produced the numbers in the paper. Default
is a single seed; `--10-seeds` runs the ten-reinitialization protocol.

| Cell | Command |
| --- | --- |
| YelpChi, abundant | `python train.py --dataset yelpchi --abundant` |
| YelpChi, scarce | `python train.py --dataset yelpchi --scarce --10-seeds` |
| Amazon, scarce | `python train.py --dataset amazon --scarce --10-seeds` |
| T-Finance, abundant | `python train.py --dataset tfinance --abundant --10-seeds` |
| T-Social, scarce | `python train.py --dataset tsocial --scarce --10-seeds --device cuda:0` |

Outputs land in `outputs/<dataset>-<regime>/seed-<n>/run.json` and are
summarized by `python scripts/summarize.py outputs/<dir>`.

## Datasets

Not bundled in this repository. Download the BWGNN fraud graphs from
<https://github.com/squareRoot3/Rethinking-Anomaly-Detection> and place them
under `data/` exactly as the README's *Data* section describes.

## Hardware notes

- CPU is sufficient for YelpChi / Amazon and for `--smoke`.
- T-Finance needs ~4 GB GPU memory.
- T-Social needs ~24 GB GPU memory (the full graph stays on device).

If you hit a DGL import error on Python 3.13, that is expected — DGL still
does not import on 3.13 and is gated by a Python-version marker in
`requirements.txt`. Use Python 3.10–3.12 for the DGL datasets.

## What changed since submission

Nothing — `v1.0` is the snapshot under review. Any subsequent commits on
`main` are post-submission work and should not be cited.

## Contact

Open an issue on this repository, or contact the corresponding author.