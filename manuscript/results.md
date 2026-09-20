# Manuscript figure and result archive

## Scope and maintenance

- Paper source: [`handmade.tex`](handmade.tex); bibliography: [`references.bib`](references.bib).
- All figures used by the paper live in [`figures/`](figures/). No parent-directory assets are needed to compile.
- This document records **existing reported values**, not new runs or a recomputed aggregate. Table values are transcribed from the author-written TeX. Sensitivity values and experiment IDs were supplied directly by the author in the frozen-sensitivity handoff.
- Metrics are percentages. `±` for SOAR denotes population SD across fixed-split training reinitializations, as specified in the manuscript. Published baseline SD conventions are those of their sources.
- Keep main-table recipes and diagnostic sweeps separate. Do not merge runs across source versions, masks, graph modes, or method variants. Amazon's archived sweeps use **hetero** graphs; they must not be relabeled as homo results.
- Missing original arrays are explicitly marked **PENDING**. No values have been reconstructed from image coordinates. Summary mean/SD values cannot reconstruct seed-level samples, box plots, or training trajectories.
- User-approved scope: archive and publish the available images and numeric summaries now; leave unavailable arrays pending.

### Provenance limits

| Field | Status |
| --- | --- |
| Project | SOAR manuscript archive |
| Method lineage / promotion status | Not independently verified from executable run artifacts; this archive does not promote a candidate |
| Experiment IDs | Recorded below where provided by the author |
| Model/trainer/evaluation source hashes | PENDING: not supplied with the numeric summaries |
| Split/mask hash | PENDING: not supplied |
| Split seed | 2 in the frozen sensitivity handoff |
| Graph-mode alignment | Not independently certified; Amazon archived results explicitly retain relations (hetero) |
| Asset SHA-256 | Recorded at the end; these are image hashes, **not** experiment-source hashes |

### Build

From the repository root:

```sh
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error handmade.tex
bibtex handmade
pdflatex -interaction=nonstopmode -halt-on-error handmade.tex
pdflatex -interaction=nonstopmode -halt-on-error handmade.tex
```

The compiled paper and LaTeX intermediates are ignored by Git; figure PDFs remain versioned. The existing Method reference `fig:framework` has no corresponding figure label and produces a nonfatal warning. It is not repaired by inventing a figure or changing the Method text.

## Figure inventory

Figure numbers are the current compiled numbering; TeX labels are the stable identifiers.

| Figure / TeX label | Content | Asset included by TeX | Other archived formats | Numeric coverage |
| --- | --- | --- | --- | --- |
| 1 / `fig:motivation` | YelpChi band relevance and class orientation | `fig_motivation.png` | Original PNG only | PENDING: per-relation/band arrays |
| 2 / `fig:reliability` | YelpChi orientation stability | `fig_reliability.png` | Original PNG only | 1,000 stratified draws stated in caption; per-point stability arrays PENDING |
| 3 / `fig:framework` | SOAR framework overview | `fig_framework.pdf` | PNG/SVG/GIF and draft variants under `figures/framework/` | Schematic |
| 4 / `fig:beta` | YelpChi DC alignment and performance | `fig_beta.png` | PDF, SVG under `figures/yelpchi/` | All six beta performance mean/SD pairs below; trajectories and box-plot samples PENDING |
| 5 / `fig:sensitivity_s` | YelpChi/Amazon filter-order bars | `fig_S_bars.png` | PDF, SVG under `figures/backbone/` | S grid 1–10; all bar/line values PENDING |
| 6 / `fig:sensitivity_lambda` | Abundant-label residual-weight sensitivity | `fig_lambda.png` | PDF/PNG/SVG under `figures/{yelpchi,amazon}/` | All 10 points per dataset, both metrics and SDs below |
| 7 / `fig:sensitivity_tau` | Scarce-label support saturation sensitivity | `fig_tau_scarce.png` | PDF/PNG/SVG under `figures/{yelpchi,amazon}/` | All 6 points per dataset, both metrics and SDs below |

Figures 5 and 6 remain **single-column**, with YelpChi and Amazon side by side. Each selected experimental figure has PNG/PDF/SVG versions; superseded `draft` lambda plots and alternative S line plots are not used or archived. Existing loose Amazon PNG duplicates were identified by byte equality and consolidated into their dataset folders.

## Main-table protocol and values

These tables mirror the author-written manuscript; they are not replacements for reproducible run records. Baseline values are published references, not matched reruns. EOGFD's reproduced BWGNN row and BWGNN's original row remain separate.

- Abundant training ratio: 40%; scarce: 1%, except T-Social at 0.01%.
- Amazon indices 0–3304 are excluded from train/validation/test evaluation, not graph propagation.
- The remaining eligible nodes are split into 33% validation and 67% test by the second stratified draw.
- Epoch and threshold selection: validation Macro-F1, thresholds 0.05–0.95 at 0.05 intervals (19 values).
- SOAR main results: ten reinitializations on YelpChi, Amazon and T-Finance; five on T-Social abundant and four compatible runs on T-Social scarce. No incompatible source-version seed is added to the four-run mean.
- Amazon scarce uses concatenation, a shared two-class head, band-confidence refinement, and beta=0, unlike the default mean-fusion independent-head configuration. These differences are retained rather than silently normalized away.
- Training defaults: hidden width 64; dropout 0.5; Adam learning rate 0.01; weight decay 0; maximum 100 epochs; patience 20. The exact positive-class weighting rule remains unspecified in the current experimental setup.

### Dataset statistics (`tab:data`)

| Dataset | Nodes | Adjacency entries | Relations | Features | Fraud (%) |
| --- | --- | --- | --- | --- | --- |
| YelpChi | 45,954 | 8,097,302 | 3 | 32 | 14.53 |
| Amazon | 11,944 | 9,557,648 | 3 | 25 | 6.87 |
| T-Finance | 39,357 | 42,445,086 | 1 | 10 | 4.58 |
| T-Social | 5,781,065 | 146,211,016 | 1 | 10 | 3.01 |

### Main configurations (`tab:config`)

Abundant-supervision recipe displayed in the paper. Scarce-specific settings remain in the implementation paragraph and `configs/paper.json`.

| Hyperparameter | YelpChi | Amazon | T-Finance | T-Social |
| --- | --- | --- | --- | --- |
| Filter order $S$ | 3 | 3 | 7 | 7 |
| Encoder layers | 1 | 1 | 1 | 1 |
| Residual weight $\lambda$ | 1 | 1 | 1 | 1 |
| DC weight $\beta$ | 0.05 | 0.05 | 0.10 | 0.10 |
| Support saturation $\tau_n$ | 20 | 20 | 20 | 20 |

Displayed paper-table values below follow the current TeX. `--` is unavailable / OOM / OOT / not reported. Unmarked baselines follow EOGFD. Best/second-best marks are not copied here.

### Abundant supervision: main comparison (`tab:abundant`)

| Group | Method | Venue | YelpChi AUROC | YelpChi Macro-F1 | Amazon AUROC | Amazon Macro-F1 | T-Finance AUROC | T-Finance Macro-F1 | T-Social AUROC | T-Social Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical classifiers | MLP | -- | 66.52 | 57.57 | 89.80 | 79.17 | 87.15 | 70.57 | 56.96 | 64.00 |
| Classical classifiers | SVM | -- | 70.37 | 70.77 | 90.51 | 90.71 | 78.16 | 76.23 | -- | -- |
| General-purpose GNNs | GCN | ICLR'17 | 56.51 | 54.31 | 83.49 | 67.47 | 64.43 | 70.74 | 87.35 | 59.88 |
| General-purpose GNNs | ChebyNet | NeurIPS'16 | 78.19 | 65.72 | 94.64 | 91.94 | 88.45 | 80.81 | 85.52 | 64.77 |
| General-purpose GNNs | ChebyNetII | NeurIPS'22 | 79.45 | 66.94 | 97.47 | 92.26 | 92.69 | 86.34 | 72.66 | 59.58 |
| General-purpose GNNs | GAT | ICLR'18 | 57.20 | 54.64 | 93.18 | 91.18 | 73.00 | 53.86 | 89.06 | 69.01 |
| General-purpose GNNs | GATII | ICLR'22 | 77.97 | 65.22 | 93.18 | 91.98 | 87.49 | 71.22 | -- | -- |
| General-purpose GNNs | GraphSAGE | NeurIPS'17 | 78.31 | 65.49 | 86.95 | 74.17 | 67.12 | 52.71 | 70.80 | 59.77 |
| Heterophily-aware GNNs | GPRGNN | ICLR'21 | 82.25 | 63.19 | 93.72 | 80.66 | 54.82 | 56.25 | 82.79 | 49.23 |
| Heterophily-aware GNNs | FAGCN | AAAI'21 | 74.23 | 61.18 | 95.00 | 87.29 | -- | -- | -- | -- |
| Dedicated graph fraud detection | GraphConsis | SIGIR'20 | 69.83 | 58.70 | 87.41 | 75.12 | 91.42 | 73.46 | 71.25 | 56.55 |
| Dedicated graph fraud detection | CARE-GNN | CIKM'20 | 76.19 | 63.32 | 90.67 | 86.39 | 93.79 | 82.59 | 78.91 | 51.87 |
| Dedicated graph fraud detection | PC-GNN | WWW'21 | 79.87 | 63.00 | 95.86 | 89.56 | 91.23 | 63.18 | 68.45 | 52.17 |
| Dedicated graph fraud detection | H2-FDetector | WWW'22 | 88.77 | 69.44 | 96.89 | 83.92 | 94.31 | 83.84 | 88.56 | 78.89 |
| Dedicated graph fraud detection | Grad | WWW'25 | 89.60 | 77.10 | 97.50 | 92.15 | 95.90 | 87.90 | -- | -- |
| Dedicated graph fraud detection | HUGE | AAAI'25 | 82.50 | 68.30 | 94.20 | 87.50 | 89.80 | 78.40 | 86.50 | 71.20 |
| Spectral / frequency-aware GFD | AMNet | IJCAI'22 | 78.19 | 66.93 | 94.78 | 91.25 | 88.41 | 72.81 | -- | -- |
| Spectral / frequency-aware GFD | BWGNN | ICML'22 | 89.67 | 76.44 | 97.42 | 91.72 | 94.35 | 86.87 | 95.20 | 83.98 |
| Spectral / frequency-aware GFD | GHRN | WWW'23 | 89.57 | 77.54 | 97.07 | 92.36 | 95.77 | 87.92 | 90.60 | 68.28 |
| Spectral / frequency-aware GFD | BioGNN | WWW'24 | 88.89 | 74.71 | 96.60 | 90.98 | 95.21 | 87.83 | 93.25 | 81.40 |
| Spectral / frequency-aware GFD | SEC-GFD | AAAI'24 | 83.70 | 70.62 | 97.54 | 91.87 | 96.03 | 87.75 | 94.42 | 79.35 |
| Spectral / frequency-aware GFD | EOGFD | TEVC'26 | 89.72 | 77.80 | 97.53 | 92.80 | 96.20 | 91.11 | 97.50 | 92.59 |
| Spectral / frequency-aware GFD | EGNN | ICML'26 | 88.10 | 76.89 | 96.32 | 91.52 | 95.39 | 89.60 | 99.69 | 95.40 |
| Ours | SOAR | -- | 92.28 | 79.30 | 97.82 | 93.02 | 97.10 | 92.23 | 99.12 | 94.98 |

### Scarce supervision: main comparison (`tab:scarce`)

| Group | Method | Venue | YelpChi AUROC | YelpChi Macro-F1 | Amazon AUROC | Amazon Macro-F1 | T-Finance AUROC | T-Finance Macro-F1 | T-Social AUROC | T-Social Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical classifiers | MLP | -- | 59.83 | 53.90 | 83.62 | 74.68 | 82.93 | 61.00 | 56.35 | 50.03 |
| Classical classifiers | SVM | -- | 62.92 | 60.47 | 81.62 | 83.49 | 71.47 | 67.69 | 50.06 | 57.69 |
| General-purpose GNNs | GCN | ICLR'17 | 54.06 | 52.48 | 82.85 | 67.93 | 74.37 | 57.51 | 59.61 | 52.34 |
| General-purpose GNNs | ChebyNet | NeurIPS'16 | 71.48 | 63.13 | 87.60 | 85.74 | 85.53 | 77.20 | 70.02 | 52.59 |
| General-purpose GNNs | ChebyNetII | NeurIPS'22 | 71.67 | 61.66 | 91.89 | 90.36 | 92.89 | 80.25 | 71.94 | 58.95 |
| General-purpose GNNs | GAT | ICLR'18 | 50.95 | 50.27 | 73.45 | 60.84 | 52.04 | 53.15 | 68.55 | 47.03 |
| General-purpose GNNs | GATII | ICLR'22 | 72.13 | 61.30 | 66.05 | 56.58 | 86.99 | 70.61 | 75.06 | 62.83 |
| General-purpose GNNs | GraphSAGE | NeurIPS'17 | 67.58 | 58.41 | 75.37 | 70.78 | 66.35 | 59.03 | 59.69 | 57.91 |
| Heterophily-aware GNNs | GPRGNN | ICLR'21 | 54.19 | 46.07 | 80.77 | 51.14 | 89.57 | 80.47 | 70.30 | 49.24 |
| Heterophily-aware GNNs | FAGCN | AAAI'21 | 70.92 | 46.27 | 92.07 | 84.82 | -- | -- | -- | -- |
| Dedicated graph fraud detection | GraphConsis | SIGIR'20 | 66.41 | 56.79 | 74.11 | 68.59 | 90.28 | 71.73 | 65.29 | 52.45 |
| Dedicated graph fraud detection | CARE-GNN | CIKM'20 | 71.96 | 61.25 | 63.87 | 88.30 | 90.73 | 76.25 | 68.32 | 53.31 |
| Dedicated graph fraud detection | PC-GNN | WWW'21 | 72.47 | 59.82 | 90.40 | 79.86 | 90.76 | 62.06 | 59.84 | 51.14 |
| Dedicated graph fraud detection | H2-FDetector | WWW'22 | 71.92 | 63.45 | 90.05 | 80.10 | 90.65 | 83.27 | -- | -- |
| Dedicated graph fraud detection | Grad | WWW'25 | 71.50 | 65.50 | 91.90 | 91.00 | 93.00 | 85.00 | -- | -- |
| Dedicated graph fraud detection | HUGE | AAAI'25 | 68.50 | 58.60 | 85.30 | 79.20 | 84.70 | 72.50 | 72.80 | 61.40 |
| Spectral / frequency-aware GFD | AMNet | IJCAI'22 | 72.56 | 62.83 | 89.14 | 88.56 | 87.28 | 70.84 | -- | -- |
| Spectral / frequency-aware GFD | BWGNN | ICML'22 | 72.71 | 65.23 | 89.45 | 90.92 | 91.15 | 84.89 | 84.40 | 75.93 |
| Spectral / frequency-aware GFD | GHRN | WWW'23 | 71.76 | 64.30 | 90.27 | 89.16 | 89.18 | 73.20 | 83.28 | 65.37 |
| Spectral / frequency-aware GFD | BioGNN | WWW'24 | 72.57 | 64.93 | 91.46 | 84.36 | 92.53 | 83.14 | 84.15 | 76.75 |
| Spectral / frequency-aware GFD | SEC-GFD | AAAI'24 | 72.29 | 61.91 | 88.25 | 91.26 | 93.42 | 87.63 | 73.81 | 64.14 |
| Spectral / frequency-aware GFD | EOGFD | TEVC'26 | 71.67 | 65.99 | 92.66 | 91.49 | 93.70 | 87.78 | 85.24 | 80.37 |
| Ours | SOAR | -- | 78.21 | 67.07 | 91.83 | 90.05 | 95.05 | 90.75 | 94.34 | 85.13 |

### YelpChi component results (`tab:ablation_yelp`)

| Variant | 40% AUROC | 40% Macro-F1 | 1% AUROC | 1% Macro-F1 |
| --- | --- | --- | --- | --- |
| SOAR (Full) | 92.28±0.17 | 79.30±0.23 | 78.21±0.60 | 67.07±0.39 |
| Magnitude | 91.32±0.18 | 79.11±0.27 | 73.85±0.56 | 63.85±0.63 |
| Orientation off | 90.21±0.26 | 77.75±0.27 | 73.65±0.29 | 63.52±0.61 |
| Reliability off | 91.81±0.11 | 78.92±0.28 | 76.16±0.62 | 65.68±0.55 |
| Signed only | 77.87±0.32 | 65.89±0.33 | 73.74±1.33 | 62.95±1.45 |

### Amazon component results (`tab:ablation_amazon`)

| Variant | 40% AUROC | 40% Macro-F1 | 1% AUROC | 1% Macro-F1 |
| --- | --- | --- | --- | --- |
| SOAR (Full) | 97.82±0.19 | 93.02±0.32 | 91.83±1.05 | 90.05±0.64 |
| Magnitude | 97.75±0.18 | 92.81±0.61 | 91.03±1.39 | 89.56±0.80 |
| Orientation off | 97.68±0.22 | 93.02±0.24 | 90.86±1.10 | 89.81±0.68 |
| Reliability off | 97.85±0.20 | 92.99±0.58 | 90.91±1.54 | 88.51±1.34 |
| Signed only | 95.56±0.07 | 87.03±0.26 | 92.12±0.75 | 87.80±1.13 |

The component summaries are the existing tested controls, not newly constructed one-factor interventions. In particular, Magnitude also modifies the readout and disables directional consistency where required. No standalone DC ablation row is inferred from these tables.

## Frozen sensitivity summaries

Source: author-provided **Frozen sensitivity figures** handoff. Ten reinitializations per setting; split seed 2. Means and population SDs are archived at the supplied precision. These settings are diagnostic locks, not assertions that every highlighted value optimizes both metrics.

| Dataset / sweep | Source experiment ID | Training ratio / graph | Fixed parameters | Reference |
| --- | --- | --- | --- | --- |
| YelpChi lambda | `s09_r27_yelp_ab_lam10_b0p1` | 40%, multi-relation | beta=0.1, tau_n=20 | lambda=4 |
| YelpChi beta | `s09_r25_yelp_ab_lam4_beta` | 40%, multi-relation | lambda=4, tau_n=20 | beta=0.1 |
| YelpChi tau_n | `s09_r29_yelp_sc_tau` | 1%, multi-relation | lambda=4, beta=0.1 | tau_n=20 |
| Amazon lambda | `s09_r28_am_ab_lam10` | 40%, hetero | beta=0.1, tau_n=20 | lambda=1 |
| Amazon tau_n | `s09_r30_am_sc_tau` | 1%, hetero, r19 scarce configuration | lambda=2, beta=0 | tau_n=20 |

Amazon tau_n=20 reuses `s09_r19` Full seeds 0–9, according to the handoff. Its source and mask hashes have not been supplied, so compatibility is recorded as the author's provenance statement, not independently verified here. The handoff's high-level Amazon lock table leaves beta unspecified; the specific Amazon lambda source row explicitly states beta=0.1, which is the value recorded above. There is no Amazon beta sweep in this archive.

### YelpChi lambda — Figure 5, left

| lambda | AUROC | Macro-F1 |
| ---: | ---: | ---: |
| 0 | 90.14±0.12 | 77.75±0.26 |
| 1 | 92.29±0.14 | 79.37±0.30 |
| 2 | 92.30±0.17 | 79.42±0.30 |
| 3 | 92.32±0.12 | 79.42±0.22 |
| 4 (reference) | 92.27±0.15 | 79.44±0.28 |
| 5 | 92.28±0.15 | 79.42±0.36 |
| 6 | 92.27±0.12 | 79.50±0.18 |
| 7 | 92.25±0.14 | 79.46±0.33 |
| 8 | 92.21±0.16 | 79.38±0.33 |
| 9 | 92.20±0.11 | 79.36±0.23 |

### YelpChi beta — Figure 3, performance summaries

| beta | AUROC | Macro-F1 |
| ---: | ---: | ---: |
| 0 | 92.30±0.17 | 79.50±0.34 |
| 0.01 | 92.32±0.16 | 79.45±0.36 |
| 0.05 | 92.29±0.12 | 79.48±0.28 |
| 0.1 (reference) | 92.27±0.15 | 79.44±0.28 |
| 0.5 | 92.06±0.16 | 79.07±0.40 |
| 1 | 91.82±0.22 | 78.68±0.37 |

**PENDING:** the epoch-by-epoch reliability-weighted alignment for each beta, aggregation/smoothing details, and ten seed-level metric observations per beta (or exact box statistics). The six mean/SD pairs do not reproduce the left trajectories or the right box-plot quartiles/whiskers.

### YelpChi tau_n — Figure 6, left

| tau_n | AUROC | Macro-F1 |
| ---: | ---: | ---: |
| 1 | 77.56±0.54 | 66.66±0.42 |
| 5 | 77.53±0.56 | 66.63±0.43 |
| 10 | 77.56±0.53 | 66.65±0.41 |
| 20 (reference) | 77.54±0.53 | 66.68±0.41 |
| 50 | 76.58±0.54 | 66.11±0.60 |
| 100 | 75.30±0.54 | 64.85±0.65 |

### Amazon lambda — Figure 5, right

| lambda | AUROC | Macro-F1 |
| ---: | ---: | ---: |
| 0 | 97.73±0.20 | 92.95±0.39 |
| 1 (reference) | 97.86±0.15 | 92.93±0.34 |
| 2 | 97.78±0.17 | 92.94±0.24 |
| 3 | 97.83±0.15 | 92.57±0.40 |
| 4 | 97.92±0.12 | 92.66±0.47 |
| 5 | 97.82±0.18 | 92.75±0.67 |
| 6 | 97.86±0.12 | 92.45±0.45 |
| 7 | 97.90±0.14 | 92.32±0.39 |
| 8 | 97.89±0.18 | 92.64±0.66 |
| 9 | 97.81±0.17 | 92.42±0.49 |

### Amazon tau_n — Figure 6, right

| tau_n | AUROC | Macro-F1 |
| ---: | ---: | ---: |
| 1 | 90.69±1.46 | 88.56±1.40 |
| 5 | 90.62±1.72 | 88.49±1.37 |
| 10 | 90.66±1.43 | 88.76±1.28 |
| 20 (reference) | 91.83±1.05 | 90.05±0.64 |
| 50 | 91.06±1.25 | 89.88±0.93 |
| 100 | 91.05±1.38 | 89.91±0.62 |

## Pending numeric sources

| Figure / data | Available now | Still required for exact reproduction |
| --- | --- | --- |
| Fig. 1 motivation | Original PNG and TeX caption | Energy, relevance and orientation arrays; relation/band ordering; source/configuration |
| Fig. 2 reliability | Original PNG; caption specifies 1,000 draws | Label-budget grid, all relation/band stability values, reference statistics and sampling details |
| Fig. 3 DC alignment | PNG/PDF/SVG; beta performance mean/SD tables | Epoch trajectories, per-seed distributions, aggregation/smoothing details |
| Fig. 4 S bars | PNG/PDF/SVG; S=1,…,10; both datasets | Exact AUROC/Macro-F1 bar values, source ID/configuration/split role, and numeric mapping for the dashed average line |
| Figs. 5–6 lambda/tau | All plotted mean/SD summaries for both datasets | Original per-seed records and executable-source/mask hashes |

The archived figures remain unchanged. Missing numeric values are not interpolated, digitized, or substituted from another plot version. The S figure contains a dashed average curve; its exact axis mapping should be confirmed from the original plotting source before regenerating it.

## Asset checksums

SHA-256 below verifies archived image identity only. Paths are relative to `manuscript/`.

| File | SHA-256 |
| --- | --- |
| `figures/amazon/pdf/fig_lambda.pdf` | `df15a6cbfd9b5fb4ff05da291d32e7bb349b30456c8d9ae198029bcf6386b21c` |
| `figures/amazon/pdf/fig_tau.pdf` | `e0fc1f5a12ddc6a094ae5ac8f5b32eedb22e6ab847f3feccfed12f9d4444b652` |
| `figures/amazon/png/fig_lambda.png` | `b7e7b7a6036dd98305cddc4b1d43b96fe76cee6a3928247d421fa518b57d2ad7` |
| `figures/amazon/png/fig_tau.png` | `3e52e4de536071cf3db0584269054cd44315e2babc5ccc702b34c109a84a9d4d` |
| `figures/amazon/svg/fig_lambda.svg` | `adce6dc4d350cf3c8d29c7599d2767b0207988f0b36dc55b9d2de0c839aa4729` |
| `figures/amazon/svg/fig_tau.svg` | `57cf3eb71962c5e48c910ed2c25e4667432cb47d880c5ba84c1729bfaf54c765` |
| `figures/backbone/pdf/fig_S_bars.pdf` | `e6f3db701a6d457adf89a1215bcc72ddeedceffb018e0a2a784c6cad3cd5d879` |
| `figures/backbone/png/fig_S_bars.png` | `58dcef2b7ffbc78a5d0d38b8dfe07ba16315a7b6ac21ae0de2c9e632b8e633a0` |
| `figures/backbone/svg/fig_S_bars.svg` | `4d150529b76f2a30d213f265189ae59327782b8527c85e07f73bedb0d2005a1e` |
| `fig_motivation.png` | `8ded1fdaac40109f7c638c86ecd3e7c111bd7159ac19ada802f068f352262559` |
| `fig_reliability.png` | `9c950f9cea3f16cae371f4f3b254a76292c35a9559398fe516786e9ae52546a8` |
| `fig_framework.pdf` | `b6ab6eac5316600a0343ef255c7585e18a779b9adcafc2ef66f386edf0d8ace6` |
| `fig_beta.png` | `8e42abdd5ecae1996df277a6a4ba369870c0bd892244c7d1c3a83139af024602` |
| `fig_S_bars.png` | `3ef9a0e629d834a43a824f8c8a3efd0019f8df5bc27e9d113395c0539e966eba` |
| `fig_lambda.png` | `b56d14044db2af1650157440cc864f90cc4dc1859a8027f505c12a4346d53cbc` |
| `fig_tau_scarce.png` | `e85d0e3906384ed6afc81a6257626aa3902cdb21cd95ad88255f3de8c00c7ca9` |
| `figures/yelpchi/pdf/fig_beta_dc.pdf` | `a16549075bdd76eccbecb8b1acf94418cb7e8d503d95b8a7114edf4786177013` |
| `figures/yelpchi/pdf/fig_lambda.pdf` | `5a5ca4fcfa25fe79f3d6e89ae427e6b7b5361342d248e023f0a0851d15f9987e` |
| `figures/yelpchi/pdf/fig_tau.pdf` | `68d0bbbaa6f13b5bf77978ae6b8b5b69e40fc7ab612723a402cf1eb5ccc4cbe0` |
| `figures/yelpchi/png/fig_beta_dc.png` | `d2d9823dd6d4ed028a9edbb15280a065a4746346a982b68c5931d164c71a23e1` |
| `figures/yelpchi/png/fig_lambda.png` | `f155268aaa55235bc79c49bb5f4109b2f95983ef672150552765c04768933010` |
| `figures/yelpchi/png/fig_tau.png` | `951c8843b458055d33b25f83e16040a5df86f03ff0bb1358e73b58307288e320` |
| `figures/yelpchi/svg/fig_beta_dc.svg` | `ae4652d8cadf0b6ef5e31c27320e6585a864df8ad9dbfb39767732d2df275f5e` |
| `figures/yelpchi/svg/fig_lambda.svg` | `edc2ee7e3b371622b39c6a41e8b4b94f693a87c878fde248df6326dcb44b5e96` |
| `figures/yelpchi/svg/fig_tau.svg` | `05f558e3404f5298eab4ee75604e7f66854d065bccec37b385a64a5dffb2258b` |
