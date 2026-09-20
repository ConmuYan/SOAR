# Data

Datasets are **not** redistributed. Download the BWGNN fraud graphs and place them here:

| Dataset | Path | Source |
| --- | --- | --- |
| YelpChi | `data/YelpChi.mat` | [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) |
| Amazon | `data/Amazon.mat` | same |
| T-Finance | `data/tfinance` | DGL binary from BWGNN / DGL fraud graphs |
| T-Social | `data/tsocial` | same |

Frozen splits are written automatically on the first training run, or in bulk:

```bash
python protocol.py --all
```

`--scarce` on T-Social is the paper's 0.01% label setting. The split seed is 2 and cannot vary.
